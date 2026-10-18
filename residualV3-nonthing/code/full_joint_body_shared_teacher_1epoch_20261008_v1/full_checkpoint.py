"""Two preallocated slots; commit the paired cursor only after both fsyncs.

FP32 masters reproduce live BF16 weights exactly at update boundaries. Storing
them once avoids a redundant BF16 copy in each complete optimizer checkpoint.
The bounded reader hides the unused preallocated tail from PyTorch's ZIP reader.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from joint_model import cpu_tree


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def digest_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def digest_tree(value):
    if torch.is_tensor(value):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=digest_tensor(value))
    if isinstance(value, dict):
        return {str(key): digest_tree(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [digest_tree(item) for item in value]
    return value


def prefix_sha(path, size):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = size
        while remaining:
            block = stream.read(min(8 * 2**20, remaining))
            assert block, 'Truncated checkpoint'
            result.update(block); remaining -= len(block)
    return result.hexdigest()


class BoundedReader(io.BufferedIOBase):
    def __init__(self, path, size):
        super().__init__(); self.file = Path(path).open('rb'); self.size = size
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.file.tell()
    def seek(self, offset, whence=0):
        if whence == 2: offset += self.size; whence = 0
        position = self.file.seek(offset, whence)
        assert 0 <= position <= self.size
        return position
    def read(self, size=-1):
        remaining = self.size - self.tell()
        return self.file.read(remaining if size < 0 else min(size, remaining))
    def readinto(self, buffer):
        value = self.read(len(buffer)); buffer[:len(value)] = value; return len(value)
    def close(self):
        if not self.closed: self.file.close()
        super().close()


def reserve(root, capacity_gib):
    root = Path(root); items = []
    for slot in [0, 1]:
        directory = root / 'checkpoints' / f'slot_{slot}'
        directory.mkdir(exist_ok=True)
        for arm in ['joint', 'body']:
            path = directory / f'{arm}.state.bin'
            if not path.exists():
                with path.open('xb') as stream:
                    os.posix_fallocate(stream.fileno(), 0, int(capacity_gib * 2**30))
                    os.fsync(stream.fileno())
            assert path.stat().st_size == int(capacity_gib * 2**30)
            items.append(dict(path=str(path), bytes=path.stat().st_size))
    for arm in ['joint', 'body']:
        path = root / 'final' / f'{arm}.weights.reserve'
        if not path.exists() and not (root / 'final' / f'{arm}_epoch_01.safetensors').exists():
            with path.open('xb') as stream: os.posix_fallocate(stream.fileno(), 0, 2 * 2**30)
        if path.exists(): items.append(dict(path=str(path), bytes=path.stat().st_size))
    emergency = root / 'metadata_emergency.reserve'
    if not emergency.exists():
        with emergency.open('xb') as stream: os.posix_fallocate(stream.fileno(), 0, 2**28)
    atomic(root / 'disk_reservation.json', dict(status='reserved', items=items,
           complete_state_slots=2, old_experiment_weights_preserved=True,
           format='Bounded PyTorch state inside preallocated files; use restore()'))


def snapshot(opt, root, cursor, protocol_sha, order_sha, plan_sha):
    masters = {name: value.detach().cpu() for name, _, value in opt.pairs}
    parameters = {}
    for name, value, _ in opt.pairs:
        live = value.detach().cpu()
        assert torch.equal(live, masters[name].to(live.dtype)), name
        parameters[name] = digest_tensor(live)
    optimizers = [cpu_tree(value.state_dict()) for value in opt.optimizers]
    rng = np.random.get_state()
    return dict(format_version=1, cursor=cursor, step=cursor,
                protocol_sha256=protocol_sha, order_sha256=order_sha,
                plan_sha256=plan_sha, masters=masters, optimizers=optimizers,
                parameter_hashes=parameters,
                master_hashes={k: digest_tensor(v) for k, v in masters.items()},
                optimizer_hashes=digest_tree(optimizers),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=dict(name=rng[0], keys=rng[1].tolist(), pos=int(rng[2]),
                               gaussian=int(rng[3]), cached=float(rng[4])))


def save(root, arm, slot, cursor, opt, proof):
    root = Path(root); path = root / 'checkpoints' / f'slot_{slot}' / f'{arm}.state.bin'
    start = time.monotonic()
    state = snapshot(opt, root, cursor, proof['protocol_sha256'],
                     proof['order_sha256'], proof['plan_sha256'])
    with path.open('r+b') as stream:
        stream.seek(0); torch.save(state, stream); size = stream.tell()
        assert size <= path.stat().st_size, 'Checkpoint exceeded reserved capacity'
        stream.flush(); os.fsync(stream.fileno())
    descriptor = dict(path=str(path), logical_bytes=size, reserved_bytes=path.stat().st_size,
                      sha256=prefix_sha(path, size), arm=arm, slot=slot,
                      cursor=cursor, step=cursor, full_masters_optimizers_rng=True,
                      actual_bf16_weights_reconstruct_exactly=True,
                      parameter_hashes=state['parameter_hashes'],
                      seconds=time.monotonic() - start, **proof)
    del state
    atomic(path.with_suffix('.json'), descriptor)
    return descriptor


def restore(descriptor, opt, proof):
    assert all(descriptor[key] == value for key, value in proof.items())
    path = Path(descriptor['path']); size = descriptor['logical_bytes']
    assert prefix_sha(path, size) == descriptor['sha256'], 'Checkpoint checksum mismatch'
    with BoundedReader(path, size) as stream:
        state = torch.load(stream, map_location='cpu', weights_only=True)
    assert state['cursor'] == descriptor['cursor']
    assert set(state['masters']) == {name for name, _, _ in opt.pairs}
    with torch.no_grad():
        for name, value, master in opt.pairs:
            master.copy_(state['masters'][name]); value.copy_(master)
            assert digest_tensor(value) == state['parameter_hashes'][name], name
            assert digest_tensor(master) == state['master_hashes'][name], name
    assert len(state['optimizers']) == len(opt.optimizers)
    for optimizer, saved in zip(opt.optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
    assert digest_tree([value.state_dict() for value in opt.optimizers]) == state['optimizer_hashes']
    torch.set_rng_state(state['cpu_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
    random.setstate(state['python_rng'])
    rng = state['numpy_rng']
    np.random.set_state((rng['name'], np.asarray(rng['keys'], dtype=np.uint32),
                         rng['pos'], rng['gaussian'], rng['cached']))
    assert torch.equal(torch.get_rng_state(), state['cpu_rng'])
    assert torch.equal(torch.cuda.get_rng_state(), state['cuda_rng'])
    assert random.getstate() == state['python_rng']
    current = np.random.get_state()
    assert current[0] == rng['name'] and np.array_equal(current[1], rng['keys'])
    assert current[2:] == (rng['pos'], rng['gaussian'], rng['cached'])
    return dict(status='passed', cursor=state['cursor'],
                all_bf16_parameters_bitwise_equal=True, all_fp32_masters_bitwise_equal=True,
                all_optimizer_tensors_and_groups_bitwise_equal=True,
                cpu_cuda_python_numpy_rng_restored=True)


def commit(root, descriptors, cursor):
    root = Path(root); assert set(descriptors) == {'joint', 'body'}
    slots = {value['slot'] for value in descriptors.values()}; assert len(slots) == 1
    for arm, value in descriptors.items():
        assert value['cursor'] == cursor and value['arm'] == arm
        assert Path(value['path']).stat().st_size >= value['logical_bytes']
        assert json.loads(Path(value['path']).with_suffix('.json').read_text()) == value
    pointer = dict(status='committed', cursor=cursor, step=cursor, slot=slots.pop(),
                   artifacts=descriptors, joint_and_body_same_data_cursor=True,
                   committed_unix_time=time.time())
    atomic(root / 'latest_checkpoint.json', pointer)
    return pointer


def final_weights(root, arm, opt):
    from safetensors.torch import save as encode
    root = Path(root); path = root / 'final' / f'{arm}.weights.reserve'
    destination = root / 'final' / f'{arm}_epoch_01.safetensors'
    if destination.exists(): return str(destination)
    values = {name: value.detach().cpu().contiguous() for name, value, _ in opt.pairs}
    encoded = encode(values); assert len(encoded) <= path.stat().st_size
    with path.open('r+b') as stream:
        stream.write(encoded); stream.truncate(len(encoded)); stream.flush(); os.fsync(stream.fileno())
    path.replace(destination)
    atomic(root / 'final' / f'{arm}_manifest.json', dict(path=str(destination),
           sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
           dtype='BF16', includes_all_trainable_body_and_head_parameters=True,
           frozen_assets_referenced_in_protocol=True, epochs=1))
    return str(destination)
