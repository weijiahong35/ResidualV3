"""Atomic complete training checkpoints, including FP32 master weights and Adam."""
import os
from pathlib import Path
import torch
from runtime.utils import dump, sha, now


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_checkpoint(folder, head, opt, step, protocol_sha256, reason):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=1, step=step, protocol_sha256=protocol_sha256,
                 head=cpu_tree(head.state_dict()),
                 masters={name: cpu_tree(master) for name, _, master in opt.pairs},
                 optimizer=cpu_tree(opt.optimizer.state_dict()),
                 cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state() if next(head.parameters()).is_cuda else None,
                 utc=now(), reason=reason)
    temp = folder / 'resume_writing.pt'
    with temp.open('wb') as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    latest, previous = folder / 'resume_latest.pt', folder / 'resume_previous.pt'
    if latest.exists():
        pending_previous = folder / 'resume_previous.partial'
        pending_previous.unlink(missing_ok=True)
        os.link(latest, pending_previous)
        os.replace(pending_previous, previous)
    os.replace(temp, latest)
    directory = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    dump(folder / 'checkpoint_manifest.json', dict(step=step, path=str(latest), sha256=sha(latest),
         previous_path=str(previous) if previous.exists() else None,
         previous_sha256=sha(previous) if previous.exists() else None,
         includes=['BF16 head', 'FP32 masters', 'Adam state', 'step', 'RNG'], reason=reason, utc=now()))


def load_checkpoint(folder, head, opt, protocol_sha256):
    folder = Path(folder)
    for name in ('resume_latest.pt', 'resume_previous.pt'):
        path = folder / name
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert state['format_version'] == 1 and state['protocol_sha256'] == protocol_sha256
            assert 20000 <= state['step'] <= 30000
            assert set(state['masters']) == {n for n, _, _ in opt.pairs}
            assert all(torch.isfinite(t).all() for t in state['head'].values())
            assert all(torch.isfinite(t).all() for t in state['masters'].values())
            for key, value in head.state_dict().items():
                assert value.shape == state['head'][key].shape
            head.load_state_dict(state['head'], strict=True)
            with torch.no_grad():
                for n, parameter, master in opt.pairs:
                    master.copy_(state['masters'][n])
                    assert torch.equal(parameter.cpu(), master.to(parameter.dtype).cpu())
            opt.optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'])
            if state['cuda_rng'] is not None:
                torch.cuda.set_rng_state(state['cuda_rng'])
            return state['step'], str(path)
        except Exception as exc:
            dump(folder / (name + '.load_error.json'), dict(error=repr(exc), utc=now()))
    raise RuntimeError('No valid complete recovery checkpoint; refusing to restart from epoch2 silently')
