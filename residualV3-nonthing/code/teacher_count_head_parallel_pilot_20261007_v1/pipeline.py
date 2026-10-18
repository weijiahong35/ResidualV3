"""Two RAM packs, direct pinned packing, and bounded asynchronous H2D staging."""
from dataclasses import dataclass
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Lock, Thread
import time
import torch


@dataclass
class HostPack:
    ids: torch.Tensor
    docs: torch.Tensor
    positions: torch.Tensor
    eligible: torch.Tensor
    aux: torch.Tensor
    final: torch.Tensor
    offset: int


@dataclass
class DevicePack:
    ids: torch.Tensor
    docs: torch.Tensor
    positions: torch.Tensor
    eligible: torch.Tensor
    context: torch.Tensor
    last: torch.Tensor
    offset: int
    ready: object
    host: object
    timing: dict

    def wait(self):
        stream = torch.cuda.current_stream(self.context.device)
        stream.wait_event(self.ready)
        for value in [self.ids, self.docs, self.positions, self.eligible, self.context, self.last]:
            value.record_stream(stream)

    def release(self):
        # Called after the consumer's body stream has completed.
        self.ready.synchronize()
        self.host = None


def allocate_host(rows, budget, width=4096, layers=6, pinned=True):
    total = sum(len(row['input_ids']) for row in rows)
    assert 0 < total <= budget
    options = dict(device='cpu', pin_memory=pinned)
    ids = torch.zeros((1, budget), dtype=torch.long, **options)
    docs = torch.full((1, budget), -1, dtype=torch.long, **options)
    positions = torch.zeros((1, budget), dtype=torch.long, **options)
    aux = torch.empty((1, budget, (layers - 1) * width), dtype=torch.bfloat16, **options)
    final = torch.empty((total, width), dtype=torch.bfloat16, **options)
    if total < budget:
        aux[:, total:].zero_()
    eligible, offset = [], 0
    for document, row in enumerate(rows):
        size = len(row['input_ids'])
        ids[0, offset:offset + size] = torch.tensor(row['input_ids'])
        docs[0, offset:offset + size] = document
        positions[0, offset:offset + size] = torch.arange(size)
        eligible.extend(offset + i for i, value in enumerate(row['loss_mask'][:-16]) if value)
        offset += size
    assert eligible
    choices = torch.tensor(eligible, dtype=torch.long, **options)
    return HostPack(ids, docs, positions, choices, aux, final, total)


def fill_host(host, rows, fetch):
    offset = 0
    for row in rows:
        hidden = fetch(row)
        size = len(row['input_ids'])
        assert hidden.shape == (size, host.aux.shape[-1] // host.final.shape[-1] + 1,
                                host.final.shape[-1])
        assert hidden.device.type == 'cpu' and hidden.dtype == torch.bfloat16
        host.aux[0, offset:offset + size].copy_(hidden[:, :-1].flatten(1))
        host.final[offset:offset + size].copy_(hidden[:, -1])
        offset += size
        del hidden
    assert offset == host.offset


@torch.no_grad()
def upload(host, device, stream, target, draft, timing=None):
    with torch.cuda.stream(stream):
        tensors = [x.to(device, non_blocking=True) for x in
                   [host.ids, host.docs, host.positions, host.eligible]]
        aux = host.aux.to(device, non_blocking=True)
        final = host.final.to(device, non_blocking=True)
        context = draft.hidden_norm(draft.fc(aux))
        last = target.model.norm(final)
        tensors.extend([context, last])
        del aux, final
        ready = torch.cuda.Event()
        ready.record(stream)
    return DevicePack(*tensors, host.offset, ready, host, timing or {})


class PackPipeline:
    def __init__(self, provider, train, packs, depth=2, budget=65536, *, target, draft):
        assert depth == 2
        self.provider, self.train, self.packs = provider, train, packs
        self.budget = budget
        self.target, self.draft = target, draft
        self.stage_lock = Lock()
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.stream = torch.cuda.default_stream(device=self.device)
        self.queue = Queue(maxsize=depth)
        self.slots = BoundedSemaphore(depth)
        self.cancel, self.finished = Event(), Event()
        self.lock = Lock()
        self.error = None
        self.resident = self.peak_resident = 0
        provider.begin_epoch([train[index] for indices in packs for index in indices])
        self.thread = Thread(target=self._produce, name='pinned-pack-upload', daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            torch.cuda.set_device(self.device)
            for indices in self.packs:
                while not self.cancel.is_set():
                    if self.slots.acquire(timeout=.1):
                        break
                else:
                    return
                with self.lock:
                    self.resident += 1
                    self.peak_resident = max(self.peak_resident, self.resident)
                rows = [self.train[index] for index in indices]
                start = time.perf_counter()
                host = allocate_host(rows, self.budget)
                allocated = time.perf_counter()
                if self.cancel.is_set():
                    return
                fill_host(host, rows, self.provider)
                packed = time.perf_counter()
                if self.cancel.is_set():
                    return
                with self.stage_lock:
                    if self.cancel.is_set():
                        return
                    item = upload(host, self.device, self.stream, self.target, self.draft,
                                  dict(host_allocate_seconds=allocated-start,
                                       host_fill_seconds=packed-allocated))
                    item.ready.synchronize()
                    item.host = None
                del host
                self.queue.put((indices, rows, item))
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()

    def get(self):
        while True:
            try:
                result = self.queue.get(timeout=.1)
                result[2].wait()
                return result
            except Empty:
                if self.finished.is_set():
                    if self.error is not None:
                        raise self.error
                    raise RuntimeError('Pack upload pipeline exhausted unexpectedly')

    def release(self):
        with self.lock:
            self.resident -= 1
        self.slots.release()

    def close(self):
        self.cancel.set()
        self.thread.join()
        self.stream.synchronize()
        while True:
            try:
                item = self.queue.get_nowait()[2]
                item.release()
            except Empty:
                break
