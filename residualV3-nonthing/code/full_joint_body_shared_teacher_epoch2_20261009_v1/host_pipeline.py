"""Two bounded CPU packs; trainable context compression stays on the learner."""
import importlib.util
from pathlib import Path
from queue import Queue,Empty
from threading import BoundedSemaphore,Event,Thread
import time
import torch

_path=Path(__file__).resolve().parent.parent/'teacher_count_head_parallel_pilot_20261007_v1/pipeline.py'
_spec=importlib.util.spec_from_file_location('frozen_host_pack_helpers',_path)
_helpers=importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name]=_helpers;_spec.loader.exec_module(_helpers)

class HostPipeline:
    def __init__(self,provider,source,packs):
        self.provider,self.source,self.packs=provider,source,packs
        self.queue=Queue(maxsize=2);self.slots=BoundedSemaphore(2)
        self.cancel=Event();self.finished=Event();self.error=None
        provider.begin_epoch([source[i] for indices in packs for i in indices])
        self.thread=Thread(target=self.produce,daemon=True);self.thread.start()
    def produce(self):
        try:
            for indices in self.packs:
                while not self.cancel.is_set():
                    if self.slots.acquire(timeout=.1):break
                else:return
                rows=[self.source[i] for i in indices];started=time.perf_counter()
                host=_helpers.allocate_host(rows,65536)
                allocated=time.perf_counter();_helpers.fill_host(host,rows,self.provider)
                timing=dict(host_allocate_seconds=allocated-started,host_fill_seconds=time.perf_counter()-allocated)
                while not self.cancel.is_set():
                    try:self.queue.put((indices,rows,host,timing),timeout=.1);break
                    except __import__('queue').Full:pass
                del host
        except BaseException as exc:self.error=exc
        finally:self.finished.set()
    def get(self):
        while True:
            try:return self.queue.get(timeout=.1)
            except Empty:
                if self.finished.is_set():
                    if self.error is not None:raise self.error
                    raise RuntimeError('Host pipeline exhausted')
    def release(self):self.slots.release()
    def close(self):
        self.cancel.set();self.thread.join(timeout=300)
        assert not self.thread.is_alive(),'Host pipeline did not finish'
        while not self.queue.empty():self.queue.get_nowait()

@torch.no_grad()
def upload(host,target):
    begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize();wall=time.perf_counter();begin.record()
    ids,docs,positions,eligible,aux,final=[x.to('cuda',non_blocking=True) for x in [host.ids,host.docs,host.positions,host.eligible,host.aux,host.final]]
    last=target.model.norm(final);del final
    end.record();end.synchronize()
    return dict(ids=ids,docs=docs,positions=positions,eligible=eligible,aux=aux,last=last,offset=host.offset),dict(h2d_wall_seconds=time.perf_counter()-wall,h2d_cuda_seconds=begin.elapsed_time(end)/1000)
