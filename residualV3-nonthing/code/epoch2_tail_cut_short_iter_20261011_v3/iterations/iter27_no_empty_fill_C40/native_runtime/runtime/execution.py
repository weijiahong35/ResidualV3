"""Explicit repair-only CUDA Graph. Target/drafter body stays eager in every arm."""
import torch
import os
from .observe import scope
from .events import capture_events

ARMS=('dense_combined','static_dense')
GRAPH_ARMS=ARMS
SUITES={'static_joint':ARMS}

def run_eager(*args, **kwargs):
    from repair.native_launch import run_eager as current
    return current(*args, **kwargs)

class RepairGraph:
    def __init__(self,head,weight,basis,*,arm,capture=False,fp64=False):
        self.head=head;self.weight=weight;self.basis=basis;self.arm=arm;self.capture=capture;self.fp64=fp64
        self.graph=None;self.captures=0;self.replays=0;self.fallbacks=0
        self.replay_events=[torch.cuda.Event(enable_timing=True) for _ in range(3)] if os.environ.get("DIAG_KIND")=="profile" else None
    def _signature(self,inputs,q):
        return tuple((tuple(t.shape),t.dtype,t.device) for t in inputs)+(tuple(q.shape),q.dtype,q.data_ptr())
    def __call__(self,inputs,q):
        signature=self._signature(inputs,q)
        if self.graph is None:
            self.signature=signature;self.static=tuple(x.clone() for x in inputs)
            current=torch.cuda.current_stream(self.weight.device);side=torch.cuda.Stream(device=self.weight.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(3):run_eager(self.head,self.weight,self.basis,self.static,q,arm=self.arm,fp64=self.fp64,capture=self.capture)
            current.wait_stream(side);torch.cuda.synchronize(self.weight.device)
            self.graph=torch.cuda.CUDAGraph()
            self.stage_events=[]
            with capture_events(self.stage_events):
                with torch.cuda.graph(self.graph,stream=side):
                    self.result=run_eager(self.head,self.weight,self.basis,self.static,q,arm=self.arm,fp64=self.fp64,capture=self.capture)
            current.wait_stream(side);self.captures+=1
        elif signature!=self.signature:
            raise RuntimeError('Graph input shape/q storage changed; no silent eager fallback')
        # Input copies are part of every invocation and measured. Seed/position/anchor are never baked in.
        if self.replay_events:self.replay_events[0].record()
        with scope('repair_input_copy'):
            for dst,src in zip(self.static,inputs):dst.copy_(src)
        if self.replay_events:self.replay_events[1].record()
        with scope('repair_graph_replay'):self.graph.replay()
        if self.replay_events:self.replay_events[2].record()
        self.replays+=1
        return self.result

