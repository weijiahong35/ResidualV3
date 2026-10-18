"""Read-only worker RPC snapshots outside performance windows; no idle signals."""
import torch
from runtime.utils import now


def install():
    from vllm.v1.worker.gpu_worker import Worker
    def snapshot(self):
        from runtime.native import MODEL
        assert MODEL is not None
        assert not torch.cuda.is_current_stream_capturing()
        torch.cuda.synchronize()
        h=MODEL.head._tail_aggregate.detach().cpu().tolist() if MODEL.head is not None else [0]*16
        return dict(counts=h,blocks=sum(h),repaired_positions=sum(i*v for i,v in enumerate(h)),
                    scope='cumulative device counters; includes warmup; window deltas only',utc=now())
    Worker.repair_tail_snapshot=snapshot
    from fastapi import APIRouter,Request
    from vllm.entrypoints.openai import api_server
    previous=api_server.build_app
    router=APIRouter()
    @router.get('/repair_tail_stats')
    async def stats(request:Request):
        result=await request.app.state.engine_client.collective_rpc(method='repair_tail_snapshot',timeout=30.)
        assert len(result)==1
        return result[0]
    def build(*args,**kwargs):
        app=previous(*args,**kwargs);app.include_router(router);return app
    api_server.build_app=build
