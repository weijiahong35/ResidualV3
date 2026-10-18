"""Register the two immutable-in-flight shared BF16 buffers without copying.
Storage references stay alive until all GPU copies/updates complete. Metadata
continues to use the ordinary small pinned copies. No additional hidden cache.
"""
import torch
from cuda.bindings import runtime as cuda

class RegisteredStoragePool:
    def __init__(self):self.buffers={}
    def __enter__(self):return self
    def pin(self,tensor):
        if tensor.dtype!=torch.bfloat16:return tensor.pin_memory()
        assert tensor.is_shared() and tensor.device.type=='cpu'
        storage=tensor.untyped_storage();pointer=storage.data_ptr();size=storage.nbytes()
        if pointer not in self.buffers:
            error=cuda.cudaHostRegister(pointer,size,0)[0]
            assert error==cuda.cudaError_t.cudaSuccess,(error,size)
            self.buffers[pointer]=tensor.detach()
        assert tensor.is_pinned()
        assert len(self.buffers)<=4,'Expected only aux/final for two shared slots'
        return tensor
    def __exit__(self,*exc):
        torch.cuda.synchronize()
        for pointer in self.buffers:
            assert cuda.cudaHostUnregister(pointer)[0]==cuda.cudaError_t.cudaSuccess
        self.buffers.clear()
