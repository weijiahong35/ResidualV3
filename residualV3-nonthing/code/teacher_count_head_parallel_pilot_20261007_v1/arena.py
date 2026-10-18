"""Pilot-only fixed physical workspace, shared by all autograd threads.

The global allocator suballocates one live cudaMalloc allocation. It accepts
only the one-device/default-stream protocol already used by pipeline v8.
No unused separate reservation tensor is added to the training footprint.
"""
import ctypes
import json
from pathlib import Path


def install(gib):
    import torch
    assert not torch.cuda.is_initialized()
    path = Path(__file__).with_name('fixed_workspace.so')
    allocator = torch.cuda.memory.CUDAPluggableAllocator(str(path), 'arena_alloc', 'arena_free')
    torch.cuda.memory.change_current_allocator(allocator)
    library = ctypes.CDLL(str(path))
    library.arena_init.argtypes = [ctypes.c_size_t]
    library.arena_init.restype = ctypes.c_int
    library.arena_owns.argtypes = [ctypes.c_void_p]
    library.arena_owns.restype = ctypes.c_int
    for name in ['arena_capacity', 'arena_current', 'arena_peak', 'arena_allocations']:
        getattr(library, name).restype = ctypes.c_size_t
    error = library.arena_init(int(gib * 2**30))
    if error:
        raise RuntimeError(f'Cannot acquire fixed workspace: CUDA error {error}')
    return allocator, library


def stats(library):
    return {name: int(getattr(library, 'arena_' + name)())
            for name in ['capacity', 'current', 'peak', 'allocations']}


def small_gate(library):
    import torch
    torch.manual_seed(111)
    x = torch.randn(32, 64, device='cuda', requires_grad=True)
    w = torch.randn(64, 64, device='cuda', requires_grad=True)
    xc, wc = x.detach().cpu().requires_grad_(), w.detach().cpu().requires_grad_()
    y = (x @ w).square().mean()
    yc = (xc @ wc).square().mean()
    y.backward(); yc.backward()
    torch.cuda.synchronize()
    owned = all(library.arena_owns(t.data_ptr()) for t in [x, w, x.grad, w.grad])
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(y.cpu(), yc, rtol=1e-5, atol=2e-6)
    assert owned, 'Autograd gradient escaped the workspace'
    x.record_stream(torch.cuda.current_stream())
    return dict(passed=True, autograd_allocations_in_arena=True,
                CPU_loss_gradient_reference_passed=True, statistics=stats(library))

