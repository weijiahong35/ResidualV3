"""Exact BF16 stores at unique TopK indices, for inference only.

Candidate arithmetic remains in the original PyTorch implementation. The caller
owns and refreshes destination storage before each rollout/Graph replay.
"""
import torch
import triton
import triton.language as tl


def write_candidates(destination, indices, values):
    assert destination.dtype == values.dtype == torch.bfloat16
    assert indices.dtype == torch.int64 and indices.shape == values.shape
    assert destination.ndim == indices.ndim == 2
    assert destination.shape[0] == indices.shape[0]
    if not destination.is_cuda:
        # CPU validates storage/recurrence only; GPU tests validate the kernel.
        return destination.scatter_(-1, indices, values)
    unique_write[(destination.shape[0],)](destination, indices, values,
        destination.stride(0), destination.stride(1),
        indices.stride(0), indices.stride(1), values.stride(0), values.stride(1),
        indices.shape[1], BLOCK=triton.next_power_of_2(indices.shape[1]), num_warps=4)
    return destination


@triton.jit
def unique_write(Z, I, C, ZS0: tl.constexpr, ZS1: tl.constexpr,
                 IS0: tl.constexpr, IS1: tl.constexpr,
                 CS0: tl.constexpr, CS1: tl.constexpr,
                 K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ids = tl.load(I + row * IS0 + cols * IS1, cols < K, other=0)
    values = tl.load(C + row * CS0 + cols * CS1, cols < K, other=0)
    tl.store(Z + row * ZS0 + ids * ZS1, values, cols < K)
