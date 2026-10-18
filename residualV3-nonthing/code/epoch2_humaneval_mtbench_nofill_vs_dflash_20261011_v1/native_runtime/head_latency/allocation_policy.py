"""Opt-in head-local allocation scope, keeping deterministic algorithms enabled.

Avoid NaN/max-int debug fills on torch.empty outputs that head kernels overwrite.
Explicit zeros/ones/full, model arithmetic, RNG and target allocation stay intact.
The process flag is restored before returning, including exceptions and nesting.
This scope assumes the existing synchronous single-threaded head launch path.
"""
import functools
import os
import torch
import torch.utils.deterministic


def head_allocation_scope(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        if os.environ.get('HEAD_SKIP_EMPTY_FILL', '0') != '1':
            return fn(*args, **kwargs)
        assert not torch.is_grad_enabled(), 'inference only'
        deterministic = torch.are_deterministic_algorithms_enabled()
        previous = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.utils.deterministic.fill_uninitialized_memory = False
            return fn(*args, **kwargs)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = previous
            assert torch.are_deterministic_algorithms_enabled() == deterministic
    return call
