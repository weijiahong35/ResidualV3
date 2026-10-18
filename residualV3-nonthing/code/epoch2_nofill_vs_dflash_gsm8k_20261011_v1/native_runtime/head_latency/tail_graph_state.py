"""Keep tail outputs outside vLLM's shared target/draft graph scratch pool."""
import torch


def allocate_tail_state(head, sizes, score_columns):
    # Allocate before CUDA capture. The target graph runs between drafting and
    # rejection sampling and can reuse the global graph pool's scratch storage.
    # A retained Python view alone is insufficient to export those scratch
    # outputs across that boundary.
    assert not torch.cuda.is_current_stream_capturing()
    device = head._tail_aggregate.device
    head._tail_graph_stats = {
        n: (torch.empty((n, 17), device=device, dtype=torch.int32),
            torch.empty((n, 16), device=device, dtype=torch.float32),
            torch.empty((n, 15), device=device, dtype=torch.float32),
            torch.empty(3, device=device, dtype=torch.int32),
            torch.empty((n, 15, score_columns), device=device, dtype=torch.float32),
            torch.empty(n, device=device, dtype=torch.int32))
        for n in sizes
    }


def record_tail_state(head, batch_size, state, *, capturing):
    assert state[-1].shape == (batch_size,)
    if capturing:
        exported = head._tail_graph_stats[batch_size]
        for target, source in zip(exported, state):
            assert target.shape == source.shape and target.dtype == source.dtype
            target.copy_(source)
        head._tail_last_stats = exported
    else:
        # Eager parity references must not overwrite a captured graph's output.
        head._tail_last_stats = state


def bind_tail_state(head, batch_size):
    state = head._tail_graph_stats[batch_size]
    assert state[-1].shape == (batch_size,), (batch_size, state[-1].shape)
    head._tail_last_stats = state
    return state
