"""Prefetch the unchanged cache while the main stream finishes head preparation."""
import torch


def prefetch(head,raw,ids):
    from .tail_sparse_q45 import prepare
    assert getattr(head,'_tail_cache_prefetch_requested',False)
    main=torch.cuda.current_stream();side=head._tail_stream
    # Candidate IDs are now ready. Raw logits and these IDs stay live through
    # the causal loop. Cache kernels only read them; main-stream repair later
    # writes the distinct transposed clone, never this original raw tensor.
    side.wait_stream(main)
    with torch.cuda.stream(side):
        head._tail_sparse_cache=prepare(raw,ids)
    head._tail_cache_prefetched=True
