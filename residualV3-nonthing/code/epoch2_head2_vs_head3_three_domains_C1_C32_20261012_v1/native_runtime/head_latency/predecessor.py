"""Fixed-weight vocabulary features, GPU only; actual predecessor stays causal.

Nothing is persisted to disk or added to checkpoint state. Exact tables use
the same row count for each original LayerNorm/Linear call as the runtime.
"""
import time
import torch
from runtime.components import normalized_rows


def encode(head, rows):
    return head.previous_proj(normalized_rows(rows, eps=head.eps))


def make_table(head, batch_size=None):
    assert not head.training and not head.use_soft
    weight = head.feature_weight
    table = weight.new_empty((head.vocab_size, head.width))
    with torch.no_grad():
        if batch_size is None:
            for start in range(0, head.vocab_size, 4096):
                stop = min(start + 4096, head.vocab_size)
                table[start:stop].copy_(encode(head, weight[start:stop]))
            return table
        n = int(batch_size)
        assert 1 <= n <= 40
        groups = 32 if weight.is_cuda else 1
        chunk = groups * n
        inputs = weight.new_zeros((chunk, head.hidden_size))
        outputs = weight.new_empty((chunk, head.width))

        def compute():
            for start in range(0, chunk, n):
                outputs[start:start+n].copy_(encode(head, inputs[start:start+n]))

        graph = None
        if weight.is_cuda:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Construct fixed-weight lookup before runtime Graph capture')
            for _ in range(3):
                compute()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                compute()
        for start in range(0, head.vocab_size, chunk):
            count = min(chunk, head.vocab_size-start)
            inputs[:count].copy_(weight[start:start+count])
            if count < chunk:
                inputs[count:].zero_()
            if graph is None:
                compute()
            else:
                graph.replay()
            table[start:start+count].copy_(outputs[:count])
        if graph is not None:
            torch.cuda.synchronize()
            del graph
    return table


def prepare_lookup(head, *, exact=False, batch_sizes=(1,7,40)):
    tick = time.monotonic()
    if exact:
        if not hasattr(head, '_previous_tables'):
            head._previous_tables = {}
        for n in batch_sizes:
            if n not in head._previous_tables:
                table = make_table(head, n)
                # Retain references with a nonpersistent buffer, so checkpoints
                # remain unmodified and .to() behavior is explicit.
                name = f'_previous_encoding_batch{n}'
                head.register_buffer(name, table, persistent=False)
                head._previous_tables[n] = table
    elif not hasattr(head, '_previous_encoding'):
        head.register_buffer('_previous_encoding', make_table(head), persistent=False)
    if head.feature_weight.is_cuda:
        torch.cuda.synchronize()
    return time.monotonic() - tick


def predict_encoded(head, hp, hard, qp, *, fused_swiglu=False):
    parts = {'hidden': hp, 'hard_previous': hard, 'q': qp}
    x = head.fusion(torch.cat([parts[n] for n in head.branch_names], -1))
    for block in head.blocks:
        if fused_swiglu:
            from .activation import block_forward
            x = block_forward(block,x)
        else:
            x = block(x)
    latent = head.output_norm(x)
    direction = normalized_rows(head.direction_head(latent), eps=head.eps)
    dose = (head.dose_max * torch.sigmoid(head.dose_head(latent).squeeze(-1).float())).to(latent.dtype)
    return direction, dose
