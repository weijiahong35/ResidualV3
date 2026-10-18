"""Diagnostic event spans only; preserve native arithmetic and fused kernel boundaries."""
import os
from contextlib import contextmanager
import torch

CONTEXT = None

@contextmanager
def detail_scope(model, n):
    global CONTEXT
    previous = CONTEXT
    enabled = os.environ.get('HEAD_PROFILE_DETAIL') == '1' and torch.cuda.is_current_stream_capturing()
    CONTEXT = (model, n, {}) if enabled else None
    try:
        yield
    finally:
        CONTEXT = previous

@contextmanager
def head_stage(name):
    if CONTEXT is None:
        yield
        return
    model, n, counts = CONTEXT
    serial = counts.get(name, 0)
    counts[name] = serial + 1
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        model._short_latency_spans.setdefault(n, {})[name+f'#{serial}'] = (start, end)

def sample_metadata(input_batch):
    n = input_batch.num_reqs
    lengths = input_batch.seq_lens_cpu_upper_bound[:n]
    return dict(n=n, input_tokens=input_batch.num_tokens, draft_tokens=input_batch.num_draft_tokens,
        context_min=int(lengths.min()), context_max=int(lengths.max()),
        full_fixed_decode=bool(input_batch.num_draft_tokens > 0 and input_batch.num_tokens == 16*n))

def install_target_components(timer):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original_sample = GPUModelRunner.sample
    def sample(self, hidden_states, input_batch, grammar_output):
        metadata = sample_metadata(input_batch)
        original_projection = self.model.compute_logits
        def projection(*args, **kwargs):
            return timer('target_lm_projection', lambda:original_projection(*args, **kwargs), metadata)
        self.model.compute_logits = projection
        try:
            return original_sample(self, hidden_states, input_batch, grammar_output)
        finally:
            self.model.compute_logits = original_projection
    GPUModelRunner.sample = sample
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    original_reject = RejectionSampler.__call__
    def reject(self, logits, input_batch, *args, **kwargs):
        return timer('target_rejection', lambda:original_reject(self, logits, input_batch, *args, **kwargs),
            sample_metadata(input_batch))
    RejectionSampler.__call__ = reject
