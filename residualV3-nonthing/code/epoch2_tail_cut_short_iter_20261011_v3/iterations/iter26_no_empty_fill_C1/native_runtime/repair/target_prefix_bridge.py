"""Process-local causal proposal-prefix transfer; installed packages stay unchanged.

The stop decision is made inside the existing repair graph before later candidates
can affect it. Later raw samples are discarded, never used to choose the length.
The scheduler receives a genuinely shorter list, including the native bonus slot.
"""
import os
import numpy as np
import torch

COUNTERS = dict(draft_lists=0, scheduled_proposals=0, possible_proposals=0,
                target_graph_rows=0, variable_batches=0, prefix_contract_checks=0,
                decode_calls=0, decode_logical_rows=0, decode_graph_rows=0,
                decode_eager_calls=0, decode_full_graph_calls=0,
                decode_piecewise_graph_calls=0, native_q_reference_checks=0)
HISTOGRAM = [0] * 16
IN_DECODE = False
DECODE_MAX_QUERY = 0


def trim_prefixes(rows, lengths):
    assert len(rows) == len(lengths)
    result = []
    for row, length in zip(rows, lengths):
        length = int(length)
        assert len(row) == 15 and 1 <= length <= 15
        assert all(int(token) >= 0 for token in row[:length])
        result.append(row[:length])
    return result


def install():
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
    from vllm.v1.worker.gpu.async_utils import async_copy_to_np
    from vllm.v1.outputs import DraftTokenIds
    old_set = DraftTokensHandler.set_draft_tokens
    old_get = DraftTokensHandler.get_draft_tokens

    def set_tokens(self, batch, tokens):
        if os.environ.get('TARGET_PREFIX_MODE')=='DEVICE':
            from runtime.native import MODEL
            from .device_prefix import cuts
            n=len(batch.req_ids)
            # The scheduler needs only its fixed slot count; actual proposal
            # IDs already reside in req_states.draft_tokens on the GPU.
            cuts(MODEL)[batch.idx_mapping.long()]=MODEL.head._tail_last_stats[-1][:n]
            self._adaptive_prefix=False
            return old_set(self,batch,tokens)
        if os.environ.get('TARGET_PREFIX_MODE', 'FULL') != 'ADAPTIVE':
            self._adaptive_prefix = False
            return old_set(self, batch, tokens)
        from runtime.native import MODEL
        assert MODEL.head is not None and not batch.has_structured_output_reqs
        assert os.environ.get('TAIL_MODE') == 'PIPELINE'
        self._adaptive_prefix = True
        self.req_ids = batch.req_ids
        n = len(self.req_ids)
        assert tokens.shape == (n, 15)
        cuts = MODEL.head._tail_last_stats[-1][:n]
        assert cuts.shape == (n,)
        packed = torch.cat((tokens.to(torch.int32), cuts[:, None]), dim=1)
        current = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            self._prefix_np = async_copy_to_np(packed)
            packed.record_stream(self.copy_stream)
            self.copy_event.record()

    def get_tokens(self):
        if not getattr(self, '_adaptive_prefix', False):
            return old_get(self)
        self.copy_event.synchronize()
        rows = self._prefix_np[:, :15].tolist()
        lengths = self._prefix_np[:, 15].tolist()
        trimmed = trim_prefixes(rows, lengths)
        COUNTERS['draft_lists'] += len(trimmed)
        COUNTERS['possible_proposals'] += 15 * len(trimmed)
        return DraftTokenIds(self.req_ids, trimmed)

    DraftTokensHandler.set_draft_tokens = set_tokens
    DraftTokensHandler.get_draft_tokens = get_tokens

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
    previous_sample = RejectionSampler.__call__
    def sample(self, logits, batch, draft_logits=None):
        if batch.num_draft_tokens:
            lengths = np.diff(batch.cu_num_logits_np) - 1
            active_lengths = [int(x) for x in lengths if x > 0]
            assert all(1 <= x <= 15 for x in active_lengths)
            COUNTERS['scheduled_proposals'] += sum(active_lengths)
            COUNTERS['variable_batches'] += int(len(set(active_lengths)) > 1)
            for length in active_lengths:
                HISTOGRAM[length] += 1
            if os.environ.get('LOCAL_CONTRACT_CHECKS') == '1':
                from runtime.native import MODEL
                inputs, processed, state, fp64 = MODEL.diagnostic_latest
                ids, scores, previous = state
                raw, hidden, anchor, pos, idx, temp, seeds, step = inputs
                # scores are before temperature; the verifier consumes processed
                # logits. Re-run the stock sampler independently, including the
                # startup dummy temperature (0.9), not just formal T0/T1.
                from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
                reference_q = torch.empty_like(draft_logits)
                reference_ids = gumbel_sample(
                    scores.reshape(-1, scores.shape[-1]), idx, temp, seeds,
                    pos + 1, apply_temperature=True,
                    output_processed_logits=reference_q,
                    output_processed_logits_col=step, use_fp64=fp64)
                if MODEL.head is not None and os.environ.get('TAIL_MODE','FULL15')!='FULL15':
                    cuts=MODEL.head._tail_last_stats[-1]
                    kept=(torch.arange(15,device=ids.device)[None,:]<cuts[:,None]).reshape(-1)
                else:
                    kept=torch.ones(ids.numel(),device=ids.device,dtype=torch.bool)
                assert torch.equal(reference_ids[kept], ids.reshape(-1)[kept]), 'native submitted proposal IDs differ'
                row_for_idx = {int(idx.reshape(ids.shape)[j, 0]):j for j in range(ids.shape[0])}
                for j, length in enumerate(lengths):
                    length = int(length)
                    if length <= 0:
                        continue
                    mapped = int(batch.idx_mapping_np[j])
                    if os.environ.get('TARGET_PREFIX_MODE')=='DEVICE':
                        from .device_prefix import cuts
                        length=min(length,int(cuts(MODEL)[mapped]))
                    source = row_for_idx[mapped]
                    start = int(batch.cu_num_logits_np[j])
                    scheduled = batch.input_ids[batch.logits_indices[start:start+length+1]].long()
                    assert torch.equal(scheduled[1:], ids[source, :length]), 'scheduled prefix differs from generated IDs'
                    assert int(scheduled[0]) == int(anchor[source]), 'verification anchor differs'
                    assert torch.equal(draft_logits[mapped, :length], reference_q[mapped, :length]), 'verifier did not use native temperature-processed repaired q'
                    COUNTERS['prefix_contract_checks'] += 1
                    COUNTERS['native_q_reference_checks'] += 1
        return previous_sample(self, logits, batch, draft_logits)
    RejectionSampler.__call__ = sample

    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    previous_graph = ModelCudaGraphManager.run_fullgraph
    def graph(self, desc, *args, **kwargs):
        COUNTERS['target_graph_rows'] += int(desc.num_tokens)
        return previous_graph(self, desc, *args, **kwargs)
    ModelCudaGraphManager.run_fullgraph = graph

    # Ragged verification uses native piecewise CUDA graphs with FlashAttention.
    # Count the actual invocation, not only its dispatch descriptor.
    previous_pw_graph = ModelCudaGraphManager.run_pw_graph
    def pw_graph(self, *args, **kwargs):
        result = previous_pw_graph(self, *args, **kwargs)
        if IN_DECODE:
            from runtime.native import MODEL
            MODEL.target_execution_counts.inc('replays')
        return result
    ModelCudaGraphManager.run_pw_graph = pw_graph

    import vllm.v1.worker.gpu.model_runner as runner
    previous_execute = runner.GPUModelRunner.execute_model
    def execute(self, scheduler_output, *args, **kwargs):
        global IN_DECODE,DECODE_MAX_QUERY
        from runtime.native import MODEL
        keys=list(scheduler_output.num_scheduled_tokens)
        warmup=bool(keys) and all(rid.startswith('_warmup_') for rid in keys)
        if MODEL is not None and MODEL.head is not None and hasattr(MODEL.head,'_tail_sampling_override'):
            head=MODEL.head
            if warmup!=head._tail_sampling_override_host:
                head._tail_sampling_override.fill_(int(warmup))
                head._tail_sampling_override_host=warmup
        values = list(scheduler_output.num_scheduled_tokens.values())
        IN_DECODE = bool(scheduler_output.scheduled_spec_decode_tokens) and bool(values) and max(values) <= 16
        DECODE_MAX_QUERY=max(values) if values else 0
        if IN_DECODE:
            COUNTERS['decode_calls'] += 1
            COUNTERS['decode_logical_rows'] += int(sum(values))
        try:
            return previous_execute(self, scheduler_output, *args, **kwargs)
        finally:
            IN_DECODE = False
    runner.GPUModelRunner.execute_model = execute
    previous_dispatch = runner.dispatch_cg_and_sync_dp
    def dispatch(*args, **kwargs):
        result = previous_dispatch(*args, **kwargs)
        if IN_DECODE:
            from vllm.config.compilation import CUDAGraphMode
            desc = result[0]
            if desc.cg_mode == CUDAGraphMode.NONE:
                COUNTERS['decode_eager_calls'] += 1
            else:
                COUNTERS['decode_graph_rows'] += int(desc.num_tokens)
                key = ('decode_piecewise_graph_calls' if desc.cg_mode == CUDAGraphMode.PIECEWISE
                       else 'decode_full_graph_calls')
                COUNTERS[key] += 1
        return result
    runner.dispatch_cg_and_sync_dp = dispatch

    from vllm.v1.worker.gpu_worker import Worker
    def snapshot(self):
        return dict(COUNTERS, histogram=HISTOGRAM.copy(), scope='host scheduling counters; take window deltas; excludes timing of RPC')
    Worker.target_prefix_snapshot = snapshot
    from fastapi import APIRouter, Request
    from vllm.entrypoints.openai import api_server
    prior_build = api_server.build_app
    router = APIRouter()
    @router.get('/target_prefix_stats')
    async def stats(request: Request):
        result = await request.app.state.engine_client.collective_rpc(method='target_prefix_snapshot', timeout=30.)
        assert len(result) == 1
        return result[0]
    def build(*args, **kwargs):
        app = prior_build(*args, **kwargs)
        app.include_router(router)
        return app
    api_server.build_app = build
