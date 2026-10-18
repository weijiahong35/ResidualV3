"""Process-local vLLM head integration, using stock Gumbel and rejection sampling."""
import json
import os
from pathlib import Path
import torch
from safetensors.torch import load_file
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from Baseline_sampling import sample_block, POLICY
from Baseline_models import DSparkHeadBaseline as HEAD_CLASS
FAMILY = 'dspark'

class BaselineDraft(DFlashQwen3ForCausalLM):
    def load_weights(self, weights):
        state = load_file(os.environ['BASELINE_CHECKPOINT'], device='cpu')
        body = {k[5:]:v for k,v in state.items() if k.startswith('body.')}
        incoming = dict(weights)
        if set(body) - set(incoming):
            raise RuntimeError('body checkpoint contains unknown DFlash weights')
        incoming.update(body)  # raw and head arms use the exact same optional body overlay
        result = super().load_weights(incoming.items())
        self.head = None
        if os.environ['BASELINE_ARM'] == 'head':
            state = ({k[5:]:v for k,v in state.items() if k.startswith('head.')}
                     if any(k.startswith('head.') for k in state) else state)
            self.head = HEAD_CLASS(top_k=int(os.environ['BASELINE_TOP_K'])) if FAMILY == 'dflash2' else HEAD_CLASS()
            self.head.to(device=self.lm_head.weight.device, dtype=torch.bfloat16)
            self.head.load_state_dict(state, strict=True)
            self.head.eval()
        return result

    def after_share(self, target):
        target = target.get_language_model() if hasattr(target, 'get_language_model') else target
        assert self.lm_head is target.lm_head
        assert self.model.embed_tokens is target.model.embed_tokens
        assert self.draft_id_to_target_id is None
        assert self.lm_head.weight.dtype == torch.bfloat16
        record = dict(precision_policy=POLICY, head_dtype='bfloat16', full_target_head_shared=True,
                      actual_sampled_feedback=True, arm=os.environ['BASELINE_ARM'], family=FAMILY)
        (Path(os.environ['BASELINE_TRIAL'])/'model_contract.json').write_text(json.dumps(record,indent=2))

    def forward(self, input_ids, positions, inputs_embeds=None):
        self.query = input_ids
        return super().forward(input_ids, positions, inputs_embeds)

    def aligned_sample(self, hidden, choose):
        with torch.autocast('cuda', enabled=False):
            raw = torch.mm(hidden, self.lm_head.weight.T)
            n = raw.shape[0] // 7
            anchor = self.query[:n*8].reshape(n,8)[:,0].long()
            ids, scores, previous = sample_block(self.head, raw.reshape(n,7,-1),
                hidden.reshape(n,7,-1), anchor, sample_step=choose)
        return ids.reshape(-1), scores.reshape_as(raw), previous, anchor


def install():
    from vllm.model_executor.models import ModelRegistry
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    original = dflash.load_dflash_model
    def load(target, config):
        model = original(target, config)
        model.after_share(target)
        return model
    dflash.load_dflash_model = load
    ModelRegistry.register_model('DFlashDraftModel', BaselineDraft)
    def sample(self, hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
        assert draft_logits is not None and hidden_states.shape[0] % 7 == 0
        def choose(logits, j):
            sel = torch.arange(j, hidden_states.shape[0], 7, device=hidden_states.device)
            return gumbel_sample(logits, idx_mapping[sel], temperature, seeds,
                positions[sel]+1, apply_temperature=True, output_processed_logits=draft_logits,
                output_processed_logits_col=draft_step[sel], use_fp64=self.use_fp64_gumbel)
        ids, logits, previous, anchor = self.model.aligned_sample(hidden_states, choose)
        # Diagnostics only during smoke, never during the timed measurement window.
        if os.environ.get('BASELINE_CONTRACT_CHECKS') == '1':
            reference = torch.empty_like(draft_logits)
            expected = gumbel_sample(logits, idx_mapping, temperature, seeds, positions+1,
                apply_temperature=True, output_processed_logits=reference,
                output_processed_logits_col=draft_step, use_fp64=self.use_fp64_gumbel)
            assert torch.equal(ids, expected), 'step vs stock batched sample mismatch'
            active = idx_mapping.unique().long()
            assert torch.equal(draft_logits[active], reference[active]), 'verifier q mismatch'
            blocks = ids.reshape(-1,7)
            assert torch.equal(previous[:,0], anchor)
            assert torch.equal(previous[:,1:], blocks[:,:-1]), 'feedback not actual samples'
            row = dict(step_vs_batch_equal=True, processed_q_equal=True, feedback_equal=True,
                non_top1=int((ids != logits.argmax(-1)).sum()),
                temperature=float(temperature[idx_mapping[0]].item()),
                bf16_scores=logits.dtype==torch.bfloat16)
            with (Path(os.environ['BASELINE_TRIAL'])/'sampling_contract.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        return ids
    DraftModelSpeculator.sample_draft = sample
