"""Small BF16 arithmetic/storage equivalence check, not a throughput test."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'native_runtime'))
import torch
from repair.model import DualFeedbackRepair, rollout
from head_latency.optimized import optimized_rollout, prepare_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    rows = []
    with torch.no_grad():
        for seed in (42, 57):
            torch.manual_seed(seed)
            head = DualFeedbackRepair(hidden_size=32, width=16, vocab_size=64,
                basis_rank=8, depth=3, q_topk=16, dose_init=.2, use_soft=False).to(torch.bfloat16).eval()
            embedding = torch.randn(64, 32).to(torch.bfloat16)
            basis = torch.randn(64, 8).to(torch.bfloat16)
            head.bind_features(embedding, basis)
            for n, length in ((1, 7), (3, 11), (4, 15)):
                raw = torch.randn(n, length, 64).to(torch.bfloat16)
                hidden = torch.randn(n, length, 32).to(torch.bfloat16)
                anchor = torch.randint(64, (n,))
                pristine = raw.clone()
                for temperature in (0, 1):
                    def run(variant):
                        gen = torch.Generator().manual_seed(100 + seed)
                        def choose(z, j):
                            return z.argmax(-1) if temperature == 0 else torch.multinomial(z.float().softmax(-1), 1, generator=gen).squeeze(-1)
                        if variant == 'baseline':
                            return rollout(head, raw, hidden, None, basis, anchor, choose)
                        return optimized_rollout(head, raw, hidden, None, basis, anchor, choose, variant=variant)
                    original = run('baseline')
                    for variant in ('storage', 'triton_write'):
                        prepare_variant(head, variant)
                        actual = run(variant)
                        checks = dict(logits_equal=torch.equal(original.logits, actual.logits),
                            tokens_equal=torch.equal(original.tokens, actual.tokens),
                            predecessors_equal=torch.equal(original.predecessors, actual.predecessors),
                            raw_unchanged=torch.equal(raw, pristine),
                            actual_feedback=torch.equal(actual.predecessors[:, 1:], actual.tokens[:, :-1]))
                        rows.append(dict(seed=seed, batch_size=n, proposals=length, T=temperature, variant=variant, **checks))
                        if variant in ('storage','triton_write'):
                            assert all(checks.values()), rows[-1]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(status='passed', scope='CPU BF16 toy storage equivalence; CPU fallback validates storage/recurrence only; the Triton kernel is not exercised; GPU gates pending', rows=rows), indent=2) + '\n')
    print('CPU storage equivalence passed', len(rows), 'checks; GPU/native gates pending')


if __name__ == '__main__':
    main()
