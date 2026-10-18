# V20.6 Dual-V10 — Soft-Logit Training / Greedy Evaluation

## Source of truth

This implementation was rebuilt against the uploaded V10 source
`SpecLoRA_residual_V10_independent_branches_no_lmhead.zip`.

The V10 information stem and V10 progressive fusion architecture are retained.
For p2, p3 and p4, V20.6 owns TWO complete independent V10 branches:

- Bias/NLA branch: predicts V10 NLA severity per speculative slot.
- Repair branch: predicts V10 soft rejection-conditioned repair direction.

Only the input information stem is shared.  Risk/Repair branch parameter objects
are disjoint.

## Training semantics

Training never uses argmax, greedy acceptance labels, top-p, top-k or sampling.

From raw Target and Draft logits:

```
p = softmax(target_logits / T_dist)
q = softmax(draft_logits  / T_dist)
TV = 0.5 * |p-q|_1
overlap = sum(min(p,q)) = 1-TV
NLA = -log(overlap)
```

With `T_dist=1.0` this uses the ordinary full logits directly.

Soft speculative structure:

```
reach_j        = product_{i<j} overlap_i
first_reject_j = reach_j * (1-overlap_j)
survival_j     = product_{i<=j} overlap_i
EAL            = sum_j survival_j
```

Repair target follows V10:

```
positive residual = [p-q]_+
repair_dist_j = [p-q]_+ / TV
repair_hidden_j = repair_dist_j @ lm_head.weight
block repair = first_reject-weighted pool(repair_hidden_j)
repair_direction = normalize(block repair @ fixed_projector)
```

Stage1 loss:

- Bias: V10 EAL-sensitivity-weighted SmoothL1 on NLA.
- Repair: V10 rejection-mass-weighted cosine direction loss.

Stage2/Stage3 correction loss:

- soft Target->Draft KL only;
- weighted by detached V10 EAL-sensitivity weights;
- no hard AUF / argmax CE in training.

## Evaluation semantics

Final speculative decoding evaluation is greedy:

```
target_token = argmax(target_logits)
draft_token  = argmax(draft_logits)
accepted = target_token == draft_token
```

Greedy is evaluation/runtime only.

## DFlash execution

- Base DFlash weights are frozen.
- lm_head is frozen/unchanged.
- D1+D2 use the V10 fused compiled probe.
- D3..D5 use the V10 fused Base tail for reference/base rollout.
- final norm + lm_head stay outside the D3..D5 compiled tail.
- corrected rollout becomes progressive only from D3 onward.
- only anchor-level Draft states `[A,L,H]` are retained for Predictor history.

## Residual correction

- external `ResidualMoLE`; Base decoder source is not mutated;
- correction after the complete frozen MLP;
- p2 -> D3, p3 -> D4, p4 -> D5;
- 8 experts, Top-2 by default;
- Repair branch controls expert identity;
- Bias/NLA branch controls correction gate using continuous soft signals.

## Validation performed in the build environment

- all Python files: `compileall` PASS;
- all shell scripts: `bash -n` PASS;
- standalone `test_softlogit_train.py`: PASS;
- evaluation-only greedy semantics test: PASS;
- V10 shared-stem state shapes: exact match;
- p2/p3/p4 V10 fusion core shapes: exact match for BOTH new branches;
- Risk/Repair shared parameter objects per pK: 0;
- full integration test requires `transformers` and the local speculators CUDA stack; the build environment does not include `transformers`.

Old V20.5.x checkpoints are intentionally incompatible. Retrain Stage1 -> Stage2 -> Stage3.
