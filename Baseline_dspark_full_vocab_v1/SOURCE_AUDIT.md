> 2026-09-24 更新：以下记录是修复前的来源审计。当前已补 BF16 与 native T0/T1 入口；最新使用与限制见 README.md。旧离线结果不改写。

# DSpark source and paper audit

Audit date: 2026-09-23. This is a **DSpark Markov and confidence head baseline** on a frozen precomputed DFlash body. It is not a full end-to-end DSpark reproduction.

- [DSpark paper §3.1–3.3](https://arxiv.org/html/2607.05147): the sequential Markov variant adds a low-rank predecessor-dependent full-vocabulary bias to unary logits. Its confidence head predicts acceptance overlap, and its full system schedules verifier prefix length using calibrated confidence and a hardware cost curve.
- Local `src/speculators/models/dspark/model_definitions.py`: `MarkovHead` has verifier-vocab W1 and draft-vocab W2; `ConfidenceHead` is linear over the backbone hidden plus previous-token embedding. This baseline directly instantiates these local classes with **both vocabularies 151,936** and rank 256.
- Local `src/speculators/models/dspark/core.py`: training conditions on the true preceding response token. This baseline trains likewise, then its held-out test starts from the verified anchor and feeds each head's own Top1 to the next step.
- Local `src/speculators/models/dspark/metrics.py` and `src/speculators/models/metrics.py`: loss is 0.1 CE + 0.9 TV + 1.0 confidence BCE with exponential position decay, as configured in earlier local DSpark training. Local CE uses the target-logit argmax as its hard label. The paper writes the ground-truth response token, a documented difference.
- [DFlash paper](https://arxiv.org/abs/2602.06036) and local `src/speculators/models/dflash/core.py`: the cached body is DFlash. Its earlier 32k token embedding/mapping behavior remains in the frozen hidden states; this head-only baseline cannot claim a fully retrained full-vocabulary DFlash body.
- The frozen Qwen3-8B `lm_head.weight` [151936,4096] reconstructs unary and target logits. The head checkpoint contains no target embedding or LM-head copy and no draft-to-target token mapping.
- Local vLLM contains `qwen3_dspark.py` and `dspark/speculator.py`, but this baseline's test currently measures offline Greedy EAL and an offline Soft EAL proxy only. No native T=0/T=1 acceptance, throughput, confidence scheduler, or CUDA Graph result is claimed. A native benchmark needs a converted checkpoint and graph-tested serving path.

All 2368 train and 64 sealed final trajectory groups use target-regenerated OpenMath responses. `Baseline_prepare_cache.py` reproduces the source 8192-token padding and anchor RNG, and checks every group against the compact cache. Train/final group overlap is zero. The fixed final epoch is selected before final testing.

## Online variant

`Baseline_train_online.py` recomputes target hidden states from the local target vLLM for each regenerated-response trajectory, deletes the generated file immediately after loading, and recomputes the DFlash body rather than using a hidden snapshot. It updates the DFlash body and DSpark Markov/confidence heads, with frozen target embedding and LM head. This remains a **DSpark-head baseline on a DFlash body**: it does not implement DSpark's full confidence-guided runtime scheduler. `Baseline_test_online.py` measures offline live-forward behavior only.
