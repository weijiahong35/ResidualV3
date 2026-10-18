> 2026-09-24 更新：以下记录是修复前的来源审计。当前已补 BF16 与 native T0/T1 入口；最新使用与限制见 README.md。旧离线结果不改写。

# DFlash2 source and paper audit

Audit date: 2026-09-23. I found [DFlash2's technical blog](https://inco.ai/blog/dflash2/) and [public Z-Lab inference source](https://github.com/z-lab/dflash/blob/main/dflash/model.py), but no separate DFlash2 paper or public training objective. The [DFlash paper](https://arxiv.org/abs/2602.06036) covers its predecessor backbone.

- Z-Lab `CandidateSelector` forms unary Top-`K` candidates from the **full verifier vocabulary** and scores each with `unary(candidate) + dot(predecessor_codebook(previous) * hidden_projection(hidden), successor_codebook(candidate))`. It begins with the verified anchor and walks the chosen tokens sequentially. This baseline implements that formula with `K=16`, rank 256 and both token codebooks of size 151,936.
- At serving, DFlash2 can choose only from the unary Top-16 set. “Full vocabulary” means the unary LM head and codebooks span all 151,936 token IDs, not that the selector reranks all 151,936 at each step. Test never inserts the target into the candidate set.
- [Speculators DFlash2 documentation](https://docs.vllm.ai/projects/speculators/en/v0.8.0/user_guide/algorithms/dflash2/) labels its training objective experimental: teacher-force the previous token, use hard candidate CE, and replace the weakest Top-`K` candidate with the target when absent **during training only**. This baseline follows that selector-head rule using target-logit Top1; the unary/body objective is absent because the backbone is frozen.
- Full DFlash2 additionally inserts grouped dynamic causal convolutions around attention and MLP in every draft layer. A frozen final-hidden cache cannot reproduce those convolutions. This is a **head-only selector baseline** and must not be labeled full DFlash2.
- The two 151936×256 token codebooks are free parameters. Their parameter count must not be equated with the blog's lightweight-selector parameter claim; the Speculators audit explicitly cautions against that inference.
- The frozen Qwen3-8B `lm_head.weight` [151936,4096] reconstructs unary and target logits. The head checkpoint contains no target embedding or LM-head copy and no draft-to-target token mapping. The cached DFlash body itself retains its historical 32k input embedding/mapping behavior.
- The local vLLM checkout lacks a DFlash2 model/speculator. This folder's test measures offline Greedy EAL and an offline Soft EAL proxy only. No native T=0/T=1 acceptance, throughput, or CUDA Graph result is claimed. Native use requires a source-faithful serving integration with sampled self-conditioning and graph contract checks.

All 2368 train and 64 sealed final trajectory groups use target-regenerated OpenMath responses. `Baseline_prepare_cache.py` reproduces the source 8192-token padding and anchor RNG, and checks every group against the compact cache. Train/final group overlap is zero. The fixed final epoch is selected before final testing.

## Online variant

`Baseline_train_online.py` recomputes target hidden states from the local target vLLM for each regenerated-response trajectory, deletes the generated file immediately after loading, and recomputes the DFlash body rather than using a hidden snapshot. It updates the DFlash body and the DFlash2-style candidate selector head, with frozen target embedding and LM head. This is still **not a full DFlash2 reproduction**: the DFlash backbone here lacks DFlash2 grouped dynamic causal convolutions, and the local vLLM has no DFlash2 serving implementation. `Baseline_test_online.py` measures offline live-forward behavior only.
