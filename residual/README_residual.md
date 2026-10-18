# Residual V20.6 Dual-V10

Use `V20_6_SOFTLOGIT_DUAL_V10.md` as the architecture/training contract.

Main rules:

1. Real V10 lagged anchor-level information flow.
2. p2/p3/p4 each own two complete independent V10 predictors: Bias/NLA + Repair.
3. Training is raw-logit/soft-distribution based; no greedy labels or sampling.
4. Greedy argmax is used only by final speculative-decoding evaluation/runtime.
5. Base DFlash and lm_head remain frozen.
6. External Residual MoLE is applied after the complete frozen MLP.
7. Base reference rollout keeps V10 fused D1+D2 and D3..D5 compile boundaries.

Fresh training is required:

```bash
bash residual/train_openmath_stage1_residual.sh
bash residual/train_openmath_stage2_residual.sh
bash residual/train_openmath_stage3_residual.sh
```

Then do a small greedy runtime test before the full benchmark:

```bash
NUM_SAMPLES=20 bash residual/test_openmath_residual.sh
```
