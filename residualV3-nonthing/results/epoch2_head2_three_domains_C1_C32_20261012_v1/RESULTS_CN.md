# E2 前两层推理：三领域 C1/C32

E2三层训练权重只在推理省去第三层，保留前两层和原输出映射，没有两层重训。独立单训DFlash vs HEAD2_NOFILL；固定15proposal、无尾部截断，BF16、batch-invariant、CUDA Graph；T0/T1、max2096/context4096。新专家头训练保持三层。
GSM8K C1=200/C32=1319，HumanEval=164、MT-Bench=80首轮。每组同卡同配置对比；头训练可使用共享显存余量，正式评测仍要求宿主独占空卡。AC为接受长度，非准确率；吞吐单位token/s。

## gsm8k

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
<!-- log: gsm8k_C1_formal.log -->
| gsm8k | C1 T0 | 7.479→8.329 | +11.38% | 180.3→194.2 | +7.68% |
| gsm8k | C1 T1 | 6.336→7.394 | +16.70% | 153.2→172.5 | +12.66% |
<!-- log: gsm8k_C32_formal.log -->
| gsm8k | C32 T0 | 7.609→8.530 | +12.10% | 3036.1→3199.1 | +5.37% |
| gsm8k | C32 T1 | 6.483→7.554 | +16.51% | 2610.3→2841.2 | +8.85% |

即时日志：gsm8k_C1_smoke.log / gsm8k_C1_formal.log；gsm8k_C32_smoke.log / gsm8k_C32_formal.log。

## humaneval

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
<!-- log: humaneval_C1_formal.log -->
<!-- log: humaneval_C32_formal.log -->

即时日志：humaneval_C1_smoke.log / humaneval_C1_formal.log；humaneval_C32_smoke.log / humaneval_C32_formal.log。

## mt-bench

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
<!-- log: mt-bench_C1_formal.log -->
| mt-bench | C1 T0 | 3.145→3.367 | +7.05% | 77.2→80.3 | +4.06% |
| mt-bench | C1 T1 | 2.821→3.068 | +8.76% | 69.3→73.1 | +5.54% |
<!-- log: mt-bench_C32_formal.log -->
| mt-bench | C32 T0 | 3.148→3.367 | +6.96% | 948.1→970.1 | +2.32% |
| mt-bench | C32 T1 | 2.777→3.095 | +11.45% | 886.7→891.1 | +0.50% |

即时日志：mt-bench_C1_smoke.log / mt-bench_C1_formal.log；mt-bench_C32_smoke.log / mt-bench_C32_formal.log。

