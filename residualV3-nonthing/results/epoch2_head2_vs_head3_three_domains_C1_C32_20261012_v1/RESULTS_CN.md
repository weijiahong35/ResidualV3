# E2三领域：三层头与两层头直接对照

两者使用同一个E2联合训练DFlash主体、同一套三层训练权重；两层仅推理省去第三层，未重训。两个头均NOFILL、FULL15，无尾部截断。
每组两臂在同一独占空卡连续测量。BF16、batch-invariant、CUDA Graph，T0/T1，max2096/context4096；GSM8K C1=200/C32=1319，HumanEval164，MT-Bench80首轮。
AC为接受长度，吞吐单位token/s。变化均为(两层/三层−1)×100%；仅使用本组已通过隔离的配对窗口。
独立单训DFlash对照继续保存在 ../epoch2_head2_three_domains_C1_C32_20261012_v1/RESULTS_CN.md；原三层正式结果保留。

## gsm8k

| 数据集 | 设置 | AC：三层→两层 | AC变化 | 吞吐：三层→两层 | 吞吐变化 |
|---|---|---:|---:|---:|---:|
| gsm8k | C1 T0 | 8.515→8.329 | -2.18% | 196.6→192.8 | -1.93% |
| gsm8k | C1 T1 | 7.489→7.394 | -1.26% | 173.8→171.8 | -1.20% |

即时日志：gsm8k_C1_smoke.log / gsm8k_C1_formal.log。当前状态：complete。

| gsm8k | C32 T0 | 8.652→8.530 | -1.41% | 3239.0→3194.5 | -1.38% |
| gsm8k | C32 T1 | 7.699→7.554 | -1.89% | 2876.6→2838.1 | -1.34% |

即时日志：gsm8k_C32_smoke.log / gsm8k_C32_formal.log。当前状态：complete。

## humaneval

| 数据集 | 设置 | AC：三层→两层 | AC变化 | 吞吐：三层→两层 | 吞吐变化 |
|---|---|---:|---:|---:|---:|
| humaneval | C1 T0 | 6.344→6.261 | -1.32% | 148.7→147.1 | -1.06% |
| humaneval | C1 T1 | 5.688→5.491 | -3.45% | 133.3→129.4 | -2.92% |

即时日志：humaneval_C1_smoke.log / humaneval_C1_formal.log。当前状态：complete。

| humaneval | C32 T0 | 6.343→6.257 | -1.35% | 2176.1→2140.9 | -1.61% |
| humaneval | C32 T1 | 5.568→5.537 | -0.56% | 1893.3→1988.6 | +5.03% |

即时日志：humaneval_C32_smoke.log / humaneval_C32_formal.log。当前状态：complete。

## mt-bench

| 数据集 | 设置 | AC：三层→两层 | AC变化 | 吞吐：三层→两层 | 吞吐变化 |
|---|---|---:|---:|---:|---:|
| mt-bench | C1 T0 | 3.413→3.367 | -1.35% | 80.8→80.0 | -1.07% |
| mt-bench | C1 T1 | 3.121→3.068 | -1.72% | 73.9→72.8 | -1.44% |

即时日志：mt-bench_C1_smoke.log / mt-bench_C1_formal.log。当前状态：complete。

| mt-bench | C32 T0 | 3.415→3.368 | -1.38% | 985.8→981.4 | -0.44% |
| mt-bench | C32 T1 | 3.153→3.104 | -1.56% | 957.4→892.4 | -6.79% |

即时日志：mt-bench_C32_smoke.log / mt-bench_C32_formal.log。当前状态：complete。

