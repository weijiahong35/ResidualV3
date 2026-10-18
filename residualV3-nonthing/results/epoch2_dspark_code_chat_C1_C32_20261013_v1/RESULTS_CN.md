# E2：DSpark代码与对话六套数据重测

只测E2独立单训DFlash与E2联合主体＋完整三层FULL15_NOFILL；两臂主体权重不同。
C1/C32，T0/T1，固定15 proposal；BF16、batch-invariant、CUDA Graph，无尾部截断。
最大生成2048；五套数据context4096，Arena-Hard-v2 context12288以容纳原始长输入。
固定真实KV：C1 52GiB、C32 44GiB，smoke与formal相同，utilization=0.92。
至少78000MiB空闲且宿主无GPU进程才启动；最多同时占两张空卡。
AC为包含bonus的平均接受长度；吞吐为整窗输出token/s，不是任务准确率。

当前状态：complete。新数据与作者的固定提交、抽样数、抽样顺序、首轮处理、nonthinking、2048生成上限、种子980406对齐。
引擎仍为本项目vLLM，proposal仍为15（DSpark原论文为7），并额外测T0及C32；这不是DSpark完整方法复现。
未重复扩充题目；MBPP作者选出的256个来源行中有255个不同题面，按原样保留。

[即时控制器log](parallel/controller.log) · [状态](status.json) · [数据审计](DATASET_AUDIT.json) · [固定协议](plan.json)

| 领域 | 数据集 | 每窗来源行数 | 不同题面数 | 上下文上限 |
|---|---|---:|---:|---:|
| code | humaneval | 164 | 164 | 4096 |
| code | mbpp | 256 | 255 | 4096 |
| code | livecodebench | 500 | 500 | 4096 |
| chat | mt-bench | 80 | 80 | 4096 |
| chat | alpaca | 500 | 500 | 4096 |
| chat | arena-hard-v2 | 500 | 500 | 12288 |

每个模型×温度×并发使用同一组2000个来源行；全批共48正式窗口、16000个请求。
旧HumanEval/MT-Bench使用max2096、seed42，保留为历史；本批按新请求重新测，不将跨批TPS相除。

[作者TASKS与默认参数](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py)；
[作者首轮处理与抽样](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py)。

## humaneval

[C1 smoke](humaneval_C1_smoke.log) · [C1 formal](humaneval_C1_formal.log) · [C32 smoke](humaneval_C32_smoke.log) · [C32 formal](humaneval_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| humaneval | C1 T0 | 5.542→6.344 | +14.48% | 134.4→148.6 | +10.57% |
| humaneval | C1 T1 | 4.840→5.731 | +18.43% | 117.5→134.4 | +14.38% |
| humaneval | C32 T0 | 5.542→6.343 | +14.46% | 1999.1→2175.2 | +8.81% |
| humaneval | C32 T1 | 4.810→5.678 | +18.03% | 1793.4→1987.0 | +10.80% |

## mbpp

[C1 smoke](mbpp_C1_smoke.log) · [C1 formal](mbpp_C1_formal.log) · [C32 smoke](mbpp_C32_smoke.log) · [C32 formal](mbpp_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| mbpp | C1 T0 | 4.975→5.694 | +14.46% | 121.2→134.2 | +10.70% |
| mbpp | C1 T1 | 4.325→5.029 | +16.28% | 105.8→118.8 | +12.36% |
| mbpp | C32 T0 | 4.980→5.694 | +14.32% | 1968.3→2131.3 | +8.28% |
| mbpp | C32 T1 | 4.269→4.986 | +16.80% | 1458.8→1816.8 | +24.54% |

## livecodebench

[C1 smoke](livecodebench_C1_smoke.log) · [C1 formal](livecodebench_C1_formal.log) · [C32 smoke](livecodebench_C32_smoke.log) · [C32 formal](livecodebench_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| livecodebench | C1 T0 | 6.140→6.733 | +9.67% | 143.8→152.1 | +5.78% |
| livecodebench | C1 T1 | 4.494→5.204 | +15.81% | 105.7→118.1 | +11.71% |
| livecodebench | C32 T0 | 6.140→6.737 | +9.72% | 1732.4→1768.8 | +2.10% |
| livecodebench | C32 T1 | 4.495→5.117 | +13.82% | 1181.7→1276.7 | +8.03% |

## mt-bench

[C1 smoke](mt-bench_C1_smoke.log) · [C1 formal](mt-bench_C1_formal.log) · [C32 smoke](mt-bench_C32_smoke.log) · [C32 formal](mt-bench_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| mt-bench | C1 T0 | 3.145→3.413 | +8.52% | 77.3→81.2 | +5.12% |
| mt-bench | C1 T1 | 2.765→3.190 | +15.35% | 68.0→75.8 | +11.53% |
| mt-bench | C32 T0 | 3.148→3.415 | +8.48% | 963.7→987.1 | +2.42% |
| mt-bench | C32 T1 | 2.779→3.185 | +14.61% | 824.3→900.3 | +9.23% |

## alpaca

[C1 smoke](alpaca_C1_smoke.log) · [C1 formal](alpaca_C1_formal.log) · [C32 smoke](alpaca_C32_smoke.log) · [C32 formal](alpaca_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| alpaca | C1 T0 | 2.930→3.192 | +8.93% | 72.0→75.9 | +5.43% |
| alpaca | C1 T1 | 2.633→2.945 | +11.86% | 64.7→70.0 | +8.14% |
| alpaca | C32 T0 | 2.930→3.192 | +8.95% | 1149.9→1191.6 | +3.63% |
| alpaca | C32 T1 | 2.634→2.958 | +12.27% | 1005.5→1095.2 | +8.92% |

## arena-hard-v2

[C1 smoke](arena-hard-v2_C1_smoke.log) · [C1 formal](arena-hard-v2_C1_formal.log) · [C32 smoke](arena-hard-v2_C32_smoke.log) · [C32 formal](arena-hard-v2_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| arena-hard-v2 | C1 T0 | 3.341→3.542 | +6.01% | 80.5→82.8 | +2.86% |
| arena-hard-v2 | C1 T1 | 2.681→2.919 | +8.86% | 64.6→68.1 | +5.47% |
| arena-hard-v2 | C32 T0 | 3.340→3.542 | +6.05% | 1213.7→1220.9 | +0.59% |
| arena-hard-v2 | C32 T1 | 2.674→2.912 | +8.91% | 957.8→977.7 | +2.08% |

