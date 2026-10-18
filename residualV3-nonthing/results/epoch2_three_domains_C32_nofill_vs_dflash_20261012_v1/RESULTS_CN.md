# E2 三领域 C32 正式补测

**已完成：12正式窗口、6252请求；最终审计通过。** 182份源码、10份权重/依赖SHA核验通过；所有正式窗隔离及draft/target CUDA Graph通过；1563题T0最终输出逐题一致。两臂采用各自E2训练权重，主体权重不同。MT-Bench仅首轮，未计算HumanEval pass@1或MT-Bench质量评分。

[最终审计](FINAL_AUDIT.json) · [宿主进程退出核验](parallel/FINAL_PROCESS_SNAPSHOT.json)。本批GPU服务、控制器及报告器已退出，后续高并发默认C32，历史C40保留。

独立单训 DFlash vs 联合训练 DFlash＋FULL15_NOFILL；固定15 proposal，无尾部截断。
BF16、batch-invariant、CUDA Graph；T0/T1，max2096/context4096。新冻结代码，权重与既有正式版一致。AC为平均接受长度，不是准确率；吞吐单位token/s。

数量：GSM8K=1319、HumanEval=164、MT-Bench=80（首轮问题），与原C40题目及顺序相同，无重复扩充。C32 formal KV44GiB、smoke38GiB。所有GPU0–7允许，只用宿主无进程且free≥78000MiB的空卡。

[控制器log](parallel/controller.log)、[当前状态](status.json)、[代码与设置](plan.json)。

## 数学：GSM8K

[smoke log](gsm8k_C32_smoke.log) · [formal log](gsm8k_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| GSM8K | C32 T0 | 7.609→8.652 | +13.70% | 2978.9→3183.1 | +6.86% |
| GSM8K | C32 T1 | 6.483→7.699 | +18.75% | 2549.9→2837.1 | +11.26% |

## 代码：HumanEval

[smoke log](humaneval_C32_smoke.log) · [formal log](humaneval_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| HumanEval | C32 T0 | 5.542→6.343 | +14.46% | 1981.0→2167.8 | +9.43% |
| HumanEval | C32 T1 | 4.682→5.568 | +18.94% | 1635.4→1878.9 | +14.89% |

## 对话：MT-Bench

[smoke log](mt-bench_C32_smoke.log) · [formal log](mt-bench_C32_formal.log)

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| MT-Bench | C32 T0 | 3.148→3.415 | +8.48% | 952.5→982.7 | +3.17% |
| MT-Bench | C32 T1 | 2.777→3.151 | +13.47% | 889.2→953.5 | +7.23% |

