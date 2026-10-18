# 尾部截断：短测实现迭代

使用已有epoch2联合权重、三层no-soft修复头、冻结的后修复q45预测头及原阈值。只改推理实现，不改训练或模型参数。

对照为FULL15；独立训练DFLASH的正式结果见epoch1/epoch2总比较。短测为GSM8K nonthinking：C1 16题／C40 80题，最多256 token，block16；8题事件诊断不计吞吐证据。短测不替代200／1319题正式评测。

推荐保留前驱输入拼接、采样最大值复用和方向/幅度后处理融合版；六份状态拷贝融合没有体现稳定端到端优势，已恢复原拷贝。恢复后的177份运行时Python文件及评估脚本与已完成短测的iter21逐字节相同。

[推荐实现及对应短测](SELECTED_IMPLEMENTATION.json) ／ [修改说明](IMPLEMENTATION_REVIEW_CN.md) ／ [恢复校验](RESTORED_RUNTIME_GATE.json)

## 推荐实现同窗短测

| 迭代 | C | T | FULL15 AL | 截断 AL | AL变化 | FULL15 TPS | 截断 TPS | TPS变化 | 隔离 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iter20_fused_input_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 179.096 | 177.151 | -1.086% | True |
| iter20_fused_input_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.843 | 165.554 | -0.773% | True |
| iter21_fused_input_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2899.821 | 2864.693 | -1.211% | True |
| iter21_fused_input_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2671.803 | 2648.496 | -0.872% | True |

## 全部迭代（包括未采用原型）

| 迭代 | C | T | FULL15 AL | 截断 AL | AL变化 | FULL15 TPS | 截断 TPS | TPS变化 | 隔离 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| iter01 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2901.792 | 2635.348 | -9.182% | True |
| iter01 | 40 | 1 | 7.543002 | 7.586493 | +0.577% | 2649.403 | 2477.754 | -6.479% | True |
| iter02_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 183.122 | 164.438 | -10.203% | True |
| iter02_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 170.710 | 154.391 | -9.560% | True |
| iter03 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2861.258 | 2666.198 | -6.817% | True |
| iter03 | 40 | 1 | 7.564586 | 7.564687 | +0.001% | 2627.037 | 2439.372 | -7.144% | True |
| iter06_bounded | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2895.637 | 2664.561 | -7.980% | True |
| iter06_bounded | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2678.652 | 2487.441 | -7.138% | True |
| iter07_device_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 178.726 | 175.031 | -2.068% | True |
| iter07_device_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 165.929 | 164.357 | -0.947% | True |
| iter08_device_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2878.711 | 2840.367 | -1.332% | True |
| iter08_device_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2659.219 | 2658.545 | -0.025% | True |
| iter09_feature_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 178.038 | 175.393 | -1.485% | True |
| iter09_feature_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.054 | 163.989 | -1.244% | True |
| iter10_feature_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2873.302 | 2846.860 | -0.920% | True |
| iter10_feature_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2673.472 | 2632.473 | -1.534% | True |
| iter11_dense_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 178.965 | 176.051 | -1.628% | True |
| iter11_dense_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.559 | 165.029 | -0.919% | True |
| iter12_dense_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2890.811 | 2840.174 | -1.752% | True |
| iter12_dense_C40 | 40 | 1 | 7.543002 | 7.586493 | +0.577% | 2655.315 | 2636.683 | -0.702% | True |
| iter13_readout_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 178.185 | 175.955 | -1.252% | True |
| iter13_readout_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.249 | 164.427 | -1.096% | True |
| iter14_readout_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2881.035 | 2848.772 | -1.120% | True |
| iter14_readout_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2667.183 | 2629.124 | -1.427% | True |
| iter15_bulk_entropy_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2902.382 | 2847.198 | -1.901% | True |
| iter15_bulk_entropy_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2677.033 | 2654.612 | -0.838% | True |
| iter16_epilogue_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 179.470 | 176.819 | -1.477% | True |
| iter16_epilogue_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.671 | 165.606 | -0.639% | True |
| iter17_epilogue_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2914.757 | 2851.988 | -2.154% | True |
| iter17_epilogue_C40 | 40 | 1 | 7.543002 | 7.586493 | +0.577% | 2678.654 | 2650.223 | -1.061% | True |
| iter18_sample_max_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 178.893 | 176.339 | -1.428% | True |
| iter18_sample_max_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.509 | 165.087 | -0.854% | True |
| iter19_sample_max_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2908.698 | 2867.083 | -1.431% | True |
| iter19_sample_max_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2683.056 | 2653.867 | -1.088% | True |
| iter20_fused_input_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 179.096 | 177.151 | -1.086% | True |
| iter20_fused_input_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 166.843 | 165.554 | -0.773% | True |
| iter21_fused_input_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2899.821 | 2864.693 | -1.211% | True |
| iter21_fused_input_C40 | 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2671.803 | 2648.496 | -0.872% | True |
| iter22_fused_export_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699% | 179.324 | 177.181 | -1.195% | True |
| iter22_fused_export_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.300% | 167.080 | 165.618 | -0.875% | True |
| iter23_fused_export_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2903.620 | 2842.587 | -2.102% | True |
| iter23_fused_export_C40 | 40 | 1 | 7.543002 | 7.586493 | +0.577% | 2671.715 | 2633.366 | -1.435% | True |

实现进展：

- iter01：未修改Top256缓存、稀疏q45、ragged target整图；有数值校验但吞吐下降。
- iter02：前9位置批量预测，保留原每位置cumsum布局；C40 smoke因未预捕获的PW尺寸失败。C1完成，吞吐仍降。
- iter03：target请求数／token数二维图选择，尺寸与原PW捕获桶对齐；C40短测仍慢约7%。
- iter04/05：8题CUDA/CPU事件诊断，不能当速度证据。定位到截断长度D2H同步阻断CPU/GPU重叠；C1读回等待约8ms，FULL15使用占位长度无需该等待。
- iter06：精确有界BF16直方图缓存及高优先级repair流；含fallback边界校验、真实q45校验。
- DEVICE控制：将有效前缀边界留在GPU采样器，保留原概率比检验、RNG和bonus逻辑。target仍执行固定补齐图，不能声称物理减少target全部计算；目的是隔离并消除CPU等待。

- iter07/08：C1/C40有效前缀边界留在GPU；C1短测通过，吞吐损失由约10%降至约1–2%，尚未净提速。
- iter09/10：只计算前13位置缓存，合并q45特征拼接；完整头逐值校验通过，C1/C40短测完成，但没有稳定净收益。
- iter11/12：前10个保证修复的位置使用无门控快速算子；完整头逐值校验通过。C1吞吐下降约0.9–1.6%，C40下降约0.7–1.8%。
- 扫描融合原型：根据本机ATen的Sklansky加法树复现cumsum；78组不同批次数量逐值一致，另进行完整头和真实q45校验；N1仍使用原确定性CUB。

- 融合q45预测MLP：108组预测输出逐值相同；单次微测从约0.034–0.044ms降至0.009ms。完整头和端到端结果以对应probe、迭代状态为准，算子微测不代表吞吐收益。

- iter13/14：融合预测MLP后，C1吞吐仍降约1.1–1.3%，C40约1.1–1.4%；没有把算子提速写成端到端收益。
- iter15：前9个位置的合批熵计算改为原8192块的FP32归约顺序，减少一次全词表读取；6组完整头校验通过。短测结果见上表和即时日志。
- iter16/17：保留原PyTorch方向范数归约，融合后续BF16分母、方向除法和幅度。127组组件及12组实际配置完整头校验通过。完整头微测N1 T0约2.03→1.87ms、N40 T0约6.50→6.33ms；C1/C40短测仍无截断净收益。
- iter18/19：原1024分块的采样器顺带输出未加噪logits最大值，供q45熵计算复用；12组完整头逐值通过。端到端是否受益以本页短测结果为准。
- 前驱输入组装原型：合并前驱编码表查找与三个分支的拼接，不改变LayerNorm或矩阵运算。先测完整头，不提前宣称吞吐收益。
- iter20/21：前驱输入合并通过12组完整头与真实smoke；单并发完整头约1.6–1.9%时间缩短，C1/C40短测仍无截断净收益；保留这一已验证实现。
- iter22/23：六份图外状态拷贝合并，24组逐字节拷贝与12组全部图状态/有效输出通过；组件较快，短测没有体现稳定端到端优势，不选用。源码与负结果存档，开发实现已恢复原拷贝。
- 未提速原型：活跃请求集中计算让MLP慢约2.6–4.2%；提前到第5个位置启动预测让完整头慢约0.4–0.9%；混合长度的target词表投影原型也更慢。均未接入当前运行路径，源代码和负结果存档。

失败的cumsum并行原型及证据保留。改动必须先通过采样/修复后q/前驱/截断校验，然后进入短测。当前未宣称端到端净提速。

## 即时入口

- [C1 smoke](single_card_C1_smoke.log) / [C1短测](single_card_C1_short.log)
- [C40 smoke](single_card_C40_smoke.log) / [C40短测](single_card_C40_short.log)
- [主状态与每轮状态](status.json)
- [短测原始汇总](SHORT_RESULTS.json)
- [算子校验和计时](probes/)
- [磁盘整理记录](../../../residualV3_main_v1/maintenance/storage_cleanup_20261011_v2/RESULTS_CN.md)

## 原型处理记录

| 原型 | 状态 | 核验组数 | 证据或说明 |
|---|---|---:|---|
| compact_head_01 | rejected_for_latency | 16 | MLP prediction slower by 2.6–4.2% after packing, even with fewer active row tiles. Prototype sources retained; development main path restored. |
| target_logits_02 | not_selected | 12 | C40 uniform cut10 vocabulary projection saves about18%, but mixed lengths are 4.4% slower with BM128 and12.6% slower with BM64. Not installed in live evaluation. |
| readout_fused_01 | verified | 108 | probes/readout_fused_01/SUMMARY.json |
| full_head_16_bulk_entropy | verified | 6 | Exact original8192 per-row entropy order for large early batch; one logits read instead of separate max/mass passes. No precision relaxation. |
| full_head_17_early5 | not_selected | 6 | Matched EARLY5 is slower than EARLY9 by about0.44–0.85%; keep EARLY9. Frozen sources and negative timings retained. |
| output_epilogue_01 | not_selected_full_head_failed | 126 | Random microcases passed, but real N40 norm FP32 reduction differed by9.54e-7 on a BF16 rounding boundary. One denominator row differed, changing413 direction values; full_head_18 rejected before deployment. |
| full_head_19_epilogue_diagnosis | root_cause_verified | 1 | 定位到方向L2归约树变化：FP32误差9.54e-7落在BF16分母舍入边界，幅度计算一致。保留原PyTorch norm，只融合后续操作。 |
| iter15_bulk_entropy_C40 | complete_no_net_gain | 6 | C40同窗短测：T0接受长度-0.375%、TPS-1.901%；T1接受长度+0.290%、TPS-0.838%。合批熵未显示稳定净收益。 |
| output_epilogue_02 | verified_whole_head | 127 | 127组方向/幅度逐值相同，包括之前真实N40失败样本。完整头12组也逐值通过；继续核验实际有界缓存配置的完整头时间。 |
| full_head_20_partial_epilogue | numeric_verified_timing_configuration_corrected | 12 | C1/C40×T0/T1×no-cut/cut10/learned三种情况通过。组件计时的缓存变量拼写误用默认TopK；实际bounded缓存配置由full_head_21重测，避免当成最终路径计时。 |
| full_head_21_bounded_epilogue | verified_actual_runtime_configuration | 12 | 实际bounded缓存13位置、合批9位置，保留原norm并合并方向/幅度。完整头N1 T0约2.03→1.87ms，N40 T0约6.50→6.33ms；新旧同窗微测，不等同TPS收益。 |
| cache_tuning_01 | micro_verified_full_head_pending | 56 | 仅改变整数直方图分块。N40约0.923→0.794ms；N1约0.070→0.065ms。含重复值、窄分布和全范围fallback样本，完整头核验运行中。 |
| full_head_22_tuned_cache | not_selected | 12 | 缓存本身较快，但同窗完整头反而慢约0.05–0.35%，没有接入；开发原型已核对存档后删除。 |
| full_head_23_sampled_max | whole_head_verified_short_pending | 12 | Same-input complete head exact for N1/N40 × T0/T1 × no-cut/cut10/learned. Matched head micro improvement 0.05–0.86%; end-to-end short tests required. |
| iter16_epilogue_C1 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.6990658367185931, 'tps_delta_percent': -1.4771197093686905, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': -0.2997612781358905, 'tps_delta_percent': -0.6390282977707162, 'isolation': True}] |
| iter17_epilogue_C40 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.37466468519433693, 'tps_delta_percent': -2.1535031823761486, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': 0.5765681014038737, 'tps_delta_percent': -1.0613952549513805, 'isolation': True}] |
| full_head_24_input_assembly | whole_head_verified_short_pending | 12 | Only predecessor BF16 table lookup and branch copies fused. Original norm/GEMM retained. N1 whole-head pipeline 1.60–1.87% faster; N40 -0.18% to +0.04% time changes. End-to-end tests pending. |
| iter18_sample_max_C1 | complete_no_net_gain |  | C1 T0 AL -0.699%, TPS -1.428%; T1 AL -0.300%, TPS -0.854%. |
| full_head_25_state_export | test_reference_scope_corrected |  | 附加断言把当前v3全部状态与多算末两行q45的旧v2比较。未变动拷贝内核；随后24逐字节拷贝及12组当前eager/图全部状态通过，旧v2有效输出比较继续保留。 |
| full_head_26_state_export | not_selected_end_to_end_gain_unproven | 12 | 24拷贝逐字节与12完整头/图状态通过，组件较快；新C1/C40短测未体现稳定端到端优势。推荐保留input/final融合版，状态拷贝原型源码和负结果已存档，开发路径恢复六次原拷贝。 |
| iter19_sample_max_C40 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.37466468519433693, 'tps_delta_percent': -1.4307016634379388, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': 0.2896008375197434, 'tps_delta_percent': -1.0879124051851763, 'isolation': True}] |
| iter20_fused_input_C1 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.6990658367185931, 'tps_delta_percent': -1.085896949969467, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': -0.2997612781358905, 'tps_delta_percent': -0.7725894014360857, 'isolation': True}] |
| iter21_fused_input_C40 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.37466468519433693, 'tps_delta_percent': -1.2113727503387683, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': 0.2896008375197434, 'tps_delta_percent': -0.8723173474923906, 'isolation': True}] |
| engine_async_scheduling | not_enabled_by_installed_dflash |  | Local vLLM config/vllm.py lines1028–1050 explicit async whitelist does not include dflash; enabling flag would raiseValueError. No bypass or global package change applied. |
| iter22_fused_export_C1 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.6990658367185931, 'tps_delta_percent': -1.195270300958018, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': -0.2997612781358905, 'tps_delta_percent': -0.8750193082439295, 'isolation': True}] |
| iter23_fused_export_C40 | complete_no_net_gain |  | [{'T': 0, 'acceptance_delta_percent': -0.37466468519433693, 'tps_delta_percent': -2.1019889699188665, 'isolation': True}, {'T': 1, 'acceptance_delta_percent': 0.5765681014038737, 'tps_delta_percent': -1.435391680970055, 'isolation': True}] |

用户目录当前约531.50GB；上限600GB。epoch1/epoch2和长训断点保留。


## 2026-10-11 续接：iter24/25 q45输出融合

2026-10-11续接：q45输出融合通过180组特征、12组完整头token/真实前驱/verifier q/cut及同版本Graph状态检查；固定E2权重/阈值，仅空卡GPU7。 新短测（FULL15为同权重实现对照）：C1 T0 AL-0.699%、TPS-2.311%、隔离True；C1 T1 AL-0.300%、TPS-1.286%、隔离True；C40 T0 AL-0.375%、TPS-0.862%、隔离True；C40 T1 AL+0.290%、TPS-0.906%、隔离True。 本轮未形成稳定净吞吐收益；保留iter20/21选用状态，新实现及负结果存档。 target仍是DEVICE有效prefix加固定补齐图；latent未测。

[校验日志](probes/full_head_27_fused_q45/probe.log) / [特征门禁](probes/full_head_27_fused_q45/FEATURE_GATE.json) / [C1即时log](iterations/iter24_fused_q45_C1/single_card_C1_short.log) / [C40即时log](iterations/iter25_fused_q45_C40/single_card_C40_short.log)

源码在新冻结迭代；新增 `tail_q45_epilogue.py`，仅与 q45 特征输出融合有关。每次服务启动要求空卡且空闲显存≥78000MiB；原开发177份runtime和选用实现未覆盖。头SHA `6635f429…944f`，联合主体SHA `ad4ab70f…8168`，aux SHA `b3609423…fa42`；完整身份与源文件SHA见各迭代 `native_protocol.json`。C1=16/C40=80、max256、T0/T1、BF16、Graph；不是正式长测。

| C | T | FULL15 AL | CUT AL | AL变化 | FULL15 TPS | CUT TPS | TPS变化 | 隔离 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8.011261 | 7.955257 | -0.699% | 181.762 | 177.562 | -2.311% | True |
| 1 | 1 | 7.446502 | 7.424180 | -0.300% | 168.660 | 166.491 | -1.286% | True |
| 40 | 0 | 8.540297 | 8.508300 | -0.375% | 2881.878 | 2857.029 | -0.862% | True |
| 40 | 1 | 7.564586 | 7.586493 | +0.290% | 2653.249 | 2629.204 | -0.906% | True |


## 2026-10-11 结构复核与局部预填充优化

2026-10-11结构复核：当前保留源码177文件冻结；发现确定性模式torch.empty调试预填充被录入头部Graph。仅头部范围关闭预填充，保留确定性算法/权重/BF16算序/完整采样。完整头4组与Q45停修12组数值合同通过。人工C1 T0/T1 1.854→1.307ms、1.920→1.359ms；C40 6.272→5.455ms、7.026→6.207ms。微测不是服务开销；追加真实服务分段与同主体开关头成本复核。 已完成短窗：C1 T0 FULL15_NOFILL/FULL15 AL+0.000% TPS+1.697% 隔离True；C1 T1 FULL15_NOFILL/FULL15 AL+0.000% TPS+1.433% 隔离True；C1 T0 Q45_NOFILL_CUT/FULL15_NOFILL AL-0.699% TPS-0.812% 隔离True；C1 T1 Q45_NOFILL_CUT/FULL15_NOFILL AL-0.300% TPS-1.052% 隔离True；C40 T0 FULL15_NOFILL/FULL15 AL+0.000% TPS+1.509% 隔离True；C40 T1 FULL15_NOFILL/FULL15 AL+0.000% TPS+0.852% 隔离True；C40 T0 Q45_NOFILL_CUT/FULL15_NOFILL AL-0.375% TPS-2.002% 隔离True；C40 T1 Q45_NOFILL_CUT/FULL15_NOFILL AL+0.290% TPS+0.086% 隔离True。 iter26/27仅C1=16/C40=80、max256；新旧FULL直接对照与新CUT/FULL分开。未启动正式长测/训练；latent未核验；选用iter20/21暂保持。

[微测日志](probes/full_head_29_no_empty_fill/audit.log) / [短窗比较](probes/full_head_29_no_empty_fill/IMPLEMENTATION_COMPARISON.json) / [C1即时日志](iterations/iter26_no_empty_fill_C1/single_card_C1_short.log) / [C40即时日志](iterations/iter27_no_empty_fill_C40/single_card_C40_short.log)

头SHA6635f429…944f；主体SHAad4ab70f…8168；两者及aux/阈值未变。完整代码、协议、权重SHA见各冻结迭代native_protocol.json；源代码均独立copy，未覆盖保留版本。服务启动要求空卡，GPU7，Graph与隔离门禁照常执行。

| C | T | 比较 | AL变化 | TPS变化 | 隔离 |
|---:|---:|---|---:|---:|---|
| 1 | 0 | FULL15_NOFILL / FULL15 | +0.000% | +1.697% | True |
| 1 | 1 | FULL15_NOFILL / FULL15 | +0.000% | +1.433% | True |
| 1 | 0 | Q45_NOFILL_CUT / FULL15_NOFILL | -0.699% | -0.812% | True |
| 1 | 1 | Q45_NOFILL_CUT / FULL15_NOFILL | -0.300% | -1.052% | True |
| 40 | 0 | FULL15_NOFILL / FULL15 | +0.000% | +1.509% | True |
| 40 | 1 | FULL15_NOFILL / FULL15 | +0.000% | +0.852% | True |
| 40 | 0 | Q45_NOFILL_CUT / FULL15_NOFILL | -0.375% | -2.002% | True |
| 40 | 1 | Q45_NOFILL_CUT / FULL15_NOFILL | +0.290% | +0.086% | True |


## 2026-10-11 真实服务成本复核完成

2026-10-11真实服务复核：同E2联合主体开关头诊断已完成；C1 T0 原/新服务头1.843/1.299ms，完整propose相对关头增量1.846/1.302ms；C1 T1 原/新服务头1.885/1.342ms，完整propose相对关头增量1.877/1.330ms；C40 T0 原/新服务头6.187/5.371ms，完整propose相对关头增量5.756/4.977ms；C40 T1 原/新服务头6.967/6.143ms，完整propose相对关头增量5.876/5.060ms。事件诊断不是TPS证据；同worker真实输入token/q重放通过，满并发行汇总，context轨迹不同，增量为观察估计。独立单训DFlash正式基线不变。

[真实服务分段与增量成本报告](probes/full_head_32_native_cost_snapshot/RESULTS_CN.md)

<!-- FUSED_NATIVE_FOLLOWUP -->

## 2026-10-11 概率准备与截断组合验证

2026-10-11概率准备与截断组合验证：C1 complete；C1 T0 FULL15_PREPARED/FULL15_NOFILL AL+0.000% TPS-0.487% 隔离True；C1 T1 FULL15_PREPARED/FULL15_NOFILL AL+0.000% TPS-0.065% 隔离True；C1 T0 Q45_PREPARED_CUT/FULL15_PREPARED AL-0.699% TPS-0.633% 隔离True；C1 T1 Q45_PREPARED_CUT/FULL15_PREPARED AL-0.300% TPS-0.143% 隔离True；C40 complete；C40 T0 FULL15_PREPARED/FULL15_NOFILL AL+0.000% TPS+0.055% 隔离True；C40 T1 FULL15_PREPARED/FULL15_NOFILL AL+0.000% TPS+1.346% 隔离True；C40 T0 Q45_PREPARED_CUT/FULL15_PREPARED AL-0.375% TPS-1.389% 隔离True；C40 T1 Q45_PREPARED_CUT/FULL15_PREPARED AL+0.290% TPS-1.741% 隔离True。固定权重/阈值，C1=16/C40=80,max256，空卡GPU0；共同概率准备融合与CUT/FULL收益分开。冻结iter28/29，未覆盖选用实现；不新增正式长测或训练。

[即时结果与日志](probes/full_head_37_native_fused_followup/RESULTS_CN.md)。
