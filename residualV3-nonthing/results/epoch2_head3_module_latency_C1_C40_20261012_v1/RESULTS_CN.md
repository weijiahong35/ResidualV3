> 覆盖复核：采集已结束，但 C40 T1 缺少满40并发的主体/raw LM/完整头直接分段；C40 T0仅3份稳态满并发组件。混合并发与头重放不能填补这些同边界空项。详见 [覆盖表](TIMING_COVERAGE_CN.md)。本次复核没有重跑GPU。

# E2三层 C1/C40 模块耗时

固定原三层正式显存口径：C1 KV52GiB、C40 KV44GiB，memory utilization=0.92。没有预算扫描，没有占位显存。
原E2联合主体＋三层NOFILL头，FULL15，BF16、batch-invariant、CUDA Graph，T0/T1，GSM8K C1=16/C40=80、max2096/context4096，保留原生EOS。
服务分段直接来自真实混合长度请求；头内模块来自同worker保存的首个满并发真实状态重放。两个边界分别列出。单位为ms/整批decode block，不是ms/token。
细分头重放插入较多事件，会扰动耗时；同时给出同一状态粗计时总头和细事件总头。嵌套/重叠跨度不得相加，结果不是正式AC/TPS。

## C1 T0，KV=52GiB

| 真实服务分段（满并发decode） | 样本数 | 均值ms | 中位数ms | P90 ms |
|---|---:|---:|---:|---:|
| drafter主体query | 574 | 4.4603 | 4.4493 | 4.5285 |
| drafter原始LM投影 | 574 | 1.5442 | 1.5380 | 1.5941 |
| 完整头＋草稿采样 | 574 | 1.2974 | 1.2974 | 1.3005 |
| drafter完整query图 | 598 | 7.3354 | 7.3236 | 7.4192 |
| 完整propose（含准备/等待） | 598 | 8.0578 | 8.0169 | 8.1217 |
| drafter context KV | 598 | 0.2552 | 0.2396 | 0.2468 |
| target forward图（与decode验证配对） | 582 | 31.9358 | 31.8387 | 32.3633 |
| target LM投影 | 582 | 1.4624 | 1.4612 | 1.4878 |
| target原生验证/拒绝采样 | 582 | 0.0882 | 0.0870 | 0.0952 |
| target LM＋验证整体 | 582 | 1.5648 | 1.5631 | 1.5923 |
| target forward起点至验证终点（含间隙） | 582 | 33.5109 | 33.4170 | 33.9229 |

NVML观察峰值73057MiB；PyTorch窗口峰值reserved=69.887GiB。两者测量范围不同。

同一真实状态重放：粗计时完整头中位数1.3119ms；细事件完整头3.5033ms；事件扰动观察差+2.1914ms。

| 头内模块（同真实状态重放，合计15proposal） | 均值ms | 中位数ms |
|---|---:|---:|
| head.topk_and_raw_cast | 0.1135 | 0.1137 |
| head.probability_normalization | 0.0581 | 0.0584 |
| head.candidate_basis_gather | 0.0069 | 0.0072 |
| head.q_moment | 0.0114 | 0.0113 |
| head.q_projection | 0.0148 | 0.0143 |
| head.hidden_projection | 0.0484 | 0.0481 |
| head.hidden_norm | 0.0189 | 0.0184 |
| head.prepare_total | 0.3069 | 0.3062 |
| head.logits_buffer_copy | 0.0127 | 0.0123 |
| head.previous_input_assembly | 0.1065 | 0.1075 |
| head.fusion_projection | 0.2101 | 0.2099 |
| head.block0.norm_up_silu | 0.1858 | 0.1864 |
| head.block0.down_residual | 0.1518 | 0.1516 |
| head.block0.total | 0.4918 | 0.4925 |
| head.block1.norm_up_silu | 0.1855 | 0.1853 |
| head.block1.down_residual | 0.1505 | 0.1495 |
| head.block1.total | 0.4896 | 0.4895 |
| head.block2.norm_up_silu | 0.1837 | 0.1833 |
| head.block2.down_residual | 0.1494 | 0.1505 |
| head.block2.total | 0.4857 | 0.4854 |
| head.output_norm | 0.1194 | 0.1198 |
| head.direction_dose_projection | 0.1245 | 0.1260 |
| head.direction_dose_finish | 0.1932 | 0.1925 |
| head.predict_total | 2.6889 | 2.6890 |
| head.correction_write | 0.1687 | 0.1690 |
| head.sampling | 0.1463 | 0.1475 |
| head.output_stack | 0.0079 | 0.0082 |

norm＋up＋SiLU、down＋residual、direction＋dose均按实际融合边界报告，不拆造内部时间。按有效并发/上下文分组及CPU跨度见ANALYSIS.json与逐行原始记录。

## C1 T1，KV=52GiB

| 真实服务分段（满并发decode） | 样本数 | 均值ms | 中位数ms | P90 ms |
|---|---:|---:|---:|---:|
| drafter主体query | 587 | 4.4611 | 4.4472 | 4.5421 |
| drafter原始LM投影 | 587 | 1.5447 | 1.5391 | 1.5907 |
| 完整头＋草稿采样 | 587 | 1.3397 | 1.3394 | 1.3425 |
| drafter完整query图 | 663 | 7.3821 | 7.3708 | 7.4748 |
| 完整propose（含准备/等待） | 663 | 8.0947 | 8.0609 | 8.1785 |
| drafter context KV | 663 | 0.2531 | 0.2376 | 0.2458 |
| target forward图（与decode验证配对） | 647 | 31.9551 | 31.8577 | 32.4522 |
| target LM投影 | 647 | 1.4620 | 1.4592 | 1.4893 |
| target原生验证/拒绝采样 | 647 | 0.1032 | 0.1014 | 0.1116 |
| target LM＋验证整体 | 647 | 1.5793 | 1.5770 | 1.6077 |
| target forward起点至验证终点（含间隙） | 647 | 33.5447 | 33.4492 | 34.0441 |

NVML观察峰值73057MiB；PyTorch窗口峰值reserved=69.969GiB。两者测量范围不同。

同一真实状态重放：粗计时完整头中位数1.3533ms；细事件完整头3.5453ms；事件扰动观察差+2.1920ms。

| 头内模块（同真实状态重放，合计15proposal） | 均值ms | 中位数ms |
|---|---:|---:|
| head.topk_and_raw_cast | 0.1138 | 0.1137 |
| head.probability_normalization | 0.0582 | 0.0584 |
| head.candidate_basis_gather | 0.0070 | 0.0072 |
| head.q_moment | 0.0118 | 0.0123 |
| head.q_projection | 0.0142 | 0.0143 |
| head.hidden_projection | 0.0487 | 0.0492 |
| head.hidden_norm | 0.0197 | 0.0195 |
| head.prepare_total | 0.3075 | 0.3072 |
| head.logits_buffer_copy | 0.0121 | 0.0123 |
| head.previous_input_assembly | 0.1084 | 0.1075 |
| head.fusion_projection | 0.2092 | 0.2089 |
| head.block0.norm_up_silu | 0.1855 | 0.1853 |
| head.block0.down_residual | 0.1535 | 0.1536 |
| head.block0.total | 0.4923 | 0.4925 |
| head.block1.norm_up_silu | 0.1840 | 0.1843 |
| head.block1.down_residual | 0.1517 | 0.1516 |
| head.block1.total | 0.4902 | 0.4895 |
| head.block2.norm_up_silu | 0.1830 | 0.1833 |
| head.block2.down_residual | 0.1501 | 0.1505 |
| head.block2.total | 0.4852 | 0.4854 |
| head.output_norm | 0.1194 | 0.1198 |
| head.direction_dose_projection | 0.1249 | 0.1260 |
| head.direction_dose_finish | 0.1932 | 0.1935 |
| head.predict_total | 2.6867 | 2.6860 |
| head.correction_write | 0.1669 | 0.1669 |
| head.sampling | 0.1902 | 0.1905 |
| head.output_stack | 0.0080 | 0.0082 |

norm＋up＋SiLU、down＋residual、direction＋dose均按实际融合边界报告，不拆造内部时间。按有效并发/上下文分组及CPU跨度见ANALYSIS.json与逐行原始记录。

## C40 T0，KV=44GiB

| 真实服务分段（满并发decode） | 样本数 | 均值ms | 中位数ms | P90 ms |
|---|---:|---:|---:|---:|
| drafter主体query | 3 | 8.3644 | 8.3436 | 8.4320 |
| drafter原始LM投影 | 3 | 4.1981 | 4.1841 | 4.2193 |
| 完整头＋草稿采样 | 3 | 5.3030 | 5.3023 | 5.3047 |
| drafter完整query图 | 23 | 17.9228 | 17.8821 | 18.1178 |
| 完整propose（含准备/等待） | 23 | 19.3149 | 19.1570 | 19.7667 |
| drafter context KV | 23 | 0.5217 | 0.4680 | 0.6396 |
| target forward图（与decode验证配对） | 13 | 59.5450 | 59.5220 | 60.1575 |
| target LM投影 | 13 | 4.2090 | 4.2004 | 4.2412 |
| target原生验证/拒绝采样 | 13 | 0.2457 | 0.2447 | 0.2478 |
| target LM＋验证整体 | 13 | 4.4750 | 4.4657 | 4.5087 |
| target forward起点至验证终点（含间隙） | 13 | 64.0304 | 63.9969 | 64.6375 |

NVML观察峰值76689MiB；PyTorch窗口峰值reserved=70.537GiB。两者测量范围不同。

同一真实状态重放：粗计时完整头中位数5.3046ms；细事件完整头7.5472ms；事件扰动观察差+2.2426ms。

| 头内模块（同真实状态重放，合计15proposal） | 均值ms | 中位数ms |
|---|---:|---:|
| head.topk_and_raw_cast | 1.6884 | 1.6886 |
| head.probability_normalization | 1.3170 | 1.3179 |
| head.candidate_basis_gather | 0.0070 | 0.0072 |
| head.q_moment | 0.0628 | 0.0625 |
| head.q_projection | 0.0161 | 0.0164 |
| head.hidden_projection | 0.0516 | 0.0512 |
| head.hidden_norm | 0.0336 | 0.0338 |
| head.prepare_total | 3.2110 | 3.2102 |
| head.logits_buffer_copy | 0.3916 | 0.3912 |
| head.previous_input_assembly | 0.1221 | 0.1229 |
| head.fusion_projection | 0.2638 | 0.2652 |
| head.block0.norm_up_silu | 0.2131 | 0.2140 |
| head.block0.down_residual | 0.1973 | 0.1966 |
| head.block0.total | 0.5650 | 0.5642 |
| head.block1.norm_up_silu | 0.2090 | 0.2099 |
| head.block1.down_residual | 0.1947 | 0.1946 |
| head.block1.total | 0.5617 | 0.5622 |
| head.block2.norm_up_silu | 0.2115 | 0.2120 |
| head.block2.down_residual | 0.1906 | 0.1905 |
| head.block2.total | 0.5576 | 0.5581 |
| head.output_norm | 0.1262 | 0.1260 |
| head.direction_dose_projection | 0.1466 | 0.1464 |
| head.direction_dose_finish | 0.2354 | 0.2355 |
| head.predict_total | 3.0463 | 3.0474 |
| head.correction_write | 0.2269 | 0.2273 |
| head.sampling | 0.4702 | 0.4710 |
| head.output_stack | 0.0083 | 0.0082 |

norm＋up＋SiLU、down＋residual、direction＋dose均按实际融合边界报告，不拆造内部时间。按有效并发/上下文分组及CPU跨度见ANALYSIS.json与逐行原始记录。

## C40 T1，KV=44GiB

| 真实服务分段（满并发decode） | 样本数 | 均值ms | 中位数ms | P90 ms |
|---|---:|---:|---:|---:|
| drafter完整query图 | 28 | 18.6961 | 18.7064 | 18.8861 |
| 完整propose（含准备/等待） | 28 | 20.0494 | 20.0315 | 20.3283 |
| drafter context KV | 28 | 0.5062 | 0.4669 | 0.5686 |
| target forward图（与decode验证配对） | 18 | 59.8234 | 59.4990 | 60.9928 |
| target LM投影 | 18 | 4.2123 | 4.2015 | 4.2444 |
| target原生验证/拒绝采样 | 18 | 0.5365 | 0.5356 | 0.5410 |
| target LM＋验证整体 | 18 | 4.7695 | 4.7580 | 4.8007 |
| target forward起点至验证终点（含间隙） | 18 | 64.6035 | 64.2622 | 65.7888 |

NVML观察峰值76689MiB；PyTorch窗口峰值reserved=72.652GiB。两者测量范围不同。

同一真实状态重放：粗计时完整头中位数6.0375ms；细事件完整头8.2851ms；事件扰动观察差+2.2476ms。

| 头内模块（同真实状态重放，合计15proposal） | 均值ms | 中位数ms |
|---|---:|---:|
| head.topk_and_raw_cast | 1.6820 | 1.6824 |
| head.probability_normalization | 1.3145 | 1.3138 |
| head.candidate_basis_gather | 0.0073 | 0.0072 |
| head.q_moment | 0.0625 | 0.0625 |
| head.q_projection | 0.0158 | 0.0154 |
| head.hidden_projection | 0.0516 | 0.0512 |
| head.hidden_norm | 0.0338 | 0.0338 |
| head.prepare_total | 3.2022 | 3.2041 |
| head.logits_buffer_copy | 0.3913 | 0.3912 |
| head.previous_input_assembly | 0.1241 | 0.1249 |
| head.fusion_projection | 0.2629 | 0.2621 |
| head.block0.norm_up_silu | 0.2126 | 0.2130 |
| head.block0.down_residual | 0.1979 | 0.1976 |
| head.block0.total | 0.5667 | 0.5663 |
| head.block1.norm_up_silu | 0.2117 | 0.2109 |
| head.block1.down_residual | 0.1973 | 0.1976 |
| head.block1.total | 0.5623 | 0.5622 |
| head.block2.norm_up_silu | 0.2124 | 0.2120 |
| head.block2.down_residual | 0.1915 | 0.1925 |
| head.block2.total | 0.5604 | 0.5612 |
| head.output_norm | 0.1267 | 0.1270 |
| head.direction_dose_projection | 0.1477 | 0.1485 |
| head.direction_dose_finish | 0.2351 | 0.2355 |
| head.predict_total | 3.0552 | 3.0546 |
| head.correction_write | 0.2300 | 0.2294 |
| head.sampling | 1.2033 | 1.2032 |
| head.output_stack | 0.0082 | 0.0082 |

norm＋up＋SiLU、down＋residual、direction＋dose均按实际融合边界报告，不拆造内部时间。按有效并发/上下文分组及CPU跨度见ANALYSIS.json与逐行原始记录。

