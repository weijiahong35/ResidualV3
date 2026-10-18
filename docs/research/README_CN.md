# 研究证据：只读这些

预实验不等于论文贡献。这里保留能回答具体问题的证据，以及会阻止错误结论的重要负结果；旧执行方案、重试日志、未收敛探索和未建立额外价值的结果明细不再放在当前项目里。

| 已得到的有限认识 | 证据 | 不能据此声称 |
|---|---|---|
| 冻结头确实利用前驱对应，改善同状态分布 | [R1配对对照](../../residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/results/RESULTS_CN.md)：置换−正确TV约0.1791，raw−正确约0.2509 | 条件失配占主导、全部增益原因已解释 |
| 当前头一好一坏，往往仍存在共同改善动作 | [R4可行见证](../../residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4/results/RESULTS_CN.md)：163组中153组 | 所有失败都由信息不足强迫、这些动作已能被网络学到 |
| 一部分共同需求与原头实际移动有有限联系 | [R7移除对照](../../residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/20261015T075350Z_mismatch_r7/results/RESULTS_CN.md)：η约0.2310，κ约0.0347 | 全体收益解释率、训练来源或部署因果 |
| 共同偏差在两个主体中存在，换到指定草稿历史后仍有条件平均收益 | [R8迁移检验](../../residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8/results/RESULTS_CN.md) | 标签探针可部署、每条历史安全、长期自生成或TPS收益 |
| 有限历史族既有安全余量，也有高收益必须取舍的边界 | [R9边界](../../residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/runs/20261015T144624Z_mismatch_r9/results/RESULTS_CN.md)及保留的数值界 | 一般训练因果、未见历史保证、原basis可达 |

[未获支持与未完成事项](LIMITATIONS_CN.md)集中记录失败方向，不再保留每轮启动包。正式AC/TPS及E1/E2对照仍以项目首页的正式报告为准。
