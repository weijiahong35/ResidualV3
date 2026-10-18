# ResidualV3 项目入口

这是已展开源码的项目入口。训练、推理与重要报告可直接浏览；服务器独有的大数据与恢复资产不包含在Git中。

| 要看什么 | 入口 |
|---|---|
| 方法、训练、正式AC/TPS及相关工作 | [组会汇总](GROUP_MEETING_20261013_CN.md) |
| 最新六数据集正式对照 | [48窗/24配对结果](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/RESULTS_CN.md) |
| E1/E2统一实现对照 | [E1/E2结果](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/RESULTS_CN.md) |
| E1训练代码 | [E1入口目录](residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/) |
| E2训练代码 | [E2入口目录](residualV3-nonthing/code/full_joint_body_epoch2_aux_compare_20261010_v4/) |
| 正式推理及评测代码 | [六数据集冻结实现](residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/) |
| 四个最终权重 | [Release下载](https://github.com/weijiahong35/ResidualV3/releases/tag/v2.0-archive) |
| 机制与基底预实验结论 | [最终报告入口](residualV3-nonthing/current/05_research/) |
| 完整实验账本 | [Nonthinking](EXPERIMENTS_NONTHINKING.md) · [Thinking](EXPERIMENTS_THINKING.md) |
| 本次删了什么 | [清理记录](maintenance/cleanup_20261018_v1/) |
| 恢复退役旧代码 | 仅服务器保留归档，本仓库不附带 |

## 目录含义

`residualV3-nonthing/code/` 和 `runtime_bundle/` 中部分早期helper仍被E1/E2训练入口调用，按依赖保留。
`residual/` 仍被主库DFlash适配器导入，保留；独立的ResidualV2与EAGLE旧源已归档。
`residualV3_main_v1/research/` 中原始概率数据、模型、最终表格和负结果保留。已结束研究的展开代码已归档，复跑前先按RESTORE恢复。
`output/`、`regeneration_standalone/`、训练数据及完整恢复断点包含唯一输入或恢复资产，本轮没有将它们当垃圾删除。

## 打包范围

Git仓库已展开当前运行代码及其依赖、必要配置/basis、重要报告与表格；E1/E2四个最终权重在Release单独提供。
不把服务器的全部历史记录、逐状态概率银行、环境副本、编译缓存、完整恢复断点和计时临时模型再次上传。

AC/AL是接受长度，不是答题准确率。原概率警报、科学负结果、未判定与资源限制均继续保留。
