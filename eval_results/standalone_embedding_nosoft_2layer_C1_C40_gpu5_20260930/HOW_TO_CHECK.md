新模型：N_no_soft_2layer，从零训练384步。旧A权重仅作为评估基线。

总进度：status.json / run.log
训练进度：N_no_soft_2layer/training/status.json
正式训练日志：N_no_soft_2layer/training.log
前置检查：N_no_soft_2layer/pipeline_status.json、pipeline.log、initialization_native_smoke.log、smoke.log
统一评估日志：C1/C1.log、C40/C40.log
完成判据：根status.json的status为complete，两个formal/summary.json均complete且各6个T0/T1窗口。RESULTS.md分别报告接受长度和吞吐。failed查看stage与对应日志，不用smoke吞吐判断效果。
