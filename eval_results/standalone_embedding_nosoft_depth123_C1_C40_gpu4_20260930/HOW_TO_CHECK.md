仅 N_no_soft_1layer 新训练384步。N_no_soft_2layer、N_no_soft_3layer 从旧epoch3复制，证明见各_reference/reuse_proof.json。

总进度：status.json / run.log
前置检查进度：N_no_soft_1layer/pipeline_status.json
正式训练：N_no_soft_1layer/training.log、training/status.json
统一评估：C1/C1.log、C40/C40.log；顺序RAW→去软1层→去软2层→去软3层，T0/T1，正向一轮。
完成判据：根status.json为complete，两个formal/summary.json均complete且各8个窗口，RESULTS.md/results.json生成。分别比较acceptance length和吞吐。
