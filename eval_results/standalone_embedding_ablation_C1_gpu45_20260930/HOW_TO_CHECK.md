当前统一评估顺序：RAW→A→B→C→D→E→F→G。
所有训练已完成，不再训练。smoke和正式评估各包含全部8个模型，每个模型T0/T1，共16窗口。每阶段RAW只跑一次，不按模型重复RAW。

只需看一个日志：unified_RAW_A_G_C1/C1.log
状态：unified_RAW_A_G_C1/status.json（根status.json同步）。waiting_gpu=等待GPU5外部任务释放；SMOKE=全部模型接线验收；FORMAL=正式测试；complete=全部完成。
检查：python unified_RAW_A_G_C1/check_run.py unified_RAW_A_G_C1
结果：unified_RAW_A_G_C1/RESULTS.md，完成后同步根RESULTS.md。
旧逐组测试目录是历史记录，不混入当前汇总。
