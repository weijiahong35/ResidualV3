有效版本v2，卡2–5；v1因卡1外部任务及总控制器清理错误在smoke阶段失败，无新头训练，证据保留。
四卡分片采集128提示实际原生到达状态的XTX/XG/GG，拟合五个可预测基底。卡2依次RR_KL、RR_OVERLAP_LOW；卡3 RR_OVERLAP；卡4 RR_MARGIN；卡5 RR_OVERLAP_HIGH。
五新组均rank512、no-soft两层Top256、原q输入基底、随机初始化、128×3=384步OUTPUT；普通GG三个控制复用卡0v3新头，不重训旧N。
CPU实际双进程故障隔离测试：单卡失败不提前中止其他卡；native显存启动失败独立等待并重试，旧attempt保留。外部进程只记录不操作。
进度status.json、workers/gpu2..5/collect_status.json；实际采集collection/fit/progress.json。训练training_workers/各arm/train_status.json和training/status.json。
完成判断status.json=complete、completion_validation.json=complete且formal_windows=40、training_steps_each=384、controller_cleanup.json.finished=true。
C1卡4=20/C40卡5=300，RAW/LONG/GG3/RR5各T0/T1共40正式窗口；smoke不是测速。结果FORMAL_RESULTS.csv和REPORT.txt。
无持久hidden/logit缓存；只存聚合矩阵、基底、标量及必要权重。固定随机输入投影+线性回归不是全部输入信息充分性的证明。
