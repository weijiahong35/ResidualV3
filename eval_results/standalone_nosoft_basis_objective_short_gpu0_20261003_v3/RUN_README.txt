有效运行：本目录v3，GPU0；控制器PID4016527。
GPU5准备版本未启动；GPU3 v2在native加载前因预留叠加而失败，其记录保留，未训练任何新头。
主进度：status.json；采集进度：collection/fit/progress.json和collection_fit.log。
顺序：native共同采集 -> 三种基底拟合 -> 32提示几何/oracle诊断 -> 初始化native校验 -> 三个新头同状态短训128×3 -> C1/C40 smoke与正式T0/T1。
短训进度：training/status.json，各新组训练384步；旧长训头只读参照，不重训。
正式评测：C1/formal和C40/formal；每套RAW/LONG_BASE/KL_B512/OVERLAP_B512/MARGIN_B512各T0/T1，共20个正式窗口。
完成判断：根status.json为complete；completion_validation.json为complete且formal_windows=20；三组training步骤均384；controller_cleanup.json.finished为true。
最终结果：FORMAL_RESULTS.csv、REPORT.txt；几何诊断：DIAGNOSTIC_RESULTS.json。
若status为failed，查看error及对应阶段log；显存不足可显示waiting_for_memory。额外进程只记录，正式污染窗口不作为干净TPS收益。
基底rank数值/对齐门槛失败会停在诊断，不自动改rank或启动更大训练。
hidden/logits在线使用；只保存聚合梯度矩阵、basis、标量诊断和必要权重，无持久hidden/logit缓存。
