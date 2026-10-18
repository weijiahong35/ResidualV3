当前有效调度：单个完整任务占一张空卡，vLLM target、DFlash及修复头在同一卡上；不再拆新采集/训练分片。
2026-10-03用户要求重组后，卡1 RR_KL、卡2 RR_OVERLAP_LOW、卡3 RR_OVERLAP、卡4 RR_MARGIN保留正在进行的完整128×3=384步训练；卡5 RR_OVERLAP_HIGH已完成384步，不重训。
之前的四分片原生采集已全部完成：4096个block，128提示T0/T1，256/256输出及usage与GG采集一致；五个基底拟合已完成，全部复用。
卡0在启动前核实为空，完整32提示T0/T1留出几何诊断先在卡0运行。全部训练完成后，卡0依次完成C1 smoke、C1正式、C40 smoke、C40正式。每个阶段开始前等待该卡为空；阶段中若出现外部进程，仅记录和标记性能污染。
原恢复调度PID242089仅暂停自身，未暂停/停止其四组训练子进程；新调度在四组训练清理完成后自动退出旧调度。闲置watcher PID338526已停。运行中冻结源码未改。
主进度：status.json（与single_card_status.json同内容）；调度：single_card_controller.log；诊断：single_card_geometry.log及single_card_diagnostic/collection/diagnostic/；训练：training_workers/各arm_recovery01/training/各arm/status.json。
完成判断：status.json=complete，completion_validation.json=complete且formal_windows=40、training_steps_each=384，controller_cleanup.json.finished=true。
C1=20/C40=300，RAW/LONG/GG3/RR5各T0/T1，共40正式窗口；smoke不提供测速结论。结果FORMAL_RESULTS.csv、REPORT.txt。GG三个已完成控制与旧长训N只读复用，不再训练。
物理卡号和UUID记录在各阶段protocol/command中。GPU0诊断的新增入口仅扩展设备号校验到0..7，原始protocol.py在磁盘未改，device_schema_extension.json记录该差异。
hidden在线计算、/dev/shm读取即删；无持久hidden/logit缓存。训练恢复的服务器KV预算差异会记录，不声称完整hidden bitwise parity。
旧调度状态、README、ACTIVE_RUN和失败attempt保留为历史证据；恢复状态文件不再是当前主进度。

2026-10-04比较口径：以原始384步短训N为主参照，LONG_BASE只作长训参考；历史旧N正式成绩按checkpoint完整SHA检索并去重，详见analysis_short_baseline_20261004/COMPARISON.txt与AUDIT.json。旧N历史面板与新RR面板不同，按各自RAW并列展示仍只作描述，不构成同面板因果比较。用户要求不重新测试，未新增旧N训练或测试。
