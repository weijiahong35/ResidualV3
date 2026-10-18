# 2026-10-11 23:11北京时间：跨域评测续跑记录

已完成HumanEval C1四个正式窗口；T0/T1 AL分别+14.48%/+21.50%，TPS分别+10.85%/+17.48%。这些结果保留，当前总正式进度4/16，没有最终完成审计。

HumanEval C40之前的smoke失败：FULL15_NOFILL启动期间，其他用户的GPU7进程PID3573779进入，CUDA显存不足；此外该尝试的DFLASH隔离记录也出现外来进程。整组尝试仅为预检，不能作为正式速度证据。日志、协议及部分结果已完整移入prior_attempts，逐文件SHA核验。

MT-Bench C1在GPU0继续原正式评测，不中断、不重跑。原v3控制器负责MT-Bench与根报告；独立v4恢复任务只接管已失败的HumanEval C40。所有GPU0–7的宿主空卡均可用，仍须无GPU进程且空闲至少78000MiB。两路使用同一GPU租约锁，避免自己的评测互抢。已完成C1的79份文件SHA保留；原183份native源码、数学合同及权重不变。只重排未完成对照；如果外来进程在两臂切换时进入，保存整组尝试并重选空卡；数值合同失败不会自动放宽或忽略。

恢复任务完成后，由仍在运行的原控制器或恢复任务完成四组/16窗最终审计。当前状态以实时JSON和log为准，历史PID不能直接视为活跃。

[总状态](status.json)、[MT-Bench C1正式log](mt-bench_C1_formal.log)、[HumanEval恢复log](parallel/controller_v4.log)、[原主控制器log](parallel/controller_v3.log)、[恢复身份](parallel/RECOVERY_MANIFEST.json)、[恢复准备审计](parallel/RECOVERY_READY_AUDIT.json)。

[失败HumanEval C40 smoke日志](humaneval/C40/prior_attempts/occupation_2026-10-11T151046.423046_0000/humaneval_C40_smoke.log)。
