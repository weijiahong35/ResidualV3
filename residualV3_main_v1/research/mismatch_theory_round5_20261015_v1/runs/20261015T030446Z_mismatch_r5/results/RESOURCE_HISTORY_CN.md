# R5资源、恢复与保留事项

本轮GPU采集只有一次启动，36题全部完成，没有GPU失败、重试或换卡。累计计费 **0.367712753 GPU小时**，包含启动、编译及共卡运行，低于本轮8小时上限。采集结束时间为2026-10-15 03:46:19 UTC（北京时间11:46:19）。GPU预算不会因本文件或后续CPU分析另起账本。

启动前重新检查全部8张可见卡的UUID、显存和预约，按阶段预计26GiB工作集加4GiB余量选择可共用的卡，未使用空卡gate。实际使用GPU7、UUID `GPU-d74003c2-c31c-a723-8de9-4b1e1a41cf3f`；准入时已有外来PID1373911。自有显存峰值21768MiB，低于32GiB。原正式六数据集与CPU收尾的终态重新核验，未重启旧控制器、parallel或收尾，也未管理外来进程。完整命令、UID/PID/PGID/startticks及全部卡快照见 [GPU_BUDGET.json](../logs/GPU_BUDGET.json)。

CPU累计上限24小时，数值作业由同一个并发安全账本计入监督器与其子进程。监督入口在import前设置OMP/OPENBLAS/MKL/NUMEXPR为1，GPU工作进程另设置Torch intra/inter-op为1。核心分析只用单个CPU worker；tokenizer及驱动/IO辅助线程另存记录。GPU阶段实际观察到的自有进程树OS线程峰值为7，三次实时线程名称及CPU ticks快照见 [LIVE_GPU_THREAD_SCOPE_AUDIT.json](../logs/LIVE_GPU_THREAD_SCOPE_AUDIT.json)。这不是对所有时刻全部OS线程数的全局证明，也不把未测量线程改称IO线程。

必须保留一处本轮记录缺口：锁前一条协议元数据写入命令在CPU监督入口外导入了common.py/NumPy，未显式设定四个线程环境变量，未留下该进程全生命周期线程观测。该命令未做模型、矩阵或概率分析；没有证据确认计算线程超限，但也不能出具“全run线程无条件合规”证书。后续直接元数据命令均使用显式线程环境包装。见 [METADATA_THREAD_RECORD_GAP.json](../logs/METADATA_THREAD_RECORD_GAP.json)。核心数值合同通过不消除此记录缺口。

新增磁盘按用户本轮授权上限 **700GiB** 管理。写前检查本轮共享gross预约账本与宿主物理余量，定期扫描实际分配字节；保留失败分片、临时缓存和副本，预约不因文件删除而重置。小型元数据留有独立8GiB保守额度；收尾及报告也计入。瞬时精确峰值没有连续测量，因此分别报告采样实际峰值与保守写入上界。最终CPU累计、磁盘实际分配及gross上界以 [RUN_AUDIT.json](RUN_AUDIT.json) 和 [COMPLETION.json](../COMPLETION.json) 的闭合账本为准，不能用某一中途快照冒充最终数字。

锁前有两项已定位的接口恢复，失败日志均保留，没有更换题目或探针：

1. 当前tokenizer默认返回BatchEncoding，首次输入审计沿旧默认假设失败。核对正式prepare.py后显式使用同样的return_dict=False；修订后36题与原面板token逐一相符。见 [TOKENIZER_INTERFACE_DIAGNOSIS.json](../analysis/TOKENIZER_INTERFACE_DIAGNOSIS.json)。
2. NumPy连续化会把0维scalar的shape从()提升成(1,)，首次旧dev读取的hidden_norm哈希因而失败。修正读取端保留原shape，单独scalar合同通过后才恢复；失败分片位于logs/partial_dev_scalar_reader_revision0，第一次成功pilot已完成的题在18题分析中直接KEEP。见 [SCALAR_TENSOR_HASH_CONTRACT.json](../analysis/SCALAR_TENSOR_HASH_CONTRACT.json)。

上述恢复均在科学锁之前；新36题打开后没有更换科学代码、资格、采样、阈值、统计或案例规则。来源复制的初始SHA与资源入口修订后的SHA分别记载，不把最初复制清单误称最终字节身份。见 [PRELOCK_RESOURCE_SOURCE_REVISIONS.json](../logs/PRELOCK_RESOURCE_SOURCE_REVISIONS.json) 与 [LOCK.json](../protocol/LOCK.json)。

来源核验角色补存了已经阅读的同一Banerjee2005 PDF/文本，SHA与锁定阅读记录一致；未重新搜索或更换版本。单次宿主监督器及子进程CPU实测0.264527秒，账本保守记1秒，不重复累计worker。见 [BANERJEE_LOCAL_SNAPSHOT_RECEIPT.json](../logs/BANERJEE_LOCAL_SNAPSHOT_RECEIPT.json)。小型非监督元数据/只读工具另以初始及闭合保守秒数记账。

R1–R4、冻结权重/basis/源码、安装环境及正式结果保持只读。实际使用的完整旧数据、201项冻结来源及旧签名报告核SHA；23项大型资产核stat并继承既有可信身份，未伪称本轮重新计算全部模型权重SHA。R4线程超限、部分证明事后重构及CPU/GPU投影差异等历史披露仍在旧文件中；R2原NATIVE_BRIDGE_AUDIT的passed=false保持不变。科学结果完成、数据身份通过与资源记录的有限范围分开记录。

最终宿主审计仅核查自有真实身份退出和共享lease释放，不发送终止他人进程的信号。静态全文件SHA清单覆盖本轮数据、源码和报告；CPU/disk最终账本及闭合元数据由COMPLETION另签。沙箱PID记录带有命名空间，不能将PID2/6当成宿主PID；GPU及最终宿主核查使用真实宿主命名空间。收尾后不继续GPU工作，不自动训练或开启第六轮。
