# R6完成：历史、失配与修复成败

固定48题K64=fit16/eval48完成；旧R4 fresh48训练、旧R5 fresh36开发；全部10次限定诊断拟合、T1–T4、完整统计/4类真实图/固定自然对与资源签名已交付。见[RESULTS_CN.md](results/RESULTS_CN.md)。

- HN：0.25937115564182783；98⅓% CI=[0.2095802625430688, 0.3104230511338766]；覆盖=True；supported_within_predefined_population。
- HP：-0.0034874643017817077；98⅓% CI=[-0.02503449407495782, 0.017126899324501798]；覆盖=True；uncertain_interval_includes_zero。
- HC：0.0006764240811061173；98⅓% CI=[-8.008350149641198e-05, 0.002235379994811327]；覆盖=True；uncertain_interval_includes_zero。

T4固定12对，safe松弛候选两支严格改善raw=11对。松弛可行不是原basis/dose可达；未找到不是不可修。HP/HC全eval与HN预定novel人群分开。有限预测器失败不证明历史无信息；T3不能代替T2，ridge B stop7达到1000上限，未证收敛。

预算GPU=0.996440/8h；CPU账本=1.369036/24h；实际新增磁盘约11.425/700GiB，gross写入上界约107.022GiB。自有GPU/CPU实验退出，共享lease释放，未触碰他人任务。数值线程1；驱动/IO线程另记，不伪称全时刻精确线程生命周期证书。

R2原NATIVE_BRIDGE_AUDIT passed=false、R1条件主导/R2几何/R3半强度/R4增量未支持及R4/R5资源披露均保留。本轮没有native/RNG/TPS或真实头训练，没有在线gate，没有标签无关修复分布。不要重跑已完成stage、换特征/seed/阈值/模型救结果、扩大K或补题。不自动开启第七轮；后续需明确新研究授权。

RUN_FILE_MANIFEST全量签静态文件，COMPLETION另签最终元数据例外。冻结大权重本轮只作继承哈希/stat，未冒充新全量SHA。
