# 第二轮执行 chat 启动 prompt

复制以下分隔线后的全文。此次是新一轮限定执行授权；原研究按归档保留。

---

你负责继续 ResidualV3 的 mismatch 理论研究。仓库：`/data/weijiahong/speculators-main-lora`。

先完整阅读：

1. `residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/ARCHIVE_INDEX_CN.md`
2. `residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/EXECUTION_GUIDE_CN.md`
3. `residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/THEORY_PLAN_CN.md`

上轮原run为 `residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2`。其A/B已完成，HF参考120题留出上置换−正确TV=0.179090、联合raw−正确=0.250882；条件KL未证明主导，约12.69%状态恶化。native C因三个不同工程缺陷退出，成功验证记录为0。这些是已归档事实，不能改写或重跑旧主分析。

**我的上层问题仍是大小模型分布失配。** 这轮要进一步问：误差相近时，允许的修正范围为什么不同，实际头为什么有时成功、有时失败？AR头是研究载体。标准KL/TV公式、前驱有用和Top256本身都不能自动成为新贡献。

我授权你按新任务书自主完成新run中的资料核验、代码修复、CPU检查、冻结模型pilot、限定dev/新留出实验、低容量诊断拟合和归档。不要只给计划或再次问是否可以开始。旧NEXT_CHAT禁止继续的是旧run；本次已授权新的限定轮次，使用新目录、新协议和新预算，保留旧gate/源码/结果原样。

执行顺序：

1. 校验归档和原始资产；实时核对原六数据集FINAL/COMPLETION与宿主资源。创建 `mismatch_theory_round2_20261014_v1/runs/<UTC时间>_<标识>/`。原批次已完成，不重新启动controller/parallel/收尾，不沿用旧空闲PID快照。
2. 优先修native测量桥接。scheduler提供权威request/轮次/预算/proposal长度；GPU worker提供真实token和同轮实际p/q。CPU[-1]是特定接口的长度占位，不能当真实token，也不能只删检查后放过错配。显式区分warmup、普通/零输出prefill、proposal验证、停止后输出和HTTP usage。
3. 先完成一条B0请求且得到真实verifier记录，再跑每域2旧dev题×B0/B1/B2；原始/优化头parity须有固定容差和越界失败动作，不能只记字段。整数输出计数必须闭合，概率期望与随机接受按统计不确定性核验。只有health/anchor不算桥接完成。
4. 第二条工作线按THEORY_PLAN检查放宽修正集合：只改raw Top256 logits时，尾部总质量可变、内部条件比例保持。其TV oracle下限是 `p(U)·TV(p(.|U),q(.|U))`，需先验证采样支持及数值身份。它是放宽集合的标准数学工具，不能称真实rank/dose头的可学习最优解。
5. 用每域12条旧dev题（包含上述2条）开发新量和固定低容量解释模型。旧C/H可重放采新字段，不能重开旧120题主分析。M0含raw TV/位置/域/候选覆盖等，M1只增加尾部形状风险；比较新留出上的配对预测损失，不能用代数上界或相关性代替新发现。oracle特征需要target，不能宣传在线gate。
6. 新test默认每域16题共48，排除上轮全部192题和本轮调试题。MT-Bench80题已用64，只剩16；先去重和暴露审计再锁定。新test仅是本轮机制分析留出，不称从未用于评测/训练。锁定代码、特征、正则、阈值和统计后一次打开；按prompt分组，负结果不扩搜。
7. 桥接通过后检验native reached局部改善和实际接受/发出，三臂各自轨迹分开。Δ_N是主要局部迁移量；B2−B1接受proposal/验证轮为关键后果，实际输出/全部调度轮为辅助后果。仅Δ_N为正不能写接受收益已迁移。HF/native的EOS、算术和测度身份保留；G∏b中的G不含已实现接受事件，不能误用普通轨迹∏overlap。带插桩数据不报告服务TPS。
8. 按任务书七个问题交结果、图表、claim–evidence、审计和NEXT_CHAT。bridge若失败只关闭native支路，继续独立HF理论诊断；负结果和未解决问题同样完整交付。方法改动最多提出两项有证据驱动的后续建议，本轮不训练新头/主体。

**新轮资源：** 同时最多1张实时确认空闲、无预约冲突的GPU；总累计8 GPU小时，桥接≤2小时且计入总额，所有启动/失败/重试/采集/清理计费；新增磁盘≤50GiB、CPU worker≤4。预留保存退出时间，到限停止核实身份的自有工作。只管理记录UID/PID/PGID/启动时间/CMD/GPU UUID的自有进程，不改安装环境、原冻结资产或他人作业。

**工程自主修复：** 不沿用“累计三次失败必停”。每次重启都须有已定位原因、代码改动、对应CPU检查和下一验收点；不同缺陷可在预算内继续。无新证据不重复启动；同一缺陷连续两次修订仍无定位进展或桥接2小时用尽则退出该支路，不能换名字重置。CPU约4小时是进展检查点，不是到点强停。

执行过程中简短报告发现和下一步，持续推进授权工作。不要把“文献没搜到同名方法”“公式成立”“工程终于运行”写成理论创新；成功标准是可靠的新解释、可反驳的边界，以及它能否联系到真实接受过程。
