# 资料阅读指南：只读取能改变候选判断的证据

目标是抽取“具体异常→近似环节假说→区分解释的证据→方法”，不以论文标题或最终加速比代替机制。每篇记录版本、日期、核查章节、作者主张、实验证据、我们的推断、与候选的重合。尚未核查的内容写待核。

## 先读本地证据

1. [R1结果](../../mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/results/RESULTS_CN.md)：前驱置乱、单温度、首位置。不可将置乱剩余收益命名为独立系统成分。
2. [R2结果](../../mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2/results/RESULTS_CN.md)：Top256内外坐标贡献、几何预测负结果。覆盖/可行空间不等于来源。
3. [R3结果](../../mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3/results/RESULTS_CN.md)与[R6结果](../../mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6/results/RESULTS_CN.md)：统一减半和新增历史预测未获推荐/确认，防止重复。
4. [R8结果](../../mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8/results/RESULTS_CN.md)、[R9结果](../../mismatch_theory_round9_20261015_v1/runs/20261015T144624Z_mismatch_r9/results/RESULTS_CN.md)：数据口径、共同改善及取舍的有限族边界。两者均已完成，不重开stage。
5. [冻结事实入口](../../../../PAPER_IDEATION_ACL2027_PROMPT_20261013_CN.md)与[训练loss源码](../../../../residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/losses.py)：核实raw的teacher-argmax CE与repair完整概率OUTPUT监督。训练差异是竞争解释来源，尚非因果结论。
6. [既有文献核查](../../mismatch_theory_followup_20261016_v1/MECHANISM_SOURCE_SCAN_20261016_CN.md)：只复用出处、事实和负结果；其“优先输出适配”推荐已由本包撤回。旧thinking梯度证据不升级为E2 nonthinking实测。

R4/R5/R7仅在候选确实涉及对应已知结论时补读原报告，不重算共同修正量。R8数值源与身份核对见执行指南及 `qa/INPUT_AUDIT.json`。

## 先比较这些一手论文

| 原文与建议位置 | 抓住的具体环节 | 必须问的问题 |
|---|---|---|
| [Domino](https://arxiv.org/html/2605.29707v1)、[DSpark](https://arxiv.org/html/2607.05147v1)，引言、方法、消融 | 并行预测缺少实际块内条件，轻量顺序修正；DSpark另涉及训练目标与调度 | 我们的候选是否只是同一信息缺失的改名？不能把对方简化成只加一个token |
| [AngelSpec](https://arxiv.org/html/2607.25852v1)，§1、§3.2、§3.4–3.5 | 不同负载对应不同预测难度和草稿设计；已有hidden与实际前驱残差修正 | 作者怎样从负载差异连接设计？熵关联是否被当成独立因果？ |
| [EAGLE-3](https://arxiv.org/html/2503.01840v3)，§1图3–4、§3、§4.2 | 特征回归约束与递推输入不一致 | 去掉约束后的第一步收益、后续步代价，以及training-time test分别支持哪条解释？借鉴证据链，不能照搬机制 |
| [DFlare](https://arxiv.org/html/2606.02091v1)，§1、§4、§6.1–6.3 | 共享融合条件可能限制草稿各层利用target信息 | 增深收益饱和怎样定位到具体条件方式？哪些比较仍有混杂？ |
| [DCOCO](https://arxiv.org/html/2610.12570v1)，§1–3、附录G | 候选中已有合适token，但并行草稿选择不对 | “候选存在、重选即可”已有多大覆盖？其贪心/候选路径与我们的完整概率设定有何差别？ |
| [DistillSpec](https://arxiv.org/html/2310.08461v2)、[LK Losses](https://arxiv.org/html/2602.23881v1)，目标函数与消融 | 数据、蒸馏距离与接受目标的关系 | “硬标签与概率匹配不同”是否已经足以覆盖候选？额外主张究竟是哪一个条件？ |
| [GKD正式稿](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)，§1、§3–4 | 学生自身生成状态与训练状态不一致 | P/D/F差异是否只是已知状态分布变化？没有训练干预时能说到哪一步？ |

本表是已有定向核查的入口，不代表穷尽检索或复现结果。2026-10-16写作阶段再次打开并核对了EAGLE-3、DFlare、DCOCO原文；其他行沿用已有核查，执行时核查与候选最接近的章节。

## 随候选选择的补充阅读

- 概率/容量/监督：[MiniLLM](https://arxiv.org/html/2306.08543v4)、[Scaling Distillation](https://arxiv.org/html/2502.08606v2)、[Mismatch Matters](https://arxiv.org/html/2608.09836v1)。区分容量限制、目标选择、优化和泛化；不要仅用“模型小”归因。
- 目标信息利用或logit修正：[SD²](https://arxiv.org/html/2511.09844v1)、[FlexDraft](https://arxiv.org/html/2605.20022v1)、[xPress](https://arxiv.org/html/2608.02438v1)。额外target信息与重新映射原信息必须分开。xPress具体损失系数先核代码，当前不引用未解歧义的数值。
- 熵/训练难度：[EDSD](https://aclanthology.org/2026.acl-long.2145.pdf)，§3.1–3.2、§4.3。熵是候选描述或课程信号；p=q时高熵也没有分布失配。
- 位置/连续接受：[D-PACE](https://arxiv.org/html/2605.18810v1)、[VSD](https://arxiv.org/html/2602.05774v1)。仅在候选确实依赖位置目标时读；旧档案不提供新策略整段轨迹证据。

候选冻结前可针对具体错误、近似环节及同义词进行最多8组补充查询，优先论文原文、正式会议稿、作者官方代码。保存检索式、日期和支持位置；不因没找到同名词就认定新颖。若原文不可获得，保留novelty_unknown，不能靠摘要写细节。

## 读完必须留下什么

文献矩阵每行包含：异常、作者解释、真正改变的因素、关键对照、支持范围、未排除的解释、对当前候选的影响。报告不能只写“某某论文也提到mismatch”。优先回答：**这个候选是已知问题、已知机制的新适用边界，还是尚有可区分的新来源假说？**
