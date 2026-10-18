# 定向原文矩阵

核查日2026-10-16；仅按列明章节核查，不是穷尽综述或新颖性认证。作者解释、作者干预证据与本项目推断分开。下列HTML均为固定arXiv版本，GKD为正式会议PDF；网页获取使用只读browser。本轮没有复现作者实验。

|来源与实际阅读范围|异常→作者所指近似|作者改变及关键证据|未排除的解释；本项目使用边界|
|---|---|---|---|
|[Domino v1](https://arxiv.org/html/2605.29707v1)，2026-05-28，§1、3、4、5.3|并行草稿缺块内实际条件；单独最终监督可能绕过主体|GRU汇总已生成前缀＋低秩logit修正；teacher forcing和base-anchored训练对照|作者证据在其设置；本地只能把条件缺失列作竞争来源，不能把它简化成只补一个前驱|
|[DSpark v1](https://arxiv.org/html/2607.05147v1)，§1、3.1–3.3、4.3|边缘式并行预测混合不同续写，后缀不协调|顺序transition、CE/TV/位置目标、置信度与调度；固定长度分析区分部分调度作用|非单一因素全因果分解；候选若仅称多模态碰撞就是已有认识|
|[AngelSpec v1](https://arxiv.org/html/2607.25852v1)，§1、2.3、3.2–3.5|负载和条件利用方式影响草稿；只依前驱的固定transition受限|DFly比较Markov与hidden＋前驱SwiGLU残差，同时考虑条件融合和训练目标|负载熵关联不是独立熵干预；本地输出层主因不能由头收益推断|
|[EAGLE-3 v3](https://arxiv.org/html/2503.01840v3)，2025-04-23，§1图3–4、3、4.2|特征回归约束限制首步；解除后递推输入偏离训练|去约束的首步改善和后步恶化，TTT及融合消融形成不同预测|可借鉴证据链；本地并行主体没有同样的递推特征过程，不能照搬原因|
|[DFlare v1](https://arxiv.org/html/2606.02091v1)，§1、3.3、4、6.1–6.3|共享融合条件下，增深/增加target层的收益饱和|逐层融合、异构KV、位置loss及对应消融/深度曲线|说明具体条件利用假说可检验；本地没有这些干预，不能仅凭raw错误指定融合为来源|
|[DCOCO v1](https://arxiv.org/html/2610.12570v1)，§1–3、附录G|top1错时Top16仍有正确候选|并行候选对打分与prefix-first选路；附录G的proposal为确定点质量|候选已存在/重选不是新解释；与本地完整随机概率诊断口径不同|
|[DistillSpec v2](https://arxiv.org/html/2310.08461v2)，§3–5.2|蒸馏数据和距离改变草稿与target匹配|比较不同divergence与teacher/student生成训练数据，最合适距离依任务|已经覆盖“监督/距离影响接受”的宽泛主张；不证明本地硬标签致错|
|[LK Losses v1](https://arxiv.org/html/2602.23881v1)，2026-02-27，§3–5、6.1|有限模型下KL与overlap最优取舍不同|玩具分布、梯度与KL/TV/混合目标训练对照|已有目标选择机制；实际本地是argmax加D-PACE且已有联合训练，不能当普通KL实验|
|[GKD，ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)，§1、3–4|固定训练前缀与学生生成状态不同|同起点比较student数据比例与距离；teacher概率监督student状态|来源分布改变是已知竞争解释；本轮固定P/D/F比较不是训练因果干预|

补充一次查询：`"D-PACE" late positions weighting draft speculative decoding confidence uncertainty`；仅使用原论文及作者/项目官方代码定位，不依赖返回的二手网页。

|补充近邻|实际核查与影响|
|---|---|
|[D-PACE v1](https://arxiv.org/html/2605.18810v1)，2026-05-12，§1、3、5.2–5.3|作者已讨论固定权重失配，并用平滑的prefix乘积和suffix和分配信用，包含组件/平滑消融。本地同形式权重不能称固定衰减；“后部训练预算不足”只是待检验解释，公式的非零下限既不证明充分训练，也不证明欠训练。K1若场景无覆盖，不得据此继续推荐。|

这些是有边界的定向阅读；没有以“未搜到同名论文”判断创新。未调用更多补充查询。

## 本地证据与负结果

已读取R1、R2、R3、R6、R8、R9的实际报告与R8采集源码；同轮测度及资产分别保留。R1单温度无法复现头全部收益，但没有排除所有校准；条件KL主导未证。R2几何增量未支持。R3统一半剂量整体不优。R6历史风险/方向预测未确认。R8共同探针跨主体/来源有限迁移不识别形成原因。R9共享预测的安全余量和高收益取舍可并存，不识别训练原因。

本地源码 `residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/losses.py`：`dpace_loss`调用teacher-argmax CE，`output_loss`使用完整概率overlap的prefix代理；`eager_objective`显式验证差异。`fused_kernels.py:376`说明CE目标为argmax。这个事实提供监督竞争解释，存档的argmax频数r并非真实训练标签分布或加权最优解。

R8 `hf_frozen_adapter.py:body_states`只以C构造anchor＋mask，固定raw q不读块内H；`collect_r8.py:TargetTree`的p读取完整H。`body_states`同时有共享target LM head和共享融合ctx；旧概率不足以在这几种近似之间定位责任。此前“优先输出适配”的建议已撤回，不采纳其推荐。

旧R2原native `passed=false`、各轮资源/数值披露继续保留。本轮GPU=0，不能撤销这些警报。

AI辅助检索、原文抽取及实现；未做全面撤稿、利益冲突或出版状态审查，不将预印本提升为已复现事实。
