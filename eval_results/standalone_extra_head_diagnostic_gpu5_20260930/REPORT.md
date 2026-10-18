# 额外修复头：资料核查与首轮在线诊断

本报告先诊断现有修复头，不实现或训练新的额外头。固定现有 no-soft 2层 epoch3，8条训练子集＋8条既有 architecture_dev，每条32锚点、7位置。目标hidden由vLLM在线生成，DFlash在线计算；每条结束释放状态；没有持久化hidden/logits缓存。测试C1/C40未用于新增结构选择。开发集此前已使用，不能当作全新最终测试。

## 核实的资料

- DFlash 2 官方说明：https://inco.ai/blog/dflash2/ 。候选选择用原始分数加上前驱/候选的低秩上下文交互；另有块内动态卷积。候选选择器和改backbone的卷积是两个不同干预。
- 官方实现：https://github.com/z-lab/dflash/blob/main/dflash/model.py ，`CandidateSelector` 的两个词表codebook及hidden projection为可学习参数；采样分布在候选支持上重新计算。它没有把显式q当作修正增量条件，不能照搬来替代本项目的q输入要求。
- DSpark 论文：https://arxiv.org/html/2607.05147v1 ，§3.1采用可学习低秩Markov偏置，默认rank256；另讨论RNN。官方实现：https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/markov_head.py 。这些仍然是连续低秩模型，不能用“连续/不连续”区分其与现有头。
- 本项目当前512维B来自训练128的加权未中心化方向二阶矩；方向为normalize((p−q)W)，有first-reject权重。它是冻结的分布mismatch方向基底，不是通用全hidden空间。但direction本身是连续输出，dose只限制幅度。
- 不能直接套用论文参数/时延数字：按公开codebook形状，在本项目V=151936、r=256时，两张完整可训练表就有77,791,232参数；正式新结构不能无论证引入它们。

## 已有native证据

旧C_no_soft三层：C1 T0接受长度3.9545→4.0692（+2.90%），吞吐−7.18%；C40 T0 4.0345→4.1915（+3.89%），吞吐−2.20%。greedy并非接受长度没有改善，而是吞吐未盈利。

仅用AL/TPS构造的有效周期成本比约为1.1086（C1）、1.0623（C40），不是kernel profiler结果，且受prefill/调度/序列长度影响。若该代理成本不变，打平RAW所需AL约4.3840和4.2859；新增头会额外提高成本要求。

## 首轮在线诊断结果

这些是固定已有答案前缀的单步诊断，不是native acceptance length，不能据此宣布额外头有效。`greedy_prefix_compatible`仅保留已有答案前缀与target greedy一致的行；first_position为每个锚点的第一草稿位置。

|数据/口径|位置数|RAW top1一致率|修复top1一致率|修对/改错|Top256覆盖|dose网格oracle一致率|
|---|---:|---:|---:|---:|---:|---:|
|train_probe/teacher_forced_all|1792|56.58%|68.02%|236/31|99.16%|71.76%|
|train_probe/greedy_prefix_compatible|1478|64.01%|72.53%|151/25|99.46%|75.51%|
|train_probe/first_position|256|89.84%|90.23%|2/1|100.00%|91.02%|
|dev/teacher_forced_all|1792|61.10%|71.76%|220/29|99.72%|74.89%|
|dev/greedy_prefix_compatible|1542|68.09%|75.16%|137/28|99.87%|78.27%|
|dev/first_position|256|87.11%|86.33%|2/4|100.00%|88.67%|

dose网格为0、0.25、0.5、1、2、4、8倍当前dose，并限制在原上限内，逐行使用teacher挑最好；仅作乐观诊断，不能部署或视作学会。

开发集全部1792位置的分布重叠均值：RAW 0.55963 → repair 0.70561；top1一致率61.10%→71.76%。第一位置：223/256→221/256，修对2、误改4；分布重叠0.87733→0.87852。样本量小，但说明分布改善与greedy首错改善可分离，不能只看离线loss。

## 可达性与幅度

对每个样本的候选矩阵 C[I]∈R^(256×512) 做FP64 SVD。对已有修正再增加仅target候选的margin0.5，解最小范数系数。检查总系数范数≤0.8‖hidden‖，再通过相同BF16候选logit更新验证top1。此oracle使用标签，只验证单候选集的表示能力，不验证网络能学会。

|抽样|错误案例数|rank256|幅度范围内构造成功且BF16纠正|条件数中位数|所需系数范数/上限中位数|
|---|---:|---:|---:|---:|---:|
|train_probe|32|32|26|80.36|0.579|
|dev|32|32|32|81.09|0.633|

抽样规则：每条前4个前缀兼容且target在Top256内的错误，另加2个正确控制，总96例。并非随机总体估计。构造成功是“可表示”的充分证据；构造超幅度不是“不可能修对”的证明，因为仍可联合调整其他候选。

开发集dose均值约0.4087，最大0.6953，未观察到接近0.8的饱和；不能据此认定其他数据也无幅度限制。现有学习映射、初始化/优化、训练目标和候选几何仍可能影响泛化。

修复自身CUDA Graph计时（包含七步greedy argmax，目标服务器驻留但空闲；不含draft/target forward及verifier）：batch1约1.160ms，batch40约3.181ms。该计时不是完整native性能，不与论文周期时延直接对比。

## 对额外头结构的约束与下一阶段

目前不支持把“冻结512基底表达不了这些错误”作为主要理由，也没有验证任何新头已经有效。更值得对比的是候选条件化的排序残差与继续使用共享基底预测方向的学习差异。建议保留现有direction+dose路径作为对照，在同一Top256上加入候选逐项打分的研究原型；它必须实际读取当前q及候选ID对应的表示，并保留hidden和实际hard前驱。

候选形式仅是待验假设：z_j(v)=u_j(v)+δ_old,j(v)+1[v∈I_j]·rθ(h_j,E[prev],q_j[I_j],I_j,v)。rθ输出每候选额外分数，不是把同一个512维direction网络简单加深。候选外保持原logits，仍按全词表的最终z采样和交给verifier。

下一阶段必须有同预算、同初始化策略、按prompt/group划分的对照：

1. 不加头；冻结旧头下训练一个仍通过C[I]输出的附加方向分支，作为同基底容量对照。
2. 候选条件化双线性残差：共享小投影，交互上下文/实际前驱与候选原embedding，显式q条件；不使用完整可训练V×r表。
3. 若2对难例仍不足，再考虑共享的逐候选非线性标量打分器；不能仅因oracle有空间就直接堆MLP。

先固定OUTPUT目标，避免把结构收益和换损失混为一谈。比较开发集correct→wrong、wrong→correct、前缀兼容位置、第一位置及分布重叠，同时测新增Graph耗时；如收益只来自训练集或首位置伤害加重，则否决。只有泛化和延迟都过关才做正式从零训练/native C1、C40。若仅在换目标后改善，必须作为单独因素报告。

历史 `residualV2_preexp31` 的 dspark/dflash2 风格rank头CV曾未稳定获益；其32k支持、旧状态和旧权重不同，不是当前头的直接否定证据，但要求保留失败对照，不能只凭论文命名选结构。

## 证据与清理

- preregistration.json：数据面板、checkpoint与诊断脚本SHA256。
- summary.json、row_metrics.json、geometry_cases.json：统计量与oracle抽样。
- hidden_cleanup.json：16请求、16次unlink，剩余0；cleanup.json确认专属进程退出、tmpfs目录移除。
- 未修改当前任何训练、native模型或运行中的GPU4评估；未训练新的额外头。
