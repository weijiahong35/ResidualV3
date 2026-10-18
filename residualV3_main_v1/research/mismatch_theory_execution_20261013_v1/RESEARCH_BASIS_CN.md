# ResidualV3：mismatch 理论研究基础与证据边界

版本：2026-10-13 v1。用途：给执行 chat 的研究依据；不是论文初稿，不是实验结果。
配套：[执行任务书](EXECUTION_GUIDE_CN.md)、[启动 prompt](START_PROMPT_CN.md)。

## 1. 用户意图与研究范围

用户已明确：现有模型已经有可观的接受长度与吞吐收益；当前薄弱环节是理论解释和相对近邻的贡献，而不是必须重新发明一个头。应以大小模型的分布失配为研究对象，从理论问题、可检验推论走向方法解释。AR 修复头是方法实现，不能成为研究问题的全部。

中心问题：**大小模型失配中有哪些结构能被现有 ResidualV3 利用和修复？这种修复怎样影响连续前缀接受，能解释现有收益的哪些部分，又在哪些条件下失败？**

研究假设：给定计算和信息约束，草稿与 target 的失配包含不同成分；其中部分残差可被现有头利用和修复，而修复发生的状态及其前缀可达性影响实际接受收益。更强的“可预测”主张需要独立留出预测任务，不能仅凭头有效得出。此假设尚未证实，不是执行者必须“做成功”的结论。

当前交付是研究执行任务书。执行阶段以现有模型与实验为基础，可开展独立诊断、对照、轻量探针；不自动开始论文写作，不把机制研究替换成模块搜索或大规模复现。

## 2. research canon：已知事实

1. target 为冻结 Qwen3-8B；当前论文主体为 nonthinking。主体一次并行计算 block16（anchor＋15 proposal），修复按位置使用实际采样前驱。
2. 修复头读当前位置 hidden、实际前驱的冻结 target embedding、原始 q 的 basis 特征。三个 256 维 SwiGLU 残差块；新增可训练参数 4,353,537。输入与输出 basis 不同。
3. 在原始 Top256 候选上施加 logit 残差，最终从修复后的完整词表分布采样。候选残差稀疏不等于 q 在其余词上的概率为零，也不等于只在 Top256 采样。
4. 独立 DFlash 与联合主体＋头从同一起点分别追加两轮训练；主结果两臂主体权重不同。完整配方收益不能全归因于推理开头。
5. 修复训练使用 teacher-prefix 前驱；正式推理使用实际采样前驱。OUTPUT 是 teacher-prefix 上的前缀 overlap 乘积代理，不是未经条件限定的 native 接受长度。
6. 新六数据集批次的五套数据已形成 20 个有效配对；上一个只读核验中其 AC/TPS 增幅均为正。实时有效窗数曾从 42 降至 40，不能沿用旧快照；原因和现状见任务书。
7. 20 配对的描述性位置结果：第 2–15 个 proposal 的累计存活率全部提高，首位置在 9 个配对中下降。它来自不同整体模型各自轨迹，属于待解释观察，不是单头因果机制证据。
8. E1 有同联合主体开关头的历史 AC 证据；其 acceptance-only 协议不是当前 E2 吞吐证据。不能把“缺当前归因”说成“历史从未做过”。
9. 两层对照只是同三层权重推理跳第三块，没有重训；模块时间有 C1/C40，不能填 C32。尾部停修、mismatch 专家等历史负结果必须保留。

上述事实以任务书列出的源码、protocol、summary 和总账为来源；接手必须核验模型与数据身份。thinking、RAW/E1 历史、新旧 max2096/max2048 批次分别处理。

## 3. 文献：按理论问题读，不按模块名称排列

以下来源已在构思阶段定位并核对过相应内容。执行 chat 必须自行打开原文并记录阅读范围、版本、URL、访问时间及必要的局部摘录。新下载资料放本研究 runs 目录，不能改原有文献快照。无法访问时用已存原文并标版本；不能用摘要代替方法核查。本文献表是定向检索，不是系统综述。

| ID | 一手来源与优先阅读范围 | 应回答的问题与边界 |
|---|---|---|
| L01 | [Leviathan et al., Fast Inference from Transformers via Speculative Decoding, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)，§3 | 固定状态接受为何等于 overlap？几何级数速度分析用了哪些简化？不能把 iid 接受假设当当前实测事实。 |
| L02 | [DistillSpec, ICLR 2024](https://arxiv.org/html/2310.08461v2)，§3–4、Theorem 4.1、§5.2、Appendix A.2 | 训练轨迹分布与散度怎样分别影响 draft–target 对齐？任务质量不等于分布保真度；其接受指标与本项目 AC 是否一致？ |
| L03 | [Online Speculative Decoding, ICML 2024](https://arxiv.org/pdf/2310.07177v4)，§3–4 | 训练/服务分布偏移如何造成失配？在线更新证据不等于本项目有在线学习。 |
| L04 | [A Theoretical Perspective for Speculative Decoding Algorithm, NeurIPS 2024](https://arxiv.org/pdf/2411.00841)，§2.2、§3 | 条件 TV 与拒绝数关系；注明忽略 draft 成本、长 lookahead 等假设。不可直接推导 block15 的 TPS。 |
| L05 | [Cho & Hariharan, On the Efficacy of Knowledge Distillation, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.pdf)，§5 | 容量不匹配有哪些实验证据？这是视觉分类研究，不能直接证明 Qwen3/DFlash 的容量下限。 |
| L06 | [Stanton et al., Does Knowledge Distillation Really Work?, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf)，§4–6 | 泛化、模仿保真度、可辨识性与优化困难怎样区分？足够容量也可能拟合失败。 |
| L07 | [LK Losses, 2026 v1](https://arxiv.org/html/2602.23881v1)，§3–4 | 有限模型中 KL 下降为何不保证接受提高？不能用训练损失下降替代实际 overlap/存活。 |
| L08 | [SpecTr](https://arxiv.org/html/2310.15141v2)，§3–4；[Block Verification](https://arxiv.org/html/2403.10444v3)，§2–4 | 单 token 最大耦合与整块最优接受不是同一问题。核查符号：最大相等概率=1−TV，最小不等概率=TV；不要照抄 HTML 中可能的符号错误。 |
| L09 | [D-PACE](https://arxiv.org/html/2605.18810v1)，§3、§7；[Spec-AUF](https://arxiv.org/html/2607.01893v1)，§4、§6；[Acceptance-Aware Draft Model Training](https://arxiv.org/html/2609.24150v1)，§3 | 前缀价值、动态位置权重、teacher-prefix overlap 乘积已有何种先例？已有 OUTPUT 形式不是新理论。 |
| L10 | [Non-Autoregressive Neural Machine Translation, ICLR 2018](https://arxiv.org/pdf/1711.02281)，§2.3；[Hydra](https://arxiv.org/html/2402.05109v2) | 独立预测、序列依赖与多种合理续写混合的历史认识。不要声称首次发现依赖缺失。 |

近邻是理论解释的竞争对象，不只比较收益数字：

| 来源 | 必须保留的事实 |
|---|---|
| [Domino v1](https://arxiv.org/html/2605.29707v1) | GRU 累计前缀＋logit 残差；主表与匹配训练表要分开。 |
| [DSpark v1](https://arxiv.org/html/2607.05147v1) | 默认 Markov 与 RNN 变体分开；离线 AC 与完整生产调度 TPS 分开。 |
| [DFlash-2 作者报告](https://inco.ai/blog/dflash2/)及[官方模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2) | 选择头与卷积完整版分开；Top16 草稿支持限制不等于 target 输出失真。 |
| [AngelSpec / DFly v2](https://arxiv.org/html/2607.25852v2)，§2.3、§3.2–3.3、§5 | 已有 hidden＋实际前驱修正、前缀 overlap 目标、成本分析。v2 明确补充该目标来源；不要静默混用 v1/v2。 |
| [xPress v1](https://arxiv.org/html/2608.02438v1)，§3–5 | 条件失配、主体表示复用、Jacobi 修正、vLLM bs32 已有研究；发布 DFlash baseline 与匹配重训 Markov baseline 身份不同。 |
| [DSpine v1](https://arxiv.org/html/2609.36173v1)，§3–4、Appendix C | 每层注入与最终顺序候选选择都有；不同方法 T1 proposal 口径须单独审计。 |

新近 arXiv 工作按预印本处理，除非另核实发表状态。上述链接不是性能已独立复现的证明。不以跨论文 AC/TPS 直接排名。

## 4. 理论对象、推导及其限制

### 4.1 统一符号

- C：固定已验证上下文与已知 anchor；i=1…15 是 proposal 位置。
- H=X_<i：块内实际前缀；s=(C,H,i)。
- p_H：target 在该状态上的下一 token 分布；q_H：该方法实际用于抽样的 proposal 分布。
- raw q_i(C) 在主体计算后固定；联合主体的 raw 与独立 DFlash 的 raw 是不同预测，必须分别标注。
- μ：明确指定的前缀分布。teacher-prefix、target 分支、某一方法的 draft-prefix、已接受前缀分别记录，不能互换。
- Z：方法可见信息。raw 为预计算状态；当前 repair 增加实际前驱，但没有完整块内历史状态。

所有 T1 数学量都使用采样变换后的 p/q。原始 logits、温度后的分布、过滤后的 proposal 必须有不同字段。T0 用贪心一致性与首拒绝；可附带计算 T1 诊断距离，但不把它当 T0 接受概率。

### 4.2 固定状态的一步失配

\[
D(s)=\operatorname{TV}(p_s,q_s)=\tfrac12\sum_v|p_s(v)-q_s(v)|,
\quad a(s)=\sum_v\min(p_s(v),q_s(v))=1-D(s).
\]

这是已有恒等式，不是新贡献。一次 T1 随机拒绝不等于高 TV，更不等于语义错误。平均距离改善也不能直接证明整轮加速。

### 4.3 条件变化与拟合偏差的精确 KL 分解

固定 C,i 和前缀测度 μ，令 \(\bar p=\mathbb E_{H\sim\mu}p_H\)，q 是对这些 H 不变的原始预测，则：

\[
\mathbb E_H\mathrm{KL}(p_H\|q)
=\mathbb E_H\mathrm{KL}(p_H\|\bar p)+\mathrm{KL}(\bar p\|q).
\]

证明：在 log(p_H/q) 中加减 log(\bar p)，第二项交换有限和与期望即可。全 support 的 softmax 下有限；数值实现不允许悄悄 epsilon 裁剪并称精确。

第一项反映不同 H 所要求的 target 分布变化；第二项保留 q 对平均分布的拟合偏差。q 不假定等于 target 边缘。第二项不能仅凭大小参数差归为“不可约容量误差”，还可能含优化、数据与目标偏差。

当 H 来自 target 自身分布时，第一项可解释为相应条件互信息；其他 μ 下仍是 barycenter 分解，但不得不加限定地称 target 自然分布下的互信息。有限 K 分支可得到经验分布下的精确分解，却不等于总体真实分解；报告 K 的敏感性与抽样不确定性。

位置 i=1 的块内 H 为空，因此第一项为零。后续项不保证随位置单调。当前 repair 可在固定前驱的子组重复分解，但自然 token 分组稀疏时应报告不可估，不强行给“剩余历史占比”。KL 分解不能当 TV 加法分解或接受长度分解。

### 4.4 信息不可区分的 TV 限制

给定 Z=z，H,H' 从同一 μ(·|z) 独立抽样，任意只读取 z 的预测 q_z 满足：

\[
\mathbb E_{H|z}\mathrm{TV}(p_H,q_z)
\geq \tfrac12\mathbb E_{H,H'|z}\mathrm{TV}(p_H,p_{H'}).
\]

由 TV 三角不等式即得。它是平均一步风险下界/overlap 上界，不是整块 AC 下界。样本 U-statistic 的点估计不是总体严格下界；人工挑选前缀只能做例子。已接受前缀测度依赖方法，不可用两种方法各自的 RHS 当共同结构常量。

更完整的风险概念：固定 μ 与可见信息 Z，模型类 F 是所有 q_Z 的子类，可将实际风险写成“所有 q_Z 的最优风险＋F 相对这个最优值的差距＋当前训练结果相对 F 最优值的差距”。这只是非负概念分解，不意味着三项可由当前日志识别。训练→部署的 μ 变化另作分析。

### 4.5 从一步失配到连续前缀

标准逐 token 拒绝、固定 block、先忽略 EOS/输出预算终止时：

\[
S_k=\sum_{x_{1:k}}\prod_{i=1}^k\min\{p_i(x_i|x_{<i}),q_i(x_i|x_{<i})\},
\quad \mathbb E[K]=\sum_{k=1}^{15}S_k.
\]

定义 ν_(i−1) 为前 i−1 个 proposal 全接受条件下的前缀分布，则
\(S_i=S_{i-1}\mathbb E_{H\sim\nu_{i-1}}[1-\mathrm{TV}(p_H,q_H)]\)。S_(i−1)=0 时后续全零，不定义该条件分布。

另一种无歧义的采样估计：从该方法实际 proposal 联合分布 Q 抽整块，令 b_i=min(1,p_i(x_i|H)/q_i(x_i|H))，则 \(S_k=\mathbb E_{X\sim Q}\prod_{i\leq k}b_i\)。不要把 b_i（已抽样 token 的验证概率）与 a_i（整词表 overlap）混淆。

本项目 AC 通常包含一枚修正/bonus token。只有完整可发出该 token 且计数定义一致时，才用 AC=1+∑S_i；遇 EOS、预算、变长或特殊轮次需单列。teacher-prefix overlap 乘积是代理。普通 target/draft-prefix 平均 overlap 的乘积也不普遍等于 S_k。

### 4.6 什么可能形成新贡献

已知：失配存在、一步 overlap 恒等式、独立性缺失、前缀价值、以上标准分解。

待检验：在真实 E2 上，哪类分布误差占重要地位；已有头在哪类状态降低误差、在哪类状态失败；这些规律能否在锁定留出上解释/预测连续接受收益，且比仅用原始置信度、位置或平均 TV 更有解释力。

“低秩残差”“可预测方向”“可修复质量”“普遍容量鸿沟”均不得预设为事实。若研究只覆盖 Qwen3-8B＋DFlash，就把结论限定于该对象；AR 头外观相似既不能否定也不能证明理论贡献。

## 5. claim–evidence 表

| ID | 主张 | 现有证据 | 状态/仍需什么 |
|---|---|---|---|
| C1 | 整体联合配方提高当前已测 AC/TPS | 新批次 COMPARISON、正式 summary | 已支持于已完成配对；整批仍待最终审计 |
| C2 | 所有收益均来自推理修复 | 主表主体不同；E1 有历史开关头 AC | 不应声称；需要 E2 同主体同状态对照 |
| C3 | 大小模型失配由多种限制产生 | L02、L05、L06、L07 | 文献支持一般可能性；本项目占比未测 |
| C4 | 前缀条件变化解释本项目主要误差 | 分解与 TV 界只证明可能性 | 假设；实验 A，不强求占主导 |
| C5 | 当前头利用了与实际前缀相联系的残差信息 | 整体收益只是间接线索 | 假设；实验 B，留出验证、前驱置换与简单校准参照 |
| C6 | 一步改善能解释 native 前缀收益 | 位置存活描述、标准递推 | 待验证；实验 C，μ 与 terminal 明确 |
| C7 | OUTPUT 函数形式首次提出 | L09、DFly 已有相关形式 | 不应声称 |
| C8 | 已优于主流 AR 修复头 | 跨论文协议不同、本地移植身份复杂 | 不应由跨论文涨幅推断；只能报告已核验条件下对照 |
| C9 | 风险高即值得修、oracle 可修即能学会修 | 历史诊断与专家/尾部负结果不支持此跳跃 | 不应声称；分别报告风险、可修复潜力、实际收益 |

## 6. 论证图与实验契约

论证顺序：受限草稿逼近 target → 明确同状态与前缀测度 → 分辨条件变化/拟合偏差 → 检查可学习修复 → 验证连续接受后果 → 解释现有方法并限定范围。

竞争解释：联合主体本身变强；简单概率校准已足够；teacher-prefix 改善不能迁移；收益主要来自数据/训练改变；位置构成不同造成伪相关。正结果必须对这些解释提供区分证据，负结果也回答问题。

| 部分 | purpose / inputs | 允许结论 | 禁止结论 / validation |
|---|---|---|---|
| 来源核查 | L01–L10、近邻原文 | 已知知识与开放假设 | 不把预印本或摘要当独立复现；锁版本/阅读范围 |
| 实验 A | 冻结 raw、固定 C、多分支 p | 指定 μ 与样本下的误差结构 | 不声称真实不可约容量下限；检查分解闭合与 K 偏差 |
| 实验 B | 同主体同 H 的 p/q/raw/repair，分支间前驱置换 | 配对局部分布改善及对前缀对应信息的依赖 | 不冒充独立模型全收益归因；置换仍可能改变联合输入分布，不推导贡献百分比 |
| 实验 C | 各臂自己的 native proposal 与验证 | 有限块存活和局部风险的关系 | 不混 teacher 代理；检查 EOS/bonus 与计数 |
| 综合结论 | 锁定留出与所有负结果 | 支持/削弱/不确定的具体解释 | 不用相关性写“全部由…导致”；不替论文编写已证明定理 |
