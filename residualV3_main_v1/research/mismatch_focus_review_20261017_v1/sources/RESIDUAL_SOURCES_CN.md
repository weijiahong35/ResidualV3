# 残差修复、低秩方向与数据加权几何：一手来源核查

日期：2026-10-16 UTC / 2026-10-17 UTC+8。职责：为研究收缩提供文献事实、理论适用边界和竞争解释；这是来源备忘录，不是新增执行指令。没有训练、加载模型、采样或修改旧资产。已读仓库 `AGENTS.md`。

## 最重要的判断

**“从误差选方向比只看 W 选方向更好”不是一个尚无人讨论的大理论。** 残差蒸馏已有 ResKD/Caprese；梯度选方向已有 LoRA-GA；数据加权几何已有 CorDA/EVA/FWSVD/SVD-LLM；data-weighted teacher-update 子空间理论已有 SAD-LoRA。它们使用的数学对象与本项目并不相同，但足以否定泛化的首次性叙述。

本项目还有一个具体、没有被当前 E/G/R 实验分开的判断：**优势是否需要真实目标预测与草稿预测的对应关系，还是仅仅因为 E 看过实际数据、候选词和概率分布，而 G 把整个词表均匀对待？** 文献支持把它作为竞争解释；不支持预先宣布哪一个成立。

本地已读 [RESULTS_CN.md](../../mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse/results/RESULTS_CN.md)：固定参考成本阳性；尺度匹配后仍阳性；27 次头拟合未达到学习优势判据。不能用这些文献把本地不确定改写成“只是优化问题”。

## 逐篇核查

### 1. Caprese：有明确来源的误差，先证明低秩可补，再补真实功能

- WHY：FF 稀疏近似损害长推理表现；所研究的误差明确来自被替换的 FF 计算。
- HOW：用小型低秩线性分支拟合完整 FF 与压缩 FF 的输出残差，之后进行端到端表示蒸馏。
- WHAT：正文 Figure 2 先检查残差的低秩可近似性；§4 再检验任务表现与延迟。§3.1 的目标是固定样本上的低秩回归 MSE，不是概率 TV 或最小隐藏改动。该残差来源可直接定位，不能据此推断我们大小模型残差的来源。
- 对我们的含义：可借鉴“已知具体近似→残差结构→定向修复”的论述；不能只借“残差低秩”作为新意。
- 读范围：摘要、§1/Figure 2、§2.2、§3.1–3.5、§4 开头。版本 v3，未在本次核实录用状态。
- 原文：[Scalable LLM Reasoning Acceleration with Low-rank Distillation](https://arxiv.org/html/2505.07861v3)。

### 2. ResKD：让额外学生专门学习未被原学生学会的部分

- WHY：教师与学生有性能差距。
- HOW：以教师减学生的 logits 残差训练更小的 res-student，叠加后继续修复；另有按样本节省计算的策略。
- WHAT：图像分类上的多个教师/学生配置与残差学生比较。§V-A 用 MLP、Lipschitz 激活与损失梯度讨论 informativeness gap；不是证明任意教师残差都低秩、跨域复用或安全改善。
- 对我们的含义：附加残差模块及“不要重新学习整个教师”已有直接先例。需明确我们新增的可检验结构，而非把 residual head 本身当理论。
- 读范围：摘要、方法 residual 定义、§V-A 理论开头与 Table IX。没有逐行复核其全部推导。
- 原文：[ResKD: Residual-Guided Knowledge Distillation](https://arxiv.org/html/2006.04719)。

### 3. SAD-LoRA：最直接覆盖“方向与系数应分开”的理论近邻

- WHY：只做输出蒸馏，没有显式控制 LoRA 所占子空间。
- HOW：用数据加权参考更新 `ΔW_T Σ_x^(1/2)` 选子空间，再用主角损失约束训练中的方向。
- WHAT：§4 对单线性层的加权平方误差精确分成子空间偏离、子空间内系数偏离和秩尾误差；谱间隙决定最优子空间是否唯一。§5 使用受控合成谱和 RoBERTa/GLUE 消融。
- 边界：理论是平方误差；大教师提供 logits，而参数谱目标来自与学生同结构的参考模型。不能直接套成 masked softmax/TV、完整 LLM 训练定理，也不保证经验方向跨域有效。
- 对我们的含义：简单误差分解和“方向对齐重要”创新空间很拥挤；可借鉴其让合成可控实验检验理论，而非用最终分数反推解释。
- 读范围：摘要、§3、§4 全部、§5.1–5.4、§6、Appendix B 相关证明。按预印本处理。
- 原文：[SAD-LoRA: Spectral Alignment for Low-Rank Knowledge Distillation](https://arxiv.org/html/2607.04306v1)。

### 4. LoRA-GA：实际梯度方向已经被用于选择低秩空间

- WHY：LoRA 起始更新未必接近全参数微调的梯度。
- HOW：从训练损失梯度的 SVD 构造 A/B 初始化，以逼近全微调首步更新，并单独处理尺度稳定。
- WHAT：§3.1/Theorem 3.1 优化的是低秩参数化的一步有效梯度与给定全梯度的 Frobenius 距离。Theorem 3.2 的尺度结论使用随机正交方向等假设。随后开展微调任务实验。
- 对我们的含义：`(p−q)W` 与误差/梯度方向有联系，但当前全词表方向不是 masked Top256 动作的精确梯度。“从任务梯度选方向”本身不新；初始化理论不自动保证最终学习收益。
- 读范围：摘要、§3 目标与定理、Appendix A 两个证明及实验描述；未复核所有表格。
- 原文：[LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/html/2407.05000v1)。

### 5. PiSSA：只看 W 主方向也可改进训练，不能把它等同随机对照

- WHY：LoRA 随机初始化可能使早期更新方向与尺度不佳。
- HOW：用 W 的主要奇异分量初始化可训练适配器，冻结剩余权重。
- WHAT：多模型/任务的同设置微调与量化比较；§7 明确把进一步理论解释列为未解决问题。它不是一个证明 W 主方向对任意任务最优的定理。
- 对我们的含义：W 几何是合理基线，但不能据此把均匀 W 几何当成所有非误差方法的最强代表；PiSSA 的可训练因子也不同于我们固定输出 B。
- 读范围：摘要、§1、§3、§7 及量化附录相关段落。
- 原文：[PiSSA](https://arxiv.org/html/2404.02948)。

### 6. CorDA：最直接的“数据分布解释”近邻

- WHY：同一模型不同任务激活不同方向，纯 W 分解未利用任务上下文。
- HOW：收集输入激活 `C=XXᵀ`，分解 `WC` 并反变换；按知识保持或任务适配目的选择不同谱部分。
- WHAT：§3 给出构造，§4/Table 6–7 与附录比较数据选择和分解方式。作者论述来自数据上下文改变方向，而不需要教师与学生的真实残差配对。其 `WC` 不是与 `W C^(1/2)` 完全等价的最优风险解。
- 对我们的含义：E 胜均匀 G 可能仍只是“见过数据分布”的优势。因此应该区分数据加权几何与真实误差关系，而不是再次比较均匀 W 或随机空间。
- 读范围：摘要、§1、§3、消融表文字、Appendix C。
- 原文：[CorDA](https://arxiv.org/html/2406.05223v3)。

### 7. EVA：理论结论必须跟它优化的量一起读

- WHY：低秩初始化应利用下游激活，而不是只用随机、W 或一批梯度。
- HOW：激活 SVD 初始化下投影，按解释方差分配层间秩。
- WHAT：NeurIPS 2025 正式版 Theorem 3.1 是给定秩的激活方差最大化；梯度能量分析使用输入与上游梯度独立假设，§3.4 的 NTK 近似还用弱相关和近似各向同性。并报告多模型任务与训练效率。
- 对我们的含义：数据驱动方向有理论支撑并不必然依赖 teacher/student 错误。方差/梯度能量最优也不等于 TV 修复最优。
- 读范围：正式版 pp.1–5、Appendix H 中梯度独立假设；旧 arXiv v3 仅作版本对照，正文论述采用正式版。
- 原文：[EVA, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/file/41d33bd41fd44bd9dba0e092047cf213-Paper-Conference.pdf)。

### 8. VeRA：随机方向也可能足够，参数节省不证明误差结构

- WHY：低维适配可能不需要训练完整低秩矩阵。
- HOW：跨层共享固定随机矩阵，训练缩放向量。
- WHAT：GLUE、生成、指令与图像任务，另比较共享/独立随机矩阵及初始化。缩放向量可改变行列相对作用，因此并非与我们固定正交随机 B 完全相同。
- 对我们的含义：修复头有效不能反推误差基底是原因；随机臂有效也不能反推误差不具结构。
- 读范围：摘要、§3、§4.6。
- 原文：[VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/html/2310.11454)。

### 9. FWSVD：几何重建好，不等于任务误差小

- WHY：W 的小奇异分量也可能影响任务；普通 SVD 的重建目标与任务目标不同。
- HOW：用训练数据上平方梯度估计参数重要性；为易于求解，简化成每行相同权重的 Fisher 加权 SVD。
- WHAT：§3 按奇异值组做干预；§4 的目标是加权参数重建平方误差；§5 验证压缩模型任务表现。这里使用经验 Fisher 近似和行级简化，不是所有参数的精确 Fisher 风险定理。
- 对我们的含义：数据加权是有原理的竞争解释。更小系数范数不能直接翻译成更小任务风险。
- 读范围：摘要、§2–4、§5 概述、局限。
- 原文：[Language model compression with weighted low-rank factorization](https://arxiv.org/html/2207.00112)。

### 10. SVD-LLM：通过加权让数学目标真正对应观测误差

- WHY：即使做激活缩放，小奇异值也未必对应小输出误差。
- HOW：用校准输入的 `XXᵀ` 构造白化，分解 `WS`，再作连续层的低秩恢复。
- WHAT：§3.1/Theorem 3.2 与 Corollary 3.3：在 `SSᵀ=XXᵀ` 且可逆的设定，截断造成的样本输出 Frobenius 误差由截断奇异值平方和决定。之后检验压缩误差与语言模型表现。
- 对我们的含义：要先说清方向选择对哪个风险最优。其结论不证明当前 normalized 残差 PCA 最小化 softmax 精确修正成本。
- 读范围：摘要、§3.1、定理证明、ASVD 对照与 Appendix A.5。
- 原文：[SVD-LLM](https://arxiv.org/html/2403.07378v5)。

### 11. ASVD 与 LoftQ：补充定位，不作本项目关键因果证据

- [ASVD](https://arxiv.org/abs/2312.05821)：本次读一手摘要，作者把激活异常值与不同层敏感性作为压缩问题，采用激活感知缩放和校准。正文方法细节主要由 SVD-LLM 的明确比较交叉定位，没有冒称全文已核验。
- [LoftQ](https://arxiv.org/abs/2310.08659)：本次读一手摘要及 [ICLR 2024 论文入口](https://openreview.net/pdf?id=LzPWWPAdY4)。交替量化与低秩近似，补偿量化造成的 W 差异。它展示“来源明确的误差→对应补偿”，不证明任意大小模型差异都同类。

### 12. 新近直接相关：Predictive Geometry of Hidden Trajectories in Transformers

- WHY：隐藏空间的欧氏大小未必反映预测影响。
- HOW：固定模型的终端损失向隐藏状态回拉，得到 Jacobian/Fisher 几何。
- WHAT：局部 Hessian 写成 Fisher 回拉项与残差项；残差控制依赖低损失参考轨迹、下游导数有界等条件。包含扰动敏感性、压缩/剪枝和蒸馏检查。
- 对我们的含义：更小隐藏改动不是天然的更小功能损伤；“预测相关几何”已有直接理论邻居。不能把局部二阶结论泛化成我们有限步 masked softmax 安全保证。
- 读范围：摘要、§1、局部几何定理/证明段、压缩对照、蒸馏与局限段。2026-09-29 预印本，未核实录用；正文中的 NeurIPS checklist 不等于录用。
- 原文：[Predictive Geometry of Hidden Trajectories in Transformers](https://arxiv.org/html/2609.37717v1)。

### 13. 新近名称相似但问题不同：LLM residual learning KD

[Knowledge Distillation for Large Language Models through Residual Learning](https://www.amazon.science/publications/knowledge-distillation-for-large-language-models-through-residual-learning) 的作者机构页面明确标 ICLR 2026。该工作的出发点是教师也可能误导学生：先压缩教师表示，再通过残差目标与真实标签监督帮助学生超越错误教师知识。与本项目“把目标分布当验证基准”的任务不同。

读取范围：作者机构摘要和论文搜索摘录。OpenReview PDF 返回浏览器验证页，因此没有声称核查完整公式、实验和因果证明。不能凭同名 residual 把它当成本研究的重复，也不能用二手笔记补全未读细节。

## 论述可以借什么，不能借什么

|可借鉴的写法|本项目必须补的区别|
|---|---|
|Caprese/LoftQ：先锁定造成误差的近似，再设计补偿|目前尚未定位大小模型差异的具体形成原因，不能挪用其因果确定性|
|SAD-LoRA：分清子空间可达性和系数学习|本项目成本结果与学习结果已分离，不能用前者覆盖后者|
|CorDA/EVA：同一 W 对不同数据有不同有效方向|需要判断真实 p/q 错误对应关系是否提供额外信息|
|FWSVD/SVD-LLM：数学目标应对应想解释的误差|当前最小系数范数只针对规定修正，不是任意 TV 改善的全局成本|
|VeRA：随机参数化也可能有效|原方法有效本身不证明误差提取这一步必要|

研究判断：值得保留的最小问题是“真实错误关系是否带来超出数据加权几何的修复信息”。这是给主代理的来源结论，不预判结果、不宣称创新已成立，也不自动授权新增实验。

## 数学与对照提醒（本备忘录的推理，不是作者结论）

1. 当前 E 以单位误差向量外积构造二阶矩；整条误差向量乘 −1，外积不变。因此随机翻整条符号不能破坏这个 B，不能当有效负对照。
2. 打散 p/q 配对与打散误差向量、token 维度是不同干预，保持的候选集合、概率边际、熵、域/位置分布也不同。具体方案应明确保留什么、破坏什么，不能统称“shuffle 就排除几何”。
3. 统一一个尺度只排除一种整体增益差别；它没有匹配方向各自的伸缩、候选分布或初始化函数，也不能证明训练收益的来源。
4. 用同一固定数据检验新竞争解释属于回顾性探索。冻结新设计能减少继续追结果，却不会把旧数据变成新确认。

## 检索和阅读审计

通过 web 检索和原文 open/find，检索日为 2026-10-17；没有使用聚合网站的自动论文解读作为证据。发现时出现的第三方页只用于定位，再转一手 arXiv、会议论文或作者机构页。没有声称系统综述或穷尽全部近期论文。

主要查询（实际执行）：

- `site.arxiv.org Caprese 2505.07861 residual compensation speculative decoding`
- `site.arxiv.org SAD-LoRA 2607.04306`
- `site.arxiv.org ResKD residual knowledge distillation 2006.04719`
- `site.arxiv.org CorDA 2406.05223 EVA LoRA data driven initialization`
- `"Knowledge Distillation for Large Language Models through Residual Learning"`
- `site.arxiv.org "Parameter Efficient Fine-tuning via Explained Variance Adaptation"`
- `site.arxiv.org "Fisher" "weighted" "low-rank" "compression"`
- `site.arxiv.org "ASVD" "SVD-LLM" activation`
- `site.arxiv.org "ASVD: Activation-aware Singular Value Decomposition"`
- `site.arxiv.org "LoftQ" "LoRA-Fine-Tuning-aware Quantization"`

直接打开起点：LoRA-GA 2407.05000v1、PiSSA 2404.02948、VeRA 2310.11454、FWSVD 2207.00112、SVD-LLM 2403.07378v5、EVA 2410.07170v3 与正式版。定向检索词包括 `Theorem`、`Assumption`、`covariance`、`isotropic`、`min_`、`distillation`。逐篇读范围已在上方披露。

未完成而不假装完成：没有复现实验、重算作者数字、逐行审查全部证明、核实所有论文最终会议状态；SAD-LoRA 与最新 predictive geometry 按预印本使用。没有证据声称它们已证明本项目的具体失配来源。

## 下一步候选设计的来源审核（给主代理，非执行授权）

**值得作为唯一一次短门禁，条件通过。** 固定旧折、Top256、r(.5)，加入只看 q 的候选 Fisher 几何，并以保留加权边际的错配误差基作辅助，能够检验“真实错误关系是否比指定的数据几何更有修复信息”，直接回答限定的修复原因问题。它不应升级为“已找到失配形成原因”。F 是局部曲率，E 是全词表误差方向；二者并非只有一个因素不同。错配也改变残差范数及候选对应，因此即使 E 同时胜出，也只支持超出这两种竞争解释，不能称纯因果证明。非均匀权重下普通置乱不保加权边际，需精确构造并审计；trace 近零保留 NA。主成本比较沿旧完整权重与旧场景，不挑域、阈值或秩。若 E 不胜指定几何对照，就停止“误差关系特殊性”主张；不追加训练或紧缩预算救阳性。这个门禁有用，但单独通过仍不足以证明实际接受率收益。
