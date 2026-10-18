# 并行草稿与轻量修正：一手资料核验

检索日：2026-10-16 UTC / 2026-10-17 UTC+8。本文件只做来源核验和论证结构提取，不作新实验决定。已读仓库AGENTS；未运行实验、未加载模型、未修改旧资产。

## 核验口径与检索记录

- 指定六篇均实际打开arXiv摘要页和HTML正文，以下列出所读章节。均按预印本记录，没有核验到会议录用，不能写已录用。
- 先直开2605.29707、2607.05147、2607.25852、2608.20530、2608.02438、2608.05448；xPress的v2 URL失败，摘要页确认只有v1，随即读取v1，不能将失败解释为论文不存在。
- 实际广泛查询：`site:arxiv.org speculative decoding "residual" "subspace" 2026`；`site:arxiv.org speculative decoding "logit" "correction" 2025 2026`；`site:arxiv.org parallel drafter "mismatch" 2026`；`site:arxiv.org "DFlash2"`；`DFlash2 speculative decoding Inco AI 2026`；`speculative decoding lightweight correction residual 2026 paper`；`parallel drafter candidate reranking low rank 2026 arxiv`；`speculative decoding gradient subspace 2025 2026`。
- 搜索召回DCOCO、DFlash2、TreeSpark后，均另开一手正文。搜索中的博客转述、社交帖、一般LLM自纠错和与推测解码无关的几何论文，未用作实质依据。本检索并非穷尽性系统综述。

## 指定六篇

### Domino

**元数据：** Jianuo Huang、Yaojie Zhang、Qituan Zhang、Hao Lin、Hanlin Xu、Linfeng Zhang；*Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding*；v1，2026-05-28。[记录页](https://arxiv.org/abs/2605.29707)；[正文](https://arxiv.org/html/2605.29707v1)。读§1、§3、§4、§5.3、§7。

- WHY：并行预测缺少已采样前缀；完整AR又要反复执行大草稿网络和词表投影。
- HOW：并行DFlash骨干＋GRU汇总前缀＋低秩logit残差；teacher forcing和逐步从base转final监督的课程。
- WHAT/支撑：延迟分解、接受长度、head/课程消融共同支持设计；“前缀条件化”来自明确概率分解。没有给出解释所有失配来源的定理。
- 本地关系（判断）：低秩残差修正已有；固定误差基底需证明额外价值，不能单靠“我们也加了小头”。其low-rank输出映射可训练，不等于误差统计构造的固定空间。

### DSpark

**元数据：** Xin Cheng、Xingkai Yu、Chenze Shao等33位作者；*DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*；v1，2026-07-06。[记录页](https://arxiv.org/abs/2607.05147)；[正文](https://arxiv.org/html/2607.05147v1)。读§2、§3.1–3.2、§4.2–4.3、§5。

- WHY：独立边缘分布会组合出互不相容的后继；固定长块验证浪费后缀预算。
- HOW：并行骨干＋低秩Markov或RNN顺序头；另以接受置信度和硬件成本调度验证。
- WHAT/支撑：条件接受率随位置的曲线、层数/块长/head形式对照，以及部署测量；概率因子分解解释“补入什么信息”，没有将整体现象归结为一个普遍因果定理。
- 本地关系（判断）：共享低秩token转移已经存在；多领域平均提升不能作为本地跨域复用的独有贡献。§4.3称首位置差异严格来自容量，但其架构比较不能自动排除一切训练差异，应标为作者解释。

### AngelSpec

**元数据：** Hong Liu、Rui Cen、Junhan Shi等13位作者；*AngelSpec: Towards Real-World High Performance Inference with Speculative Decoding*；v1 2026-07-28，v2 2026-07-29。[记录页](https://arxiv.org/abs/2607.25852)；[正文](https://arxiv.org/html/2607.25852v2)。读§1、§2.2及loss消融、§3.1–3.2。

- WHY：任务分布和系统负载改变合适的草稿结构；共享target context与块内独立也有约束。
- HOW：按数据/结构做互补专业化；DFly混合层条件特征＋前驱hidden残差；接受相关目标和调度。
- WHAT/支撑：TV与单位置接受的恒等关系、训练目标/结构实测对照。不能简化成“只研究熵”，也不能把恒等关系当新定理。
- 本地关系（判断）：基于当前hidden和前驱的轻量修正已经覆盖；本地固定输出空间的错误信息价值属于更窄问题，不能凭该文确认本地误差来源。

### LiLiCorr

**元数据：** Matan Rusanovsky、Yoav Miron、Roy Uziel、Omer Belhasin、Hao Guo、Ran Zilberstein、Maor Ashkenazi、Michael Elad；*LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding*；v1 2026-08-20，v2 2026-09-22。[记录页](https://arxiv.org/abs/2608.20530)；[正文](https://arxiv.org/html/2608.20530v2)。读§2、§3.1–3.3、附D/E。

- WHY：各位置候选各自合理，却可能不相容；全链总分与只接受前缀的验证规则有差别。
- HOW：Top-K候选网格用小Transformer共同编码，输出入/出向量，余弦分数关联邻接候选；神经评分并行，选择顺序执行。
- WHAT/支撑：候选及目标成本目标、固定总权重的loss消融；附E证明其惩罚是期望目标成本，并强调不是匹配完整分布。
- 本地关系（判断）：支持区分“分布接近”和“候选决策”的目标，但不能据此重写本地训练目标为成功解释。该文对DFlash2的固定token分数描述与当前官方含context gate的公式不一致，见下。

### xPress

**元数据：** Zheng Wang、Davis Wertheimer、Yu Chin Fabian Lim、Mudhakar Srivatsa、Raghu K. Ganti、Minjia Zhang、Naigang Wang；*xPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding*；v1，2026-08-03。[记录页](https://arxiv.org/abs/2608.02438)；[正文](https://arxiv.org/html/2608.02438v1)。读§1、§4.2–4.3、§5.2。

- WHY：已有Top-K候选覆盖，但没有前驱信息决定选谁。
- HOW：低秩前缀线性混合输出logit残差；Jacobi同步迭代替代逐token网络执行。
- WHAT/支撑：定理1在hidden固定和确定性并列规则下，证明最多B次迭代达到同一refiner的序贯贪心结果；迭代次数/速度/接受长度实测补充。
- 边界：定理不证明修正更接近target，也不保证实际少于B次；少轮足够属于实测。
- 本地关系（判断）：这是收缩理论承诺的范例：只证明本方法一项精确性质，其他靠受控证据；不需要完整解释大小模型差距。

### DBLast

**元数据：** Amirmohammad Karimi、Chao Gao、Negar Hassanpour；*DBLAST: Dependent Block Drafting for Stochastic Speculative Decoding*；v1，2026-08-05。[记录页](https://arxiv.org/abs/2608.05448)；[正文](https://arxiv.org/html/2608.05448v1)。读§1–3、§4.1–4.4。

- WHY：随机目标的多种延续，使独立块采样更脆弱；块似然目标不等于接受前缀长度。
- HOW：CP式潜类别混合诱导块内依赖；接受长度启发的log域替代目标。
- WHAT/支撑：联合分布与条件化公式、与proposal一致的验证论证；独立/依赖×NLL/AL四格消融分离两个因素，按目标确定性展示差异。
- 边界：AL使用替代目标，不能表述成直接精确优化所有真实在线接受长度；类别低秩不等于固定残差子空间。
- 本地关系（判断）：提供“结构变化与目标变化分开”的证据模板；并不替本地证明基底复用。

## 新检索的近邻与反向约束

### DCOCO（最新、必须纳入近邻）

**元数据：** Roy Eisenstadt、Harel Lidar、Tal Gonen、Omer Luxembourg、Itamar Zimerman、Lior Wolf、Eli David、Ilan Naiman；*Contextual Correction of Parallel Drafts for Speculative Decoding*；v1，2026-10-08。[记录页](https://arxiv.org/abs/2610.12570)；[正文](https://arxiv.org/html/2610.12570v1)。读§1–4、§5.3、附B组件消融。

- WHY：Top-K中常有正确候选，独立选择没有利用已选前驱；串行网络修正又有延迟。
- HOW：候选与邻位hidden组成状态，共享MLP一次算全部候选对残差，随后前缀优先选路径。
- WHAT/支撑：候选覆盖；同一评分下比较greedy、全链总分及位置折扣；去pair/candidate状态与冻结骨干消融。
- 边界：同分数对照支持所测设置的选择规则，不是证明greedy对任何验证目标全局最优。
- 本地关系（判断）：泛泛“候选里有答案、重新选对”已覆盖；论文的理论承诺主要是前缀结构对方法的约束，而非完整失配起源。

### DFlash 2（作者官方技术博客，非论文录用证据）

**元数据：** Inco AI；*DFlash 2: Keep Drafting Parallel*；页面引用给出2026年8月；当前页面访问日2026-10-17，未核验历史快照。[官方原文](https://inco.ai/blog/dflash2/)。读“Right Tokens”“Lightweight Path Selector”“Suffix Decay”。

- WHY：Top-K覆盖与Top1有差距；即使oracle选择，后缀候选覆盖仍下降。
- HOW：邻接候选低秩双线性匹配、hidden条件门控；另外在骨干加入局部卷积。
- WHAT/支撑：selector单独对照，以及骨干深度/候选覆盖分析；没有固定误差基底复用定理。
- 版本提醒：当前公式为 `U(b)+<A(a)⊙H(h_t),B(b)>`，包含上下文。LiLiCorr v2称其分数只依赖token身份；两者可能对应不同版本，不能引用后者来断言当前DFlash2没有上下文。未核历史代码，差异原因未判。
- 本地关系（判断）：“局部候选适配”是严肃竞争解释；固定误差基底若只适配常见候选，可能不代表方向性错误的特殊价值。

### TreeSpark（关联但不同问题）

**元数据：** Huapeng Zhou、Huayu Wang、Xinyu Wang；*TreeSpark: Calibrated, Load-Adaptive Draft Trees for Semi-Autoregressive Speculative Decoding*；v1。arXiv记录显示提交2026-08-12，ID月份2609存在表面不一致，按页面如实记录、不自行修日期。[记录页](https://arxiv.org/abs/2609.22098)；[正文](https://arxiv.org/html/2609.22098v1)。读§1–2、§3摘要性方法描述。

- WHY：树节点若只按位置边缘分布排序，会忽略不同父节点；固定树大小忽略负载。
- HOW：复用现有Markov头按父节点计算条件概率，校准边接受并按路径存活展开/停止；无放回采样配递归拒绝。
- WHAT/支撑：同预算/不同条件化、校准和树大小对照；保分布依赖采样与验证的一致性。
- 本地关系（判断）：复用已算信息的价值可以通过具体对照建立，但这是树排序与验证预算的机制，不是误差空间复用证据。

## 可供综合者使用的边界

1. 在已读近邻中，“低成本修正”“补前驱”“Top-K重选”“跨域有效”都已有明确实例，单独命名不足以构成额外认识。
2. 理论支撑有不同层次：概率因子分解解释结构缺什么；既有恒等式连接目标与评价；有限收敛定理保证算法性质；受控消融判断某项设计是否带来收益。不能把这些统称为“证明了失配原因”。
3. 本地当前成本结果与这些论文都不矛盾。已有论文也不能替本地排除候选几何、尺度和训练目标的解释。
4. 上述是阅读后的比较判断，不是“已穷尽文献、证明无人研究固定误差基底”。

## 对拟议收缩线的近邻审查补充

根节点拟议问题：真实p−q配对是否提供超出候选/数据几何的低代价修正结构；以train-only masked-q Fisher几何和pW/qW保边际拆配作对照，不重训头、不收紧预算找阳性。以下是比较判断，不是论文已证明的结论。

- **可能新增的认识：** 上述并行草稿工作主要证明加入条件信息、候选关系或改变选择规则的用途；它们没有在这里核验的实验中分开“固定修正空间来自正确错误配对”与“只适配当前候选集合”。若拟议对照成立，可支持正确草稿—目标配对对固定空间的额外统计价值；这是更窄的来源问题。
- **第一处容易过度：** pW/qW拆配同时毁掉上下文与候选对应关系。胜过全局拆配，只能说明配对信息重要，不能直接认定“跨题反复出现的是同一种错误机制”。必须说明拆配保持了哪些条件；候选分布/位置/样本权重失配可能是替代解释。
- **第二处容易过度：** masked-q Fisher是一个具体候选几何基准，不是所有无误差几何方法的上界；E胜它不能写“排除了全部几何解释”。E还使用目标数据，需拆配对照保留这个信息预算。
- **第三处容易过度：** 单位化残差的构造下，拆配可能改变残差范数分布及近零比例；真实与拆配空间差异必须连同这些量报告，不能把归一化诱导差异全称为方向复用。非均匀样本权重下，普通置乱不一定保留加权边际。
- **论文充分性：** 若阳性，新增的是限定条件下的统计结构与修正代价解释。它仍未解释原头为何有效、未保证实际修复收益、也没有建立跨域训练迁移。可以成为理论分析一节，但目前不能据此承诺足够支撑完整方法主论文。原27次拟合阴性须保持可见。
- **阴性解释：** 若候选几何追平E，现有“省幅度”可由该几何基准解释，不等于修复头无效；若拆配追平E，应放弃正确误差关系具有额外价值的当前叙述。此时不自动转训练目标或紧预算寻找阳性。
