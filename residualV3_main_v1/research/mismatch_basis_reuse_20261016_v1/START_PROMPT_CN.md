# 修正方向复用检验启动 prompt

复制分隔线以下全文作为执行指令。本包包含明确的CPU小头拟合；当前制文任务没有执行训练。GPU预算仍为0。

---

你接手ResidualV3的“误差提取方向是否有额外修复价值”检验。仓库 `/data/weijiahong/speculators-main-lora`。

**只回答一个问题：同一个错误、同一项规定概率修正，从真实误差提取的方向是否比只看输出权重几何的方向需要更小幅度，并且这种差别能被同预算小头在折外题上利用？** 不把多领域有效、谱集中、原头平均有效、更多历史输入或任意oracle存在当成答案。

先读仓库AGENTS.md及：

```text
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/README_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/EXPERIMENT_METHOD_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/THEORY_CONTRACT_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/protocol/HEAD_CONTRACT_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/qa/INPUT_AUDIT.json
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/qa/CORE_QUESTION_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/qa/FEASIBILITY_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/sources/READING_GUIDE_CN.md
```

主要概率来源只读R8：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

本prompt由用户明确用作启动指令时，授权在新run写CPU代码、读静态W/E与存档、执行数学核验、方向构造、规定成本分析、27次CPU小头拟合、评价及收尾。**这是新包明确增加的小头对照训练，不是重启旧包。** 不训练主体或大模型、不加载/前向运行LLM、不采新数据、不用GPU、不启动native/TPS、不安装环境、不重启旧stage/controller/parallel/CPU收尾。只管理核实身份的自有CPU进程。

全包所有run累计6 CPU进程树小时、16GiB自有RSS、8GiB新增磁盘；单数值worker、数值库及torch线程1，import前设置。默认python缺torch，使用现有 `/data/weijiahong/miniconda3/envs/dflash/bin/python`，禁CUDA。当前已有文件头/元数据审核，**尚无性能门禁实测**；完整计时乘1.5后须≤剩余CPU预算80%并保留20分钟收尾，计时和失败也计账。超限就partial_resource_limited，不能减少科学面板/臂/seed/epoch或改用GPU。

执行顺序固定：

1. **锁身份与数学合同。** 重验36题/69C、历史/target_row、原档SHA、W/E张量身份、折划分和源码。先通过小词表数学/数值核验。不得把数值未定写不可修。
2. **完整性能门禁。** 按META身份预选首group SHA的C1、P唯一target最多C、全来源唯一target最多C，重复去重；计时包括完整softmax、pW、W Gram/eigh、候选SVD、三臂类型训练step、P/D/F评价与收尾。禁止看科学收益选工程题。
3. **构造固定空间。** 每域完整group SHA排序，零基序号mod3；3折各24训练题、12折外题。E仅用该折训练题P源 `(p−q)W` 单位方向的加权非中心二阶矩；G仅用全词表中心化W的Gram；R用三个预设随机正交矩阵。均512方向。所有臂的q输入投影固定用G，只改输出方向来源。
4. **规定参考修正的最小代价。** 对折外P源全部i3/8/15自然槽，固定t=.25/.5/1概率插值；t=.5为主。参考分布候选内向p移动，尾部保留q条件形状。用候选映射A的SVD求实现同一分布的最小系数范数，并报告.8||h||预算内参考收益、连续界及全部未定。另做train-only单个全局尺度匹配的副成本分析。
5. **27次匹配CPU小头拟合。** 3fold×E/G/R×3seed；同一初始化、输入、样本权重/顺序、AdamW lr3e−4、batch128、12epochs、其余按JSON。原NOFILL结构CPU FP32，hidden4096/width256/depth3/rank512、use_soft=false、dose上界.8。精确全词表forward KL的动作相关项为 `log(q_tail+sum_S q*exp(delta))−sum_S p*delta`；tail不能重归一到Top256。epoch12唯一最终模型，不选最好seed，不早停、不临时改学习率。
6. **全部折外评价与收尾。** 主P源，D/F仅固定外推；q始终J。完整词表FP64 TV、top1、改善/损伤/覆盖与初始扰动；5000次按域整题配对bootstrap仅描述。按固定判定表写结论，完成核心审核、资源/文件审计、COMPLETION与NEXT_CHAT，然后退出自有作业。

数据精确过滤是`pool=='fresh_eval'`，位置3/8/15。P主面板9936槽，P/D/F共29808槽；不是独立样本数。按槽→C→位置→题→域固定权重，保留自然重复H和等H评价敏感性。EOS当前位置保留；不按头成败、reached、novel或P-fit匹配筛样本。所有数据已暴露，折内隔离不使其变成新确认。uint16 BF16位模式→FP32→FP64，不能直接数值astype。

严守结论边界：

- 参考路径安全是数学定义保证，不是E来源证据。
- 最小代价只针对规定r(t)，不等于任意同等TV改善的全局最小代价；有限三点收益不是完整动作族最优值。
- 数值截断/残差/预算边界保留未判定，不直接写∞或不可修；浮点保护区间不叫严格认证。
- 尺度匹配只做成本，不能据此排除训练收益的尺度/优化解释。
- 新头使用固定G输入和KL代理目标；它不同于原BF16联合训练的OUTPUT目标，不直接证明原部署因果或接受率/TPS。
- D_cost主要场景公共参考改善≥.01，有限双边成本覆盖≥90%，至少两域各4有定义题。D_cost≥log1.10且描述区间下界>0、两域正向，只是双方可解覆盖子集的条件成本线索；有未定项时总体log比界可为±∞，不能称全场景优势。P全体D_learn≥.01、描述区间下界>0、三seed总体正向、两域正向且E自身改善>0为可利用性支持。其余按JSON/方法文档逐项报告，不能改门槛。
- 成本支持但学习未实现，或只有学习支持但成本解释未支持，必须分开。oracle见p而头可能不能区分H，信息/目标冲突与优化都可能导致前者。条件成本沿用完整面板原权重条件化，不能筛选后重新等C/题权重。阴性不重试，区间宽不能说不存在；仅胜R不能支持误差信息特殊性。

输出只写：

```text
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/runs/<UTC>_basis_reuse/
```

先简报实际可读资产、固定面板与授权范围，然后执行，不只再写计划。每阶段最多2次有定位技术修复，保存diff/失败/耗时，不能换run重置预算。每次改方案与最终报告都按AGENTS审核；不通过先修声明或设计，不能用更多实验掩盖问题。

交付全部成本/预算/尺度/小头/P-D-F表、3张核心图、最多6个按SHA固定排序案例、claim–evidence、数值/资源/文件审计。第一页≤600中文字回答额外价值有无、证据在哪层、是否仅尺度、哪些未判定、下一步或停止理由。必要工作未做为partial；全部固定工作完成但阴性/未定可complete。不得自动下一轮。
