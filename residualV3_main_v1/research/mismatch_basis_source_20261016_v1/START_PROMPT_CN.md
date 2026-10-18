# 基底信息来源对照：启动 prompt

复制分隔线以下全文给执行 chat。当前包是文档交付，尚未执行新实验；只有用户明确把本prompt作为启动指令时才运行。

---

你接手ResidualV3的“基底为什么更省幅度”来源对照。仓库`/data/weijiahong/speculators-main-lora`。

**只回答一个问题：从正确大小模型预测配对中提取的基底，是否比只看候选概率几何、或保留两边统计但拆开对应关系的基底，更省完成同一规定修正所需的幅度？** 不把谱集中、多领域有效、任意oracle可修、旧头平均收益当答案。

先读AGENTS.md及本包：

```
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/README_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/EXPERIMENT_METHOD_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/THEORY_CONTRACT_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/qa/INPUT_AUDIT.json
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/qa/CORE_QUESTION_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/qa/FEASIBILITY_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/sources/READING_GUIDE_CN.md
```

只读旧资产：

```
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

旧E/G成本比约.515，整体增益匹配后系数成本比约.693；学习优势.002892 TV、区间跨0。r(.5)现有幅度内可行质量E约99.8%、G约99.0%。不能据此预设基底缺方向、路由必要或学习失败原因。

**本prompt被用户明确启动后，授权新run内CPU代码、静态W/缓存读取、数学核验、12个新基底构造、固定成本分析与收尾。** 不训练任何头、路由或主体，不加载/前向运行LLM，不采新数据、不用GPU、不启动native/TPS、不安装环境、不重启旧stage/controller/parallel/收尾。旧run只读；只管理核实身份的自有进程。

本包所有run累计6 CPU进程树小时、16GiB自有RSS、8GiB新增磁盘；单数值worker、所有数值库线程1，import前设置，CUDA_VISIBLE_DEVICES为空。使用现有`/data/weijiahong/miniconda3/envs/dflash/bin/python`。计时/失败/修复/审计也计账，不换run重置。当前只有头/元数据审核，完整性能门禁未测。

执行顺序固定：

1. **锁身份、数学和代码。** 重验36题/69C/207 C-i/9936 P自然槽/2571唯一(C,target_row)，三折每折24训练题/12折外题。精确P来源、pool=='fresh_eval'、i3/8/15；fold0/1/2折外槽3168/3312/3456。核旧依赖SHA与W tensor身份、旧E训练题与SHA。旧脚本ROOT会写旧run，只能移植纯函数，禁止直接运行或导入其写入路径。先完成小词表数值合同。
2. **完整性能门禁。** 按身份预选首group SHA题C1和P唯一target最多C（并列固定排序）。测全词表读入/softmax/参考、F曲率、稀疏N耦合/矩阵、一轮4096阶eigh、映射/SVD、训练尺度和汇总。预测必须计入12个新基底及训练/评价映射；1.5倍剩余科学工作预测≤剩余CPU80%，且另留1200秒收尾。首折真构造后重核。超限partial_resource_limited，不删臂/种子/题/折、不换rank或GPU。
3. **构造三类方向。** E只读旧E0/E1/E2。F只看相同训练候选/q/W：W_Sᵀ[diag(qS)−qS qSᵀ]W_S，逐C/i trace归一、原槽权重累加。N0/1/2在训练同域同位置保p/q加权边际，按协议固定PCG64种子与圆周半周平移的区间交叠耦合拆配；零误差保权重不重归一。每折F+三个N，共12组新512维正交基底。三N全报告，不挑最差N衬托E，不输出置换p值。
4. **完成同一参考成本评价。** 全部折外P，只用r(.5)，raw q始终J、Top256/尾部固定logits/完整归一化、.8幅度不变。缓存没有完整p/q、pT或d，须逐C只读R8 logits：uint16 BF16位模式→FP32→FP64全词表softmax，尾部直接求和，禁止1−sum(pS)替代极小尾概率。同C/i复用SVD、同target行复用概率。A=W_S Bᵀ，计算实现同一d的最小系数范数/hidden范数，完整重建核验；全板旧E仅作身份/数值对齐。另做train-only单个整体增益匹配，匹配后称系数成本，不用于原.8预算可行性。
5. **固定汇总、判定、收尾。** 保留自然重复H，槽→C→位置→题→域原权重，另做等H评价。公共场景是同一参考改善≥.01，不按成本或头表现选样本。F/E用其双边有效交集；三N先取E/N0/N1/N2共同有效交集、每槽平均log比再聚合；另列E/F/N全交集敏感性。5000次域内整题配对bootstrap仅描述。按固定表出全部结果、审核、COMPLETION及NEXT_CHAT后退出，不自动下一轮。

严守边界：

- 只研究规定r(.5)的最小幅度，不是任意同TV改善的最小成本。参考安全是定义保证；不是E来源证据。
- 原正交尺度成本可对应修正向量范数；整体gain匹配后只是系数单位，不代表更小实际位移或节省算力/训练时间。
- F是一个明确几何对照而非所有几何方法的最优上界；N会同时改变语境、候选对应和误差范数。胜出不能叫唯一语义原因。
- F/N有效秩不足、截断边界退化、SVD截断、残差或预算边界未定均保留；不补随机方向，不写∞或不可修。control_uninformative不算E胜出。
- V_F及三N共同V_N对公共场景的可靠正成本覆盖均≥90%、至少两域各≥4有定义题；F比较及三N汇总在原/匹配尺度均≥log(1.10)、数值保护下界不跨此量级、描述区间下界>0、至少两域正向，且三个N总体均正，才记限定来源价值支持。具体bootstrap/NA规则以JSON及方法为准。
- 没明显领先只能说未建立额外价值，不能说等价或解释充分。旧题折外不是新确认；缺失质量的总体log比可能无界。学习和部署价值本包未测。

输出只写：

```
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/runs/<UTC>_basis_source/
```

先用白话简报实际资产、可做/不可做与授权范围，再实施，不只再写计划。每阶段最多2次定位技术修复，保存失败/diff/耗时；每次改方案和最终报告前按AGENTS保存核心问题审核，不通过修设计或声明，不能用追加实验掩盖。

交付F/N构造与拆配审计、全部原/尺度成本和覆盖表、逐题逐域结果、两张核心图、claim–evidence、数值/资源/文件审计、COMPLETION和NEXT_CHAT。第一页≤600中文字回答“真实配对有无额外价值、证据在哪层、是否只是整体增益、哪些未定、停止理由”。必要工作缺失为partial；全固定工作完成但阴性/统计未定可complete。禁止自动下一轮。
