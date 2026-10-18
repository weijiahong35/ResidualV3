# 执行指南

## 0. 授权、位置及唯一数值合同

本次任务只交付文档。用户明确启动START_PROMPT后，在本包 `runs/<UTC>_basis_reuse/` 写代码和结果。新prompt允许27次CPU小头拟合；不允许大模型/主体训练、LLM前向、采样、GPU、native/TPS、旧作业重启、环境安装。旧目录只读。

数值规则以 `protocol/DESIGN_SPEC.json` 为准；若与Markdown有冲突，必须先修文档并保存核心审核，不能择有利值执行。模型权重只按静态张量读取，不实例化Transformers模型。

## 1. 运行前锁定

1. 读AGENTS及本包方法/理论/协议/审核；建立输入/代码/资源账。复制文档入run，记录SHA。
2. 用 `tools/audit_inputs.py` 重验元数据并和归档审计比较。运行首次读数值前核对各C raw/target/META的旧COMPLETION SHA；静态W/E各计算一次tensor-data SHA，归档前后stat。作者阶段只核验头和stat，不能冒称已核验全部大张量数据。
3. 使用现有 `/data/weijiahong/miniconda3/envs/dflash/bin/python`。import前设CUDA_VISIBLE_DEVICES为空、OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS线程1；torch intra/inter-op均1、CPU device、无DataLoader子进程。默认shell python无torch，不能据此安装新环境。
4. 先冻结36题fold及24/12映射、所有seed、构造/训练/评价代码。元数据预选的first group SHA C1、P唯一target最多C为计时输入；再加全P/D/F唯一target最多C（并列完整SHA、anchor升序），仅由META决定。若重合去重。禁止看科学效果挑工程题。

## 2. 数据、权重和缓存

- uint16为BF16位模式，先uint32左移16再view FP32，之后按合同转FP64；不能 `.astype(float)` 当数值解码。
- NPZ mmap不提供成员逐行映射。逐C读raw/target成员，唯一target行归一化后复用。禁止整档复制、全库完整p/q FP64矩阵或加载全模型。
- P基底构造只需2571个每C去重后的target行（实际以新审计为准）；构造缓存可存pW、qW、单位残差、q_S/p_S、logq_S/logq_T、hidden、previous ID、原权重和行身份。输入去重仅省算力，不丢自然槽频数。
- D/F评价不做pW或训练。按C单次读完整p，评价所有臂/seed缓存的系数；不为每个头反复解压target.npz。
- 静态W/E各BF16约1.16GiB。分块解码W计算投影/Gram；E只读所需前驱与候选词行，不同时保留多份全词表FP64权重。G的Gram可FP64累计；任何混精度近似须通过固定数值检查，不能静默换构造算法。
- 构造矩阵E/G均4096×4096，使用现有scipy.linalg.eigh确定性求前512；不同折依次处理。R使用固定PCG64高斯+薄QR。不能改rank节约时间。
- 用manifest生成每折N_train及w，assert权重和=1；三个来源分开。完整(C,H,i)身份必须与target_row一致。

## 3. 完整性能门禁

只计head step会漏掉W乘法/分解等主成本。固定工程C实际计时必须覆盖：

1. NPZ解压、BF16解码、全softmax及SHA；
2. 唯一P pW/qW、4096二阶矩更新；W均值/Gram一个固定词表块的吞吐；一次4096×4096谱分解（可作为正式G构造保留，计入总账）；
3. 候选A构造、SVD及三个参考动作成本；
4. 完整128batch的小头forward/backward/AdamW至少10步，三个臂类型都覆盖，FP32/FP64投影漂移检查；
5. 完整P/D/F逐C读取和所有预测合并评价、写入、bootstrap/绘图量级。

工程训练使用一次性副本和固定前128训练身份，计时权重销毁，不作为正式初始化/模型。训练和构造不得用评价结果选超参；工程输出只读时间、内存、字节量、数值错误，不以收益开关任务。

计时前完整E尚未构造：网络三臂的计算图和尺寸完全相同，E类型计时明确使用固定seed2026101696的R作为等维占位映射，G/R使用已构造矩阵；不得称已测真实E效果。完整E的pW/二阶矩/谱分解另按上项实测吞吐计入。正式E完成后复核实际每步耗时与FP32/FP64漂移，更新剩余预算门禁；超限仍partial，不缩面板。按[头合同](protocol/HEAD_CONTRACT_CN.md)复制纯CPU网络并保存新旧源码SHA，禁用旧rollout、BF16/native路径。

按元数据真实唯一行/C/i数、3E+G+3R、27次拟合的确切step数外推，分别累加解码、投影、Gram/eigh、SVD、拟合、评价、5000次聚合和至少20分钟收尾；整个剩余工作估计乘1.5。该值须≤剩余CPU预算80%，峰值RSS≤16GiB且新增磁盘≤8GiB。已消耗计时/失败也扣预算。

未达到门禁：记录partial_resource_limited，完成已测审计与NEXT_CHAT，不缩题、不砍臂/seed/epoch、不偷偷用GPU或多线程、不换run重置6h。禁止依据科学阴性延长训练。

## 4. 顺序执行

S0 数学合同和小词表检查→S1 身份/代码/完整计时→S2 构造E/G/R和训练集尺度→S3 所有P折外参考成本→S4 27次CPU小头拟合→S5 全P/D/F折外TV与附表→S6 聚合、判定、审核、收尾。

S3阴性不跳过S4；S4阴性不重做S2。两层证据必须齐。每阶段最多2次有明确定位的技术修复，记录原异常、修复diff、重验和资源；科学阴性不是技术故障。失败不能换seed、改学习率、改数据。

每次正式拟合从配对初始化文件加载；归档init SHA、参数量、输出B/input-G SHA、shuffle SHA、最终SHA和12epoch步数。可在epoch边界恢复同一作业且保存优化器与RNG，不能把重复训练当恢复。最多保留最终模型和一个恢复checkpoint，防磁盘膨胀。

CPU自有进程记录PID、starttime、cmd、run绝对路径；仅管理身份一致的自有进程。不查询/终止旧GPU作业，不启动旧controller/parallel/收尾。预算触达先停自有数值worker，再保存partial账目。

## 5. 必交文件

- `protocol/RUN_LOCK.json`、`SPLITS.json`、`CODE_HASHES.json`、`TRAINING_SCHEDULE.json`。
- `analysis/BASIS_CONSTRUCTION.json`：每fold训练身份、权重、有效秩、谱/正交误差、随机seed、全局尺度；不把谱作为主结论。
- `results/REFERENCE_COSTS.csv`、`REFERENCE_BUDGET_GAINS.csv`、`SCALE_MATCH_COSTS.csv`：连续数值、保护区间、预算状态、全部分母。
- `results/HEAD_OOF.csv`、`HEAD_BY_PROMPT.csv`、`SOURCE_SHIFT.csv`：所有seed、P主面板及D/F，epoch0/12，收益/损伤/覆盖；原始行可压缩。
- `results/DECISION.json`、`RESULTS_CN.md`、最多6个固定排序案例（按group SHA、anchor、i、slot取首6，不能挑好例）。
- 三张核心图：同一动作E/G成本配对；预算内参考收益与未定覆盖；折外小头差异及损伤。跨域只分面，不另选主线。
- `qa/NUMERICAL_AUDIT.json`、`qa/FINAL_CORE_REVIEW_CN.md`、`RESOURCE_AUDIT.json`、`FILE_AUDIT.json`、`COMPLETION.json`、`NEXT_CHAT.md`。

RESULTS第一页≤600中文字：误差方向是否有额外价值；证据是成本/学习哪一层；是否仅统一尺度；哪些仍未判定；下一步或停止理由。完整结果保留不利域/seed、数值NA和学习目标限制。审核不通过就改表述再审，不能换实验补阳性。

COMPLETION区分execution_status和scientific_decision。必需工作没做完=partial，全部做完但结果否定/未定可complete；不得以审核文档存在宣称实验完成。结束退出自有作业，不自动下一轮。
