# 执行指南

## 0. 边界与入口

当前用户请求制文，不能据此运行科学分析。用户明确启用START_PROMPT后，执行chat在本包`runs/<UTC>_basis_source/`创建新run。本包累计6 CPU进程树小时、16GiB自有RSS、8GiB新增磁盘，单数值worker、所有数值库线程1、GPU0；失败、计时、审计、重试和子进程均计账，不换run重置。

只允许写新分析代码、读取静态W/旧存档、数学核验、F/N构造、规定成本评价和收尾。禁止训练任何头/路由/主体，禁止LLM加载或前向、采新题、GPU、native/TPS、环境安装、旧controller/stage/parallel/收尾重启。原27次拟合与旧成本表封存。

`protocol/DESIGN_SPEC.json`固定数字与选择规则；方法/数学合同固定含义。若冲突先停止依赖步骤、定位修正文档并重做核心审核，不得挑有利版本。新的科学结果不足不是修改协议的理由。

## 1. 输入路径与数据契约

仓库`/data/weijiahong/speculators-main-lora`。只读旧run：

```
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

旧basis_reuse的`data/cache/*.json`中`records`是P自然槽；`all_records`还有D/F，不作为本包面板。每条包括group/domain/anchor/fold/source/i/slot/H/target_row/pindex。NPZ字段：h(3,4096)、S(3,256)、logqs(3,256)、logqt(3)、pS(n_unique,256)、pW/unit(n_unique,4096)、qW(3,4096)、zero、P_target_rows。缓存没有完整p/q、没有单独pT或d(.5)。

旧`data/bases/E{0,1,2}.npy`为固定E，SHA见新INPUT_AUDIT。仅需静态`lm_head.weight`，无需embedding、原头checkpoint或主体。W路径/offset/header结构记录取旧`protocol/INPUT_MANIFEST.json`的`static_tensors`（或最终`protocol/VERIFIED_INPUTS.json`的`metadata.static_tensors`，不是其中的`model_source`）；tensor数据SHA验真状态取最终`qa/STATIC_AND_SOURCE_AUDIT.json`并与`VERIFIED_INPUTS.metadata.static_tensors`核对。早期INPUT_MANIFEST的未验布尔值不能覆盖最终记录，也不能因此丢掉结构来源。新run独立复验文件头、offset范围及tensor数据SHA。

旧`data/costs/*.json.gz`和`results/REFERENCE_COSTS.csv`是按(C,i,target_row)合并的成本记录，有count，不是自然槽表。必须以缓存自然槽身份展开配对或汇总原权重；不得对旧成本行直接等权，也不得把slot0代表行当全部H。

## 2. 锁身份与移植代码

1. 新run保存文档快照/SHA、CODE_HASHES、SPLITS、旧依赖manifest、RUN_LOCK和CPU账本。统计36题/69C/207 C-i/9936槽/2571唯一(C,target_row)，核三折身份和槽数。
2. 重新读取R8的69份META，精确筛P/fresh_eval/i3,8,15/valid/not prior_EOS，与缓存records逐条比较group/C/i/slot/H/target_row；target_row同C内不得跨位置错误复用。
3. 对将使用的旧cache JSON/NPZ、3E、construction/scale/manifest/成本/源码，比较旧FILE_AUDIT中的SHA；未被旧manifest覆盖的输入单列，不能冒称旧链已验证。W tensor区间SHA执行前重验一次，与旧最终SHA比较。读取R8的raw/target/META前核其旧SHA；不读F.npz、旧27个头或全仓库大文件。
4. 只读代码参考：旧common.py/data_ops.py/maps.py/references.py/costs.py。**禁止直接执行、导入调用旧带缓存/写入/guard函数**：它们ROOT绑定旧run。将必要纯函数移植到新run，旧路径命名OLD_RUN/R8，输出命名RUN；审计所有写路径都在新run。不可简单从旧目录启动pipeline。
5. `CUDA_VISIBLE_DEVICES=''`和OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/NUMEXPR_NUM_THREADS/VECLIB_MAXIMUM_THREADS/BLIS_NUM_THREADS均设1，必须在数值import前。用现有`/data/weijiahong/miniconda3/envs/dflash/bin/python`，无DataLoader子进程；若导入torch，其intra/inter-op也设1，但本任务无需torch训练。
6. 准备完整实现后冻结代码再测门禁；执行数学合同里的小词表检查。工程样本只按元数据选：完整group SHA最小题的C1、P唯一target最多C（并列group SHA/anchor升序），重合去重。工程输出仅用于身份、数值和性能，不按效果选样本。

旧E和reference_gain已暴露；本包不声称盲法。禁止读F/N成本后再换构造。

## 3. 完整性能门禁

先设监督进程与本包跨run总账，再开始计时。门禁至少覆盖：

- 工程C的完整NPZ解压、需要读入文件的SHA吞吐、BF16解码、唯一P行全词表softmax、pT/qT、r(.5)/d和完整TV。
- 工程C各位置F曲率/trace构造及4096矩阵更新；同C/i只算一次。
- 规范排序首个训练域/位置层的完整稀疏N耦合、边际核验和单位误差分块二阶矩；统计全库准确非零边数以外推，不构建稠密K。
- 一次实际4096×4096 top513 eigh；可用按身份选定的完整fold0 F矩阵，保留作为正式F0中间结果以免重算。此时仍不评估科学收益。
- 工程C的256×512映射及SVD、参考成本/保护/重建；计时含E和F/N维度一致的映射路径。若N未建可用已验证E作尺寸占位，明确不是N科学计算，N构造与分解另独立计账。
- 各折训练集整体gain计算、折外3个位置的全臂评价、自然/等H权重、5000次bootstrap、图表/写盘/审计的实测小规模吞吐与保守外推。

准确任务量：12个新基底、3个只读E，另有3次由旧unit重建M_E以审计N矩阵是否改变（不重新提取E基底）；各新基底既有训练映射（尺度）又有折外映射（成本）。SVD只需折外映射且C/i内跨H复用；F/N二阶矩不能按旧E耗时直接估。选定C的target成员仍需整成员解压，NPZ mmap不提供逐行映射。

设remaining=21600−本包已消耗CPU秒。必须同时满足`1.5*predicted_remaining_science_CPU <= .8*remaining`及`1.5*predicted_remaining_science_CPU + 1200 <= remaining`；1200秒专供收尾，预测包括尚未完成的身份核验、I/O、所有构造/分解/映射/评价/汇总，不能漏掉已知阶段。峰值自有RSS预测≤16GiB、新增磁盘≤8GiB。门禁自身耗时先扣账。

首折真实F/N完成后、全成本前再以实测更新剩余门禁。任一失败则partial_resource_limited并收尾，不降rank、不删N种子、不减折或槽、不改用GPU。监督应在消耗达到20400秒时停止自有数值worker留收尾；不能用“仍有系统空闲”突破包预算。

## 4. 内存与精确复用

按C读取旧NPZ，只解码P唯一行；可在一个C内复用完整概率，禁止全库完整FP64概率矩阵和整档复制。W以只读BF16 tensor区间映射，按候选行解码，不同时保留全词表FP64 W。

F可写成加权W行Gram减去q加权均值的外积，trace归一不可漏。N用稀疏边表和批量矩阵乘累加，不能为每边持久化4096²矩阵。一个4096² FP64矩阵128MiB，折/臂顺序处理；只保存所需基底/元数据，不保存12份完整分解工作区。

允许数学等价的复用（同C/i合权重、矩阵分块、按fold可加和的精确共享），须在小例和正式权重校验中验证等价。不允许近似SVD、混精度、截短词表、随机投影、用未声明的缓存替换合同。所有同义改写保存diff与数值检查。

## 5. 固定阶段与技术修复

S0身份/数学/代码锁→S1完整门禁→S2各折F/N及train-only尺度→S3全P折外r(.5)成本与旧E核对→S4汇总/判定/图/审核/收尾。

F/N谱、权重、耦合、有效性审核完成前不解读成本；阴性不跳过另一个必需对照。旧E全板重建成本与旧同身份记录相对误差≤1e−8（分母max(1,|旧值|)），gain绝对误差≤1e−10、原预算标签一致；超差是技术定位事项，不通过不能混用旧E和新对照成本。

每阶段最多2次有定位技术修复，记录失败、修改diff、重验和CPU，失败也计账；不得另开run清零次数。不因科学阴性改变solver/种子/分组/阈值。无法定位或超次数，保存partial_technical_failure；数学秩不足/缺资产为对应partial，不写科学反证。

只管理自有进程，记录PID/UID/starttime/cmd/绝对run路径；停止前重新核身份。若容器PID无法映射宿主只陈述可核范围，不能猜旧进程PID。不查询或管理旧GPU任务。所有自有子进程最终reap并退出。

## 6. 输出合同

- `protocol/RUN_LOCK.json`、`CODE_HASHES.json`、`SPLITS.json`、`VERIFIED_INPUTS.json`、`PERFORMANCE_GATE.json`，及不可变文档快照。
- `analysis/BASIS_CONSTRUCTION.json`：每折/臂训练题、权重、SHA、谱/边界/正交误差；`N_COUPLING_AUDIT.json`及稀疏边表、`SCALE_FACTORS.json`。
- `results/REFERENCE_COSTS.csv.gz`：每个自然槽/臂身份、r(.5)gain、原成本/保护/原预算、尺度/系数成本、全部NA原因；可另存去重计算表，但必须提供自然槽权重映射。
- `results/PAIRED_COSTS.csv`、`COVERAGE.csv`、`BY_QUESTION.csv`、`BY_DOMAIN.csv`、`E_REFERENCE_CHECK.json`、`DECISION.json`；自然/等H、全体/U内外、F支持集/N共支持集/全交集，所有分母与NA。
- 两张核心图：①F/E和三N/E规定成本比、原尺度/匹配尺度及域；②覆盖/原幅度可行性/未定质量。坐标标明几何比与系数成本单位，不能画成训练加速。
- `results/RESULTS_CN.md`与`CLAIM_EVIDENCE.md`；第一页≤600中文字，说清有无额外成本价值、哪类解释仍可行、是否只整体增益、不能证明什么、停止理由。
- `qa/NUMERICAL_AUDIT.json`、`qa/FINAL_CORE_REVIEW_CN.md`、`RESOURCE_AUDIT.json`、`FILE_AUDIT.json`、`COMPLETION.json`、`NEXT_CHAT.md`。

COMPLETION分开execution_status与scientific_decision，列必需项完成布尔值。全部工作完成但阴性/统计未定可complete；必要工作未做为partial。科学结论只来自已测部分。完成后退出自有作业，不自动下一轮。
