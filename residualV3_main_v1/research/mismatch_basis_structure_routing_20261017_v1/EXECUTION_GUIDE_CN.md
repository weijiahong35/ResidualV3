# 执行指南：只研究基底内部结构与选择

## 1. 授权边界

本次交付只有文档、prompt和设计审核。用户将START_PROMPT明确用作执行指令时，才授权在本包新run执行：CPU基底构造、固定凸优化、18次小型闭式ridge路由拟合、折外评价和收尾。它不授权修复头/LoRA/主体训练、LLM前向、新题采样、GPU、native/TPS或环境安装。

旧basis_reuse/source/R1–R9及其controller/收尾脚本只读。禁止导入带旧ROOT写入副作用的旧脚本；提取纯函数到新run，核对并保存源码SHA。不得在旧run“补文件”“续跑”。

新包所有执行run合计上限6 CPU进程树小时、16GiB自有RSS、8GiB新增盘、GPU0；同一run技术失败也计账，换run不重置。单数值worker；所有BLAS/OMP/MKL/NUMEXPR及torch线程1，在import数值库前设置；CUDA_VISIBLE_DEVICES为空。使用现有`/data/weijiahong/miniconda3/envs/dflash/bin/python`；此包不需要加载torch模型。

## 2. 已核实资产，仍未核实性能

制文阶段standard-library输入审核已实际通过，见qa/INPUT_AUDIT.json：

- 36题、69C、207 C/i、9936 P自然槽、2571唯一(C,target_row)。
- 69缓存JSON和69NPZ的完整payload SHA、三份E512 SHA、R8 META身份；静态W/embedding文件头和偏移。
- 177个C/i有多H；1402个C/i/previous类中542类含多个H。因此有信息冲突可检查，但这不是已发现的修复冲突。
- 审核仅约2.36 CPU秒、峰值约47MiB；不包含任何新基底、容量解、路由或科学评价。

未核实：静态tensor完整payload重哈希、新基底数值秩、聚类收敛、凸解收敛与全流程耗时。不能将“资产可读”写“6小时必能完成”。

主要只读路径：

```
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
residualV3_main_v1/research/mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse
residualV3_main_v1/research/mismatch_basis_source_20261016_v1/runs/20261016T172105Z_basis_source
```

W：`/data/weijiahong/huggingface/models/Qwen3-8B/model-00005-of-00005.safetensors`的lm_head.weight；embedding在对应model-00001文件。准确tensor偏移、dtype、shape从已核验manifest读取，不能硬猜文件首地址。BF16位模式解码；静态tensor可mmap只读。NPZ成员不能按mmap假定逐行读，应逐C解压、唯一行归一化、处理后释放。

## 3. 先锁定，再做完整门禁

1. 建立`runs/<UTC>_basis_structure_routing/`，保存指令快照、AGENTS、设计JSON、输入manifest、全部源码SHA和进程身份。资源账单始终写新run。
2. 重核样本、折分、自然权重、H/previous、旧E训练身份、静态W/Etensor SHA和概率。先完成小词表数学核验，不把数值错误写成不可修。
3. 固定工程题只用身份选：首完整group SHA的C1、P唯一target最多C（并列按SHA/anchor），去重。名单已在qa/INPUT_AUDIT.json。工程fold固定0；不得看收益换工程题。
4. 计时fold0的完整训练聚类、IN/OUT/RND构造、SVD/正交补和映射；之后同算法构造可直接复用。低秩/空簇是科学构造未定义，记录，不能在门禁中换rank。
5. 两个工程C运行全部14基本空间、8删除空间、E512两级方向稳定性、完整词表TV和输入构造；计时18次ridge所需Gram/解法及一个完整train标签批次。工程C可作为本轮固定面板的已计算部分，不按效果换掉。
6. 预测必须覆盖：每个唯一目标OOF22空间；每题在另外两个fold作为训练时各12空间。因此基础上界`2571*(22+2*12)=118266`个状态/空间求解，另2571次E512方向续算、数学核验、全部分解/IO、重复H展开、5000次bootstrap、作图和审计。不可仅按2571条估时。
7. 对凸求解同时量实际迭代及每步耗时；以2000步主上限、4000步E累计上限作保守预测，按最慢工程C的相应维度吞吐外推。每种规模分别计时，不拿128维速度代替512/删除空间。性能方法只看耗时/误差，不读科学汇总。
8. 完整剩余预测乘1.5后，必须≤剩余CPU预算80%，且留下至少1200CPU秒收尾；RSS与磁盘也留20%余量。基底构造完所有fold、全标签解开始前再核一次。超限为`partial_resource_limited`，不能减少题、位置、组、rank、求解上限、必要对照或改GPU。

保守上限可能拒绝执行，这是真实风险。不要用未计入的“多数应该很快收敛”保证预算。执行chat若找到严格等价的向量化/分解复用可做工程修复并重新计时，但不能借此改科学合同。

## 4. 固定执行顺序

S0身份/数学→S1工程门禁→S2所有fold聚类与基底→S3训练12组收益标签和折外14组容量→S4八项删除与方向稳定性→S5固定路由/输入等价类上界→S6完整TV、任务交叉与全部统计→S7核心问题审核/审计/退出。

每阶段最多两次有定位技术修复，保存diff、失败原因和CPU。改声明或设计要先按AGENTS保存审核；不能换目标、筛样本或参数“修科学”。实验过程中按阶段报告已完成/未定/剩余预算，避免留无人监管作业。

若任何关键库秩不足或资产缺失，执行能独立完成的面板，缺项明确NA，最终partial_asset_rank_limited。若只因固定求解上限留下宽区间，仍保留全部界并完成报告；不能将未收敛算成阴性。

## 5. 必须交付的文件

- `results/RESULTS_CN.md`：第一页≤600中文字，逐一回答Q1–Q4和证据层次。
- `Q1_DIRECTION_USAGE.csv.gz`：固定坐标、符号、幅度、稳定性；`Q1_REMOVAL.csv`单轴/整组损失界、错误类别和分母。
- `Q2_SPACES.csv`、`Q2_CONTEXT_PAIRS.csv`：完整512/单128/IN/OUT/RND的界、局部见证和全部负例。
- `Q3_ROUTING.csv`、`Q3_INPUT_ALIAS_BOUNDS.csv`：常量/领域/X0/X1/oracle、实际相同输入上界、训练标签可靠覆盖。
- `Q4_TASK_BY_GROUP.csv`、`Q4_TASK_CROSSOVER.csv`：按fold的全组表、训练选组/折外交叉差；错误类型表另列。
- `CLUSTERS_AND_RANKS.csv`、`NUMERICAL_STATES.csv.gz`、`DENOMINATORS.csv`：空簇、秩、换组比例、优化未定、expected/valid/invalid/NA、题/C/位置/槽/H及原权重。
- 4张核心图：方向使用及删除；完整空间与组比较；路由与信息上界；任务×组交叉效果。每张图有CSV数据。
- 最多6个案例：每域按SHA首两题，各取最小anchor、i3、最小有效slot。案例按身份固定，缺失记NA，不按收益挑。
- `CLAIM_EVIDENCE.md`、`qa/FINAL_CORE_REVIEW_CN.md`、数值/资源/文件审计、`COMPLETION.json`、`NEXT_CHAT.md`、`ARCHIVE_SEAL.json`。

## 6. 做完后再审核四问

最终审核逐问写：问题原文→实际执行对照→关键表格和分母→支持/未支持/未定→是否越界。先审核正文是否把“选组”写成“修复已实现”，是否把胜128写成胜512，是否把簇成分写任务适配；不通过先改结论再复审。不允许为了通过审核另开科学实验。

核验身份后退出所有自有数值作业并关闭预算账单。外部旧进程不动。没有自动下一轮；NEXT_CHAT仅写已证实内容、缺口和是否需要新授权。
