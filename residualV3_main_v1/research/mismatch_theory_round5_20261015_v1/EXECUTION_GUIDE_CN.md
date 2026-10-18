# 第五轮执行指南：先解释失配

版本：2026-10-15 v1。先完整读 [THEORY_PLAN_CN.md](THEORY_PLAN_CN.md)。用户交付 [START_PROMPT_CN.md](START_PROMPT_CN.md) 后，执行 chat 按以下范围自主完成，不只交计划。

## 1. 授权范围与明确撤下的工作

本轮允许：新run内写诊断代码、只读复用旧完整分布、旧C/H必要字段重放、冻结模型的新题采集、CPU/GPU概率分析、固定探针与置乱对照、报告和清理。当前编写chat没有启动模型。

**不训练主体或头，不做A/B/C训练，不改损失/输入/结构，不蒸馏oracle，不扩剂量网格，不新跑LP或优化见证，不开native/RNG/吞吐支路。** 无论预算剩多少，都不自动增加这些任务。R4文末“唯一训练建议”与旧“不自动第五轮”分别是旧建议和旧授权边界；收到本启动prompt后只执行这份新的机制研究范围，旧run终态保持。

新run：`residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/<UTC时间>_mismatch_r5/`。研究文档根不保存模型采集输出。所有失败、恢复和临时文件仍计入同一轮账本，不能换run重置预算。导入旧代码前设置 `PYTHONDONTWRITEBYTECODE=1`，避免写入旧目录的pycache；新缓存统一放新run。

## 2. 必读资料与复用入口

仓库 `/data/weijiahong/speculators-main-lora`，下列均为仓库相对路径：

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3
R4=residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4
DOC=residualV3_main_v1/research/mismatch_theory_round5_20261015_v1
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
```

必读顺序：

1. R4 `results/RESULTS_CN.md`、`THEORY_CARDS_CN.md`、`METHODS_CN.md`、`CLAIM_EVIDENCE.json`、`RUN_AUDIT.json`，以及 `COMPLETION.json`、`NEXT_CHAT.md`。先区分真实操作见证、放宽oracle、fit→eval和资源偏差。
2. R1 `results/RESULTS_CN.md`、`analysis/THEORY_CARDS_CN.md`；R2/R3各 `results/RESULTS_CN.md`、`THEORY_CARD_CN.md`。不重算旧主检验；理解已被削弱的解释。
3. R4 `protocol/LOCK.json`、`SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`、`SEED_TABLE.json`；R1–R3各样本清单及其他机制暴露记录。核对 DOC `qa/INPUT_AUDIT.json` 的输入哈希，变更要查因。
4. R4 `code/collect_mechanism.py`、`hf_frozen_adapter.py`、`repair_replay.py`、`numeric_gate.py`、`prepare_r4_samples.py`、`statistics_rules.py`。按需要读FROZEN原头/precision及原E1/E2训练协议，仅用于确认身份，不修改。
5. R4 `code/gpu_budget.py`、`cpu_runner.py`、`resource_identity.py`，`protocol/RESOURCE_POLICY.json`、`results/RESOURCE_HISTORY_CN.md`。先审查硬编码RUN、线程设置、旧磁盘guard，不原样运行旧入口。
6. R4 `sources/READING_RECORD.json` 索引的R1原文；重点条件失配、边缘与前缀、候选限制、标准KL/TV工具。新增数学参考见 DOC `sources/READING_GUIDE_CN.md`。做定向阅读，不拿大综述代替实验。

R4已存完整数据，典型每题/anchor资产为：`TARGET_BITS.npy`、`STATE_ARRAYS.npz`、`HEAD_ARRAYS.npz`、`HEAD_BRANCH_ARRAYS.npz`、`INPUT_ARRAYS.npz`、`BRANCH_STATES.json`、`GROUPS.json`。真实字段按LOCK与源码读取，不猜schema。target logits用BF16 uint16位保存，必须按位恢复，不能把整数值当logits。

**默认先流式读取R4资产；不因R3曾只有标量而整批重采R4。** 缺字段才固定原C/H/seed重放，输出写新run并记录原因；新字段分析不宣称旧test重新成为fresh。不能从TV标量反推出p/q。

## 3. 阶段安排：理论问题决定采什么

### A. 旧数据开发与合同

用R4旧18条dev作为实现开发集，R4已暴露48题可作新机制描述和案例来源。先跑 DOC `qa/check_math_contracts.py` 的独立副本，输出到新run。它检验数学与反例，不能称模型证据。

实现：完整分布重构；C/H/Z身份；E1经验分解；E2固定共同质量转移；E3同C标签置乱；E4完整词表账本。所有实际头输出从锁定GPU资产重构或按原实现采集，不能用不同CPU重投影替换，R4曾观察到两后端动作差异。

旧开发集必须输出覆盖预检：每域有定义题面、有效位置、fit≥2不同H的组、eval匹配状态、M=0比例、配对非零步长覆盖。只用于验证可定义性和成本，不按效果挑K、位置、阈值或换探针。

默认设置已固定：K32、fit16/eval16、主位置3/8/15、负控1/2、8次置乱、诊断步长M/2；预算不足优先减保存冗余、流式处理，不按结果缩选数据。若旧dev显示主量根本无法定义或工程不可行，记录阻碍并完成可做的探索报告，不能偷偷换成更容易显著的任务。

### B. 锁定新题、数值和统计

建立 `protocol/PROTOCOL.json`、`LOCK.json`、`SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`、`SEED_TABLE.json`、`NUMERIC_CONTRACT.json`、`STATISTICS_RULES.json`、`RESOURCE_POLICY.json`。记录本轮全部代码/输入SHA及开发决策。

锁前只核新题身份、长度、暴露与资源；不看新题模型效果。fresh打开后不得更换抽样、探针、组资格、统计或语义类型；修复测量错误要保留原输出和暴露记录，受影响结果降为探索，不重新包装确认。

### C. 一次固定新题确认

新机制留出默认 GSM8K/HumanEval/Alpaca 各12题，共36题。沿R4原面板：GSM旧200题 `final_C1.jsonl`；HumanEval/Alpaca为FORMAL各 `panels/dspark_selected.jsonl`。路径和SHA由R4 manifest展开，不能自行换下载版本。

合并R1–R4全部机制暴露、专家pilot、旧headonly train/holdout清单及新发现的相关记录。沿既有NFKC/空白/GSM后缀规则，核对完整messages与token身份；训练/预训练语义重合未知仍披露。机制未暴露不等于从未参与正式评测。

DOC输入核对时，R4 ledger剩余为GSM88、HumanEval52、Alpaca462题；这只是旧账本计数，执行前必须合并后续暴露。按group SHA排序，再用 `Random(2026101505 + sum(ord(c) for c in domain))` 洗牌取前12。不足时只允许锁前按三域共同最小合格数统一缩小，并报告精度降低；锁后不按效果补题。

固定每题anchor1/65，主体一次计算15 proposal。自然target分支K32，fit/eval各16条独立seed槽，采样T1完整词表、原EOS、不截断prompt。只需保存主位置3/8/15与负控1/2的target分布/头输入输出；生成历史仍连续到对应位置，不能跳过中间token采样。

anchor上下文形成沿R4锁定HF规则，max生成预算沿其配置记录，不能把缺失anchor补为更早位置。EOS提前终止、缺anchor、未匹配Z均保留。fit/eval独立seed不代表不同历史必然互异。不得强制换前驱、补采到碰撞或删除没有冲突的题。

seed表在fresh前一次生成：轨迹、8次置乱、bootstrap分别用独立命名空间，保存显式整数列表，不用Python不稳定的内置hash。默认master seed 2026101506，bootstrap 2026101507。

### D. 分析、案例与收尾

按E1–E4顺序回答问题，但主结果篇幅给E2/E3及真实失败案例，不让分解表或工程审计淹没解释。旧数据描述与新确认分表，不能拼成同一独立样本数。

真实案例选择规则预先固定：从主位置中按 `domain/prompt_sha/anchor/i/Z/H` 字典序取各类首例；类型为共同方向在eval有效、fit有效但eval受损、原头同Z一好一坏、尾部剩余限制。某类未出现就写未找到。每类最多2组，正负token贡献各最多8个并附完整账本路径；不按视觉冲击或增幅最大挑例子。

案例类型可用结果定义，是说明性选例，不用于该类型存在率的无偏估计。总体频率来自完整预定集合。文本语义解释附真实前文与token，不能事后发明“数学/代码特有机制”并当确认。

## 4. 统计、零值与覆盖

主集合：i∈{3,8,15}，同C完整Z身份有效，真实fit组至少2条**不同H**，且eval有匹配Z。重复自然历史保留原频数；资格按不同H计，均值按自然分支计。置乱对照沿真实资格，不能用置乱组效果筛选。

每个有效C/i内，对合格eval状态按自然频数平均；题内对有定义的主位置等权、有效anchor等权；域内对有定义题面等权，三域等权。每层都报告原始总数、合格数和被条件化掉的覆盖。没有合格状态的题面/域是未定义，不能填0。

合格组M=0、共同步长0时保留raw和零效果。微小M数值带在dev锁定：默认≤1e−12按零，不改p/q分布本身；不抬高阈值去掉难例。伤害判定采用固定TV数值带，默认1e−5；均值保留连续未阈值化值。需根据dev重构误差确认阈值足够；误差>1e−6先定位，不能靠大阈值掩盖。

两项预定主量：

|检验|量|支持条件及限制|
|---|---|---|
|H_shared|E2真实fit组探针相对raw的eval TV改善|97.5%题面区间下界>0，且三域各≥6个有定义题；只支持该探针、该条件子集|
|H_match|E3等步长真实分组相对置乱分组的eval TV改善|97.5%区间下界>0；三域各≥6个有定义且含非零配对步长的题；只支持对应方向价值|

覆盖门槛是防止极窄子集承载确认，不是功效保证；达不到仍报告效应/区间/分母，结论为覆盖不足。样本若锁前缩到每域不足6，自动降为探索，不自行降低门槛。效应虽正但极小只报告大小，无已知实质阈值不能称重要突破。

按域分层的2,000次prompt bootstrap，整个题面的anchor/分支/置乱配对一起重采。固定8次置乱先平均，不当8个独立实验；CI条件于本次固定采集与置乱实现，不单独估计有限分支采样及置乱的Monte Carlo误差。补充报告8次的描述范围，不挑最有利次数。其余量95%区间并标描述，不用它们替代未支持的主量。

同时报告eval平均改善、损伤比例、负向TV幅度、各域、位置、组大小和配对非零率。i1/2负控遵守理论计划中的不同零项规则。C合并探针只作描述；不从“不显著”推出等效，不从探针与头账本相关推出头的唯一机制。

## 5. 数值与信息边界

- 分布用实际保存BF16 logits的稳定FP64 softmax重构。KL用稳定log-softmax/log-sum-exp处理，不静默截尾、加epsilon或把词表尾部合成一桶；遇非有限值记录并定位。
- E1使用同一经验权重的m_C/m_Z；不得Z等权后仍叫自然测度分解。全样本身份与fit→eval风险分别计算。
- E2/E3每个r核查非负、和为1、fit不越界及TV增益≈a，允差固定1e−10；E1/逐token账本残差默认1e−10。失败要保存分片，不自动归一化遮蔽错误。
- 真实与置乱组保存分支索引、完整u/o或可重建来源、M、配对步长、r的SHA；eval只读取冻结fit产物，禁止重新选方向。
- 原头q、raw q、target p的C/H/位置及batch身份绑定。旧R4复制的数据核SHA，新数据同Z完整输入/完整输出检查，近似相同组不能套精确公式。
- 探针可能改U内部比例，不能标“真实头合法操作”。E4新状态只计算S/U闭式；不继承无关旧组的oracle证书。
- 本轮无runtime接受分布结论；旧native警报和未建立的理想连续存活规律保持，不借标准文献替本地实现背书。

## 6. 共卡与资源：有能容纳工作集的卡就用

**共享GPU为默认授权。无需整卡空闲、无外来进程或utilization=0；不等空卡。** 每次GPU阶段前实时查看全部可见卡UUID/free memory/有效预约，选能容纳该阶段工作集且有余量的卡。一张卡有独占预约只跳过该卡，不阻挡其他卡。不能沿用 `watch_empty_gpu.py` / `admit_when_empty.py` 空卡gate，也不固定GPU0/7。

先用旧资产做CPU开发。新采集按R4流式C1、小微批估计峰值，自有GPU显存默认上限32 GiB，阶段峰值之外留4 GiB余量；旧约26 GiB峰值仅供估计，不作为所有阶段统一门槛。纯概率分析工作集小，可用更少剩余显存。一次最多1卡，不报告共卡TPS。

有满足条件的卡立即推进，不额外询问共卡许可。若没有卡容纳最低工作集，继续CPU工作并短周期复查；不强行挤占、不终止他人任务。运行余量不足只保存/缩微批/释放自有未用缓存/停止自己的进程或迁卡，科学样本与K不随资源变化。

预算为**本轮全部run合计最多8 GPU小时、24 CPU小时、新增磁盘≤700 GiB**。编译、失败、重试、共卡慢速和GPU进程存活的等待均计入；上限不是用满目标。预计新数据远小于磁盘上限，按C/i去重、读取旧文件不复制整套资产。临时缓存、失败分片和副本均计磁盘。

CPU worker≤4，所有自有CPU计算线程合计≤8；控制器和分析进程也计CPU时间。import前设置OMP/MKL/OPENBLAS/NUMEXPR线程为1，PyTorch intra/inter-op各1，监测实际进程树；驱动/IO辅助线程单独披露，不能把计算线程超限隐藏为辅助线程。R4历史超限不继承为许可。

进程与共享预约记录UID/PID/PGID/startticks/CMD、GPU UUID、run路径和用途。只管理身份核实的自有任务；不碰原controller/parallel/收尾、其他研究任务、安装环境或旧结果。到限前保存退出，不以新进程/新卡重置账本。同一失败无新定位不重启，同一技术阶段最多2次修复重试，耗尽后收束可做结果。

## 7. 必交文件与完成标准

```text
protocol/        # 锁、样本、暴露、seeds、数值/统计/资源合同
code/ sources/ logs/ data/ analysis/
analysis/IDENTITY_AND_MEASURE_AUDIT.json
analysis/SOURCE_DECOMPOSITION.json
analysis/FIT_PROBE_ARTIFACTS.json
analysis/HELDOUT_COMMON_FLOW.json
analysis/MATCHED_PERMUTATION.json
analysis/ACTUAL_HEAD_TOKEN_LEDGER.json
analysis/COVERAGE.json
results/RESULTS_CN.md
results/MECHANISM_CASES_CN.md
results/CLAIM_EVIDENCE.json
results/RUN_AUDIT.json
results/RUN_FILE_MANIFEST.json
COMPLETION.json
NEXT_CHAT.md
```

结果第一页不堆公式，按五问逐条回答：**失配来源；为何能共同修；条件对应是否有用；原头何时失败；哪些仍解释不了。** 每条写证据类型、覆盖、反例/不确定性，不能只写平均收益。

图表最多四类：来源账本及有限K敏感性；E2留出改善/损伤；E3等步长对应对照与覆盖；真实一好一坏的token概率移动。数学合同放附录，不把例子当真实模型数据。

核心采集/分析有未完成项目时不得标全轮complete；预算/工程限制写partial并列剩余项。科学负结果可以complete。完整科学状态、资源偏差、数据身份分别记账，不合成一个模糊passed。收尾核对自有进程退出、共享预约释放、磁盘/时长、旧资产身份；不自动开第六轮或训练。
