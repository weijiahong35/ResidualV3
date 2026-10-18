# 第六轮执行指南

版本：2026-10-15 v1。先读 [THEORY_PLAN_CN.md](THEORY_PLAN_CN.md)。本指南与 [START_PROMPT_CN.md](START_PROMPT_CN.md) 是新一轮限定执行范围；当前编写chat未启动实验。

## 1. 范围与入口

允许在新run内自主完成旧数据提取、必要冻结前向重放、固定离线预测器拟合、新48题采集、T1–T4分析、报告和清理。**小型预测器是本轮新授权的研究工具；主体、真实修复头、basis、target均冻结。** 不训练新头，不改变修复输入/损失/结构，不上线gate，不开展native、RNG、TPS或剂量网格。

全轮至多12对松弛oracle诊断也是新的有限范围，不延伸到actual basis动作优化。R5“不自动第六轮”保留为旧run终态，不阻止用户交付本prompt后的新任务。

仓库 `/data/weijiahong/speculators-main-lora`。只写：

```text
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/<UTC>_mismatch_r6/
```

旧run、归档、安装环境、冻结权重、自动结果报告均只读。设置 `PYTHONDONTWRITEBYTECODE=1`，缓存改到新run；旧入口可能硬编码RUN，复制必要代码到新run并改入口后使用，不能直接运行旧控制器。

## 2. 先读什么

完整路径、证据与原文问题见 [sources/READING_GUIDE_CN.md](sources/READING_GUIDE_CN.md)。顺序：

1. R5 `RESULTS_CN.md`、`NEXT_CHAT.md`、`COMPLETION.json`、`RUN_AUDIT.json`、样本与暴露清单：确认已有事实、未见H覆盖和旧警报。
2. R4原始采集schema、原头概率重构、`tv_oracles.py`；R5共同流探针、统计和资源控制代码。先读，不重跑旧stage。
3. R1–R3未支持结果及近邻文献矩阵：防止再声称条件主导、减半普遍更好、已有普通工具是创新。
4. 探针控制与容量的两篇原文；只做定向阅读，不增加MDL、文献大综述或新预测器搜索。

先核对本目录 `qa/INPUT_AUDIT.json` 的来源哈希。该审计只验证准备时的有限输入，不替代执行时全样本身份检查或新GPU准入。

## 3. 数据与测度固定

|用途|题面|允许使用的量|
|---|---|---|
|工程|R4旧18条dev|字段、重构、合同、成本；不加入正式train/dev|
|预测训练|R4 `fresh_test` 48题|主位置3/8/15的全部自然fit/eval分支，旧标签与允许特征|
|预测开发|R5 `fresh_test` 36题|同口径状态；选风险预测器正则、固定风险阈值，不能改特征/模型族|
|新确认|R6三域各16，共48题|预测测试用全部合格eval主状态；T1只用其预定匹配novel子集|

“旧fresh”现在已暴露，只能作train/dev。以规范化完整messages、token身份及继承group SHA核对题面隔离，所有anchor/历史同属一道题。跨文件复制的同一观测去重；不同自然采样槽即使H相同也保留频数。不能把来源文件不同视作独立样本。

R4训练必须裁到R5/R6的3/8/15位置、相同自然target分支到i存活条件和概率重构口径。R4特征缺失时只允许冻结前向重放取得缺字段，输出新run，保留旧标签及观测身份；概率身份不一致先定位，不能用重采标签悄悄替换。

新题沿R5固定GSM200题面板、HumanEval/Alpaca的DeepSpec固定面板。合并R1–R5及其他已发现机制暴露、专家pilot、headonly train/holdout清单。继承NFKC/空白/GSM后缀规则和训练精确重叠检查；不声称排除了未知预训练语义重合。机制未暴露不等于未参加原正式评测。

准备时由R5继承账本减去R5新题可得GSM76、HumanEval40、Alpaca450条候选；**只是继承计数，执行前重新合并暴露**。按group SHA排序后，用 `Random(2026101506 + sum(ord(c) for c in domain))` 洗牌取前16。少于16时锁前按三域共同最小数等量缩小并披露；锁后不补题。与旧train/dev重叠的题不能进入fresh。

固定anchor1/65；15 proposal；K64=fit16/eval48；T1完整词表自然target采样、原EOS、原HF anchor形成规则及生成预算、不截prompt。仅保存主位置3/8/15及控制位置1/2的target与头输入输出；中间历史连续生成。缺anchor、EOS、未匹配Z保留，不能换位置或强制新历史。branch/置乱/投影/训练/bootstrap分别用独立命名空间，master seed `2026101516`，bootstrap `2026101517`；锁前保存显式seed表，不用Python内置hash。

## 4. 先冻结预测，再打开新题

阶段A：只读提取旧数据；建立来源/特征/标签schema、同Z身份、单位合同与旧数据覆盖。运行 `qa/check_contracts.py` 的新run副本，明确它是人工例子。不得把合同通过当模型证据。

阶段B：在R4训练、R5开发完成下述固定拟合。写模型、投影矩阵、scaler、正则选择、阈值和源码SHA。建立 `MODEL_LOCK.json`，**必须先于新题模型采集**。如果旧数据没有所需类别/字段，记录不可识别或工程阻碍，不改任务救结果。

阶段C：锁新题清单、采样、统计、资源和所有代码，随后只采一次固定fresh。feature导出器只读取白名单字段。target概率和标签单独写入label bank；预测worker配置只能指向feature bank和锁定模型，代码/访问审计确认不打开label bank、TARGET_BITS或fit产物。collector可以保存target，但不得据fresh效果调整方案。

阶段D：先生成并签名全部fresh预测、原始概率分数和同Z置乱索引；记录 `PREDICTIONS_SEALED.json`，然后由独立评估入口读取标签。T1另行使用新题fit标签，产物不能回流预测器。禁止fresh拟合、fresh标准化、改阈值、校准、特征筛选或重选模型。

旧collector若自动打印效果摘要，在新run副本中关闭；不以“人没看见”为唯一防泄漏措施。发现测量错误保留原输出及暴露记录；修正后受影响确认分析标探索，不能称重新未见。

## 5. 固定特征与预测器

### 5.1 A/B输入

A按顺序使用：raw hidden4096、previous static embedding4096、原头实际qp256、实际fusion_concat768，再加6标量：i/15、hidden norm、raw entropy、raw top1、raw top1−top2、raw Top256质量。实际维数由schema核验；预计9222维。qp/fusion均为冻结确定变换，不是独立信息源。

B=A加更早H：排除紧邻前驱后，从近到远最多13个token。每个静态embedding除以其RMS（分母下限1e−8），乘固定4096×32 Rademacher矩阵，元素为±1/√32；13个槽各32值+存在mask，共429维。不足补零与mask=0，不改成均值池化或搜索历史长度。矩阵只由独立seed生成，与标签无关。

数值列用train-only加权均值/标准差标准化，零标准差设1，mask不标准化；所有dev/fresh原样使用。静态embedding允许，target运行得到的历史hidden、logits或likelihood禁止。

白名单之外全部拒绝：当前位置p/target logits、p−q、TV/Δ、oracle/共同流M、当前或未来采样token、fit统计、seen/novel、seed/pool/promptID/groupID/hash等。ID只作join和分组，不能进预测器。H只能截到当前位置之前，不能把记录中的`token`字段混入。T3的v*另有明确例外，与真实当前采样token不同。

### 5.2 主风险模型

固定单隐藏层width32、GELU、sigmoid输出，无dropout。A/B独立拟合同一模型族，单固定seed；AdamW lr1e−3、batch512、50epoch，使用自然层级权重的二元交叉熵。训练权重与第7节测度一致，不做类别重加权/欠采样；每epoch所有状态均访问一次。记录每epoch train/dev Brier及训练损失，不按fresh早停。

权重全局归一为Σw_i=1，目标L=Σw_iℓ_i。均匀访问状态行时，batch损失固定为 `(N/b)Σ_batch w_iℓ_i`，N为训练状态行数、b为该批实际行数（含末批）；不能逐批再除该批权重和，也不能把全局小权重直接batch mean。方向分类沿用同一规则，ridge使用同一全局权重。

仅weight decay `{0,1e−4,1e−2}` 三候选，分别按R5加权dev Brier选择；差≤1e−12取较大decay。只用第50epoch模型，不搜索checkpoint、层数、宽度、学习率或seed。最多6次主拟合；报告参数量与实际迭代，固定预算结束不等于已证明收敛。NaN/数值失败属于技术问题；有限模型表现差不是“历史无信息”。

常数基线=训练加权坏状态率，另报全预测非坏的结果。B参数更多，用同一B置乱控制限制容量解释，不宣称已控制所有模型复杂度影响。

阈值：候选集合固定为该模型dev全部不同预测分数与{+∞}的并集；从中选**最小的、加权FPR≤10%的阈值**，判定score≥threshold，相同分数整体处理。A/B分别锁定。保存实际dev FPR/recall；test直接用该值，不重调为test 10%FPR。dev无负类则阈值不可定义，不能借test补。

### 5.3 同Z历史置乱

仅在fresh eval的同C/i/精确Z内，对B的整个H特征块做8次独立固定置乱，保持组内自然多重集与所有A输入。完整Z相同并核验raw/头输出一致后，可等价置乱B预测，避免重复前向。不能跨C、按标签选择可置乱组或仅保留冲突组。

每次先算逐状态Brier，再平均8次损失。单历史组贡献真实零差，缺失资产记缺测。该对照不是8样本显著性置乱检验，也不是历史语义干预。独立报告8次变化范围，不挑次数。

### 5.4 次要收益/方向模型

收益：A/B各固定ridge，目标加权MSE+`λ||w||²`，样本权重归一至和为1，λ=1e−3，截距不惩罚；使用带权LSMR的等价增广系统，atol/btol=1e−8、maxiter=1000，记录stop code与残差，不构造巨大正规矩阵。不能按fresh换λ。共2次拟合。

方向：A/B共同追加v*完整静态embedding4096、q0(v*)、qh(v*)；v*规则见理论计划。固定线性3类softmax、加权交叉熵，AdamW lr1e−3、decay1e−4、batch512、50epoch，同样train-only scaler，共2次拟合。三类输出固定；train缺类则该任务仅探索并披露，不能从test补样。报告加权logloss、混淆矩阵、每类覆盖；常数基线采用训练加权类别计数（权重重标至总数N），每类加固定0.5伪计数再归一，即(Nπ_c+0.5)/(N+1.5)，π_c为加权训练比例；N仅是平滑尺度，不是独立样本数，避免无类时无穷分数。打分logloss统一裁至[1e−12,1−1e−12]并报告裁剪次数，不能改目标分布。

T3最多4次拟合，不扩搜索。收益/方向不增加主确认检验；方向正确、预测风险正确和真实修好分别报告。

## 6. 概率身份、T1与有限oracle

按R5实际BF16 logits位重构，用稳定FP64 softmax计算全词表TV。uint16是BF16位，不是数值logits。原头q使用实际GPU保存的selected_updated_logits替换raw对应Top256，不能用不同CPU重投影冒充。检查p/q的C/H/i、raw/candidate身份及头完整输入输出。

同Z以实际完整输入确认，不以压缩向量近似碰撞或仅前驱token ID代替。hidden norm、q路/输入basis和output basis角色均保留。若同组raw或qh不一致，不能套用共享动作与输出置乱。

T1严格沿R5 u/o/M、M/2；M≤1e−12视零但不修改p/q。每个r核非负、和为1、fit TV改善等于M/2，默认残差≤1e−10；原头重构差>1e−6先定位，不扩大1e−5损伤带掩盖。novel对同C/i全部fit H去重；i1/2作为身份控制（i2同Z不能有不同更早历史），不加入主指标。

T4只看新eval主状态的不同H对。异号定义一支Δ>1e−5、另一支Δ<−1e−5；非异号要求两支均改善或均受损，排除中性。每域每类按“同长且更早H汉明距1优先，再prompt SHA/anchor/i/Z/H字典序”取前2对，去重复自然H对；不足不补其他类。全轮至多12对，保存完整候选和选择账本。

复用R4 `tv_oracles.solve_oracle`，每对权重1/2；S=raw Top256，S内概率可松弛调整，U内条件比例保持raw，仅U总质量可变，不是T1的全词表自由动作。`max_iter=100`、gap≤1e−6、feasibility≤1e−7，沿其每LP时间限制，CPU时间计总账。计算R_ind、R_shared和相对raw安全的R_safe；记录原始primal/dual、候选、逐支实际TV与认证残差。不可只从均值界宣称两支严格改善。禁止basis_witness、扩迭代救证书或新max-min优化器。求解未证保留区间，科学报告写不知道。

## 7. 统计合同

每个C/i内状态按自然频数平均；题内有定义主位置等权、有效anchor等权；域内题面等权，三域等权。训练/开发损失沿同一层级权重。HP/HC使用全部合格eval主状态，HN采用预定novel匹配子集；不能混分母或用HN覆盖筛预测测试。

|主量|定义|支持所需覆盖|
|---|---|---|
|HN|novel上的T1 TV改善|每域至少6题有定义|
|HP|Brier(A)−Brier(B)|每域至少6题含坏状态，且该域两类均存在|
|HC|平均置乱Brier−真实Brier|至少10题、每域至少2题含同Z坏/非坏异标签组|

覆盖仅决定解释力度，不改变点估计人群；门槛不是功效保证。三量各双侧**98⅓%**区间（尾部分位1/120、119/120，三项Bonferroni），同时给95%描述区间。每项“支持”要求锁定覆盖满足且主CI下界>0。分别报告，不赋予一般理论总passed；若讨论三者同时成立，必须三项各自满足，不能互补补证。

各主量按自己的预定题面人群分层做2000次完整prompt bootstrap：HN从每域有定义的HN题抽取；HP/HC从全部合格预测测试题抽取，不能限于含坏状态或异标签组的题。题内所有anchor/历史/配对/8次置乱一起抽取；覆盖独立报告。先固定置乱平均再bootstrap；CI条件于这次固定模型seed、分支与置乱采样，不包含训练seed不确定性。主CI不因覆盖低而多采题或重抽至显著。

各域原始有效题数/状态数、novel比例、损伤率、坏状态负TV幅度、PR-AUC（加权average precision）、test FPR/recall及固定10个等宽概率bin校准均必报。没有类的bootstrap样本对应分类指标记NA并报比例，不重抽到有类；主量某域未定义不能填0或偷偷两域均值。缺测、零动作、真实零差分别记录。

允许附HP/HC在novel上的预定描述，样本量不足照实；不能拿它替换全状态主量。风险识别是否实用要看效应大小、基线和误报检出，没有预定实质阈值不喊突破。

## 8. 资源与自主执行

**共卡默认允许，有能容纳阶段工作集的卡就用；不等空卡、不要求utilization=0。** 每次准入实时查所有可用UUID、free memory、有效预约和本阶段峰值估计，不固定GPU0/7；某卡有独占预约只跳过该卡。不得使用旧watch_empty/admit_when_empty入口。R5约21.3 GiB峰值仅为采集估计，非当前可用性证明。

同时最多1卡，自有显存默认上限32 GiB，峰值之外保留4 GiB余量；用C1流式/小微批，科学样本、K与位置不随资源变化。余量足够即继续，不额外问共卡许可。无卡容纳最低工作集时推进CPU工作并短周期复查；不终止他人任务，不为GPU等待而开无关实验。

全轮所有run/失败/重试合计：**8 GPU小时、24 CPU小时、新增磁盘700 GiB**。GPU进程存活期间等待、编译和共卡慢速都计入；CPU用进程树user+system累计，临时缓存/失败分片/副本计磁盘。上限不是目标，不换run重置。旧数据流式读，不复制整套。

CPU worker≤4，全部自有计算线程合计≤8。所有进程import前设置OMP/MKL/OPENBLAS/NUMEXPR=1，PyTorch intra/inter-op=1；记录实际进程树，驱动/IO线程单列。R5锁前线程记录缺口和R4历史超限不能继承为许可。

记录UID/PID/PGID/startticks/CMD、GPU UUID、用途和run路径。只管理核实身份的自有任务与预约；不触碰原controller/parallel/CPU收尾或他人作业。每阶段最多2次有新定位的技术修复重试，无新诊断不反复启动。到限保存并释放自有资源，独立科学结果保留。

## 9. 交付与结束

至少交付：

```text
protocol/{PROTOCOL,LOCK,MODEL_LOCK,SAMPLE_MANIFEST,EXPOSURE_LEDGER,SEED_TABLE,
          FEATURE_SCHEMA,STATISTICS_RULES,RESOURCE_POLICY}.json
predictors/                 # 仅离线预测器，与真实头权重分目录
analysis/{IDENTITY_AUDIT,FEATURE_ACCESS_AUDIT,PREDICTIONS_SEALED,
          NOVEL_HISTORY_STABILITY,PREDICTION_RISK,HISTORY_PERMUTATION,
          SECONDARY_PREDICTIONS,LIMITED_SHARED_SPACE,COVERAGE}.json
results/{RESULTS_CN.md,MECHANISM_CASES_CN.md,CLAIM_EVIDENCE.json,
         RUN_AUDIT.json,RUN_FILE_MANIFEST.json}
COMPLETION.json
NEXT_CHAT.md
```

结果第一页用简单中文回答：①未见历史还能共同修吗；②历史能提前判断原头修坏吗；③能判断方向吗；④失败是否仍有兼顾空间；⑤哪些不知道。区分已有事实、新确认、次要探索、未执行。

图最多四类：seen/novel覆盖与T1效应；A/B/置乱风险对照与常数基线；方向/收益预测；真实同Z一好一坏及共同空间案例。工程审计放后面，不让它代替科学答案。

负结果可以complete；必须采集/拟合/分析未完成则partial并明确剩余项，覆盖不足可完成但结论不支持。无合格T4案例可记not_applicable，有案例而未求解需记未完成。科学状态、测量身份、资源合规分别记账，不合成一个passed。

收尾核对自有进程退出、预约释放、旧资产身份、预算和文件签名。保留R2 native概率警报与既有线程/证书披露。本轮不新检验runtime无偏性，不自动开启第七轮。
