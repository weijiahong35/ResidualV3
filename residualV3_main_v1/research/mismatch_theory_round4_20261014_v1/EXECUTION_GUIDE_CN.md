# 第四轮自主执行任务书

版本：2026-10-14 v1。仓库：`/data/weijiahong/speculators-main-lora`。
先完整阅读[研究设计](THEORY_PLAN_CN.md)，再按本文件执行。[启动 prompt](START_PROMPT_CN.md)可直接交给新chat。

## 1. 用户授权与本轮范围

用户明确要求：研究同可见输入下修复一好一坏的现象，并研究“什么样的失配能修好、为什么、何时失败”；**有合适的卡就用，允许共卡，不等待整卡空闲。** 本任务书是新的第四轮授权范围，旧NEXT_CHAT中“不自动第四轮”的限制不阻挡本轮明确授权；旧run、结论及审计保持原样。

当前编写chat只交付文档与CPU核验，不启动模型。执行chat收到启动prompt后，可自主实现、使用共享GPU、固定状态重放、新字段采集、诊断用oracle求解、锁定留出验证及收尾。不要再次询问是否开始或是否可以共卡。

不训练新头/主体，不扩展结构，不搜索全局剂量，不重开RNG归因，不重跑已完成的正式评测或旧主分析。允许优化**离线诊断用的修正向量**，不保存成新模型checkpoint或声称已部署。旧状态重放用于新机制字段，不是把旧test重新当确认数据。

## 2. 资料索引：先研究，工程按需

缩写均为仓库相对路径，执行时展开：

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3
R4=residualV3_main_v1/research/mismatch_theory_round4_20261014_v1
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
```

必读顺序：

1. R3 `results/RESULTS_CN.md`、`THEORY_CARD_CN.md`、`CLAIM_EVIDENCE.json`、`RUN_AUDIT.json`，及 `COMPLETION.json`、`NEXT_CHAT.md`；核对其关键文件哈希。R3累计0.974224 GPU小时，已完成，不继承其余额。
2. R1 `results/RESULTS_CN.md`、`analysis/THEORY_CARDS_CN.md`；R2 `results/RESULTS_CN.md`、`THEORY_CARD_CN.md`。理解条件KL/平均拟合、S/U放宽下限、负结果分别回答了什么。
3. R3 `data/dev/dose_hf/*/DOSE.jsonl`、`COVERAGE.json`；示例为 `alpaca_227521df7e4c533d1fe5`。用本目录 `qa/verify_basis.py` 的计数规则复现探索线索，但不能把该CPU计数当完整输入核验。
4. R3 `code/collect_dose_hf.py`、`hf_frozen_adapter.py`、`repair_replay.py`、`dose_math.py`；FROZEN `repair/model.py`、`runtime/precision.py`、`head_latency/structure_fusion.py`。确认原无soft结构、BF16顺序和实际输入。
5. R1/R2/R3 `protocol/SAMPLE_MANIFEST.json`、R3 `protocol/LOCK.json`、`EXPOSURE_LEDGER.json`（若存在）及其他已知机制暴露记录。
6. R3 `protocol/RESOURCE_POLICY.json`、`analysis/SHARED_GPU_NATIVE_RESOURCE_AUDIT_CN.md`、`code/gpu_budget.py`、`resource_preflight.py`、`run_native_stage.py`；只复用已经按共卡修改的版本，排查所有硬编码RUN路径和旧空卡gate。
7. R1 `sources/LITERATURE_MATRIX.md`、R2 `sources/ROUND2_LITERATURE_MATRIX.md`、R3 `sources/READING_RECORD_CN.md`。定向核查已有条件修复、可见信息、候选oracle与受限修正认识；具体原文优先沿已有一手版本。不得重新做大规模综述代替本轮机制分析。

读到不存在的路径先按真实目录定位并记差异，不造文件。R3 fresh主要保存TV/dose等标量，完整概率快照只在少量dev保存；**不能从三个TV标量反推出p/q或oracle。** 必要时重放已保存C/H采集缺失完整分布，输出仅写新run。

## 3. GPU：按可用显存共卡，不等空卡

- 默认 `mode=shared`。不要求 `GPU utilization=0`、无外来进程、全卡显存清零或其他任务结束。不能沿用 `admit_when_empty.py` / `watch_empty_gpu.py` 作为本轮准入器。
- 每个GPU阶段启动前实时查询所有可见卡的UUID、总/剩余显存、进程与实际预约。优先选择满足该阶段内存估计且剩余显存最多的卡，不固定旧GPU0/7。外来进程存在本身不是冲突；有有效独占预约的卡跳过，**不能因一张卡有预约而拦住其他卡**。
- R3实测自有峰值24610 MiB可作旧配置参考，不能保证新K/求解峰值。初次目标模型采集采用逐分支/小微批流式、C1，按估计峰值加至少4 GiB余量准入；初始参考约30 GiB空余，具体以所用阶段实际配置估算入日志。30 GiB也不是统一硬门槛：较小工作集已有充分依据能容纳时即可启动；不机械要求旧36 GiB或整卡空闲。纯oracle阶段卸载target后按其更小内存需求选卡。
- 本轮默认自有显存上限32 GiB、运行剩余余量4 GiB；监测自有显存和全卡free。`gpu_memory_utilization`不是硬配额，不能代替峰值监测。native若启动，沿用R3已验证C1/eager、KV4 GiB和显式资源fraction入口，不回退到原正式评测52/44 GiB KV。
- 余量不足时，先保存并停止/缩小**自己的**微批、缓存、常驻模型或迁移到另一张满足条件的卡。数据、K、科学协议不随显存压力改变；K的分支可以逐个处理。不得终止他人进程、清他人缓存或释放他人预约。
- 预约/锁必须表达实际共享需求，不把本轮共享任务伪装成整卡独占。只管理核验过UID/PID/PGID/startticks/CMD/run路径的自有进程，僵尸和PID复用按实时身份处理。
- 若当前没有卡连最低工作集也容纳不了，继续CPU资料、数学/求解和已有数据工作并周期复查；这是内存不足等待，**不是等待空卡**。有卡满足条件即开始，不再请求共卡许可，不设长时间“观察空闲稳定”的门槛。
- 共卡wall time全部计入GPU账本；本轮不报告服务TPS，不能用共卡速度变化解释模型机制。

## 4. 新run与预算

在 `R4/runs/<UTC时间>_mismatch_r4/` 建独立代码、协议、账本与输出。原R1/R2/R3、归档包、冻结源码/权重/basis、安装环境、原正式报告均只读。不得调用会写回旧RUN的脚本；复制后改新路径，必要时禁用pycache。

本任务书提出的新轮默认上限：**8 GPU小时，同时最多1张GPU；CPU计算最多24 CPU小时，worker≤4、总线程≤8；新增磁盘≤60 GiB。** GPU启动、编译、失败、重试和共享运行均计入，不用新进程/新卡重置预算。建议把≥6 GPU小时留给身份重放、主采集、真实操作见证与新留出；native补充最多1小时，另留退出余量。预算是上限，不以用满为目标。

开跑前计算K×位置×prompt×词表的存储量。target logits按真实模型dtype无损保存或保存可无损恢复的分块；FP64概率按需流式构造。raw hidden/logits/basis按C/i去重，q_h按Z去重。不能用未经界定误差的top-k裁剪破坏TV/oracle。

每次新启动记录目的、源码SHA、预算与验收点。定位到新工程原因可在总预算内修复；同一问题无新定位/代码变化不重复启动。数值求解达到预算/固定迭代上限即报告认证区间，不进入无界求解。到限前保存并清理自有任务，完成可做报告。

## 5. 样本：旧现象复核与新机制确认分开

本次CPU按三个清单联合排除294个唯一组后，原面板剩余：GSM8K104、HumanEval68、Alpaca478；MT-Bench无新机制题。这个计数尚不包括执行者可能发现的其他机制暴露。

| 部分 | 默认样本 | 用途 |
|---|---|---|
| 旧现象复核 | R3原18条dose dev；保留全部，另索引21个候选组 | 固定C/H重放、完整身份、求解开发；不能只留21个好看组 |
| 新确认 | GSM8K/HumanEval/Alpaca各16题，共48 | 排除R1–R3全部组和所有本轮开发题；同一题的分支/状态全同簇 |
| native补充 | 仅在锁前排入的每域4题，共12，取新48题预定顺序前4 | 可选的访问状态范围核验；不是HF主任务准入条件 |

新题来源与R3相同：GSM8K用清单指向的旧200题 `final_C1.jsonl`；HumanEval/Alpaca用FORMAL各自 `panels/dspark_selected.jsonl`，不重新从大来源抽样。按既有NFKC/空白/GSM后缀规则去重，同时保存完整messages与token IDs哈希；语义/预训练重合未知要说明。

固定选择：合格剩余group排序后，用 `Random(2026101405 + sum(ord(c) for c in domain))` 洗牌，取前16；锁前合并额外暴露。仅因身份、重复、context长度等预定条件排除，不看效果换题；数量不足时锁前统一缩小并报告，不用旧test补题。

每题最多anchors1/65，15 proposal，T1完整词表、原EOS。**默认K=32条自然target分支，预先分fit16/eval16两独立seed池。** 用旧dev检查成本/碰撞覆盖；如果资源估计无法容纳，只允许在新题采集前整体改为K=16（fit8/eval8），记录原因并锁定。不因观察到更多一好一坏而选择K，不在fresh补采到出现碰撞。K固定后所有题一致，EOS自然缺失保留。

旧18题原K8状态保留原身份；新增K分支是本轮开发采样，单独记录，不改旧C/H或声称旧现象原本就有32分支。fit/eval和轨迹seed表在锁前一次生成；不得用eval标签选初始化、动作或分组阈值。

## 6. 执行阶段与验收

### A. 复核现象，并把数学合同做实

先复现CPU索引，再按原C/H重放旧dev采完整字段。原i=1/2为负控；同组各输入张量比较，完整q_raw/q_h检查。不同历史的目标差异须由实际p验证。身份不成立就撤回对应分组，保留原因；不能忽略差异继续套同Z公式。

参考数值合同：完全相同输入要求逐位或精确数值相同；完整输出TV差以dev实测核查，默认应≤2e−6。符号带 `ε=max(1e−5,10×dev最大输出/归约误差)`，必须在fresh前锁定；若ε>1e−3，先定位测量问题，不借大容差隐藏身份差异。近似组单列。

先跑 `qa/validate_oracles_cpu.py` 的小词表合同，再实现真实组求解器。直接LP、流式/切平面解之间做小规模交叉核验；检查raw可行、单分支shared=individual、相同target无共享代价、一好一坏仍可共同改善、真共享代价、尾部不可改的反例。CPU例子不是E2发现。

### B. 旧dev开发并锁协议

实现THEORY_PLAN中的R_ind/R_shared/R_safe、真实操作可行见证、fit→eval评估。求解器未认证时只给界，不能为了填满类型表强行归因。以完整dev确定计算成本、固定初始化/步数、固定K、符号带、数值误差及C_g分组/混杂对照。

固定自然权重：每prompt内先anchor等权、有效位置等权，再按自然存活分支频率加权；组权重为该位置分支占比。现象另报“至少两条不同历史”的覆盖；冲突配对率用不同历史的无序分支对为分母，并同时报组大小。独立分支评估只用fit/eval匹配Z，单独报告其覆盖与条件权重。不能删除无碰撞prompt再称全体平均。

开发结束写 `PROTOCOL.json`、`LOCK.json`、`SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`、`SOLVER_CONTRACT.json`、`RESOURCE_POLICY.json`、fit/eval seeds和分析源码SHA。保留dev决策日志。锁后仅在科学身份不变时调度/迁卡，资源变化不自动授权改K、样本或算术。

### C. 新48题固定采集与分析

对全体新题一次采集并保存足够的完整分布/输入身份。依次回答：

1. 现象在新题出现多少，覆盖什么域/位置/组大小，是否通过完整身份？
2. 逐状态、共享、安全共享最优之间差多少，求解误差多大？
3. 对失败组，是否找到真实操作内可替代当前修正的共同/安全见证？未认证比例多少？
4. fit得到的共同修正能否在eval相同Z上保留改善、伤害率怎样？
5. 修复前残差分歧与eval冲突是否呈预定关系，是否只是raw TV/覆盖/位置/组大小的差异？

有限组oracle的描述使用全部预定分支；fit→eval可泛化性另列，不把两表混成一次独立确认。统计固定2,000次按域分层prompt bootstrap，seed=2026101406；重采整个prompt的C/H/组/对，保持配对。报告95%区间、连续效果、各域和全部分母。区间跨0是不确定，不能写等效；计数/覆盖与数值证书不强行做显著性包装。

核心不确定性包括题面、有限branch、求解误差及采样测度；单个区间不能覆盖全部。C_g高低差和fit→eval真实操作见证收益若合并声称“双重确认”，用两项各97.5%区间并预先规定方向；不把其余探索表择优补成主检验。无可信最小实质阈值时仅报告大小，不用极小正差宣称理论突破。

### D. native仅作有界补充

主任务完成且锁前已排入时，最多12题，冻结FULL单臂真实采样；收集同轮raw shadow与actual q/p、访问位置和实测接受，检验HF诊断所涉及状态在native中是否有覆盖。可以逐轮做动作边界/可行见证，但没有同C多历史就不能冒充共享oracle验证。

不实现新的oracle在线采样，不将target标签喂给实际头，不重新证明RAW/FULL平均收益，不重开RNG、不做TPS。只有现有实际token/q/轮次机械身份通过才解释native；工程失败只关闭补充，不阻断核心HF成败研究。

### E. 收束

允许以负结果、有限覆盖或认证区间收束。fresh暴露后发现错误：保留原结果，隔离受影响分片，记录是否看过效应；修改科学协议后的数据不能继续称原确认。合格旧请求不重复、失败请求按固定清单恢复，不按效果更换题面或增加分支。

## 7. 输出：让用户看懂机制，而非只看完成状态

新run至少包括：

```text
protocol/                  # 协议、锁、样本、暴露、solver、共享GPU政策
code/ sources/ logs/ data/ analysis/
analysis/INPUT_OUTPUT_IDENTITY_AUDIT.json
analysis/PHENOMENON_REPORT.json
analysis/ORACLE_CERTIFICATES.jsonl
analysis/FEASIBLE_WITNESSES.jsonl
analysis/FIT_EVAL_REPORT.json
analysis/PRE_REPAIR_CONFLICT_REPORT.json
results/RESULTS_CN.md
results/MECHANISM_CASES_CN.md
results/CLAIM_EVIDENCE.json
results/RUN_AUDIT.json
results/RUN_FILE_MANIFEST.json
COMPLETION.json
NEXT_CHAT.md
```

结果第一页必须用日常语言给出：**什么类型能修、为什么；什么类型失败、已有证据指向哪种限制；哪些仍分不清。** 每项附效应、覆盖、证据层级（闭式/认证/可行见证/探索）及反证，不以“发现异质性”结束。

至少给三类真实案例：共同可修、共享限制、当前头有可行替代；若某类没证据，明确“未找到/未认证”，不编例子。案例按预定排序选，同时展示总体分布，不拿少量故事替代统计。图展示oracle层级与证书、同Z target需求差异、独立分支改善/损伤取舍；禁止用gain-gap代数散点当主证据。

收尾核对自有作业退出、共享预约释放、累计GPU/CPU时间、磁盘及冻结资产身份；原采样概率警报继续保留。下一方法建议最多一项，必须由本轮机制证据推出，不自动训练或开启第五轮。
