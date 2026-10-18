# ResidualV3 mismatch 理论研究：执行任务书

版本：2026-10-13 v1。仓库：`/data/weijiahong/speculators-main-lora`。
配套：[理论与证据基础](RESEARCH_BASIS_CN.md)、[可直接复制的启动 prompt](START_PROMPT_CN.md)。

**目标：以已有 E2 收益为起点，把“大小模型分布失配”变成定义清楚、能被反驳、能解释方法适用边界的研究。AR 修复头是研究载体。执行者应完成资料核验、诊断实现、限定实验和结论归档，而不是只交下一份计划。**

本文件授权范围是新研究目录中的工作，不是对原评测控制器的接管。本文记录的是计划与历史证据；没有新诊断已经运行或通过的含义。

## 1. 要回答的七个问题

| 编号 | 问题 | 必须交付的答案 |
|---|---|---|
| Q1 | mismatch 究竟比较什么？ | 指定同一状态上的 p/q、温度、信息集和前缀测度；区分分布失配、采样拒绝、语义错误。 |
| Q2 | 哪些失配来自条件信息变化，哪些仍是平均分布拟合偏差？ | 固定上下文的分支分解、TV 信息限制诊断及有限样本边界；不强行把拟合偏差叫容量下限。 |
| Q3 | 当前头利用了什么可修复结构？ | 同主体同状态 raw/正确前驱修复/置换前驱修复对照；简单校准参照；正负状态均报告。 |
| Q4 | 现有整体收益中，哪些能归于推理修复？ | 区分独立 DFlash、联合主体 raw、联合主体＋头；不把不同轨迹结果机械拆成训练与推理贡献百分比。 |
| Q5 | 局部分布改善怎样转为连续前缀收益？ | 明确存活、到达条件与终止规则；验证 native 接受记录及理论估计，检查 teacher-prefix 代理的限制。 |
| Q6 | 规律能否迁移到留出题目或其他状态？ | 锁定 test 的效应与区间；可选低容量预测任务；不把同 prompt 的 token 当独立样本。 |
| Q7 | 理论上较近邻多知道了什么？ | 文献已知/本项目支持/反证/尚不确定四类结论；不是再给三路输入起名字。 |

研究允许否定候选解释。若条件信息变化不是主要因素，转而如实分析拟合、校准、训练目标或部署测度差异；不能通过增模块使最初故事“成立”。

## 2. 接手先读什么

以下路径除特别说明外均相对仓库根。按顺序阅读，检索相关段落，不必一次通读所有历史日志。

### 2.1 当前状态、资产与主结果

1. 本目录 `RESEARCH_BASIS_CN.md`，再读本任务书全文。
2. `HANDOFF_20261013_CN.md`、`HANDOFF_20261013_STATE.json`、`RESIDUALV3_START_HERE.txt`：身份与保护规则。`HANDOFF_20261013_PROCESS_SNAPSHOT.json` 是旧快照。
3. `GROUP_MEETING_20261013_CN.md`、`EXPERIMENTS_NONTHINKING.md`：结果、训练和历史负结果。`EXPERIMENTS_THINKING.md` 只作有身份标注的历史参考。
4. 设 `R=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1`：读 `status.json`、`progress.json`、`COMPARISON.json`、`RESULTS_CN.md`、`plan.json`、`DATASET_AUDIT.json`、`PREFLIGHT.json`、`READY_AUDIT.json`，以及存在时的 `FINAL_AUDIT.json`。
5. 读 `R/parallel/controller_identity.json`、`DISPATCH_MANIFEST.json`、`controller.log` 末段；读有效完成窗的 `{dataset}/C{1|32}/formal/summary.json`。Arena 另读 `arena-hard-v2_C1_formal.log` 和对应 formal 状态，核对 attempt/归档关系。
6. `residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1/` 下 `README_CN.md`、`identity.json`、`status.json`、`watcher.log`，存在时读 `COMPLETION_AUDIT.json`。

**文件快照：2026-10-13 15:24 UTC／北京时间 23:24，40/48 有效正式窗、20/24 配对；Arena C1 attempt=2，控制器仍 running，CPU 收尾仍 waiting_existing_controller；FINAL_AUDIT 尚缺。** 原 Arena attempt1 因外来占用被原控制器归档，旧的 42 窗快照不能续算。`READY_AUDIT` 的 `require_complete=false`，不是最终通过。此处只核对了文件状态，没有把记录中的 PID 当成实时进程核验。

接手必须重新读实时文件；若状态改变，以有效产物和原控制器记录为准。没有 FINAL/COMPLETION 时明确说明缺失；不能自行补文件或把旧 Arena 单臂数字放回正式表。

### 2.2 当前结构、训练与测量实现

- `residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/repair/model.py`
- 同根 `native_runtime/runtime/components.py`
- `residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/head_config.json`
- `residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/losses.py`
- `residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/protocol.json`
- `residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/protocol.json`
- 当前六数据集对应 code 根下的 `native_evaluate.py`、统计和拒绝采样实现。先追踪实际 import、权重和采样处理，不根据旧目录名猜当前入口。

要核实：anchor/位置偏移、raw 与 repair logits 来源、采样变换后 p/q、Top256 之外的归一化、bonus/EOS 计数，以及 `acceptance_at_pos` 的含义。已读 summary 中该字段是累计存活 S_i，不能当条件接受率。

### 2.3 历史研究：复用经验，不重走失败路线

- `residualV3_main_v1/research/mismatch_design_reassessment_20261004_v2/REASSESSMENT_CN.txt`：旧 on-policy／反馈路线及成本教训。
- `residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2/RESULTS_CN.txt`：已完成三种子专家负结果，不再按旧 pending 处理。
- `residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/`：旧静态缓存身份与小样本限制。
- `residual/mismatch.py`、`residualV2/runtime_basis_hooks/mismatch_runtime_dump.py`：旧 mismatch 标签和 runtime 采集思路，不能直接当当前 E2 的已验证实现。
- `docs/experiment_archive_20260923/output_metadata/mismatch_v28_final_ablation/final_mismatch_summary.json`：旧 oracle／首拒绝分析；风险可预测、oracle 可修、实际能学会修要分开。
- `residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/`、`epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/`、`epoch2_head3_module_latency_C1_C40_20261012_v1/`：训练进展、跳层与计时边界。
- `Baseline_dspark_full_vocab_v1/`、`Baseline_dflash2_full_vocab_v1/`：先找最新 README、SOURCE_AUDIT、实际结果。不是默认可直接使用的官方公平基线。

### 2.4 外部文献

按 `RESEARCH_BASIS_CN.md` 的 L01–L10 与近邻表阅读一手原文。优先链条为：标准 speculative sampling → DistillSpec／Online SD → 有限容量与保真度 → 前缀目标与验证 → Domino／DSpark／DFlash-2／DFly／xPress／DSpine。

先用 `residualV3_main_v1/research/group_meeting_sources_20261013/` 的原文与审计；新网页和旧快照分版本保存。每篇填写：研究问题、失配定义、理论假设、实验能证明什么、不能证明什么、与当前假设重合处。检索遗漏的反例，不只搜支持 mismatch 的论文。计划不是系统综述，不使用“穷尽相关工作”的说法。

## 3. 自主权限、资源与保护规则

### 3.1 在本计划内可直接执行

可只读审计资料、联网核查论文；在新 run 中实现和修复诊断代码；采集冻结模型的概率与轨迹；跑 pilot、锁定实验 A–C；拟合限定的校准／低容量探针；生成独立图表、报告和恢复记录。满足下述状态与资源条件后直接运行，无需再询问“是否开始实验”。

可以根据 pilot 的成本裁剪次要分析，但必须在打开 test 前冻结裁剪决定。现有模型仅加载，正式权重、完整断点、源码、协议与自动报告全部保留。

### 3.2 本轮默认上限

以下是本任务书为首轮制定的执行上限，不是用户声称已经提供过的硬件预算：

- 同时最多 1 张确认空闲且未被预约的 GPU；累计 24 GPU 小时，包括 pilot、正式诊断、探针、native 和重试，不能按支路重新计额。
- 峰值新增磁盘 50 GiB；CPU worker 最多 4，总计算线程建议不超过 8，避免影响原 CPU 收尾。
- 建议分配：pilot ≤2 GPU 小时，核心 A/B ≤12，C ≤6，修复与缓冲 ≤4；它是上限分配，不是已实测耗时。
- 先 pilot 测算；若无法完成，优先保留 B 的关键干预及最小 A/C，输出缺失项。不得自动扩成多卡、完整 epoch、全面近邻重训或超预算搜索。

启动分片前估算剩余额度，运行中累计计时，并预留保存和退出时间；接近总上限时停止经身份核验的自有 GPU 作业，不能让在途任务无限运行。磁盘在写入前检查余量，达到上限停止采集，保留已有审计和必要记录；不得自动删除其他实验资产腾空间。

轻量拟合仅允许：在 dev 上为每个 raw 分支拟合一个全局温度参数；可选 ≤32 输入特征的正则线性／logistic 诊断探针，参数量 <10,000。均不得读取 test 标签拟合，不得作为新正式修复结构接入服务。新的头训练、结构改动、整主体训练属于后续候选方案，先交具体证据和成本，再扩展研究范围。

### 3.3 与现有任务共存

原六数据集控制器及 CPU 收尾未终结时，本研究只进行 CPU 资料、实现和小型数学验证。不能另起 `parallel.py`、重启原控制器/收尾、改原 GPU 分配、重跑已完成正式窗。

启用本研究 GPU 前核验：原控制器、子任务/重试及 CPU 收尾均已确认终态，旧资源预约已释放；使用宿主 UID、完整 CMD、启动时间和 GPU UUID 查询。失败终态允许缺最终审计，但必须记录原因并确认预约释放；仅写一条“存在问题”不构成终态。沙箱看不到进程不等于进程退出，旧 PID 消失不等于批次成功。如原批次失败，不擅自接管。无法确认则继续 CPU 工作并留下具体等待原因。

每个自有作业记录 run ID、PID/PGID、UID、CMD、启动时间、GPU UUID、输出目录。发现外来占用时只停止/暂停经身份核验的自有工作并保留断点；不得杀其他进程，不用 `killall` 或宽泛 `pkill python`。重启先查是否仍有同 run 工作，最多两次技术性重试，已完成分片不得重复。

## 4. 输出目录与恢复状态

在本目录下新建 `runs/<UTC时间>_<短标识>/`，禁止覆盖此前 run：

```text
manifest.json                  # 身份、哈希、预算累计、分片和分支状态
DECISIONS_CN.md                 # 预先决定、改变原因、反证与退出记录
protocol/PROTOCOL.json          # 采样、估计量、split、终止、统计和判据
protocol/SAMPLE_MANIFEST.json   # 数据版本、题面组、dev/test、随机种子
sources/LITERATURE_MATRIX.md    # 原文版本、阅读范围、支持/反驳关系
code/                          # 隔离的诊断与分析实现
data/                          # 最小必要记录；不同测度分目录
logs/                          # 命令、资源、错误和恢复
analysis/                      # 预注册分析脚本、聚合、图
results/RESULTS_CN.md           # Q1–Q7 的实际回答
results/CLAIM_EVIDENCE.json     # 主张、证据、边界、支持状态
results/RUN_AUDIT.json          # 完成度、数值/概率/数据检查
NEXT_CHAT.md                   # 精确恢复位置，不重复已完成实验
```

代码可复制必要最小文件或通过只读依赖调用；不得用会连带写回冻结源文件的硬链接。大型冻结权重只读引用，记录已有可信 SHA；若需计算哈希，分批限速，避免给原评测制造 I/O 竞争。不得初始化/改写仓库版本管理来替代实验身份记录。

`manifest` 必含：target、独立 body、联合 body、head、两种 basis、tokenizer 身份；实际 import 路径和源码哈希；数据来源/提交/题面组；runtime、采样、dtype、硬件；已耗 GPU 小时/磁盘；阶段状态及退出原因。状态可用 planned/running/complete/inconclusive/blocked/abandoned，含义需固定。这里是 run 的文件状态，不要求创建会话目标工具。

## 5. 实验前必须冻结的测量协议

### 5.1 样本与上下文

默认先覆盖 GSM8K、HumanEval、MT-Bench 三域，每域 64 个唯一 prompt：24 dev＋40 test。pilot 的 8 个 prompt/域从 dev 内取，不额外占 test。按题面/会话身份分组后确定 split；同 prompt 的所有位置、分支、模型臂都在同一组。记录上游重复，不能将重复题面分到两边。

GSM8K 使用旧正式数据清单，另外两域优先使用新批次固定 DeepSpec 提交的清单与首轮协议。记录模板和来源版本；不得拼接旧新 TPS。这里的 test 是本次分析的留出集，不宣称从未被项目使用或绝无训练污染；审计已知训练重合。

A/B 的上下文 C 由独立于比较臂的 target 轨迹生成：T1、预注册 seed、最多 256 输出 token，优先在第 1 与第 65 个生成 token 作为 anchor 后取最多两个 C。先按实际 runtime 核实 anchor 语义；pilot 中发现偏移可修正，打开 test 前锁定。回答提前结束则缺少的锚点不补采、不换题；terminal anchor 不取。context≤4096，超长者记录排除，不静默截断。若可用样本不足，报告精度限制，不能根据 test 效应替换样本。

每个 C 采 target-prefix 分支：pilot K=4、主分析 K=8，dev 子集 K=16 检查有限分支偏差；分支内最多 15 proposal。所有方法共享相同 C、H 与 target 分布标签。K 的最终值可因成本在 pilot 后更改一次并登记，不能看 test 后增加到显著。

### 5.2 温度、EOS 与测度

主要分布诊断 T1，按实际采样变换定义 p/q，明确 temperature/top-p/top-k。原始 logits 与最终采样分布分字段。T0 只用贪心一致性/首次不一致；对 T0 轨迹计算的 T1 TV 另列为辅助量。

分支提前生成 EOS 时，EOS 位置前的状态仍纳入该位置分布诊断，EOS 之后无记录，不重新补足分支。A/B 在位置 i 的测度为 target 分支条件于此前尚未终止的 μ_i；每个位置的存活分支数和 EOS 比例均报告。于是不同 i 的人群不同，不能把位置曲线变化全部解释为信息逐步丢失。置换只在同 C、同 i、仍存活分支之间进行。

teacher-prefix、target-branch、native 全 proposal、native reached-prefix 使用不同 `measure_id`。一个分布的期望与另一个分布的期望不能直接相减并命名为训练部署误差。teacher-prefix 数据来源不足时标缺失，不伪造 teacher 路径。

### 5.3 统计与主要判据

主要机制指标：B 中 `TV(p,q_permuted) − TV(p,q_correct)`，在 i=2…15 上按预定规则先聚合到 prompt，再等权聚合 prompt；正值表示正确的前驱对应关系有价值。必须同时报告 `TV(p,q_raw) − TV(p,q_correct)`，避免“正确比置换好但两者都比 raw 差”被写成有效修复。

默认 prompt 内先等权平均可用 C，再在 C 内等权平均可用位置及分支，锁入协议；同时报告各位置曲线与缺失覆盖。不能只挑有效果/有不同前驱的子组作主结果。首位置是负对照。

使用 prompt 分组配对 bootstrap（例如固定 seed 的 2,000 次）给效应区间；同时给各域结果、样本数与异质性。token/分支不是独立重复，原 20 个设置也不是 20 次独立重复。64 条/域是资源选择，不是已经满足检验力的保证。主要指标之外均标次要/探索性；若做多项显著性检验，预先约定校正方案。

打开 test 前登记实质效应阈值、数值容差与支持/反证/不确定规则。阈值依据 dev 的测量噪声和实际可解释量级，不按 test 结果调节；同时报告连续效应和区间，不把“未显著”直接写成无效。允许 dev 调试，test 固定一次主分析；任何事后新假设标探索性，不反复调参检验同一 test。

## 6. 实验 A：失配结构与信息限制

**估计对象：** 固定 C,i 的 target 分支分布 p_H；分别评估独立 DFlash raw 与联合主体 raw。二者在同组 H 上固定各自 q。

1. 计算每个分支的 TV/overlap、KL(p_H‖q)、target 熵；流式聚合 p̄。
2. 验证经验 KL 分解闭合：平均 KL(p_H‖q)＝平均 KL(p_H‖p̄)＋KL(p̄‖q)。报告两个分量原值，零分母时不报比例。
3. 计算分支间 TV 的 U-statistic 及其一半，作为信息不可区分限制的样本诊断；明确不是总体严格下界的保证。
4. 报告 K=4/8/16 的 dev 稳定性；i=1 的条件变化项应为零。不要强加随位置单调。

每个 C/i 的存活分支数 n<2 时，成对 TV 记缺失，不能填零。n=1 时经验 KL 条件变化项虽恒为零，但标为退化样本，不用于判断总体条件变化很小；分别报告此类样本覆盖。这与 B 的单分支置换恒等操作是不同估计对象，不能共用缺失规则。

**支持：** 在预注册测度与样本中存在稳定的条件变化量，且与后续修复效果有可解释关系。**反证/降级：** 条件变化很小或不对应收益，则不能用它解释主要增益；关注剩余拟合偏差。**无法判断：** 分支存活、数值精度、K 敏感性不足，报告区间和缺失。

此实验不能测出“大小模型参数差导致多少误差”。容量是模型类约束，还混有优化/目标/数据差异；没有专门容量控制实验就不做因果规模结论。

## 7. 实验 B：现有头是否利用与前缀相联系的信息

**核心对照不训练新头。** 固定联合主体、同 C/H、相同 hidden/raw logits/q 特征和 target p_H，只改变是否应用头、以及头读取的前驱。按位置强制重放 H；不让上一步重新采样改变下一步比较状态。

| 臂 | 输入/作用 | 解释 |
|---|---|---|
| B0 独立 DFlash raw | 原独立主体预测 | 完整配方参照，主体不同 |
| B1 联合主体 raw | 与 B2/B3 相同主体，关修复 | 同主体局部基线 |
| B2 正确前驱修复 | 使用 H 的真实前驱 | 现有冻结头的实际映射 |
| B3 置换前驱修复 | 同 C/i 的分支间置换前驱，其余固定 | 对前驱与 target 前缀对应关系的干预 |
| B4 简单校准 | 在 dev 为 raw 拟合的单一温度，test 固定 | 检查通用置信度调整是否已足够 |

B3 默认每组 4 个固定 seed 的随机排列，保留前驱边缘频率；不刻意强制置换成不同 token。只有一个分支或前驱全相同的组记录无信息/零效应并报告覆盖，不挑出后再冒充全体。i=1 的固定 anchor 置换应零效应，是代码负对照。置换仍可能改变联合输入分布，只支持该头依赖对应信息；不能称恢复了完整 target 历史或给整体收益分摊百分比。

B4 的温度只在 dev 最小化同测度 TV，默认 log 温度在预定区间 [−2,2] 内，优化方案和预算先固定；报告 test 效果，不把它说成充分排除了所有通用再拟合解释。联合/独立 raw 可各拟合一参数，身份分开。

报告 TV/overlap、KL、T0 argmax 一致性（若测）、改善/恶化状态比例；按位置、原始 TV、熵、target 在 raw Top256 内的质量作预注册分层。Top256 外概率仍会随归一化变化，不能称“支持之外不能修复”。主结果不给头读取当前 target p；p 仅用于离线标签与分析。

**支持：** 正确修复比 raw 好，且在留出上比置换好；校准参照不能解释全部观测。**反证/降级：** 置换与正确相近、正确不胜 raw，或温度已取得近似效果，则降低条件信息解释强度。**不确定：** 可区分前驱覆盖过低、区间宽、分层样本不足。

可选预测探针仅在 A/B 核心完成后进行：标签为同状态修复增量，输入仅允许推理可见特征；位置＋raw 熵为简单参照；dev 按 prompt 分组交叉验证正则，test 一次评估。使用 target TV 等分析标签作为特征时只能称 oracle 分析，不得宣传可部署预测器。无此任务就不用“可泛化预测失配/收益”的强表述。

## 8. 实验 C：从局部失配到 native 连续前缀

目的有两层：先验证测量正确，再研究局部修复是否迁移到实际轨迹。自洽公式检查本身不是新机制发现。

使用 B0、B1、B2 三臂，在同一 prompt 集合和预注册生成设置下分别运行自己的 native proposal/验证；记录它们不是同一轨迹的逐 token 配对。默认 C1、T1、15 proposal、每请求最多 512 输出 token、固定 1 个请求 seed；dev 可先检查第二 seed 的成本，再决定是否纳入。T0 可作预算内次要诊断。它是新短诊断协议，不替代 max2048/C32 正式主表。

原生 sampler/verifier 必须使用各臂实际 q。B1 若只能实现代理、不能核验正确的原生概率，则不伪造 native B1；先交明确缺口和 B 的有效证据。不能用 repair 生成 token 后再拿 raw q 当作其实际 proposal 概率。

逐轮记录：实际 proposal token、其 p/q、b_i=min(1,p_i(x_i)/q_i(x_i))、整词表 a_i=∑min(p_i,q_i)、实际逐位置验证结果、到达风险集合、S_i、有效 proposal 数、终止原因、发出 token 和 bonus/修正 token 数。

固定长度且无终止的理论为 S_k=E_Q[∏b_i]，E[K]=∑S_i。**不能用普通 draft 路径上的 ∏a_i 代替 ∏b_i。** EOS/输出预算下，分开记录两种指示：实际 reached 指示包含此前真实接受事件；路径有效指示 G_k 只由 proposal 路径中此前是否出现 EOS、块长及轮开始时剩余预算决定，不能包含实际拒绝随机数或已实现接受事件。这样，第 k 个 proposal 被接受并可发出的概率估计为 E_Q[G_k∏_{i≤k}b_i]，不会把真实接受结果重复加权。

EOS token 若由原实现计入输出则照记，EOS 之后的 G 为零；剩余预算、截断块和无法发 bonus 的轮次单列。以上估计的是可发出的 accepted proposal 数；拒绝后的修正 token、全接受后的 bonus 及实际总发出长度另按原生计数记录和核对。不要删除短轮后把其余平均说成全体 AC。

主报告给真实输出/轮、proposal 接受数、活动风险集合及分位置存活。固定长度恒等式可用预先定义的合成例子核验；不能按实际“这轮没终止”事后筛选后直接套未条件化公式。`AC=1+∑S_i` 只在计数规则确实满足时使用，不强行修正原日志来凑等式。

在原生可到达的状态上可额外 shadow 计算同主体 raw/repair 的 p/q，标明 shadow 未参与生成。它提供相同 H 上的局部比较；各臂自己的 reached-prefix 分布另列。比较 target-branch／teacher-prefix／native 时固定可比聚合规则并报告样本构成，不能把人群差异全解释为泛化失败。

**支持：** B 的局部改善在 native reached 状态也存在，并与存活/实际发出长度的变化相符。**反证/降级：** teacher/target 分支改善在 native 不成立，或局部收益集中在很少到达的后部状态。**不确定：** terminal、采样概率或轨迹身份无法可靠对齐。

带全词表插桩的时延不是服务 TPS。第一轮沿用已有正式吞吐作为背景，不发布插桩速度结论。若后续要测新实现速度，须另锁无插桩、相同 runtime 的协议，仍不得挪用旧不同 KV/Graph 批次数字。

## 9. 数值、数据和实现验收

首次 GPU pilot 前做有意义的 CPU 数学验证：

- p=q 时 TV=0、overlap=1；二元已知分布验证 TV 与拒绝接受一致。
- 两 token 例子 P(00)=P(11)=1/2：raw 独立均匀 q 的 S1=1、S2=1/2；正确条件预测为 S1=S2=1。用于检查联合路径/前缀估计，不能把它当本项目新定理。
- 构造同一最近前驱但更早历史决定下一 token 的例子，验证 Markov 可见信息不足以恢复任意完整条件分布。
- 对离散小词表穷举 KL 分解、TV 界和 E_Q∏b_i；验证误用 ∏a_i 能被非平凡反例发现。

GPU pilot 同时核对概率、偏移、头 replay 与正常一步调用一致性、显存、耗时和新增磁盘；仅对新增诊断写必要正确性测试，不复制整套训练测试。

全词表量优先在线 FP32 log-softmax/稳定聚合，只存标量、token、少量诊断向量及必要 C/H。每次处理一个 C 的有限分支和 p̄，避免长期保留所有全词表张量。小词表用 float64 参照设容差。不能 epsilon 裁剪后仍称精确 KL，也不能仅算 Top256 距离后写 full-vocab TV；若用 tail bucket，只能标聚合分布距离及下界性质。

记录 tokenizer、温度和过滤前后 logit 阶段； target 标签只能在分析路径使用。诊断代码不得使修复器在线偷看 p。浮点误差、异常值和零/小存活组保留计数，不能静默删除异常来形成正结果。

## 10. 自主执行顺序与分支退出

1. **P0，CPU：** 更新状态/资产/文献矩阵，明确 Q1–Q7，建立 run、样本清单和协议草案，完成数学验证与诊断实现。优先指出历史已经回答的部分，避免重跑。
2. **资源条件满足后，pilot：** 只用 dev，统一验证 A/B/C 所需的最低采集接口、代价和终止规则。技术问题最多两次修复重试；无效分支标 inconclusive，不反复试到出正结果。
3. **锁协议：** 冻结样本、估计量、阈值、成本裁剪与分析代码哈希。符合本授权就继续，不把锁协议变成人工确认关卡。
4. **核心运行：** A 与 B 共用同一批 target 分支及标签，先形成 B 的干预对照；随后 C。按分片保存、查重恢复；预算优先保核心、不新增模块。
5. **分析与收束：** 按预注册规则打开 test；每条假设标 supported/weakened/inconclusive，保留反例。GPU 达上限前保存并停止经身份核验的自有 GPU 作业，继续 CPU 汇总。
6. **下一轮建议：** 仅在核心证据完成后提出 ≤2 个方法完善候选，每个写“哪条发现驱动、预计改变何种失配、最小对照、可能失败原因、额外成本”。本首轮不自动训练新结构。

候选方向随证据决定：若主要为拟合/校准，考虑目标和校准；若部署测度迁移明显，先核对历史 on-policy 失败的原因；若现有头仅在特定状态有效，先检验规律可靠性，不自动重启已失败专家路由；若全套假设均弱，保留系统收益并如实收缩理论贡献。

仅以下情形需要把具体变更交回用户：超出资源上限、需要正式模型/冻结协议变更、新的全量训练或科学问题重大转向、需要干预他人/原控制器进程。等待期间继续不受影响工作。日常实现修复、负结果分析和授权范围内运行无需反复确认。

## 11. 最终交付与完成标准

交付一份可核查研究报告，而非提前写成论文结果：

1. Q1–Q7 逐项回答，给结论、证据路径、模型/测度身份、支持状态和边界。
2. 理论卡片：定义、假设、推导、估计量、何时不能使用；标准工具与新增经验发现分开。
3. 图表：A 的条件变化/拟合偏差与 K 稳定性；B 的 raw/正确/置换/校准效应及首位置负对照；C 的真实存活、局部风险与终止覆盖。缺项明确标缺，不生成模拟数据。
4. 近邻矩阵：共同问题、已有理论、各自方法近似、本文多出的可检验认识；不以跨论文涨幅排名。
5. 运行与数据审计、资源累计、技术失败/负结果、所有命令与恢复指引。
6. 一页决策：当前理论能站到哪一步；哪些不再主张；下一轮至多两个最小验证或方法完善建议。

完成不要求假设成立。理论不支持时给明确反证，同样算研究完成；但不能将缺少数据/实现失败写成科学反证。未完成分支及具体原因必须可恢复。
