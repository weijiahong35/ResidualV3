# Thinking 实验总账

更新：2026-10-13 23:39:22 UTC。由原始结果自动生成，叙述来自逐实验登记。共71条记录。

只看这份文件即可查目的、对照、设置、接受长度（AL）和吞吐（TPS）；证据链接用于追查原始记录。

## 口径

- AL 使用各正式 native 窗口的 `acceptance_length`，包含 target bonus；不是接受概率。TPS 为实际输出 token/s。
- 只收录完成的正式窗口；初始化、smoke、逐状态 oracle、算子微基准不混入正式性能表。缺失指标写“未测”，不填0。
- “污染”窗口仅保留观测值，不支持干净吞吐比较；未提供隔离证据的历史窗口标“未核验／只看AL”。
- 不合并不同数据、面板、thinking 模式、proposal 长度、实现或训练预算；跨实验绝对TPS不直接相除。每张表保留运行分组、轮次和来源。
- RAW 按该实验协议定义。Thinking EAGLE 的 RAW 是原生 EAGLE-3 投机解码；单训 DFlash 与联合主体关闭头是不同对照。
- 正负研究结果、消融权重、长训权重及恢复断点都保留。失败准备记录与有结果的实验分开查。

## 快速定位

| 编号 | 实验 | 目的 | 当前记录 |
|---|---|---|---|
| T001 | [原始短训基线 N_no_soft_2layer](#t001) | 在较低串行成本下保留mismatch修复收益。 | complete；已记录12个正式窗口 |
| T002 | [Thinking 1万条×3轮层数长训](#t002) | 验证数据规模与1/2/3层深度效果。 | complete；已记录36个正式窗口 |
| T003 | [LMHEAD／EMBEDDING硬软反馈 · C1](#t003) | 检查冻结token输入表的选择对修复效果及成本的影响。 | 未提供顶层状态；已记录12个正式窗口 |
| T004 | [LMHEAD／EMBEDDING硬软反馈 · C40](#t004) | 验证完整硬软反馈在多并发下的接受收益和开销。 | complete；已记录12个正式窗口 |
| T005 | [A–G输入／深度／尺度消融](#t005) | 逐项检查当前hidden、soft lag-1、层数、修正尺度。 | complete；已记录32个正式窗口 |
| T006 | [No-soft 1／2／3层短训](#t006) | 检查修复头加深是否换来更多实际收益。 | complete；已记录16个正式窗口 |
| T007 | [Direction／dose 分开与合并](#t007) | 检查把方向与幅度合并预测是否更有效。 | complete；已记录12个正式窗口 |
| T008 | [冻结DFlash直接揭示前驱／前缀](#t008) | 检查原冻结DFlash能否直接利用块内真实历史。 | complete |
| T009 | [前驱条件响应干预](#t009) | 区分修复头是否已学到前驱响应、响应方向是否有益。 | complete |
| T010 | [线性包络／dose-only 可达性](#t010) | 判断错误能否在当前基底和冻结输出映射内修好。 | complete |
| T011 | [完整最终头 latent oracle](#t011) | 去掉线性包络限制，检查完整LN/direction/dose头的表达上限。 | complete |
| T012 | [条件方向配对分析](#t012) | 检查前驱变化时目标修正动作是否发生变化。 | complete |
| T013 | [共享梯度传输诊断](#t013) | 检查单状态改善是否被共享更新抵消。 | complete |
| T014 | [梯度传输独立复核](#t014) | 复核direction组共享梯度负贡献。 | complete |
| T015 | [Epoch学习／latent机制诊断](#t015) | 检查训练阶段、输出权重及局部方向变化。 | complete |
| T016 | [Epoch1／2／3 native复测](#t016) | 验证简单早停能否解决T0吞吐。 | complete；已记录8个正式窗口 |
| T017 | [显式交互头](#t017) | 检查输入交互结构能否改善方向学习。 | complete；已记录8个正式窗口 |
| T018 | [PREFIX／TOKEN_TV／TOKEN_NLA损失](#t018) | 检查改变位置梯度分配是否改善接受收益。 | complete；已记录8个正式窗口 |
| T019 | [Native轨迹最终头oracle](#t019) | 在真实采样前驱上检查oracle上限及学习标签。 | complete |
| T020 | [OUTPUT／TARGET_KL／ORACLE_KL学习](#t020) | 验证oracle标签是否比直接target监督更易学。 | complete |
| T021 | [28状态极小oracle拟合](#t021) | 区分训练拟合能力与泛化限制。 | complete |
| T022 | [64提示oracle覆盖](#t022) | 扩大固定研究状态覆盖范围。 | complete |
| T023 | [16／64提示oracle学习规模](#t023) | 检查训练拟合优秀是否能迁移到提示留出。 | complete |
| T024 | [Top256／128／64](#t024) | 减小输入／修正候选支持以降低成本。 | complete；已记录16个正式窗口 |
| T025 | [输出rank256／128／64](#t025) | 检查低维输出动作空间能否额外加速。 | complete；已记录20个正式窗口 |
| T026 | [输出头低秩审计](#t026) | 检查已训练方向矩阵谱及实际动作包络。 | 未提供顶层状态 |
| T027 | [Parallel_TF及五组统一对照](#t027) | 把重SwiGLU搬到hidden/q上下文，推理仅以实际lag-1轻门控。 | complete；已记录20个正式窗口 |
| T028 | [Parallel_T0／T1真实前缀训练](#t028) | 检查learner块内采样前缀重算teacher的训练配方。 | complete |
| T029 | [目标hidden／logits／hidden+logits输入消融](#t029) | 检查现有目标前缀信息和logits输入能否改善修复。 | complete；已记录20个正式窗口 |
| T030 | [OUTPUT＋Top16候选CE](#t030) | 给旧两层修复头加入硬候选监督。 | complete；已记录4个正式窗口 |
| T031 | [长训权重：方向／dose瓶颈](#t031) | 用长训最终权重检查错误是否由固定方向限制。 | complete |
| T032 | [长训共享全参数梯度](#t032) | 检查同状态改善是否在跨提示平均梯度中被抵消。 | complete |
| T033 | [五问题预实验](#t033) | 联合检查目标信息、greedy监督、方向、梯度冲突和运行成本。 | complete |
| T034 | [OUTPUT＋greedy margin native小试](#t034) | 验证固定状态margin改善是否转化为闭环接受收益。 | complete；已记录16个正式窗口 |
| T035 | [KL／overlap／margin 任务基底](#t035) | 改变输出基底采集目标，检查大残差方向是否有接受价值。 | failed |
| T036 | [线性可预测 mismatch 基底](#t036) | 用输入-梯度低秩回归保留部署输入能预测的修正方向。 | complete；已记录40个正式窗口 |
| T037 | [非线性可预测基底](#t037) | 检查非线性预测梯度方向是否优于线性RR。 | complete；已记录4个正式窗口 |
| T038 | [可达性／可学习性瓶颈probe](#t038) | 区分动作空间上限、当前方向、误修及研究留出拟合。 | complete |
| T039 | [输入充分性／目标hidden来源 v1](#t039) | 核对部署目标hidden来源以及分支动作冲突。 | failed |
| T040 | [输入充分性／目标hidden来源 v2](#t040) | 核对部署目标hidden来源以及分支动作冲突。 | failed |
| T041 | [输入充分性／目标hidden来源 v3](#t041) | 核对部署目标hidden来源以及分支动作冲突。 | complete |
| T042 | [OUTPUT／Euclidean／Fisher方向学习](#t042) | 检验不同动作几何监督是否能改善修正方向。 | failed；已记录10个正式窗口；用户已授权停止，后续未完成窗口不算成绩 |
| T043 | [EAGLE-3 Thinking · 3 proposal](#t043) | 检查相同mismatch思路迁移到原生EAGLE-3递推。 | complete；已记录8个正式窗口 |
| T044 | [EAGLE-3 Thinking · 7 proposal](#t044) | 检查相同mismatch思路迁移到原生EAGLE-3递推。 | complete；已记录8个正式窗口 |
| T045 | [仍供其他基线使用的历史共享数据](#t045) | 仍供其他基线使用的历史共享数据，保留研究链条及依赖。 | 未提供顶层状态 |
| T046 | [全词表lag-1历史共享数据与协议](#t046) | 全词表lag-1历史共享数据与协议，保留研究链条及依赖。 | 未提供顶层状态 |
| T047 | [额外头数学／实现诊断](#t047) | 额外头数学／实现诊断，保留研究链条及依赖。 | complete |
| T048 | [standalone_five_issue_pilot_gpu5_20261003](#t048) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T049 | [standalone_gradient_validation_gpu5_20261001](#t049) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T050 | [standalone_longtrain_bottleneck_gpu45_20261003_prepare_failed_tokenizer](#t050) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T051 | [standalone_native_oracle_pair_gpu23_20261001](#t051) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T052 | [standalone_native_oracle_pair_gpu45_20261001](#t052) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T053 | [standalone_native_oracle_pair_gpu45_20261001_attempt2](#t053) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T054 | [standalone_native_oracle_pair_gpu45_20261001_attempt3](#t054) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T055 | [standalone_native_oracle_pair_gpu45_20261001_attempt4](#t055) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T056 | [standalone_native_oracle_pair_gpu45_20261001_attempt5](#t056) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T057 | [standalone_native_oracle_pair_gpu45_20261001_attempt6](#t057) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T058 | [standalone_nosoft_basis_objective_short_gpu3_shared_20261003_v2](#t058) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T059 | [standalone_nosoft_basis_objective_short_gpu5_20261003](#t059) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T060 | [standalone_predecessor_response_gpu5_20261001](#t060) | 同名成功版本之前的准备失败／废弃尝试。 | 已归档 |
| T061 | [退役记录 · standalone_nosoft_depth_10k_3epoch_gpu123_20261002](#t061) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T062 | [退役记录 · standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002](#t062) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T063 | [退役记录 · standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003](#t063) | 记录失败阶段、恢复来源，防止重复启动。 | not_launched |
| T064 | [退役记录 · standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003](#t064) | 记录失败阶段、恢复来源，防止重复启动。 | not_launched |
| T065 | [退役记录 · standalone_nosoft_predictable_basis_gpu12345_20261003_v1](#t065) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T066 | [退役记录 · standalone_nosoft_target_hidden_logits_short_gpu012_20261003](#t066) | 记录失败阶段、恢复来源，防止重复启动。 | preparation_failed |
| T067 | [退役记录 · standalone_oracle_learning_anygpu_20261001](#t067) | 记录失败阶段、恢复来源，防止重复启动。 | superseded_by_verified_split_run |
| T068 | [退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002](#t068) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T069 | [退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2](#t069) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T070 | [退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3](#t070) | 记录失败阶段、恢复来源，防止重复启动。 | failed |
| T071 | [退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4](#t071) | 记录失败阶段、恢复来源，防止重复启动。 | failed |

## 实验记录

<a id="t001"></a>

### T001 · 原始短训基线 N_no_soft_2layer

**类别／状态：**研究实验；complete；已记录12个正式窗口。

**做什么：**在较低串行成本下保留mismatch修复收益。

**对比什么：**同轮RAW、旧完整三层A、no-soft两层N。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | A_full_3layer | 4.081050 | 231.429 | +3.20% | -9.30% | 已核验 |
| C1/formal | 1 | 0 | 1 | N_no_soft_2layer | 4.081234 | 241.618 | +3.20% | -5.30% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.146 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | A_full_3layer | 3.676044 | 207.621 | +14.69% | +0.39% | 已核验 |
| C1/formal | 1 | 1 | 1 | N_no_soft_2layer | 3.685537 | 217.022 | +14.99% | +4.94% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 206.809 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | A_full_3layer | 4.198375 | 2501.051 | +4.06% | -3.64% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_no_soft_2layer | 4.191603 | 2543.239 | +3.90% | -2.01% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.034450 | 2595.510 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | A_full_3layer | 3.856576 | 2265.390 | +15.11% | +6.72% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_no_soft_2layer | 3.867179 | 2311.046 | +15.43% | +8.88% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.350280 | 2122.658 | +0.00% | +0.00% | 已核验 |

**结果解释：**T1 C1/C40 TPS分别约+4.94%/+8.88%；T0接受长度略增而TPS仍亏。是有效正结果，不是全面失败。

诊断／原报告摘录（保持原口径）：

- 每个并发分别同卡、同prompt、同seed对照；仅单轮，不能据此声称统计显著性。

**原目录／权重入口：**[eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/C40/formal/summary.json)；[RESULTS.md](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/RESULTS.md)；[experiment.json](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/experiment.json)；[N_no_soft_2layer/preregistration.json](eval_results/standalone_embedding_nosoft_2layer_C1_C40_gpu5_20260930/N_no_soft_2layer/preregistration.json)。

<a id="t002"></a>

### T002 · Thinking 1万条×3轮层数长训

**类别／状态：**研究实验；complete；已记录36个正式窗口。

**做什么：**验证数据规模与1/2/3层深度效果。

**对比什么：**每卡同轮RAW、旧短训N、新长训1/2/3层。

**实验设置：**10k×3epoch=30000步；从epoch2 BF16恢复、重建FP32 master并重置Adam；1/2/3层卡0/1/2；C1=200、C40=2000，原dataset10000后；7proposal、max4096。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/status.json)。

协议字段：`train_count=10000; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=200条，proposal=7，max_new=4096; C40=2000条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| nosoft_1layer/C1/formal | 1 | 0 | 1 | N_baseline_2layer | 4.197766 | 248.328 | +3.88% | -4.24% | 已核验 |
| nosoft_1layer/C1/formal | 1 | 0 | 1 | N_no_soft_1layer_10k | 4.381804 | 261.674 | +8.43% | +0.90% | 已核验 |
| nosoft_1layer/C1/formal | 1 | 0 | 1 | RAW | 4.041150 | 259.334 | +0.00% | +0.00% | 已核验 |
| nosoft_1layer/C1/formal | 1 | 1 | 1 | N_baseline_2layer | 3.882209 | 228.930 | +15.68% | +6.49% | 已核验 |
| nosoft_1layer/C1/formal | 1 | 1 | 1 | N_no_soft_1layer_10k | 4.135874 | 246.000 | +23.24% | +14.43% | 已核验 |
| nosoft_1layer/C1/formal | 1 | 1 | 1 | RAW | 3.356027 | 214.975 | +0.00% | +0.00% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 0 | 1 | N_baseline_2layer | 4.204197 | 2546.606 | +3.58% | -2.05% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 0 | 1 | N_no_soft_1layer_10k | 4.394266 | 2665.687 | +8.27% | +2.54% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 0 | 1 | RAW | 4.058732 | 2599.778 | +0.00% | +0.00% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 1 | 1 | N_baseline_2layer | 3.856484 | 2302.588 | +15.27% | +8.91% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 1 | 1 | N_no_soft_1layer_10k | 4.117274 | 2461.204 | +23.06% | +16.41% | 已核验 |
| nosoft_1layer/C40/formal | 40 | 1 | 1 | RAW | 3.345651 | 2114.273 | +0.00% | +0.00% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 0 | 1 | N_baseline_2layer | 4.197766 | 248.508 | +3.88% | -4.53% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 0 | 1 | N_no_soft_2layer_10k | 4.393812 | 259.825 | +8.73% | -0.18% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 0 | 1 | RAW | 4.041150 | 260.305 | +0.00% | +0.00% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 1 | 1 | N_baseline_2layer | 3.882209 | 229.052 | +15.68% | +6.06% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 1 | 1 | N_no_soft_2layer_10k | 4.147202 | 244.420 | +23.57% | +13.18% | 已核验 |
| nosoft_2layer/C1/formal | 1 | 1 | 1 | RAW | 3.356027 | 215.964 | +0.00% | +0.00% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 0 | 1 | N_baseline_2layer | 4.204197 | 2558.707 | +3.77% | -1.94% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 0 | 1 | N_no_soft_2layer_10k | 4.400906 | 2680.060 | +8.63% | +2.71% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 0 | 1 | RAW | 4.051360 | 2609.334 | +0.00% | +0.00% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 1 | 1 | N_baseline_2layer | 3.860577 | 2316.995 | +15.40% | +8.85% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 1 | 1 | N_no_soft_2layer_10k | 4.127251 | 2474.196 | +23.37% | +16.24% | 已核验 |
| nosoft_2layer/C40/formal | 40 | 1 | 1 | RAW | 3.345457 | 2128.612 | +0.00% | +0.00% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 0 | 1 | N_baseline_2layer | 4.197766 | 248.032 | +3.88% | -4.37% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 0 | 1 | N_no_soft_3layer_10k | 4.398173 | 257.403 | +8.83% | -0.75% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 0 | 1 | RAW | 4.041150 | 259.360 | +0.00% | +0.00% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 1 | 1 | N_baseline_2layer | 3.882209 | 228.643 | +15.68% | +6.05% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 1 | 1 | N_no_soft_3layer_10k | 4.164540 | 242.803 | +24.09% | +12.62% | 已核验 |
| nosoft_3layer/C1/formal | 1 | 1 | 1 | RAW | 3.356027 | 215.595 | +0.00% | +0.00% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 0 | 1 | N_baseline_2layer | 4.204197 | 2577.924 | +3.73% | -2.15% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 0 | 1 | N_no_soft_3layer_10k | 4.410069 | 2696.996 | +8.81% | +2.37% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 0 | 1 | RAW | 4.053139 | 2634.651 | +0.00% | +0.00% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 1 | 1 | N_baseline_2layer | 3.860577 | 2339.347 | +15.30% | +8.98% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 1 | 1 | N_no_soft_3layer_10k | 4.146161 | 2500.756 | +23.83% | +16.50% | 已核验 |
| nosoft_3layer/C40/formal | 40 | 1 | 1 | RAW | 3.348203 | 2146.592 | +0.00% | +0.00% | 已核验 |

**结果解释：**长训两层T1进一步受益，C40/T0也有正TPS收益；不能沿用短训T0全亏结论。20000步恢复重置Adam，非无损续训。

诊断／原报告摘录（保持原口径）：

- 单轮结果不代表统计显著；跨卡比较使用各自相对RAW的比值。

**原目录／权重入口：**[eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [nosoft_1layer/C1/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_1layer/C1/formal/summary.json)；[nosoft_1layer/C40/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_1layer/C40/formal/summary.json)；[nosoft_2layer/C1/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_2layer/C1/formal/summary.json)；[nosoft_2layer/C40/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_2layer/C40/formal/summary.json)；[nosoft_3layer/C1/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_3layer/C1/formal/summary.json)；[nosoft_3layer/C40/formal/summary.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_3layer/C40/formal/summary.json)；[REPORT.txt](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/experiment.json)；[findings.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/findings.json)；[nosoft_1layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_1layer/preregistration.json)；[nosoft_2layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_2layer/preregistration.json)；[nosoft_3layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu012_recovery_20261002/nosoft_3layer/preregistration.json)。

<a id="t003"></a>

### T003 · LMHEAD／EMBEDDING硬软反馈 · C1

**类别／状态：**研究实验；未提供顶层状态；已记录12个正式窗口。

**做什么：**检查冻结token输入表的选择对修复效果及成本的影响。

**对比什么：**RAW、LMHEAD、EMBEDDING；两个新头共同初始张量、相同anchors/answers。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [training.log](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/training.log)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| native/C1 | 1 | 0 | 1 | EMBEDDING | 4.081050 | 231.848 | +3.20% | -11.02% | 已核验 |
| native/C1 | 1 | 0 | 1 | LMHEAD | 4.073492 | 231.305 | +3.01% | -11.23% | 已核验 |
| native/C1 | 1 | 0 | 1 | RAW | 3.954497 | 260.575 | +0.00% | +0.00% | 已核验 |
| native/C1 | 1 | 0 | 2 | EMBEDDING | 4.081050 | 232.035 | +3.20% | -10.28% | 已核验 |
| native/C1 | 1 | 0 | 2 | LMHEAD | 4.073492 | 231.249 | +3.01% | -10.58% | 已核验 |
| native/C1 | 1 | 0 | 2 | RAW | 3.954497 | 258.612 | +0.00% | +0.00% | 已核验 |
| native/C1 | 1 | 1 | 1 | EMBEDDING | 3.676044 | 207.970 | +14.69% | -1.30% | 已核验 |
| native/C1 | 1 | 1 | 1 | LMHEAD | 3.686960 | 208.465 | +15.03% | -1.07% | 已核验 |
| native/C1 | 1 | 1 | 1 | RAW | 3.205114 | 210.709 | +0.00% | +0.00% | 已核验 |
| native/C1 | 1 | 1 | 2 | EMBEDDING | 3.676044 | 207.798 | +14.69% | -0.54% | 已核验 |
| native/C1 | 1 | 1 | 2 | LMHEAD | 3.686960 | 208.296 | +15.03% | -0.30% | 已核验 |
| native/C1 | 1 | 1 | 2 | RAW | 3.205114 | 208.931 | +0.00% | +0.00% | 已核验 |

**结果解释：**两种输入表效果接近；完整硬软三层版本C1开销较高。后续EMBEDDING输入被选为消融基线。

诊断／原报告摘录（保持原口径）：

- 这两个实验同时加入软反馈、简化dose，因此与旧三输入结果不能作单因素因果归因；两版之间只换输入表。
- 接受长度和吞吐分开判断。测试panel已有开发使用历史；两轮重复不是独立样本，微小差异不等于统计显著。

**原目录／权重入口：**[eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [native/C1/summary.json](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/native/C1/summary.json)；[RESULTS.md](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/RESULTS.md)；[preregistration.json](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/preregistration.json)；[initialization_native/preregistration.json](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/initialization_native/preregistration.json)；[native/preregistration.json](eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/native/preregistration.json)。

<a id="t004"></a>

### T004 · LMHEAD／EMBEDDING硬软反馈 · C40

**类别／状态：**研究实验；complete；已记录12个正式窗口。

**做什么：**验证完整硬软反馈在多并发下的接受收益和开销。

**对比什么：**RAW与两种冻结输入表。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_d1_dual_feedback_C40_gpu4_300prompts_20260930/status.json)。

正式窗口设置：C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C40 | 40 | 0 | 1 | EMBEDDING | 4.188672 | 2450.576 | +3.76% | -4.09% | 已核验 |
| C40 | 40 | 0 | 1 | LMHEAD | 4.181147 | 2445.427 | +3.57% | -4.30% | 已核验 |
| C40 | 40 | 0 | 1 | RAW | 4.036906 | 2555.205 | +0.00% | +0.00% | 已核验 |
| C40 | 40 | 0 | 2 | EMBEDDING | 4.186371 | 2454.639 | +3.90% | -3.41% | 已核验 |
| C40 | 40 | 0 | 2 | LMHEAD | 4.188916 | 2458.313 | +3.96% | -3.26% | 已核验 |
| C40 | 40 | 0 | 2 | RAW | 4.029181 | 2541.220 | +0.00% | +0.00% | 已核验 |
| C40 | 40 | 1 | 1 | EMBEDDING | 3.849618 | 2218.288 | +15.19% | +7.60% | 已核验 |
| C40 | 40 | 1 | 1 | LMHEAD | 3.861470 | 2215.519 | +15.54% | +7.47% | 已核验 |
| C40 | 40 | 1 | 1 | RAW | 3.342026 | 2061.534 | +0.00% | +0.00% | 已核验 |
| C40 | 40 | 1 | 2 | EMBEDDING | 3.855471 | 2215.918 | +15.08% | +6.37% | 已核验 |
| C40 | 40 | 1 | 2 | LMHEAD | 3.851418 | 2222.346 | +14.96% | +6.68% | 已核验 |
| C40 | 40 | 1 | 2 | RAW | 3.350280 | 2083.231 | +0.00% | +0.00% | 已核验 |

**结果解释：**T1有接受及吞吐收益；不同并发的成本不同，不能用C1微测替代C40。

诊断／原报告摘录（保持原口径）：

- 每请求最多4096输出、总长度8192、显存比例0.93，CUDA Graph。40为客户端并发上限，结束阶段并发自然下降；不等于全程40个GPU活跃序列。

**原目录／权重入口：**[eval_results/standalone_d1_dual_feedback_C40_gpu4_300prompts_20260930](eval_results/standalone_d1_dual_feedback_C40_gpu4_300prompts_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C40/summary.json](eval_results/standalone_d1_dual_feedback_C40_gpu4_300prompts_20260930/C40/summary.json)；[RESULTS.md](eval_results/standalone_d1_dual_feedback_C40_gpu4_300prompts_20260930/RESULTS.md)。

<a id="t005"></a>

### T005 · A–G输入／深度／尺度消融

**类别／状态：**研究实验；complete；已记录32个正式窗口。

**做什么：**逐项检查当前hidden、soft lag-1、层数、修正尺度。

**对比什么：**RAW；A完整三层，B无hidden，C无soft，D两者均无，E/F一／两层，G固定尺度。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | A_full_3layer | 4.081050 | 231.725 | +3.20% | -10.04% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | B_no_hidden | 4.055383 | 231.169 | +2.55% | -10.26% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | C_no_soft | 4.069197 | 239.093 | +2.90% | -7.18% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | D_no_hidden_no_soft | 4.040488 | 237.589 | +2.17% | -7.76% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | E_full_1layer | 4.077550 | 235.764 | +3.11% | -8.47% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | F_full_2layer | 4.082112 | 233.678 | +3.23% | -9.28% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | G_full_fixed_scale | 4.085558 | 232.154 | +3.31% | -9.87% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 257.589 | +0.00% | +0.00% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | A_full_3layer | 3.676044 | 207.975 | +14.69% | +0.03% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | B_no_hidden | 3.666034 | 208.126 | +14.38% | +0.11% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | C_no_soft | 3.696223 | 215.872 | +15.32% | +3.83% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | D_no_hidden_no_soft | 3.614096 | 211.406 | +12.76% | +1.68% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | E_full_1layer | 3.668826 | 211.193 | +14.47% | +1.58% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | F_full_2layer | 3.739665 | 212.937 | +16.68% | +2.42% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | G_full_fixed_scale | 3.646432 | 205.947 | +13.77% | -0.94% | 已核验 |
| unified_RAW_A_G_C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 207.903 | +0.00% | +0.00% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | A_full_3layer | 4.198348 | 2452.662 | +4.06% | -3.55% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | B_no_hidden | 4.171911 | 2428.541 | +3.41% | -4.50% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | C_no_soft | 4.191546 | 2486.957 | +3.89% | -2.20% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | D_no_hidden_no_soft | 4.160547 | 2483.624 | +3.13% | -2.33% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | E_full_1layer | 4.205654 | 2478.974 | +4.24% | -2.52% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | F_full_2layer | 4.184226 | 2459.447 | +3.71% | -3.28% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | G_full_fixed_scale | 4.173159 | 2444.008 | +3.44% | -3.89% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 0 | 1 | RAW | 4.034450 | 2542.970 | +0.00% | +0.00% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | A_full_3layer | 3.849618 | 2207.197 | +15.19% | +6.97% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | B_no_hidden | 3.817051 | 2180.462 | +14.21% | +5.67% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | C_no_soft | 3.861112 | 2260.591 | +15.53% | +9.55% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | D_no_hidden_no_soft | 3.821438 | 2248.311 | +14.34% | +8.96% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | E_full_1layer | 3.854638 | 2232.987 | +15.34% | +8.22% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | F_full_2layer | 3.853593 | 2225.714 | +15.31% | +7.86% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | G_full_fixed_scale | 3.858166 | 2217.839 | +15.44% | +7.48% | 已核验 |
| unified_RAW_A_G_C40_gpu4_300prompts/formal | 40 | 1 | 1 | RAW | 3.342026 | 2063.436 | +0.00% | +0.00% | 已核验 |

**结果解释：**形成保留hidden、原始q、实际硬lag-1，去掉soft反馈的研究方向；各项单独结果见统一正式表，不能只按训练loss选结构。

诊断／原报告摘录（保持原口径）：

- 固定正向顺序、单轮；没有逆序或重复样本，不能据此声称统计显著性。

**原目录／权重入口：**[eval_results/standalone_embedding_ablation_C1_gpu45_20260930](eval_results/standalone_embedding_ablation_C1_gpu45_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [unified_RAW_A_G_C1/formal/summary.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/unified_RAW_A_G_C1/formal/summary.json)；[unified_RAW_A_G_C40_gpu4_300prompts/formal/summary.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/unified_RAW_A_G_C40_gpu4_300prompts/formal/summary.json)；[RESULTS.md](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/RESULTS.md)；[A_full_3layer/RESULTS.md](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/A_full_3layer/RESULTS.md)；[pre_uniform_results/RESULTS.md](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/pre_uniform_results/RESULTS.md)；[unified_RAW_A_G_C1/RESULTS.md](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/unified_RAW_A_G_C1/RESULTS.md)；[unified_RAW_A_G_C40_gpu4_300prompts/RESULTS.md](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/unified_RAW_A_G_C40_gpu4_300prompts/RESULTS.md)；[A_full_3layer/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/A_full_3layer/preregistration.json)；[B_no_hidden/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/B_no_hidden/preregistration.json)；[C_no_soft/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/C_no_soft/preregistration.json)；[D_no_hidden_no_soft/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/D_no_hidden_no_soft/preregistration.json)；[E_full_1layer/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/E_full_1layer/preregistration.json)；[F_full_2layer/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/F_full_2layer/preregistration.json)；[G_full_fixed_scale/preregistration.json](eval_results/standalone_embedding_ablation_C1_gpu45_20260930/G_full_fixed_scale/preregistration.json)。

<a id="t006"></a>

### T006 · No-soft 1／2／3层短训

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**检查修复头加深是否换来更多实际收益。

**对比什么：**同预算重新训练的1/2/3层头与RAW。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | N_no_soft_1layer | 4.072184 | 244.216 | +2.98% | -4.86% | 已核验 |
| C1/formal | 1 | 0 | 1 | N_no_soft_2layer | 4.081234 | 242.459 | +3.20% | -5.55% | 已核验 |
| C1/formal | 1 | 0 | 1 | N_no_soft_3layer | 4.069197 | 239.460 | +2.90% | -6.72% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 256.702 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | N_no_soft_1layer | 3.684464 | 219.952 | +14.96% | +5.86% | 已核验 |
| C1/formal | 1 | 1 | 1 | N_no_soft_2layer | 3.685537 | 217.735 | +14.99% | +4.79% | 已核验 |
| C1/formal | 1 | 1 | 1 | N_no_soft_3layer | 3.696223 | 216.053 | +15.32% | +3.98% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 207.784 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_no_soft_1layer | 4.178889 | 2496.816 | +3.25% | -2.89% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_no_soft_2layer | 4.191603 | 2489.604 | +3.56% | -3.17% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_no_soft_3layer | 4.182531 | 2475.616 | +3.34% | -3.71% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.047409 | 2571.021 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_no_soft_1layer | 3.843533 | 2264.043 | +15.03% | +9.12% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_no_soft_2layer | 3.854312 | 2262.070 | +15.35% | +9.03% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_no_soft_3layer | 3.850293 | 2252.277 | +15.23% | +8.56% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.341307 | 2074.769 | +0.00% | +0.00% | 已核验 |

**结果解释：**短训三种深度接受长度接近；浅层串行成本较小，两层作为重点基线，不意味着全局最优。

诊断／原报告摘录（保持原口径）：

- 每个并发分别同卡、同prompt、同seed对照；仅单轮，不能据此声称统计显著性。

**原目录／权重入口：**[eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/C40/formal/summary.json)；[RESULTS.md](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/RESULTS.md)；[experiment.json](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/experiment.json)；[N_no_soft_1layer/preregistration.json](eval_results/standalone_embedding_nosoft_depth123_C1_C40_gpu4_20260930/N_no_soft_1layer/preregistration.json)。

<a id="t007"></a>

### T007 · Direction／dose 分开与合并

**类别／状态：**研究实验；complete；已记录12个正式窗口。

**做什么：**检查把方向与幅度合并预测是否更有效。

**对比什么：**旧分开N、合并M、RAW。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | M_no_soft_2layer_merged | 4.020718 | 238.529 | +1.67% | -6.77% | 已核验 |
| C1/formal | 1 | 0 | 1 | N_no_soft_2layer | 4.081234 | 241.378 | +3.20% | -5.66% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.850 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | M_no_soft_2layer_merged | 3.705062 | 218.824 | +15.60% | +5.57% | 已核验 |
| C1/formal | 1 | 1 | 1 | N_no_soft_2layer | 3.685537 | 216.888 | +14.99% | +4.63% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 207.285 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | M_no_soft_2layer_merged | 4.134469 | 2514.963 | +2.15% | -3.64% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_no_soft_2layer | 4.192562 | 2550.304 | +3.59% | -2.29% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.047409 | 2610.073 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | M_no_soft_2layer_merged | 3.816552 | 2276.530 | +14.72% | +8.12% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_no_soft_2layer | 3.862829 | 2313.254 | +16.11% | +9.86% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.326982 | 2105.548 | +0.00% | +0.00% | 已核验 |

**结果解释：**合并没有一致优势，C40退化，未替换旧N。

诊断／原报告摘录（保持原口径）：

- 每个并发分别同卡、同prompt、同seed对照；仅单轮，不能据此声称统计显著性。

**原目录／权重入口：**[eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/C40/formal/summary.json)；[RESULTS.md](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/RESULTS.md)；[experiment.json](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/experiment.json)；[M_no_soft_2layer_merged/preregistration.json](eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/M_no_soft_2layer_merged/preregistration.json)。

<a id="t008"></a>

### T008 · 冻结DFlash直接揭示前驱／前缀

**类别／状态：**研究实验；complete。

**做什么：**检查原冻结DFlash能否直接利用块内真实历史。

**对比什么：**全mask、只揭示前驱、揭示合法前缀及原修复头。

**实验设置：**8提示×32 anchors×7位置=1792冻结状态；teacher-forced／兼容前缀代理；未重训backbone、无新native测速。

**即时日志／进度：** [status.json](eval_results/standalone_dependency_reveal_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**直接填入历史退化，说明原训练mask分布外敏感；不能证明历史信息无用。

诊断／原报告摘录（保持原口径）：

- direct_reveal: OOD sensitivity: D1 and Dprefix degrade across all eight prompts; do not promote this frozen-backbone fill-in procedure as iterative decoding
- dependency_fraction: not identified; partial-reveal OOD prevents assigning a percentage of mismatch to diffusion/dependency
- current_repair: aggregate top1 60.60% to69.25%; first-position227/256 to225/256, fixes2 harms4
- next_decision: Do not expand this OOD reveal pilot to the sealed set or treat it as a target update to distill. Next structural study should learn actual-predecessor conditioning with aligned labels; any backbone remasking study needs separately validated training-mask support.
- Only8 prompt groups; anchor counts are correlated.
- Local source shows all-masked block training; complete original checkpoint training history unverified.
- Partial reveal modifies input distribution; harm does not disprove missing dependency.
- Historical exclusion is limited to recoverable records; confirmation is unobserved by this study, not proven unseen in every prior project.

**原目录／权重入口：**[eval_results/standalone_dependency_reveal_gpu5_20261001](eval_results/standalone_dependency_reveal_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_dependency_reveal_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_dependency_reveal_gpu5_20261001/preregistration.json)。

<a id="t009"></a>

### T009 · 前驱条件响应干预

**类别／状态：**研究实验；complete。

**做什么：**区分修复头是否已学到前驱响应、响应方向是否有益。

**对比什么：**同截断前缀的真实／替换前驱、原始q与修复q。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_predecessor_response_gpu5_20261001_attempt2/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已学到部分条件响应；仍有方向、幅度、第三候选竞争及误修。首次失败不是有效结果。

诊断／原报告摘录（保持原口径）：

- 只有8个研究prompt和每位置4个前驱，不能证明总体机制或选择低秩维度。
- 均匀平均的公共修正/交互能量依赖所选前驱和度量；不是概率混合分布的分解，也不是全体mismatch的因果占比。
- q加权仅用于分析度量；修复输入及采样分布没有改动。
- 存在BF16平局：替代前驱组有1个修复成功案例与原赢家分数相等，由argmax平局规则胜出；strict crossing只表示严格越过，最终correct以实际argmax为准。
- 这是冻结模型条件干预，非训练、非native接受长度或吞吐测试。

**原目录／权重入口：**[eval_results/standalone_predecessor_response_gpu5_20261001_attempt2](eval_results/standalone_predecessor_response_gpu5_20261001_attempt2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_predecessor_response_gpu5_20261001_attempt2/findings.json)；[preregistration.json](eval_results/standalone_predecessor_response_gpu5_20261001_attempt2/preregistration.json)。

<a id="t010"></a>

### T010 · 线性包络／dose-only 可达性

**类别／状态：**研究实验；complete。

**做什么：**判断错误能否在当前基底和冻结输出映射内修好。

**对比什么：**全基底包络、学到方向包络、固定方向只调dose。

**实验设置：**8研究提示、128状态；最终错误41，主分析35；有限幅度、BF16实际argmax见证；逐状态oracle，无native AL/TPS。

**即时日志／进度：** [status.json](eval_results/standalone_attainability_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**所测41个错误加0.5 margin后均找到BF16正确包络见证；固定方向仅11/41。只证明好解存在，不证明输入足够或共享网络能预测。

诊断／原报告摘录（保持原口径）：

- 在这批状态中，完整冻结基底与最终direction头的线性输出包络都存在可修正全部41个错误的向量；带0.5 logit目标间隔的解全部通过BF16排序检查。未发现这些错误被基底范围或0.8‖hidden‖上限阻挡的证据。
- 学到的输出包络中，所需系数范数/上限的中位数约4.75%，最大约30.09%。这不是已学模型的性能，而是使用目标标签构造的可达性见证。
- 固定当前direction穷举dose仅能处理11/41：6个是撤销原本正确位置的误修，5个是修正原本就错的预测。剩余30个在当前BF16正向dose范围内无解。因此单独调整dose不能解决大部分剩余错误。
- 结果把问题收窄到方向选择/更强的最终头可达性约束/训练映射与目标，但尚不能在这些原因之间归因。特别是线性包络仍放宽了LayerNorm可达流形与direction-dose共享latent关系。
- 零间隔最小范数解常落在排序平局边界；全基底只有15/41、学到包络17/41零间隔解在BF16下直接正确。0.5间隔均41/41，说明不能用连续可行但无数值余量的解宣称runtime已修复。
- 8个研究prompt的有标签oracle，不是新修复网络，也不是native接受长度或吞吐。
- 这轮仅验证同一候选集、同一范数上限的输出空间可达性，不能证明输入到latent的函数可实现该oracle方向。
- learned_span是实际最终LN/direction仿射映射的线性包络，不等于完整最终头可达集。新方向经BF16量化后可略离开理想实数包络。

**原目录／权重入口：**[eval_results/standalone_attainability_gpu5_20261001](eval_results/standalone_attainability_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_attainability_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_attainability_gpu5_20261001/preregistration.json)。

<a id="t011"></a>

### T011 · 完整最终头 latent oracle

**类别／状态：**研究实验；complete。

**做什么：**去掉线性包络限制，检查完整LN/direction/dose头的表达上限。

**对比什么：**原latent与逐状态优化pre-LN latent。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_joint_head_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**41/41错误找到BF16正确见证，主分析35/35；这是所测状态的存在性结论。

诊断／原报告摘录（保持原口径）：

- 41个当前错误全部在完整冻结LayerNorm、direction与dose共享latent约束下找到BF16正确见证，主分析35/35。不能再仅用线性包络放宽来解释上一轮结果。
- 41个错误的OUTPUT下降方向在logit层和连续最终头latent层都提高当前最难竞争者排序间隔；未观察到局部反向梯度证据。主分析latent余弦中位数约0.958。
- 方向正确不代表信号足够强。41个错误中10个latent梯度范数低于1e-6，7个低于1e-8，3个低于1e-12；这些阈值仅描述本次loss归一化下量级，不是通用判据。
- 位置6的15个错误，其latent梯度范数中位数约3.22e-7；位置0/1/3约2.23e-3、2.67e-3、2.49e-3。不能仅凭跨状态差异把衰减归因于prefix乘积，仍需控制条件分解。
- 下一研究重点收窄为共享输入映射的学习信号、跨状态梯度干扰以及训练与测试前缀差异；当前结果不支持直接扩大基底、提高dose上限或追加头。
- label-informed per-state pre-LN latent oracle, not learned input mapping or native performance
- failure is unresolved; no nonlinear impossibility certificate
- OUTPUT gradient uses teacher prefixes; continuous Jacobian is a local surrogate ignoring BF16 quantization

**原目录／权重入口：**[eval_results/standalone_joint_head_gpu5_20261001](eval_results/standalone_joint_head_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_joint_head_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_joint_head_gpu5_20261001/preregistration.json)。

<a id="t012"></a>

### T012 · 条件方向配对分析

**类别／状态：**研究实验；complete。

**做什么：**检查前驱变化时目标修正动作是否发生变化。

**对比什么：**同状态的条件分支与方向/剂量。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_direction_pairs_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**诊断前驱条件响应和修正动作冲突；不提供新的native接受长度或吞吐成绩。

诊断／原报告摘录（保持原口径）：

- pilot的41个错误中，direction参数组平均更新对24个排序有害；独立开发复核36个错误中为12个。两组自身状态的direction更新均无负贡献。此处只分析direction参数组，不能与上一轮全参数21/41混用。
- 成对分解对全部77个错误的联合方向符号与直接BF16梯度完全一致；最大按梯度范数归一化重建误差约0.000833。
- 未发现普遍近同latent或表示整体坍缩：两组不同状态latent余弦中位数0.375、0.390，95分位0.621、0.666；有害配对中没有余弦>=0.95的配对。
- 有害/有益配对的latent相似度中位数分别为pilot 0.377/0.383、复核0.415/0.413；不能把干扰简单归因于特别相似的latent。
- 约一半跨状态配对的direction贡献为负，两组都出现；这是共享参数下的局部相互作用，未证明它属于异常干扰、结构缺陷或信息不足。
- 跨prompt占约87.9%的配对，也贡献约88.0%/83.4%的负贡献；按数量校正后，不能仅凭总量说跨prompt是特殊问题。同前驱和同目标配对样本占比很小，归因应保持谨慎。
- 等权overlap下direction有害状态从pilot24降到21，但复核12升到14；不支持直接删除前缀目标。
- 弱信号现象复现：pre-normalized direction输出t处，pilot7个、复核4个小于1e-8的错误，在同状态改为等权权重后均超过阈值；该t梯度与前轮pre-LN latent梯度层级不同，不能混用数值。

**原目录／权重入口：**[eval_results/standalone_direction_pairs_gpu5_20261001](eval_results/standalone_direction_pairs_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_direction_pairs_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_direction_pairs_gpu5_20261001/preregistration.json)。

<a id="t013"></a>

### T013 · 共享梯度传输诊断

**类别／状态：**研究实验；complete。

**做什么：**检查单状态改善是否被共享更新抵消。

**对比什么：**单状态、同提示和全研究batch平均梯度。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_gradient_transport_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共享参数梯度存在方向干扰；局部导数不是实际Adam训练成绩，也不是所有latent普遍坍缩。

诊断／原报告摘录（保持原口径）：

- direction头参数组对24/41个错误的全局平均排序导数贡献为负，dose头为5/41；贡献按当前参数化定义，不能简单比较参数组或断言必须换头。
- 在相同41个错误状态上，隔离自身加权损失的参数梯度全部帮助greedy排序；合并同一prompt的28个状态后13个反向；8个研究prompt平均后21个反向。主分析35个错误对应0、13、17个反向。
- 第7个预测位置j=6的15个错误，局部latent梯度范数中位数约0.0507，包含OUTPUT权重后约3.22e-7。两者中位数不能相除当作典型衰减倍数；逐状态分解已逐项验证。
- j=6的前驱重叠乘积中位数约0.00261，意味着该位置相对等权token损失的权重中位数只有约0.261%。这是前缀目标的数学性质，不能称为反向漏接或实现错误。
- 7个梯度范数小于1e-8的错误，在同状态以1/28替代前缀权重后全部超过1e-8；这支持极弱信号来自前缀加权，而非这些状态的局部头普遍没有梯度。
- 等权overlap控制仍损害18/41的联合排序方向：相对OUTPUT改善5个，恶化2个。前缀衰减和跨状态共享参数干扰是不同问题；不能据此直接推荐替换损失。
- 真实BF16参数梯度与连续最终头诊断相容：错误状态latent梯度余弦最小约0.999949。128个状态的标签和正确性与上一轮一致。
- 一个正确对照的极小latent梯度，跨轮直接与因子分解计算差1.1%；已进行单prompt数值复核，开启求导前后logits/latent逐位相同，FP64分解恒等式通过。所有跨轮直接梯度差异均单独记录，不能把数值量级差异当作状态机制。主要参数方向计数使用直接BF16反向，未使用这个正确对照的因子近似。

**原目录／权重入口：**[eval_results/standalone_gradient_transport_gpu5_20261001](eval_results/standalone_gradient_transport_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_gradient_transport_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_gradient_transport_gpu5_20261001/preregistration.json)。

<a id="t014"></a>

### T014 · 梯度传输独立复核

**类别／状态：**研究实验；complete。

**做什么：**复核direction组共享梯度负贡献。

**对比什么：**独立梯度检查与原pilot。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_gradient_validation_v2_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**部分错误受到联合direction梯度负贡献；不同样本/参数组计数不能与全参数结果混用。

**原目录／权重入口：**[eval_results/standalone_gradient_validation_v2_gpu5_20261001](eval_results/standalone_gradient_validation_v2_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_gradient_validation_v2_gpu5_20261001/preregistration.json)。

<a id="t015"></a>

### T015 · Epoch学习／latent机制诊断

**类别／状态：**研究实验；complete。

**做什么：**检查训练阶段、输出权重及局部方向变化。

**对比什么：**epoch1/2/3冻结状态与梯度。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_epoch_learning_gpu5_20261001/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**OUTPUT前缀权重衰减及共享干扰可见，未发现普遍latent坍缩。冻结状态统计不等于native收益。

诊断／原报告摘录（保持原口径）：

- 现有头确实学到了修正：RAW到E1，训练子集top1从233/448到270/448，开发子集从263/448到303/448；OUTPUT与局部overlap同时明显改善。不能描述为结构完全学不动。
- E1到E3，训练子集OUTPUT0.53429降至0.52200，top1从270到279；开发子集OUTPUT0.53087降至0.52332、overlap0.65688升至0.67030，但全部teacher状态top1从303降至298。
- 开发top1变化约-1.12个百分点，按16prompt配对bootstrap95%区间约[-3.125,+0.670]个百分点，包含0；该小样本不能证明总体退化或过拟合。
- 开发集greedy-prefix-compatible的355状态，RAW238、E1261、E2266、E3262正确；E1到E3没有上述下降，说明全部teacher前缀统计不能直接推断native T0。
- E1到E3，开发集新增修好13个、新增弄错18个；18个新错中只有1个的局部overlap反而提高，其他17个局部overlap也下降。总体平均改善与部分状态受损是不同层次，不能说所有误修都被OUTPUT直接奖励。
- 按位置看，开发j3的top1从49/64到44/64，j6从38/64到35/64，但j4从40/64到43/64；改善不是每个位置都单调。
- 历史训练日志128step均值约0.53891、0.50800、0.49482，与继续优化的方向一致，但这些值来自变化的pre-update权重和采样锚点，不能与固定checkpoint审计数值混同。
- 48组历史训练锚点/答案token哈希全部重现，选中64个训练锚点每个epoch均见过；本轮三个模型严格共用当前online状态。跨轮RAW有1/448开发状态排序差异已单列，目标标签和E3正确性全部一致。

**原目录／权重入口：**[eval_results/standalone_epoch_learning_gpu5_20261001](eval_results/standalone_epoch_learning_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_epoch_learning_gpu5_20261001/findings.json)；[preregistration.json](eval_results/standalone_epoch_learning_gpu5_20261001/preregistration.json)。

<a id="t016"></a>

### T016 · Epoch1／2／3 native复测

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**验证简单早停能否解决T0吞吐。

**对比什么：**RAW与三个epoch头。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_epoch_native_C1_gpu5_20261001/status.json)。

协议字段：`seed=42 + prompt_index, identical across arms`。

正式窗口设置：C1=20条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| formal | 1 | 0 | 1 | E1 | 4.046137 | 239.721 | +2.32% | -6.37% | 已核验 |
| formal | 1 | 0 | 1 | E2 | 4.082986 | 241.781 | +3.25% | -5.57% | 已核验 |
| formal | 1 | 0 | 1 | E3 | 4.081234 | 241.805 | +3.20% | -5.56% | 已核验 |
| formal | 1 | 0 | 1 | RAW | 3.954497 | 256.030 | +0.00% | +0.00% | 已核验 |
| formal | 1 | 1 | 1 | E1 | 3.589787 | 211.867 | +12.00% | +2.16% | 已核验 |
| formal | 1 | 1 | 1 | E2 | 3.671652 | 216.284 | +14.56% | +4.29% | 已核验 |
| formal | 1 | 1 | 1 | E3 | 3.685537 | 217.179 | +14.99% | +4.72% | 已核验 |
| formal | 1 | 1 | 1 | RAW | 3.205114 | 207.385 | +0.00% | +0.00% | 已核验 |

**结果解释：**不支持简单早停即可解决T0；epoch3 T1仍有收益。

**原目录／权重入口：**[eval_results/standalone_epoch_native_C1_gpu5_20261001](eval_results/standalone_epoch_native_C1_gpu5_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [formal/summary.json](eval_results/standalone_epoch_native_C1_gpu5_20261001/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_epoch_native_C1_gpu5_20261001/experiment.json)。

<a id="t017"></a>

### T017 · 显式交互头

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**检查输入交互结构能否改善方向学习。

**对比什么：**旧N、ADDITIVE、INTERACTION、RAW。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_explicit_interaction_C1_gpu45_20261001/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | ADDITIVE | 4.082986 | 239.112 | +3.25% | -6.55% | 已核验 |
| C1/formal | 1 | 0 | 1 | E3 | 4.081234 | 241.631 | +3.20% | -5.57% | 已核验 |
| C1/formal | 1 | 0 | 1 | INTERACTION | 4.080983 | 239.215 | +3.20% | -6.51% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.875 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | ADDITIVE | 3.698139 | 215.531 | +15.38% | +4.18% | 已核验 |
| C1/formal | 1 | 1 | 1 | E3 | 3.685537 | 217.207 | +14.99% | +4.99% | 已核验 |
| C1/formal | 1 | 1 | 1 | INTERACTION | 3.622537 | 211.494 | +13.02% | +2.23% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 206.878 | +0.00% | +0.00% | 已核验 |

**结果解释：**两种新结构均未超过旧N；结构直觉不能替代native结果。

**原目录／权重入口：**[eval_results/standalone_explicit_interaction_C1_gpu45_20261001](eval_results/standalone_explicit_interaction_C1_gpu45_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_explicit_interaction_C1_gpu45_20261001/C1/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_explicit_interaction_C1_gpu45_20261001/experiment.json)；[ADDITIVE/preregistration.json](eval_results/standalone_explicit_interaction_C1_gpu45_20261001/ADDITIVE/preregistration.json)；[INTERACTION/preregistration.json](eval_results/standalone_explicit_interaction_C1_gpu45_20261001/INTERACTION/preregistration.json)。

<a id="t018"></a>

### T018 · PREFIX／TOKEN_TV／TOKEN_NLA损失

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**检查改变位置梯度分配是否改善接受收益。

**对比什么：**原PREFIX-OUTPUT与TOKEN_TV/TOKEN_NLA。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/status.json)。

协议字段：`train_count=128; epochs=3; training_epochs=3; objective=PREFIX,TOKEN_NLA,TOKEN_TV; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | PREFIX | 4.081234 | 241.305 | +3.20% | -5.59% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.602 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | TOKEN_NLA | 4.078549 | 241.207 | +3.14% | -5.63% | 已核验 |
| C1/formal | 1 | 0 | 1 | TOKEN_TV | 4.059641 | 240.336 | +2.66% | -5.97% | 已核验 |
| C1/formal | 1 | 1 | 1 | PREFIX | 3.685537 | 216.859 | +14.99% | +4.82% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 206.877 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | TOKEN_NLA | 3.580891 | 210.954 | +11.72% | +1.97% | 已核验 |
| C1/formal | 1 | 1 | 1 | TOKEN_TV | 3.638874 | 214.378 | +13.53% | +3.63% | 已核验 |

**结果解释：**新目标没有超过旧N；削弱前缀权重不能自动改善greedy/native。

**原目录／权重入口：**[eval_results/standalone_gradient_objectives_C1_gpu45_20261001](eval_results/standalone_gradient_objectives_C1_gpu45_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/C1/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/experiment.json)；[PREFIX/preregistration.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/PREFIX/preregistration.json)；[TOKEN_NLA/preregistration.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/TOKEN_NLA/preregistration.json)；[TOKEN_TV/preregistration.json](eval_results/standalone_gradient_objectives_C1_gpu45_20261001/TOKEN_TV/preregistration.json)。

<a id="t019"></a>

### T019 · Native轨迹最终头oracle

**类别／状态：**研究实验；complete。

**做什么：**在真实采样前驱上检查oracle上限及学习标签。

**对比什么：**T0/T1原修复和target可见的最终头oracle。

**实验设置：**每个温度8提示×4块×7位置=224状态；冻结behavior轨迹；T0/T1到达状态分别141/139。

**即时日志／进度：** [status.json](eval_results/standalone_native_oracle_pair_anygpu_20261001/status.json)。

协议字段：`seed=42 + prompt index`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**原生状态中存在可改善动作；oracle用到了target监督，不能直接作为部署输入或native测速结果。

诊断／原报告摘录（保持原口径）：

- Frozen original N_no_soft_2layer epoch3; 8 historical research prompts per temperature; paired native-state diagnostics only

**原目录／权重入口：**[eval_results/standalone_native_oracle_pair_anygpu_20261001](eval_results/standalone_native_oracle_pair_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [findings.json](eval_results/standalone_native_oracle_pair_anygpu_20261001/findings.json)；[preregistration.json](eval_results/standalone_native_oracle_pair_anygpu_20261001/preregistration.json)。

<a id="t020"></a>

### T020 · OUTPUT／TARGET_KL／ORACLE_KL学习

**类别／状态：**研究实验；complete。

**做什么：**验证oracle标签是否比直接target监督更易学。

**对比什么：**同初始头、同冻结状态三种目标，各384更新。

**实验设置：**稀疏固定旧N native轨迹；teacher仅用于监督，learner不反馈前缀；分卡版与原32步版本重放校验通过。

**即时日志／进度：** [status.json](eval_results/standalone_oracle_learning_split_anygpu_20261001/status.json)；[training.log](eval_results/standalone_oracle_learning_split_anygpu_20261001/training.log)。

协议字段：`train_count=128; epochs=3; max_anchors=1536; seed=300931`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**ORACLE_KL相对TARGET_KL仅小幅开发overlap收益，未填平oracle差距；没有learner native AL/TPS，不能据此扩大蒸馏。

**原目录／权重入口：**[eval_results/standalone_oracle_learning_split_anygpu_20261001](eval_results/standalone_oracle_learning_split_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_oracle_learning_split_anygpu_20261001/preregistration.json)。

<a id="t021"></a>

### T021 · 28状态极小oracle拟合

**类别／状态：**研究实验；complete。

**做什么：**区分训练拟合能力与泛化限制。

**对比什么：**同28状态训练与oracle标签。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_tiny_oracle_fit_anygpu_20261001/status.json)；[training.log](eval_results/standalone_tiny_oracle_fit_anygpu_20261001/training.log)。

协议字段：`block_size=7`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**极小集合可以拟合；不能证明跨提示的可预测性。

诊断／原报告摘录（保持原口径）：

- Tiny oracle fit: completed
- Architecture/input/precision/optimizer/oracle solver unchanged; fresh learner initialization.
- 四个块的完整输入+oracle 标签签名与上轮逐位一致；seed300931 的全部初始化张量也与上轮一致。
- 上轮 epoch3 对这同一批28个状态：KL=0.39271591，目标分布重叠=0.77809869，目标top1正确=25/28；每状态训练3次。
- 本轮共享网络两个seed中，每个状态分别训练35次和40次；10个旧N错误全部改对，18个旧N正确位置均保持正确。
- 不能把结果全部归因于重复次数：本轮每次只更新一个块，prompt数量及学习率所处阶段也不同。
- 结论：这批固定状态能学会，不支持立刻更换输入或增加网络模块。下一步才应区分大样本拟合与新prompt推广；尚未启动后续实验。

**原目录／权重入口：**[eval_results/standalone_tiny_oracle_fit_anygpu_20261001](eval_results/standalone_tiny_oracle_fit_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_tiny_oracle_fit_anygpu_20261001/REPORT.txt)；[findings.json](eval_results/standalone_tiny_oracle_fit_anygpu_20261001/findings.json)；[preregistration.json](eval_results/standalone_tiny_oracle_fit_anygpu_20261001/preregistration.json)。

<a id="t022"></a>

### T022 · 64提示oracle覆盖

**类别／状态：**研究实验；complete。

**做什么：**扩大固定研究状态覆盖范围。

**对比什么：**target/oracle监督的覆盖与原修复。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001/status.json)；[training.log](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001/training.log)。

协议字段：`epochs=40; block_size=7; seed=300931`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**检查覆盖上限，不产生独立最终测试或新的native吞吐成绩。

诊断／原报告摘录（保持原口径）：

- 总更新2560；耗时748.2秒。
- 72个状态块的完整输入+oracle标签签名与原oracle学习实验一致；其中24个与4/16实验一致。冻结行为、原生BF16回放及观察器轨迹校验通过。
- 正式拟合标准不变：最后两次mean KL<=0.01、max KL<=0.05、mean TV<=0.02；oracle margin>=0.5时learner margin>=0.25。
- 训练通过=False；mean KL=0.00621737，max KL=0.13442278，mean TV=0.01010114；greedy错误=3/448。
- 16 prompt：oracle KL=1.44645360，greedy错误=19/56，目标分布重叠=0.65213654。
- 64 prompt：oracle KL=1.88148008，greedy错误=20/56，目标分布重叠=0.63903100。
- 旧N：oracle KL=1.05401062，greedy错误=13/56，目标分布重叠=0.75448008。
- oracle：目标分布重叠=0.96957958，greedy错误=1/56。

**原目录／权重入口：**[eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001/REPORT.txt)；[findings.json](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001/findings.json)；[preregistration.json](eval_results/standalone_tiny_oracle_coverage64_anygpu_20261001/preregistration.json)。

<a id="t023"></a>

### T023 · 16／64提示oracle学习规模

**类别／状态：**研究实验；complete。

**做什么：**检查训练拟合优秀是否能迁移到提示留出。

**对比什么：**16/64提示训练与研究留出。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_tiny_oracle_scale_anygpu_20261001/status.json)；[training.log](eval_results/standalone_tiny_oracle_scale_anygpu_20261001/training.log)。

协议字段：`epochs=40; block_size=7; seed=300931`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**训练拟合很好而研究留出未改善；留出有历史开发使用，不是封存最终测试；唯一原因未识别。

诊断／原报告摘录（保持原口径）：

- 总更新：800/800；执行耗时：310.0 秒。
- T0、每prompt第2块7个状态；固定旧N生成；oracle三起点、每次最多120次LBFGS。
- 没有提前停止或按留出结果选择checkpoint。原始逐状态指标已独立重算。
- 24块的完整输入+oracle标签签名与上一轮一致；CPU和GPU缓存最终签名不变；观察器不改变生成轨迹。
- 训练集通过标准：最后两次检查mean KL<=0.01、max KL<=0.05、mean TV<=0.02；oracle margin>=0.5的位置learner margin>=0.25。
- 留出强通过标准：mean KL严格优于旧N，greedy错误数严格减少（旧N零错误时要求仍为零）。单指标改善只计部分改善。
- 训练：通过=True；mean KL=0.00117538；max KL=0.00663655；mean TV=0.00388594；greedy错误=0/28。
- 留出修好旧N错误=1；损坏旧N正确=7；KL改善prompt=3/8。

**原目录／权重入口：**[eval_results/standalone_tiny_oracle_scale_anygpu_20261001](eval_results/standalone_tiny_oracle_scale_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_tiny_oracle_scale_anygpu_20261001/REPORT.txt)；[findings.json](eval_results/standalone_tiny_oracle_scale_anygpu_20261001/findings.json)；[preregistration.json](eval_results/standalone_tiny_oracle_scale_anygpu_20261001/preregistration.json)。

<a id="t024"></a>

### T024 · Top256／128／64

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**减小输入／修正候选支持以降低成本。

**对比什么：**RAW与重新训练的Top256/128/64。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.922 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | Top128 | 4.080172 | 242.501 | +3.18% | -5.24% | 已核验 |
| C1/formal | 1 | 0 | 1 | Top256 | 4.081234 | 241.882 | +3.20% | -5.49% | 已核验 |
| C1/formal | 1 | 0 | 1 | Top64 | 4.075425 | 242.299 | +3.06% | -5.32% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 207.632 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | Top128 | 3.694207 | 218.503 | +15.26% | +5.24% | 已核验 |
| C1/formal | 1 | 1 | 1 | Top256 | 3.685537 | 217.307 | +14.99% | +4.66% | 已核验 |
| C1/formal | 1 | 1 | 1 | Top64 | 3.677356 | 217.532 | +14.73% | +4.77% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.038881 | 2582.816 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | Top128 | 4.182199 | 2536.007 | +3.55% | -1.81% | 已核验 |
| C40/formal | 40 | 0 | 1 | Top256 | 4.190472 | 2540.422 | +3.75% | -1.64% | 已核验 |
| C40/formal | 40 | 0 | 1 | Top64 | 4.196252 | 2549.494 | +3.90% | -1.29% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.339801 | 2112.239 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | Top128 | 3.843054 | 2291.495 | +15.07% | +8.49% | 已核验 |
| C40/formal | 40 | 1 | 1 | Top256 | 3.864553 | 2312.040 | +15.71% | +9.46% | 已核验 |
| C40/formal | 40 | 1 | 1 | Top64 | 3.854638 | 2300.615 | +15.42% | +8.92% | 已核验 |

**结果解释：**候选范围缩小基本保住AL，但没有稳定额外TPS收益。TopK不是输出基底rank。

诊断／原报告摘录（保持原口径）：

- 另有长训练在GPU1/2/3继续运行；测试面板曾经使用，单轮不构成显著性结论。
- Top128 C1 T0：接受长度下降但吞吐提高。
- Top64 C1 T0：接受长度下降但吞吐提高。
- Top64 C1 T1：接受长度下降但吞吐提高。

**原目录／权重入口：**[eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[REPORT.txt](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/experiment.json)；[findings.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/findings.json)；[Top128/preregistration.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/Top128/preregistration.json)；[Top64/preregistration.json](eval_results/standalone_nosoft_top128_top64_short_gpu45_20261002/Top64/preregistration.json)。

<a id="t025"></a>

### T025 · 输出rank256／128／64

**类别／状态：**研究实验；complete；已记录20个正式窗口。

**做什么：**检查低维输出动作空间能否额外加速。

**对比什么：**新rank头与旧512参考；固定Top64和输入512。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/status.json)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; q_topk=64; training_epochs=3; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | R128 | 4.078177 | 243.524 | +3.13% | -5.08% | 已核验 |
| C1/formal | 1 | 0 | 1 | R256 | 4.079044 | 243.290 | +3.15% | -5.17% | 已核验 |
| C1/formal | 1 | 0 | 1 | R512_reference | 4.075425 | 242.585 | +3.06% | -5.44% | 已核验 |
| C1/formal | 1 | 0 | 1 | R64 | 4.065901 | 241.535 | +2.82% | -5.85% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 256.552 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | R128 | 3.672271 | 218.262 | +14.58% | +5.20% | 已核验 |
| C1/formal | 1 | 1 | 1 | R256 | 3.680552 | 218.558 | +14.83% | +5.34% | 已核验 |
| C1/formal | 1 | 1 | 1 | R512_reference | 3.677356 | 217.880 | +14.73% | +5.02% | 已核验 |
| C1/formal | 1 | 1 | 1 | R64 | 3.635319 | 215.160 | +13.42% | +3.70% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 207.474 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | R128 | 4.180393 | 2551.194 | +3.70% | -0.90% | 已核验 |
| C40/formal | 40 | 0 | 1 | R256 | 4.187711 | 2555.393 | +3.88% | -0.74% | 已核验 |
| C40/formal | 40 | 0 | 1 | R512_reference | 4.194341 | 2552.088 | +4.05% | -0.87% | 已核验 |
| C40/formal | 40 | 0 | 1 | R64 | 4.156084 | 2526.268 | +3.10% | -1.87% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.031111 | 2574.411 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | R128 | 3.836856 | 2294.268 | +14.52% | +8.51% | 已核验 |
| C40/formal | 40 | 1 | 1 | R256 | 3.843009 | 2303.836 | +14.71% | +8.97% | 已核验 |
| C40/formal | 40 | 1 | 1 | R512_reference | 3.854638 | 2308.440 | +15.05% | +9.18% | 已核验 |
| C40/formal | 40 | 1 | 1 | R64 | 3.797389 | 2273.845 | +13.35% | +7.55% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.350280 | 2114.264 | +0.00% | +0.00% | 已核验 |

**结果解释：**无稳定额外提速；新输出基底重新拟合、全头重训，因此旧512对新组不是纯rank单因素。

诊断／原报告摘录（保持原口径）：

- Old R512 uses a historical basis; comparison to it includes refitting and rank.
- All three fresh ranks share newly collected nested bases; only output rank differs.
- Previously used panels and one measurement round; no statistical significance claim.
- Long training on other cards can affect shared CPU/IO and power conditions.

**原目录／权重入口：**[eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[REPORT.txt](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/experiment.json)；[findings.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/findings.json)；[R128/preregistration.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/R128/preregistration.json)；[R256/preregistration.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/R256/preregistration.json)；[R64/preregistration.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/R64/preregistration.json)；[basis_collection/preregistration.json](eval_results/standalone_nosoft_output_rank256_128_64_short_gpu045_20261002/basis_collection/preregistration.json)。

<a id="t026"></a>

### T026 · 输出头低秩审计

**类别／状态：**研究实验；未提供顶层状态。

**做什么：**检查已训练方向矩阵谱及实际动作包络。

**对比什么：**权重谱、线性包络及原始修复。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**权重能量谱未按真实激活／候选重要性加权，不能据其直接证明可压缩或接受收益。

**原目录／权重入口：**[eval_results/standalone_nosoft_lowrank_audit_20261002](eval_results/standalone_nosoft_lowrank_audit_20261002)。权重、basis、Adam/RNG断点均保持原位置。

<a id="t027"></a>

### T027 · Parallel_TF及五组统一对照

**类别／状态：**研究实验；complete；已记录20个正式窗口。

**做什么：**把重SwiGLU搬到hidden/q上下文，推理仅以实际lag-1轻门控。

**对比什么：**RAW、旧Top256、Parallel_TF、独立训练Parallel_T0、Parallel_T1。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/status.json)。

协议字段：`train_count=128; epochs=3; steps=384; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| comparison_5arms/C1/formal | 1 | 0 | 1 | Parallel_T0 | 4.083246 | 246.251 | +3.26% | -3.66% | 已核验 |
| comparison_5arms/C1/formal | 1 | 0 | 1 | Parallel_T1 | 4.068142 | 245.484 | +2.87% | -3.96% | 已核验 |
| comparison_5arms/C1/formal | 1 | 0 | 1 | Parallel_TF | 4.072560 | 246.231 | +2.99% | -3.67% | 已核验 |
| comparison_5arms/C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 255.600 | +0.00% | +0.00% | 已核验 |
| comparison_5arms/C1/formal | 1 | 0 | 1 | Top256 | 4.081234 | 241.647 | +3.20% | -5.46% | 已核验 |
| comparison_5arms/C1/formal | 1 | 1 | 1 | Parallel_T0 | 3.698247 | 222.264 | +15.39% | +7.44% | 已核验 |
| comparison_5arms/C1/formal | 1 | 1 | 1 | Parallel_T1 | 3.720570 | 223.306 | +16.08% | +7.95% | 已核验 |
| comparison_5arms/C1/formal | 1 | 1 | 1 | Parallel_TF | 3.703041 | 222.828 | +15.54% | +7.71% | 已核验 |
| comparison_5arms/C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 206.869 | +0.00% | +0.00% | 已核验 |
| comparison_5arms/C1/formal | 1 | 1 | 1 | Top256 | 3.685537 | 217.078 | +14.99% | +4.93% | 已核验 |
| comparison_5arms/C40/formal | 40 | 0 | 1 | Parallel_T0 | 4.195036 | 2555.353 | +3.98% | -1.09% | 已核验 |
| comparison_5arms/C40/formal | 40 | 0 | 1 | Parallel_T1 | 4.173680 | 2542.776 | +3.45% | -1.58% | 已核验 |
| comparison_5arms/C40/formal | 40 | 0 | 1 | Parallel_TF | 4.182740 | 2548.869 | +3.68% | -1.34% | 已核验 |
| comparison_5arms/C40/formal | 40 | 0 | 1 | RAW | 4.034450 | 2583.466 | +0.00% | +0.00% | 已核验 |
| comparison_5arms/C40/formal | 40 | 0 | 1 | Top256 | 4.190472 | 2533.009 | +3.87% | -1.95% | 已核验 |
| comparison_5arms/C40/formal | 40 | 1 | 1 | Parallel_T0 | 3.864058 | 2303.394 | +15.56% | +9.59% | 已核验 |
| comparison_5arms/C40/formal | 40 | 1 | 1 | Parallel_T1 | 3.851349 | 2301.321 | +15.18% | +9.49% | 已核验 |
| comparison_5arms/C40/formal | 40 | 1 | 1 | Parallel_TF | 3.852546 | 2302.847 | +15.21% | +9.56% | 已核验 |
| comparison_5arms/C40/formal | 40 | 1 | 1 | RAW | 3.343846 | 2101.877 | +0.00% | +0.00% | 已核验 |
| comparison_5arms/C40/formal | 40 | 1 | 1 | Top256 | 3.852036 | 2292.888 | +15.20% | +9.09% | 已核验 |

**结果解释：**TF较旧N有小幅提速；on-policy两组没有一致超过TF，训练慢约8倍；用户选择TF作为后续共同基线。

诊断／原报告摘录（保持原口径）：

- 卡4已取消的smoke不纳入性能结果；单轮不作显著性结论。
- 卡4已取消的smoke不纳入性能结果；单轮不作显著性结论。

**原目录／权重入口：**[eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [comparison_5arms/C1/formal/summary.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/comparison_5arms/C1/formal/summary.json)；[comparison_5arms/C40/formal/summary.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/comparison_5arms/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[REPORT.txt](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/REPORT.txt)；[comparison_5arms/REPORT.txt](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/comparison_5arms/REPORT.txt)；[experiment.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/experiment.json)；[findings.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/findings.json)；[Parallel_TF/preregistration.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/Parallel_TF/preregistration.json)；[comparison_5arms/findings.json](eval_results/standalone_parallel_context_teacher_forced_short_gpu5_20261002/comparison_5arms/findings.json)。

<a id="t028"></a>

### T028 · Parallel_T0／T1真实前缀训练

**类别／状态：**研究实验；complete。

**做什么：**检查learner块内采样前缀重算teacher的训练配方。

**对比什么：**分别训练的T0头、T1头；有效五组评测见Parallel_TF条目。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; q_topk=256; output_rank=512; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**本配方训练约慢8倍且无一致native收益；不能推论所有on-policy训练无效。

**原目录／权重入口：**[eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5/experiment.json)；[Parallel_T0/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5/Parallel_T0/preregistration.json)；[Parallel_T1/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v5/Parallel_T1/preregistration.json)。

<a id="t029"></a>

### T029 · 目标hidden／logits／hidden+logits输入消融

**类别／状态：**研究实验；complete；已记录20个正式窗口。

**做什么：**检查现有目标前缀信息和logits输入能否改善修复。

**对比什么：**同轮RAW、旧N及三个新随机初始化头；旧N不重训。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | Logits | 4.043616 | 241.003 | +2.25% | -5.33% | 已核验 |
| C1/formal | 1 | 0 | 1 | N | 4.081234 | 242.129 | +3.20% | -4.88% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.954497 | 254.558 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | TargetHidden | 4.063171 | 239.040 | +2.75% | -6.10% | 已核验 |
| C1/formal | 1 | 0 | 1 | TargetHiddenLogits | 4.033816 | 238.365 | +2.01% | -6.36% | 已核验 |
| C1/formal | 1 | 1 | 1 | Logits | 3.636204 | 215.629 | +13.45% | +4.54% | 已核验 |
| C1/formal | 1 | 1 | 1 | N | 3.685537 | 217.540 | +14.99% | +5.47% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.205114 | 206.262 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | TargetHidden | 3.747484 | 219.946 | +16.92% | +6.63% | 已核验 |
| C1/formal | 1 | 1 | 1 | TargetHiddenLogits | 3.641666 | 214.282 | +13.62% | +3.89% | 已核验 |
| C40/formal | 40 | 0 | 1 | Logits | 4.150963 | 2520.498 | +2.89% | -1.90% | 已核验 |
| C40/formal | 40 | 0 | 1 | N | 4.186050 | 2518.672 | +3.76% | -1.97% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.034450 | 2569.214 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | TargetHidden | 4.180672 | 2518.122 | +3.62% | -1.99% | 已核验 |
| C40/formal | 40 | 0 | 1 | TargetHiddenLogits | 4.136073 | 2497.735 | +2.52% | -2.78% | 已核验 |
| C40/formal | 40 | 1 | 1 | Logits | 3.778828 | 2256.640 | +12.76% | +7.52% | 已核验 |
| C40/formal | 40 | 1 | 1 | N | 3.863560 | 2282.744 | +15.29% | +8.77% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.351135 | 2098.740 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | TargetHidden | 3.829692 | 2261.829 | +14.28% | +7.77% | 已核验 |
| C40/formal | 40 | 1 | 1 | TargetHiddenLogits | 3.777999 | 2254.260 | +12.74% | +7.41% | 已核验 |

**结果解释：**TargetHidden C1/T1比N有小幅增益，C40未延续；logits和组合没有一致优势。目标hidden来自最后已验证前缀的DFlash压缩状态，不是未来teacher hidden。

诊断／原报告摘录（保持原口径）：

- 额外GPU进程仅标记；污染窗口只展示观测值，不能作为干净吞吐收益证据。历史使用面板，单轮不作显著性结论。

**原目录／权重入口：**[eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[REPORT.txt](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/experiment.json)；[findings.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/findings.json)；[Logits/preregistration.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/Logits/preregistration.json)；[TargetHidden/preregistration.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/TargetHidden/preregistration.json)；[TargetHiddenLogits/preregistration.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003_v2/TargetHiddenLogits/preregistration.json)。

<a id="t030"></a>

### T030 · OUTPUT＋Top16候选CE

**类别／状态：**研究实验；complete；已记录4个正式窗口。

**做什么：**给旧两层修复头加入硬候选监督。

**对比什么：**仅测试新头；没有本轮RAW/N复测。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT_PLUS_TOP16_CANDIDATE_CE; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | EMBEDDING | 4.118866 | 244.260 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | EMBEDDING | 3.588519 | 212.011 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | EMBEDDING | 4.210626 | 2512.536 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | EMBEDDING | 3.740936 | 2207.608 | — | — | 已核验 |

**结果解释：**正式四窗口已完成；不可与历史绝对TPS直接相除。候选缺失标签CE为0且不补入标签；不是完整DFlash-2复现。

诊断／原报告摘录（保持原口径）：

- 额外同卡进程仅标记；污染窗口不作为干净吞吐证据。历史使用面板；单轮不作显著性结论。

**原目录／权重入口：**[eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/C40/formal/summary.json)；[REPORT.txt](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/experiment.json)；[findings.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/findings.json)；[N_OUTPUT_TOP16_CE/preregistration.json](eval_results/standalone_nosoft_output_top16_ce_short_gpu3_20261003/N_OUTPUT_TOP16_CE/preregistration.json)。

<a id="t031"></a>

### T031 · 长训权重：方向／dose瓶颈

**类别／状态：**研究实验；complete。

**做什么：**用长训最终权重检查错误是否由固定方向限制。

**对比什么：**长2实际路径上回放短2/长1/长3；最终头oracle及固定方向dose网格。

**实验设置：**8新研究提示、T0/T1、冻结native状态；hidden/q/实际前驱匹配；不修改正式权重。

**即时日志／进度：** [status.json](eval_results/standalone_longtrain_bottleneck_gpu45_20261003/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**到达的39个argmax错误中38个找到完整头好解，只调dose修好5个；不能据此认定输入充分或泛化必然可学。

诊断／原报告摘录（保持原口径）：

- 冻结长训权重瓶颈诊断：真实 native 状态；不训练；诊断窗口不测速。
- 错误状态：单状态final-linear overlap上升损margin 0; 同块reached平均 1; prefix代理 1
- 错误状态：单状态final-linear overlap上升损margin 0; 同块reached平均 1; prefix代理 1
- 原有正式两层结果（独立于本次诊断）：
- 局限：latent可行解不能证明输入足够、共享网络可学或留出泛化；数值solver无解不能证明不可达。
- prefix cumprod是代理；overlap改进不保证greedy改进；局部oracle不等于native AL改进。
- 性能依据仅原有正式窗口，单轮不是显著性结论，wall/draft不是唯一硬件瓶颈定位。

**原目录／权重入口：**[eval_results/standalone_longtrain_bottleneck_gpu45_20261003](eval_results/standalone_longtrain_bottleneck_gpu45_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_longtrain_bottleneck_gpu45_20261003/REPORT.txt)；[findings.json](eval_results/standalone_longtrain_bottleneck_gpu45_20261003/findings.json)；[preregistration.json](eval_results/standalone_longtrain_bottleneck_gpu45_20261003/preregistration.json)。

<a id="t032"></a>

### T032 · 长训共享全参数梯度

**类别／状态：**研究实验；complete。

**做什么：**检查同状态改善是否在跨提示平均梯度中被抵消。

**对比什么：**单状态gradient、8提示平均gradient与target margin。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_longtrain_fullgradient_gpu45_20261003/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**单状态均有益，平均梯度的局部SGD估计伤害12/39；不是实际Adam更新成绩，也不是唯一瓶颈证明。

诊断／原报告摘录（保持原口径）：

- Full BF16 repair-head VJP, local unpreconditioned SGD transport, not Adam. Repeated first-run diagnostic prompts, not new independent testing.

**原目录／权重入口：**[eval_results/standalone_longtrain_fullgradient_gpu45_20261003](eval_results/standalone_longtrain_fullgradient_gpu45_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_longtrain_fullgradient_gpu45_20261003/REPORT.txt)；[findings.json](eval_results/standalone_longtrain_fullgradient_gpu45_20261003/findings.json)；[preregistration.json](eval_results/standalone_longtrain_fullgradient_gpu45_20261003/preregistration.json)。

<a id="t033"></a>

### T033 · 五问题预实验

**类别／状态：**研究实验；complete。

**做什么：**联合检查目标信息、greedy监督、方向、梯度冲突和运行成本。

**对比什么：**OUTPUT、MARGIN、DIRECTION、MARGIN_PCGRAD、PAST_Q、PAST_MISMATCH。

**实验设置：**长训两层副本、16步、固定behavior轨迹；历史target仅作监督；无learner真实rollout。

**即时日志／进度：** [status.json](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**固定状态训练改善大于探针迁移；不能用局部小样本结果认定方案已解决native问题。计时是算子微基准。

诊断／原报告摘录（保持原口径）：

- 表中仅实际到达状态；fixed/harmed 表示原错误被修好/原正确被误修。不是 native AL 或 TPS。
- OUTPUT: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.056493；margin变化+0.2500
- MARGIN: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.069036；margin变化+0.9240
- DIRECTION: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.065572；margin变化+0.6544
- MARGIN_PCGRAD: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.071019；margin变化+1.0539
- PAST_Q: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.069372；margin变化+0.9632
- PAST_MISMATCH: 正确 45/51 (原 40); 修好6，误修1；overlap变化 +0.068983；margin变化+0.8922
- OUTPUT: 正确 48/55 (原 48); 修好0，误修0；overlap变化 +0.008376；margin变化-0.8648

**原目录／权重入口：**[eval_results/standalone_five_issue_pilot_gpu5_20261003_v2](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [REPORT.txt](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2/REPORT.txt)；[findings.json](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2/findings.json)；[preregistration.json](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2/preregistration.json)；[static_projection_check/findings.json](eval_results/standalone_five_issue_pilot_gpu5_20261003_v2/static_projection_check/findings.json)。

<a id="t034"></a>

### T034 · OUTPUT＋greedy margin native小试

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**验证固定状态margin改善是否转化为闭环接受收益。

**对比什么：**RAW、LONG_BASE、OUTPUT16、MARGIN16。

**实验设置：**长训两层副本16步；新研究提示，C1=20/C40=300、T0/T1、7proposal/max4096。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/status.json)。

协议字段：`objective=Frozen long2 and matched 16-update OUTPUT vs OUTPUT+greedy-margin warmstarted copies; thinking=True`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | LONG_BASE | 4.478558 | 263.997 | +8.39% | -0.72% | 已核验 |
| C1/formal | 1 | 0 | 1 | MARGIN16 | 4.404849 | 259.547 | +6.61% | -2.40% | 已核验 |
| C1/formal | 1 | 0 | 1 | OUTPUT16 | 4.421542 | 260.437 | +7.01% | -2.06% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 4.131758 | 265.916 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | LONG_BASE | 4.091385 | 240.325 | +22.06% | +11.64% | 已核验 |
| C1/formal | 1 | 1 | 1 | MARGIN16 | 4.074465 | 238.985 | +21.55% | +11.01% | 已核验 |
| C1/formal | 1 | 1 | 1 | OUTPUT16 | 4.062811 | 237.755 | +21.21% | +10.44% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.351977 | 215.275 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | LONG_BASE | 4.398408 | 2680.407 | +8.68% | +2.34% | 已核验 |
| C40/formal | 40 | 0 | 1 | MARGIN16 | 4.314966 | 2626.584 | +6.62% | +0.28% | 已核验 |
| C40/formal | 40 | 0 | 1 | OUTPUT16 | 4.338318 | 2640.212 | +7.19% | +0.81% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.047137 | 2619.124 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | LONG_BASE | 4.136669 | 2480.549 | +23.60% | +15.90% | 已核验 |
| C40/formal | 40 | 1 | 1 | MARGIN16 | 4.069670 | 2444.507 | +21.60% | +14.22% | 已核验 |
| C40/formal | 40 | 1 | 1 | OUTPUT16 | 4.069796 | 2443.537 | +21.60% | +14.17% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.346741 | 2140.172 | +0.00% | +0.00% | 已核验 |

**结果解释：**没有一致胜过LONG_BASE；margin16在T0反而低于OUTPUT16。不要扩大成greedy margin已解决方向学习。

诊断／原报告摘录（保持原口径）：

- Only one formal window per arm/temperature; fresh research panel, not independent final test.
- C1/C40 and T0/T1 run on separate GPUs; ratios only within each same-card suite.
- This is validation of the small pilot copies, not full-dataset training of the new loss.
- C1/GPU4 and C40/GPU5 run concurrently; compare ratios only within matching C/T, GPU and panel.

**原目录／权重入口：**[eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[REPORT.txt](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/REPORT.txt)；[experiment.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/experiment.json)；[findings.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu45_20261003/findings.json)。

<a id="t035"></a>

### T035 · KL／overlap／margin 任务基底

**类别／状态：**研究实验；failed。

**做什么：**改变输出基底采集目标，检查大残差方向是否有接受价值。

**对比什么：**GG-KL／GG-OVERLAP／GG-MARGIN，新basis、新头；输入basis固定。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3/status.json)；[training.log](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3/training.log)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT final-distribution teacher-prefix overlap proxy ONLY; max_anchors=1536; head_chunk_anchors=32; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**basis拟合与训练完成，顶层failed是评测资源问题；有效复用正式评测在可预测基底实验，不把failed当科学负结果。

**原目录／权重入口：**[eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[preregistration.json](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3/preregistration.json)；[initialization_native/preregistration.json](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3/initialization_native/preregistration.json)；[native/preregistration.json](eval_results/standalone_nosoft_basis_objective_short_gpu0_20261003_v3/native/preregistration.json)。

<a id="t036"></a>

### T036 · 线性可预测 mismatch 基底

**类别／状态：**研究实验；complete；已记录40个正式窗口。

**做什么：**用输入-梯度低秩回归保留部署输入能预测的修正方向。

**对比什么：**GG_KL/OVERLAP/MARGIN、RR三目标、RR_OVERLAP正则0.1/1/10；旧短训N作历史参考，LONG_BASE只读。

**实验设置：**X是h/q/硬lag1固定1537维特征；输出rank512，固定旧输入basis，五新头128×3 OUTPUT；C1=20/C40=300，7proposal、8GiB KV；正式40窗口。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/status.json)；[single_card_C1_formal.log](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/single_card_C1_formal.log)；[single_card_C40_formal.log](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/single_card_C40_formal.log)；[single_card_C1_smoke.log](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/single_card_C1_smoke.log)；[single_card_C40_smoke.log](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/single_card_C40_smoke.log)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT final-distribution teacher-prefix overlap proxy ONLY; max_anchors=1536; head_chunk_anchors=32; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | KL_B512 | 4.227829 | 249.695 | +3.28% | -4.91% | 已核验 |
| C1/formal | 1 | 0 | 1 | LONG_BASE | 4.432501 | 261.911 | +8.28% | -0.26% | 已核验 |
| C1/formal | 1 | 0 | 1 | MARGIN_B512 | 4.226455 | 249.664 | +3.25% | -4.92% | 已核验 |
| C1/formal | 1 | 0 | 1 | OVERLAP_B512 | 4.228892 | 249.930 | +3.31% | -4.82% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 4.093522 | 262.588 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | RR_KL_B512 | 4.218635 | 249.191 | +3.06% | -5.10% | 已核验 |
| C1/formal | 1 | 0 | 1 | RR_MARGIN_B512 | 4.210025 | 248.706 | +2.85% | -5.29% | 已核验 |
| C1/formal | 1 | 0 | 1 | RR_OVERLAP_B512 | 4.233371 | 249.876 | +3.42% | -4.84% | 已核验 |
| C1/formal | 1 | 0 | 1 | RR_OVERLAP_HIGH_B512 | 4.231286 | 250.051 | +3.37% | -4.77% | 已核验 |
| C1/formal | 1 | 0 | 1 | RR_OVERLAP_LOW_B512 | 4.213014 | 248.784 | +2.92% | -5.26% | 已核验 |
| C1/formal | 1 | 1 | 1 | KL_B512 | 3.862491 | 227.673 | +14.15% | +4.80% | 已核验 |
| C1/formal | 1 | 1 | 1 | LONG_BASE | 4.176840 | 246.676 | +23.44% | +13.55% | 已核验 |
| C1/formal | 1 | 1 | 1 | MARGIN_B512 | 3.851641 | 226.844 | +13.83% | +4.42% | 已核验 |
| C1/formal | 1 | 1 | 1 | OVERLAP_B512 | 3.923589 | 231.100 | +15.96% | +6.38% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.383600 | 217.242 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | RR_KL_B512 | 3.929274 | 231.419 | +16.13% | +6.53% | 已核验 |
| C1/formal | 1 | 1 | 1 | RR_MARGIN_B512 | 3.933179 | 231.670 | +16.24% | +6.64% | 已核验 |
| C1/formal | 1 | 1 | 1 | RR_OVERLAP_B512 | 3.895592 | 229.038 | +15.13% | +5.43% | 已核验 |
| C1/formal | 1 | 1 | 1 | RR_OVERLAP_HIGH_B512 | 3.921878 | 231.149 | +15.91% | +6.40% | 已核验 |
| C1/formal | 1 | 1 | 1 | RR_OVERLAP_LOW_B512 | 3.904495 | 230.036 | +15.39% | +5.89% | 已核验 |
| C40/formal | 40 | 0 | 1 | KL_B512 | 4.231450 | 2167.506 | +3.19% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | LONG_BASE | 4.428481 | 2245.606 | +7.99% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | MARGIN_B512 | 4.241474 | 2167.749 | +3.43% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | OVERLAP_B512 | 4.236138 | 2153.800 | +3.30% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 4.100657 | 1834.841 | +0.00% | — | 污染 |
| C40/formal | 40 | 0 | 1 | RR_KL_B512 | 4.219609 | 2159.718 | +2.90% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | RR_MARGIN_B512 | 4.223905 | 2156.832 | +3.01% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | RR_OVERLAP_B512 | 4.227661 | 1322.176 | +3.10% | — | 污染 |
| C40/formal | 40 | 0 | 1 | RR_OVERLAP_HIGH_B512 | 4.238296 | 2163.568 | +3.36% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | RR_OVERLAP_LOW_B512 | 4.222540 | 2143.498 | +2.97% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | KL_B512 | 3.861329 | 1946.285 | +14.86% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | LONG_BASE | 4.155942 | 2071.754 | +23.62% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | MARGIN_B512 | 3.864555 | 1940.541 | +14.95% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | OVERLAP_B512 | 3.858401 | 1936.322 | +14.77% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.361905 | 959.819 | +0.00% | — | 污染 |
| C40/formal | 40 | 1 | 1 | RR_KL_B512 | 3.860587 | 1938.871 | +14.83% | — | 污染 |
| C40/formal | 40 | 1 | 1 | RR_MARGIN_B512 | 3.870227 | 1938.007 | +15.12% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | RR_OVERLAP_B512 | 3.857019 | 1077.015 | +14.73% | — | 污染 |
| C40/formal | 40 | 1 | 1 | RR_OVERLAP_HIGH_B512 | 3.864938 | 1942.297 | +14.96% | — | 已核验 |
| C40/formal | 40 | 1 | 1 | RR_OVERLAP_LOW_B512 | 3.866055 | 1945.641 | +15.00% | — | 已核验 |

**结果解释：**C1/T1 RR_KL/RR_MARGIN有小收益，C40未稳定延续；部分C40窗口污染。不能只和更弱RR控制相比，也不能用污染RAW夸大TPS。

诊断／原报告摘录（保持原口径）：

- X为部署可用h/原始q/实际硬lag1的固定特征；teacher只进入监督G。X1537维；随机输入投影不能代表全部输入的可预测性上限。
- crossfit、梯度能量不等于真实AL/TPS；受污染窗口只报告观测值；单轮小差异不称稳定提速。

**原目录／权重入口：**[eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/C40/formal/summary.json)；[REPORT.txt](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/REPORT.txt)；[preregistration.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/preregistration.json)；[initialization_native/preregistration.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/initialization_native/preregistration.json)；[native/preregistration.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/native/preregistration.json)；[single_card_diagnostic/preregistration.json](eval_results/standalone_nosoft_predictable_basis_gpu2345_20261003_v2/single_card_diagnostic/preregistration.json)。

<a id="t037"></a>

### T037 · 非线性可预测基底

**类别／状态：**研究实验；complete；已记录4个正式窗口。

**做什么：**检查非线性预测梯度方向是否优于线性RR。

**对比什么：**两折OVERLAP预测、线性控制、GG_OVERLAP；同预算新头。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/status.json)；[training.log](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/training.log)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT final-distribution teacher-prefix overlap proxy ONLY; max_anchors=1536; head_chunk_anchors=32; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096; C40=300条，proposal=7，max_new=4096。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | KRR_OVERLAP_B512 | 4.223639 | 248.599 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | KRR_OVERLAP_B512 | 3.922346 | 230.700 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | KRR_OVERLAP_B512 | 4.227833 | 2169.883 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | KRR_OVERLAP_B512 | 3.867607 | 1960.029 | — | — | 污染 |

**结果解释：**梯度预测误差小幅改善约0.62%/0.73%，native AL相对GG变化不足0.25%；无稳定额外收益。C40/T1受污染。

诊断／原报告摘录（保持原口径）：

- 原始短训N作为主基线，历史记录只读复用；不同面板/显存预算，不能按绝对AL/TPS宣布超越。
- 非线性两折OVERLAP梯度预测误差相对最佳线性控制下降0.618%/0.728%，通过预注册门槛。
- 线性C40/RAW两个窗口受外部PID1157132污染，不能把FORMAL_RESULTS.csv中+100%左右的RAW吞吐百分比作为收益。
- 结论：可预测性有小幅改善，未得到稳定额外native收益；仍不足以定位唯一瓶颈或判定整个mismatch思路失败。

**原目录／权重入口：**[eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/C1/formal/summary.json)；[C40/formal/summary.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/C40/formal/summary.json)；[analysis_completed_20261004_v1/RESULTS_CN.txt](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/analysis_completed_20261004_v1/RESULTS_CN.txt)；[preregistration.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/preregistration.json)；[initialization_native/preregistration.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/initialization_native/preregistration.json)；[native/preregistration.json](eval_results/standalone_nosoft_nonlinear_basis_pilot_gpu5_20261004_v1/native/preregistration.json)。

<a id="t038"></a>

### T038 · 可达性／可学习性瓶颈probe

**类别／状态：**研究实验；complete。

**做什么：**区分动作空间上限、当前方向、误修及研究留出拟合。

**对比什么：**冻结旧N／长训、逐状态oracle及学习探针。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1/status.json)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT final-distribution teacher-prefix overlap proxy ONLY; max_anchors=1536; head_chunk_anchors=32; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**所测好解存在与共享预测效果不同；不能把基底预实验不稳定归结为整个结构失败，唯一原因仍未识别。

诊断／原报告摘录（保持原口径）：

- 结论1（有限子集）：RAW/NL轨迹×T0/T1，每组预选32个首个argmax错误状态，OLD/GG/RR/NL/FULL4096的MARGIN系数oracle都找到32/32正确动作。这些子集复用32条提示、且状态部分重复，不是128个独立错误样本；不能外推所有到达错误。
- 结论2：例如NL/T1同一个32错误子集，NL头修好22/32，固定其方向在预注册dose网格里寻优仍22/32，自由基底系数oracle为32/32。这支持检查方向/有限修正动作选择，不能据有限剂量网格证明当前方向绝对无解。
- 结论3（全到达状态）：NL/T1有571个状态，原q的argmax错误200、正确371。原短训修好132/200、误修7/371；长训141/200、误修4/371；NL头126/200、误修7/371；直接NL预测124/200、误修22/371。长训是不同预算参考，以上是同状态反事实指标，不是各头独立native接受长度。
- 结论4：原输出基底在所测子集已有可行解，当前换基底没有提高这里的贪心可修成功率。直接预测也没有绕过困难，所以未证明单纯增加预测器容量或替换优化器即可解决。
- oracle看到当前target答案；找到解不证明部署输入充分、可学习或泛化。该平衡首错/首正确子集主要用于定位能力，不作为总体成功率。
- 同状态瓶颈定位：诊断结果，不是native AL/TPS成绩。
- 所有输入使用原始h/q/实际lag-1；teacher只用于监督、训练提示剂量标定和标记清楚的oracle。
- 直接预测器学习OVERLAP梯度，不是最优修正动作；其native RAW监督状态与头的teacher-forced训练状态不同，不能把差异只归因于网络容量。

**原目录／权重入口：**[eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1](eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INTERPRETATION_20261004.txt](eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1/INTERPRETATION_20261004.txt)；[RESULTS_CN.txt](eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1/RESULTS_CN.txt)；[preregistration.json](eval_results/standalone_nosoft_bottleneck_probe_gpu4_20261004_v1/preregistration.json)。

<a id="t039"></a>

### T039 · 输入充分性／目标hidden来源 v1

**类别／状态：**研究实验；failed。

**做什么：**核对部署目标hidden来源以及分支动作冲突。

**对比什么：**冻结Parallel_TF输入、实际分支、额外teacher未来hidden诊断。

**实验设置：**T1冻结原生状态诊断；target context是辅助层经fc/RMSNorm的anchor前一token；未来teacher hidden仅为额外诊断。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_prefix_information_gpu5_20261004_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**v3有效23配对、14提示，有限动作冲突很小，跨提示学习探针门槛未通过；不证明输入充分。v1/v2为失败观察器尝试。

**原目录／权重入口：**[eval_results/standalone_nosoft_prefix_information_gpu5_20261004_v1](eval_results/standalone_nosoft_prefix_information_gpu5_20261004_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_nosoft_prefix_information_gpu5_20261004_v1/preregistration.json)。

<a id="t040"></a>

### T040 · 输入充分性／目标hidden来源 v2

**类别／状态：**研究实验；failed。

**做什么：**核对部署目标hidden来源以及分支动作冲突。

**对比什么：**冻结Parallel_TF输入、实际分支、额外teacher未来hidden诊断。

**实验设置：**T1冻结原生状态诊断；target context是辅助层经fc/RMSNorm的anchor前一token；未来teacher hidden仅为额外诊断。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v2/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**v3有效23配对、14提示，有限动作冲突很小，跨提示学习探针门槛未通过；不证明输入充分。v1/v2为失败观察器尝试。

**原目录／权重入口：**[eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v2](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v2/preregistration.json)。

<a id="t041"></a>

### T041 · 输入充分性／目标hidden来源 v3

**类别／状态：**研究实验；complete。

**做什么：**核对部署目标hidden来源以及分支动作冲突。

**对比什么：**冻结Parallel_TF输入、实际分支、额外teacher未来hidden诊断。

**实验设置：**T1冻结原生状态诊断；target context是辅助层经fc/RMSNorm的anchor前一token；未来teacher hidden仅为额外诊断。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v3/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**v3有效23配对、14提示，有限动作冲突很小，跨提示学习探针门槛未通过；不证明输入充分。v1/v2为失败观察器尝试。

诊断／原报告摘录（保持原口径）：

- 参考/native分布差异超过TV=0.02门限而排除的配对=10；配对尝试=64。排除产生选择限制，不能推广为全部native状态。

**原目录／权重入口：**[eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v3](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v3/RESULTS_CN.txt)；[preregistration.json](eval_results/standalone_nosoft_prefix_information_gpu4_20261004_v3/preregistration.json)。

<a id="t042"></a>

### T042 · OUTPUT／Euclidean／Fisher方向学习

**类别／状态：**研究实验；failed；已记录10个正式窗口；用户已授权停止，后续未完成窗口不算成绩。

**做什么：**检验不同动作几何监督是否能改善修正方向。

**对比什么：**同Parallel_TF权重副本的OUTPUT/EUCLIDEAN/FISHER，两套训练seed。

**实验设置：**16新提示×3轮=48步／头；重新建立master/Adam；7proposal，C1=20/C40原计划300；有效评测recovery_v3。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1/status.json)；[single_card_C1_smoke.log](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1/single_card_C1_smoke.log)；[training.log](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1/training.log)。

协议字段：`train_count=16; epochs=3; max_anchors=16; head_chunk_anchors=4; proposal_length=7; native_max_tokens=4096; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

正式窗口设置：C1=20条，proposal=7，max_new=4096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| evaluation_recovery_v3/C1/formal | 1 | 0 | 1 | BASE_TF | 3.499473 | 218.663 | +2.49% | -3.80% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 0 | 1 | EUCLIDEAN_seed430002 | 3.448702 | 215.635 | +1.00% | -5.14% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 0 | 1 | FISHER_seed430002 | 3.451688 | 215.435 | +1.09% | -5.22% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 0 | 1 | OUTPUT_seed430002 | 3.456177 | 215.794 | +1.22% | -5.07% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 0 | 1 | RAW | 3.414475 | 227.310 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 1 | 1 | BASE_TF | 3.163310 | 196.676 | +12.16% | +4.60% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 1 | 1 | EUCLIDEAN_seed430002 | 3.106200 | 192.894 | +10.14% | +2.59% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 1 | 1 | FISHER_seed430002 | 3.047461 | 190.487 | +8.05% | +1.31% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 1 | 1 | OUTPUT_seed430002 | 3.088386 | 192.648 | +9.51% | +2.46% | 已核验 |
| evaluation_recovery_v3/C1/formal | 1 | 1 | 1 | RAW | 2.820287 | 188.026 | +0.00% | +0.00% | 已核验 |

**结果解释：**用户在第一个seed C1未见收益后停止；保留18个epoch权重及10个C1正式窗口。第二seed未完成、C40未测，不自动恢复。

**原目录／权重入口：**[eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [evaluation_recovery_v3/C1/formal/summary.json](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1/evaluation_recovery_v3/C1/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[preregistration.json](eval_results/standalone_nosoft_direction_geometry_pilot_gpu2_20261004_v1/preregistration.json)。

<a id="t043"></a>

### T043 · EAGLE-3 Thinking · 3 proposal

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**检查相同mismatch思路迁移到原生EAGLE-3递推。

**对比什么：**下载的Thinking EAGLE-3 RAW与新basis、新随机Parallel_TF修复头。

**实验设置：**180重生成提示×3轮=540步；no-soft两层／Top256／basis512／OUTPUT；3递推proposal；C1=20/C40=300，T0/T1。

**即时日志／进度：** [status.json](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/status.json)；[training.log](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/training.log)。

协议字段：`train_count=180; epochs=3; steps=3; objective=OUTPUT only: 1-mean(prefix cumulative product of overlap); p uses full target denominator; max_anchors=1536; basis_rank=512; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; microbatch=32`。

正式窗口设置：C1=20条，proposal=3，max_new=见原协议; C40=300条，proposal=3，max_new=见原协议。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| evaluation_recovery_v1/test_C1 | 1 | 0 | 1 | RAW | 3.103294 | 209.265 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_v1/test_C1 | 1 | 0 | 1 | REPAIR | 3.121299 | 205.642 | +0.58% | -1.73% | 已核验 |
| evaluation_recovery_v1/test_C1 | 1 | 1 | 1 | RAW | 2.924959 | 195.711 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_v1/test_C1 | 1 | 1 | 1 | REPAIR | 3.006230 | 197.819 | +2.78% | +1.08% | 已核验 |
| evaluation_recovery_v1/test_C40 | 40 | 0 | 1 | RAW | 3.123808 | 2634.031 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_v1/test_C40 | 40 | 0 | 1 | REPAIR | 3.147360 | 2639.316 | +0.75% | +0.20% | 已核验 |
| evaluation_recovery_v1/test_C40 | 40 | 1 | 1 | RAW | 2.992101 | 2509.674 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_v1/test_C40 | 40 | 1 | 1 | REPAIR | 3.063572 | 2543.873 | +2.39% | +1.36% | 已核验 |

**结果解释：**三步版本仅小幅收益；七步独立重新拟合basis与训练，不能当纯推理长度消融。teacher全词表softmax后取64000子集，不重归一化；d2t是偏移。

**原目录／权重入口：**[residualV3 eagle-3/runs/parallel_tf_180x3_gpu5_20261002](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [evaluation_recovery_v1/test_C1/summary.json](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/evaluation_recovery_v1/test_C1/summary.json)；[evaluation_recovery_v1/test_C40/summary.json](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/evaluation_recovery_v1/test_C40/summary.json)；[REPORT.txt](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/REPORT.txt)；[evaluation_recovery_v1/REPORT.txt](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/evaluation_recovery_v1/REPORT.txt)；[preregistration.json](residualV3%20eagle-3/runs/parallel_tf_180x3_gpu5_20261002/preregistration.json)。

<a id="t044"></a>

### T044 · EAGLE-3 Thinking · 7 proposal

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**检查相同mismatch思路迁移到原生EAGLE-3递推。

**对比什么：**下载的Thinking EAGLE-3 RAW与新basis、新随机Parallel_TF修复头。

**实验设置：**180重生成提示×3轮=540步；no-soft两层／Top256／basis512／OUTPUT；7递推proposal；C1=20/C40=300，T0/T1。

**即时日志／进度：** [status.json](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/status.json)；[training.log](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/training.log)。

协议字段：`train_count=180; epochs=3; steps=7; objective=OUTPUT only: 1-mean(prefix cumulative product of overlap); p uses full target denominator; max_anchors=1536; basis_rank=512; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; microbatch=32`。

正式窗口设置：C1=20条，proposal=7，max_new=见原协议; C40=300条，proposal=7，max_new=见原协议。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| test_C1 | 1 | 0 | 1 | RAW | 4.160541 | 247.258 | +0.00% | +0.00% | 已核验 |
| test_C1 | 1 | 0 | 1 | REPAIR | 4.227970 | 225.452 | +1.62% | -8.82% | 已核验 |
| test_C1 | 1 | 1 | 1 | RAW | 3.780109 | 223.820 | +0.00% | +0.00% | 已核验 |
| test_C1 | 1 | 1 | 1 | REPAIR | 3.988172 | 211.753 | +5.50% | -5.39% | 已核验 |
| test_C40 | 40 | 0 | 1 | RAW | 4.287559 | 2852.048 | +0.00% | +0.00% | 已核验 |
| test_C40 | 40 | 0 | 1 | REPAIR | 4.349563 | 2763.466 | +1.45% | -3.11% | 已核验 |
| test_C40 | 40 | 1 | 1 | RAW | 3.923138 | 2556.845 | +0.00% | +0.00% | 已核验 |
| test_C40 | 40 | 1 | 1 | REPAIR | 4.127782 | 2571.462 | +5.22% | +0.57% | 已核验 |

**结果解释：**三步版本仅小幅收益；七步独立重新拟合basis与训练，不能当纯推理长度消融。teacher全词表softmax后取64000子集，不重归一化；d2t是偏移。

**原目录／权重入口：**[residualV3 eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [test_C1/summary.json](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/test_C1/summary.json)；[test_C40/summary.json](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/test_C40/summary.json)；[REPORT.txt](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/REPORT.txt)；[preregistration.json](residualV3%20eagle-3/runs/parallel_tf_180x3_steps7_gpu5_20261002/preregistration.json)。

<a id="t045"></a>

### T045 · 仍供其他基线使用的历史共享数据

**类别／状态：**研究实验；未提供顶层状态。

**做什么：**仍供其他基线使用的历史共享数据，保留研究链条及依赖。

**对比什么：**原协议中的冻结输入／输出与对照。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共享数据不是独立的新训练胜负结论；原始依赖及审计证据保留。

**原目录／权重入口：**[eval_results/residualV3_full_vocab_unified_lag1_v1_seed42](eval_results/residualV3_full_vocab_unified_lag1_v1_seed42)。权重、basis、Adam/RNG断点均保持原位置。

<a id="t046"></a>

### T046 · 全词表lag-1历史共享数据与协议

**类别／状态：**研究实验；未提供顶层状态。

**做什么：**全词表lag-1历史共享数据与协议，保留研究链条及依赖。

**对比什么：**原协议中的冻结输入／输出与对照。

**实验设置：**冻结Qwen3-8B target/DFlash；thinking重生成128条×3轮=384更新；旧N为no-soft两层、Top256、basis512、OUTPUT、实际硬lag-1；常规评测7proposal，C1=20/C40=300，max4096/model8192。具体例外以本条协议为准。

协议字段：`seed=42; stage1_lr=1e-05; dose_lr=5e-05; weight_decay=0.0; warmup_fraction=0.05`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共享数据不是独立的新训练胜负结论；原始依赖及审计证据保留。

**原目录／权重入口：**[eval_results/residualV3_full_vocab_unified_lag1_v3_seed42](eval_results/residualV3_full_vocab_unified_lag1_v3_seed42)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/residualV3_full_vocab_unified_lag1_v3_seed42/preregistration.json)。

<a id="t047"></a>

### T047 · 额外头数学／实现诊断

**类别／状态：**研究实验；complete。

**做什么：**额外头数学／实现诊断，保留研究链条及依赖。

**对比什么：**原协议中的冻结输入／输出与对照。

**实验设置：**冻结target/DFlash／既有头，研究状态诊断；具体样本数量及学习步数以原findings／preregistration为准。此项不沿用常规128×3训练预算；未产生learner native窗口时AL/TPS未测。

**即时日志／进度：** [status.json](eval_results/standalone_extra_head_diagnostic_gpu5_20260930/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共享数据不是独立的新训练胜负结论；原始依赖及审计证据保留。

**原目录／权重入口：**[eval_results/standalone_extra_head_diagnostic_gpu5_20260930](eval_results/standalone_extra_head_diagnostic_gpu5_20260930)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_extra_head_diagnostic_gpu5_20260930/preregistration.json)。

<a id="t048"></a>

### T048 · standalone_five_issue_pilot_gpu5_20261003

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t049"></a>

### T049 · standalone_gradient_validation_gpu5_20261001

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t050"></a>

### T050 · standalone_longtrain_bottleneck_gpu45_20261003_prepare_failed_tokenizer

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t051"></a>

### T051 · standalone_native_oracle_pair_gpu23_20261001

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t052"></a>

### T052 · standalone_native_oracle_pair_gpu45_20261001

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t053"></a>

### T053 · standalone_native_oracle_pair_gpu45_20261001_attempt2

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t054"></a>

### T054 · standalone_native_oracle_pair_gpu45_20261001_attempt3

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t055"></a>

### T055 · standalone_native_oracle_pair_gpu45_20261001_attempt4

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t056"></a>

### T056 · standalone_native_oracle_pair_gpu45_20261001_attempt5

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t057"></a>

### T057 · standalone_native_oracle_pair_gpu45_20261001_attempt6

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t058"></a>

### T058 · standalone_nosoft_basis_objective_short_gpu3_shared_20261003_v2

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t059"></a>

### T059 · standalone_nosoft_basis_objective_short_gpu5_20261003

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t060"></a>

### T060 · standalone_predecessor_response_gpu5_20261001

**类别／状态：**失败与退役记录；已归档。

**做什么：**同名成功版本之前的准备失败／废弃尝试。

**对比什么：**不作科学比较。

**实验设置：**原配置和错误摘要见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已归档，未产生有效正式成绩；不是方法无效的证据。

**原始证据：** [failed_metadata.tar.gz](archives/residualV3_cleanup_20261004_v1/failed_metadata.tar.gz)。

<a id="t061"></a>

### T061 · 退役记录 · standalone_nosoft_depth_10k_3epoch_gpu123_20261002

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002/status.json)。

协议字段：`train_count=10000; epochs=3; max_anchors=1536; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**早期长训准备／失败，含权重，保留恢复来源。

**原目录／权重入口：**[eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002/experiment.json)；[nosoft_1layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002/nosoft_1layer/preregistration.json)；[nosoft_2layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002/nosoft_2layer/preregistration.json)；[nosoft_3layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_gpu123_20261002/nosoft_3layer/preregistration.json)。

<a id="t062"></a>

### T062 · 退役记录 · standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002/status.json)。

协议字段：`train_count=10000; epochs=3; max_anchors=1536; training_epochs=3; objective=OUTPUT final-distribution prefix overlap ONLY; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**外部进程触发旧控制器错误连带全停，含epoch2权重；成功版本为gpu012 recovery。

**原目录／权重入口：**[eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002/experiment.json)；[nosoft_1layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002/nosoft_1layer/preregistration.json)；[nosoft_2layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002/nosoft_2layer/preregistration.json)；[nosoft_3layer/preregistration.json](eval_results/standalone_nosoft_depth_10k_3epoch_prefetch6_gpu123_20261002/nosoft_3layer/preregistration.json)。

<a id="t063"></a>

### T063 · 退役记录 · standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003

**类别／状态：**失败与退役记录；not_launched。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003/status.json)。

协议字段：`objective=Frozen long2 and matched 16-update OUTPUT vs OUTPUT+greedy-margin warmstarted copies; thinking=True`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**未启动的资源安排，保留资产，成功版本为gpu45。

**原目录／权重入口：**[eval_results/standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu0123_20261003/experiment.json)。

<a id="t064"></a>

### T064 · 退役记录 · standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003

**类别／状态：**失败与退役记录；not_launched。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003/status.json)。

协议字段：`objective=Frozen long2 and matched 16-update OUTPUT vs OUTPUT+greedy-margin warmstarted copies; thinking=True`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**未启动的替代资源安排，成功版本为gpu45。

**原目录／权重入口：**[eval_results/standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_nosoft_output_margin_native_pilot16_gpu2345_20261003/experiment.json)。

<a id="t065"></a>

### T065 · 退役记录 · standalone_nosoft_predictable_basis_gpu12345_20261003_v1

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_predictable_basis_gpu12345_20261003_v1/status.json)。

协议字段：`train_count=128; epochs=3; objective=OUTPUT final-distribution teacher-prefix overlap proxy ONLY; max_anchors=1536; head_chunk_anchors=32; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**资源／准备失败；有效结果为gpu2345 v2，含权重不可删除。

**原目录／权重入口：**[eval_results/standalone_nosoft_predictable_basis_gpu12345_20261003_v1](eval_results/standalone_nosoft_predictable_basis_gpu12345_20261003_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_nosoft_predictable_basis_gpu12345_20261003_v1/preregistration.json)。

<a id="t066"></a>

### T066 · 退役记录 · standalone_nosoft_target_hidden_logits_short_gpu012_20261003

**类别／状态：**失败与退役记录；preparation_failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**端口准备失败，未训练；有效版本为v2。

**原目录／权重入口：**[eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003](eval_results/standalone_nosoft_target_hidden_logits_short_gpu012_20261003)。权重、basis、Adam/RNG断点均保持原位置。

<a id="t067"></a>

### T067 · 退役记录 · standalone_oracle_learning_anygpu_20261001

**类别／状态：**失败与退役记录；superseded_by_verified_split_run。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_oracle_learning_anygpu_20261001/status.json)。

协议字段：`train_count=128; epochs=3; max_anchors=1536; seed=300931`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已被校验一致的分卡split版本主动替代，非数值失败。

**原目录／权重入口：**[eval_results/standalone_oracle_learning_anygpu_20261001](eval_results/standalone_oracle_learning_anygpu_20261001)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [preregistration.json](eval_results/standalone_oracle_learning_anygpu_20261001/preregistration.json)。

<a id="t068"></a>

### T068 · 退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; q_topk=256; output_rank=512; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧准备／资源故障；成功冻结版本v5，部分旧目录含权重，全部保留。

**原目录／权重入口：**[eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [experiment.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002/experiment.json)；[Parallel_T0/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002/Parallel_T0/preregistration.json)；[Parallel_T1/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002/Parallel_T1/preregistration.json)。

<a id="t069"></a>

### T069 · 退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; q_topk=256; output_rank=512; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧准备／资源故障；成功冻结版本v5，部分旧目录含权重，全部保留。

**原目录／权重入口：**[eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [experiment.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2/experiment.json)；[Parallel_T0/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2/Parallel_T0/preregistration.json)；[Parallel_T1/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v2/Parallel_T1/preregistration.json)。

<a id="t070"></a>

### T070 · 退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; q_topk=256; output_rank=512; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧准备／资源故障；成功冻结版本v5，部分旧目录含权重，全部保留。

**原目录／权重入口：**[eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3/experiment.json)；[Parallel_T0/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3/Parallel_T0/preregistration.json)；[Parallel_T1/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v3/Parallel_T1/preregistration.json)。

<a id="t071"></a>

### T071 · 退役记录 · standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4

**类别／状态：**失败与退役记录；failed。

**做什么：**记录失败阶段、恢复来源，防止重复启动。

**对比什么：**不作科学方法比较。

**实验设置：**历史配置、错误日志及资产见原目录；不得重启旧控制器。

**即时日志／进度：** [status.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4/status.json)。

协议字段：`train_count=128; epochs=3; steps_per_new_arm=384; objective=OUTPUT,OUTPUT final-distribution prefix overlap ONLY; q_topk=256; output_rank=512; training_epochs=3; max_anchors=1536; head_chunk_anchors=32; seed=300931; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧准备／资源故障；成功冻结版本v5，部分旧目录含权重，全部保留。

**原目录／权重入口：**[eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[experiment.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4/experiment.json)；[Parallel_T0/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4/Parallel_T0/preregistration.json)；[Parallel_T1/preregistration.json](eval_results/standalone_parallel_context_onpolicy_T0_T1_short_gpu0_eval04_20261002_v4/Parallel_T1/preregistration.json)。

## 自动维护与清理记录

本文件由 `residualV3_main_v1/maintenance/experiment_ledger/ledger.py` 维护；独立CPU读进程每30秒检查一次结果文件，不修改冻结训练／评测源码，不发起新训练。
新增实验应先用该脚本 `register` 登记目的、对照、设置和结论；新目录也会自动出现，缺少说明时明确标注待补充。人工解释写入 registry.json，避免下次更新被覆盖。

刷新：`python residualV3_main_v1/maintenance/experiment_ledger/ledger.py refresh`。

2026-10-09清理清单、代码归档、权重前后核验保存在 `residualV3_main_v1/maintenance/experiment_ledger/cleanup/`；没有把负结果当作无用结果删除。

- 2026-10-12仅整理目录，无Thinking新训练/评测；全部322个正式测量窗及原始解释保留。
- 当前入口RESIDUALV3_START_HERE.txt；旧研究代码/诊断已校验归档，恢复索引archives/residualV3_research_cleanup_20261012_v4/INDEX.csv。完整整理前总账及registry在本轮maintenance/before。
