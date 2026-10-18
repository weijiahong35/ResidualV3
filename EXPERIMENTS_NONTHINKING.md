# Non-thinking 实验总账

更新：2026-10-13 23:39:23 UTC。由原始结果自动生成，叙述来自逐实验登记。共75条记录。

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
| N001 | [全数据一轮：单训DFlash与联合训练](#n001) | 验证全数据训练主体与主体+修复头的效果。 | complete；已记录16个正式窗口 |
| N002 | [最新融合：block12／8／16两层](#n002) | 检查缩短草稿与推理跳第三层是否更划算。 | complete；已记录28个正式窗口 |
| N003 | [块内尾部停修：前12／前10](#n003) | 先还原完整修复的逐块1–16长度分布、到达概率和数学期望，再检查固定前12／前10停修的代价。 | failed；已记录6个正式窗口；C40 smoke校验失败，C1成绩仍保留；两层C40在最新block/depth实验已完成。 |
| N004 | [q-only mismatch可预测性重新验证](#n004) | 重新定义当前模型的mismatch并检查q是否能识别风险／修复价值。 | 已归档 |
| N005 | [原20k逐条训练配方](#n005) | 建立新non-thinking数据下的三层标准头。 | failed |
| N006 | [20k打包65536／1536短训](#n006) | 验证打包降低anchor总预算后的训练效率与接受效果。 | complete；已记录12个正式窗口 |
| N007 | [OUTPUT／CE+D-PACE／三者融合](#n007) | 检查greedy专门监督、接受代理及融合实现。 | complete；已记录16个正式窗口 |
| N008 | [OUTPUT前缀平滑短训](#n008) | 检验D-PACE与OUTPUT位置权重联动／平滑。 | complete；已记录12个正式窗口 |
| N009 | [严格对齐：只训头 vs 联合训练](#n009) | 修复此前不同训练预算导致不可比的问题。 | complete；已记录8个正式窗口 |
| N010 | [修复前q vs 修复后q：尾部停修信号配对](#n010) | 检验PRE先看再修和POST先修再看哪种信号更能预测修复后残余错误及尾部接受机会。 | 已归档 |
| N011 | [q尾部停修设计：双阈值与额外2/4位置窗口](#n011) | 根据用户要求，在一次mismatch预测后保留数值和位置缓冲，按接受价值设计尾部停修。 | design_complete_not_run |
| N012 | [模块参数／时间／高层使用审计](#n012) | 核实4.35M头的时间花在哪，高层是否参与预测。 | complete；已记录16个正式窗口 |
| N013 | [置信度输入配对消融：q／DFlash hidden／实际前驱](#n013) | 检验哪些推理时可得信息能够预测修复后的实际首拒绝及前缀存活，为尾部停修选择信号。 | 已归档 |
| N014 | [C40 greedy文本分叉溯源](#n014) | 核查732/1319跨运行文本差异是否由头优化引入。 | 已归档 |
| N015 | [优化头 vs 单训DFlash正式对照](#n015) | 衡量优化实现对总体AL/TPS的实际价值。 | native_complete；已记录8个正式窗口 |
| N016 | [三层SwiGLU后共享latent预测头：冻结特征诊断](#n016) | 验证修复网络完成融合后的表示是否有额外拒绝与存活预测信号，作为修复价值研究的第一步。 | 已归档 |
| N017 | [优化头正式C1恢复测量](#n017) | 补齐上一正式对照C1阶段，复用已完成权重。 | complete；已记录1个正式窗口 |
| N018 | [停修并截短target验证前缀：v1](#n018) | 将少算repair与少算target分开，检验端到端截短收益。 | 已归档 |
| N019 | [停修并截短target验证前缀：v2](#n019) | 将少算repair与少算target分开，检验端到端截短收益。 | failed；已记录14个正式窗口 |
| N020 | [target前缀截短：v3仅补C40 VERIFY_STOP](#n020) | 修复正式模式Graph输出引用的批量绑定，补齐原实验缺失窗口。 | not_started_user_hold；已记录14个正式窗口 |
| N021 | [Small tile头正式测评](#n021) | 减少低并发小矩阵计算／启动开销。 | native_complete；已记录8个正式窗口 |
| N022 | [结构融合优化 v2正式测评](#n022) | 合并前驱投影、融合及SwiGLU的冗余访存／启动。 | 正式窗口已完成（顶层历史状态：native_evaluation）；已记录8个正式窗口 |
| N023 | [原实现block12／8正式测试](#n023) | 检查较短草稿对AL/TPS的影响。 | complete；已记录32个正式窗口 |
| N024 | [结构融合正式尝试 v1](#n024) | 检查早期融合实现的native行为和性能。 | 已归档 |
| N025 | [1536 anchors整块显存探针](#n025) | 比较chunk64与整1536的内存／前后向时间。 | 已归档 |
| N026 | [修复头前后向及融合loss](#n026) | 校验融合TV与15位置并行的数值／梯度。 | complete |
| N027 | [Teacher1／2与头并行初测](#n027) | 实测hidden供给及头计算耗时。 | complete |
| N028 | [Teacher1／2联合训练计时](#n028) | 实测同时训练DFlash主体+头时供数影响。 | complete |
| N029 | [共享teacher同时供两个learner v1](#n029) | 一份hidden供独立主体与主体+头更新。 | complete |
| N030 | [共享teacher供两端 v2](#n030) | 修复供数／队列安排后复测。 | complete |
| N031 | [传输／并发优化 v3](#n031) | 降低hidden读取和CPU传输开销。 | complete_with_failed_additional_attempt |
| N032 | [固定内存注册／共享供数 v4](#n032) | 避免每包pin_memory大复制。 | complete |
| N033 | [Hidden供给专项计时](#n033) | 测单／双teacher及供数瓶颈。 | complete |
| N034 | [用显存换训练时间](#n034) | 检查activation/checkpoint等实现取舍。 | complete |
| N035 | [DP2／工作区／流水线训练吞吐](#n035) | 检查teacher供给、读取线程及训练pipeline。 | 未提供顶层状态 |
| N036 | [Target／body／head整轮耗时](#n036) | 区分验证、主体、头与草稿总体成本。 | 已归档 |
| N037 | [Small tile算子profile](#n037) | 检查C1/C40小矩阵与采样成本。 | 已归档 |
| N038 | [Small tile同状态校验](#n038) | 校验不同tile前向／采样是否一致。 | 已归档 |
| N039 | [结构融合微探针 v1](#n039) | 定位前驱/fusion/SwiGLU融合的数值差异。 | 已归档 |
| N040 | [结构融合微探针 v2](#n040) | 复核融合版本及完整头时间。 | 已归档 |
| N041 | [硬前驱投影查表](#n041) | 减少逐token大维度硬前驱编码。 | native_complete；已记录8个正式窗口 |
| N042 | [前驱＋SwiGLU融合smoke](#n042) | 降低硬前驱及逐位置启动成本。 | 已归档 |
| N043 | [候选logit写回融合](#n043) | 减少全词表scatter与重复写回。 | complete；已记录8个正式窗口 |
| N044 | [完整头低延迟初试（未采纳）](#n044) | 压低完整形状头耗时且保持bitwise。 | 已归档 |
| N045 | [修复后q风险驱动的一位置延迟尾部停修](#n045) | 判断已修复前缀的拒绝风险，与下一位置修复流水线重叠，减少无用尾部修复。 | complete；已记录16个正式窗口 |
| N046 | [q45辅助训练的初始过渡记录](#n046) | 保留被epoch2 q45/latent联合训练替代的准备记录。 | 已归档 |
| N047 | [epoch1与epoch2：单训DFlash及DFlash＋三层修复头正式对比](#n047) | 比较增加一轮训练对单训DFlash和联合修复模型的接收长度与端到端吞吐的影响 | complete；已记录16个正式窗口 |
| N048 | [E2代码与对话：DSpark六套数据C1/C32重测](#n048) | 按用户要求补齐DSpark代码与对话的六套评测；只使用E2独立与联合三层权重。 | complete；已记录48个正式窗口 |
| N049 | [E2前两层推理 · 三领域C1/C32](#n049) | 验证固定E2三层权重在推理省去第三层后的同窗AC/TPS。 | incomplete_failed_humaneval_groups_no_active_workers；已记录16个正式窗口 |
| N050 | [E2三领域C1/C32 · 三层头与仅推理前两层直接对照](#n050) | 检查省去第三层后接受长度与原生吞吐的取舍。 | complete；已记录24个正式窗口 |
| N051 | [E2三层 C1/C40 · 固定旧正式显存的模块耗时](#n051) | 测量target验证、drafter主体、完整头以及头内模块的真实边界与耗时。 | acquisition_complete_with_timing_coverage_limitations |
| N052 | [E2 HumanEval/MT-Bench：独立DFlash与联合完整修复头](#n052) | 验证代码与聊天领域的native接受长度及端到端吞吐收益。 | complete；已记录16个正式窗口 |
| N053 | [新FULL15_NOFILL与E2独立单训DFlash正式长测](#n053) | 直接测量已优化FULL15联合模型对匹配epoch独立单训DFlash的接收长度及端到端吞吐。 | complete；已记录8个正式窗口 |
| N054 | [第二轮尾部截断实现优化：精确q45统计、设备端跳过及细化target图](#n054) | 修正截断未充分减少实际计算的问题，固定训练模型、预测器和阈值验证端到端吞吐 | complete；已记录16个正式窗口 |
| N055 | [尾部截断短测迭代与CPU同步诊断](#n055) | 修复尾部截断吞吐下降超过接受长度下降的问题。 | structural_reaudit_complete_no_stable_tail_cut_gain |
| N056 | [E2三领域C32：独立DFlash vs 完整NOFILL头](#n056) | 按用户新并发口径补测；保留历史C40。 | complete；已记录12个正式窗口 |
| N057 | [第二轮已训练q45/末层预测头的自适应尾部截断](#n057) | 验证第二轮内嵌训练的预测头能否在保留接收收益时减少repair和target验证工作，并获得真实端到端吞吐提升 | complete；已记录16个正式窗口 |
| N058 | [在线两种非线性辅助读出：v3计算开销校验未通过](#n058) | 在第二轮长训练内公平比较q45非线性读出与q45+repair末层256维非线性读出；辅助梯度完全隔离。 | paused |
| N059 | [第二轮内嵌两种非线性q预测器：现场校验通过、继续训练](#n059) | 比较只给修复后q45与再给三层repair末层256维表示，是否提高独立置信度读出；直接嵌入当前长训练。 | complete |
| N060 | [第二轮q45独立预测器：初次现场校验OOM（已保留并续至v2）](#n060) | 给正在进行的联合长训练添加repair后q45独立线性预测器，核验速度和主梯度隔离。 | paused |
| N061 | [第二轮续训：repair后q45独立预测器，主梯度逐位一致、计算开销+3.39%](#n061) | 测量45维修复后q统计的独立预测器能否低开销加入第二轮长训练；通过后继续。 | paused |
| N062 | [全数据第二轮：共享teacher续训单训DFlash与主体+头](#n062) | 从第一轮完成后的两组完整断点各追加全数据一轮；保存可随时暂停恢复的训练状态，并记录后续基线可对齐的训练预算。 | paused |
| N063 | [每领域一万条 · 冻结E2主体的mismatch专家头（三种子）](#n063) | 验证可见mismatch线索的修复分工是否超过通用头、随机分工和同总参数对照。 | requested_comparison_finished_timing_acquired_with_coverage_gap_baseline_incomplete |
| N064 | [mismatch专家/可见router小规模预实验](#n064) | 检验冻结E2主体上的修复动作互补性与可见输入选择价值 | complete_offline_pilot |
| N065 | [退役／维护 · full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1](#n065) | 保留准备阶段与替代关系，避免误启动。 | paused |
| N066 | [退役／维护 · full_nonthinking_packed65536_a1536_1epoch_2gpu_20261006_v2](#n066) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N067 | [退役／维护 · fused_output_dpace_20k_packed1536_2gpu_20261007_v1](#n067) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N068 | [退役／维护 · joint_dflash_head_card_timing_20261007_v1](#n068) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N069 | [退役／维护 · nosoft3_20k_3epoch_packed65536_a1536_20261006_v1](#n069) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N070 | [退役／维护 · nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2](#n070) | 保留准备阶段与替代关系，避免误启动。 | failed |
| N071 | [退役／维护 · q_only_skip_probe_20261009_v1](#n071) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N072 | [退役／维护 · q_only_skip_probe_20261009_v2](#n072) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N073 | [退役／维护 · q_only_skip_probe_20261009_v3](#n073) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N074 | [退役／维护 · storage_cleanup_20261008_v1](#n074) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |
| N075 | [退役／维护 · teacher1_vs2_activation_timing_20261008_v1](#n075) | 保留准备阶段与替代关系，避免误启动。 | 已归档 |

## 块内尾部停修：先看真实分布

完整修复的1–16块数量和加权期望，见最新 block/depth 与尾部停修条目中的分布表。两层 C40 的正式 AL/TPS 也在最新 block/depth 条目；它是三层权重跳过第三层的推理消融，未经两层重训。

前10／前12固定停修仅保留为对照；不能仅凭平均 AL 选择停止点。现有原始 native 计数已能还原每种块长的数量，先据此评估尾部机会，再考虑条件预测和成本。


## 实验记录

<a id="n001"></a>

### N001 · 全数据一轮：单训DFlash与联合训练

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**验证全数据训练主体与主体+修复头的效果。

**对比什么：**同预训练起点的独立单训主体、联合主体关头、联合主体开头；RAW是预训练DFlash。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 一张teacher在线获取hidden并共享给两张独立learner；全数据833176条、15847包；主体5层，修复头3层。联合损失为raw CE+D-PACE、repaired CE+D-PACE、repaired OUTPUT。GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/dflash_training.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 0 | 1 | DFLASH | 7.421221 | 466.901 | +0.00% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 0 | 1 | JOINT_BODY | 7.347063 | 463.465 | -1.00% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 0 | 1 | JOINT_HEAD | 8.414318 | 426.567 | +13.38% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 0 | 1 | RAW | 6.737518 | 428.965 | -9.21% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 1 | 1 | DFLASH | 6.266453 | 401.378 | +0.00% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 1 | 1 | JOINT_BODY | 6.241176 | 398.684 | -0.40% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 1 | 1 | JOINT_HEAD | 7.355848 | 377.268 | +17.38% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal | 1 | 1 | 1 | RAW | 5.741420 | 370.538 | -8.38% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 0 | 1 | DFLASH | 7.590782 | 4150.044 | +0.00% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 0 | 1 | JOINT_BODY | 7.513410 | 4109.612 | -1.02% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 0 | 1 | JOINT_HEAD | 8.545277 | 4136.474 | +12.57% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 0 | 1 | RAW | 6.857507 | 3794.485 | -9.66% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 1 | 1 | DFLASH | 6.374347 | 3494.770 | +0.00% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 1 | 1 | JOINT_BODY | 6.441937 | 3533.434 | +1.06% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 1 | 1 | JOINT_HEAD | 7.651468 | 3704.002 | +20.04% | — | 未核验／只看AL |
| evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal | 40 | 1 | 1 | RAW | 5.960614 | 3289.504 | -6.49% | — | 未核验／只看AL |

**结果解释：**两条一轮长训已完成；GSM8K正式评测见本表。不同主体权重使该比较包含训练方式差异，不能解释成同主体纯加头收益。

诊断／原报告摘录（保持原口径）：

- 单seed/单轮数学基准评测；更换面板，不能直接和旧短训数字比较长短训练收益；吞吐不作结论。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal/summary.json](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/evaluation_dspark_gsm8k_recovery_20261009_v2/C1/formal/summary.json)；[evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal/summary.json](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/evaluation_dspark_gsm8k_recovery_20261009_v2/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[evaluation_dspark_gsm8k_recovery_20261009_v2/RESULTS_CN.txt](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/evaluation_dspark_gsm8k_recovery_20261009_v2/RESULTS_CN.txt)。

<a id="n002"></a>

### N002 · 最新融合：block12／8／16两层

**类别／状态：**研究实验；complete；已记录28个正式窗口。

**做什么：**检查缩短草稿与推理跳第三层是否更划算。

**对比什么：**每个block内比较单训DFLASH、HEAD3；block16另测HEAD2，同一三层权重跳第三层。

**实验设置：**no-soft三层、Top256、basis512；冻结EMBEDDING输入表；GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。 block12=11、block8=7新proposal；HEAD2未经两层重训；最新融合与统一batch-invariant/KV预算。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/status.json)。

正式窗口设置：C1=200条，proposal=11，max_new=2096; C1=200条，proposal=15，max_new=2096; C1=200条，proposal=7，max_new=2096; C40=1319条，proposal=11，max_new=2096; C40=1319条，proposal=15，max_new=2096; C40=1319条，proposal=7，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| BLOCK12/C1/formal | 1 | 0 | 1 | DFLASH | 6.818234 | 163.024 | +0.00% | +0.00% | 已核验 |
| BLOCK12/C1/formal | 1 | 0 | 1 | HEAD3 | 7.533568 | 170.312 | +10.49% | +4.47% | 已核验 |
| BLOCK12/C1/formal | 1 | 1 | 1 | DFLASH | 5.921768 | 142.185 | +0.00% | +0.00% | 已核验 |
| BLOCK12/C1/formal | 1 | 1 | 1 | HEAD3 | 6.686528 | 152.059 | +12.91% | +6.94% | 已核验 |
| BLOCK12/C40/formal | 40 | 0 | 1 | DFLASH | 6.941473 | 3364.192 | +0.00% | +0.00% | 已核验 |
| BLOCK12/C40/formal | 40 | 0 | 1 | HEAD3 | 7.610382 | 3485.609 | +9.64% | +3.61% | 已核验 |
| BLOCK12/C40/formal | 40 | 1 | 1 | DFLASH | 5.973799 | 2915.971 | +0.00% | +0.00% | 已核验 |
| BLOCK12/C40/formal | 40 | 1 | 1 | HEAD3 | 6.912210 | 3050.189 | +15.71% | +4.60% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 0 | 1 | DFLASH | 7.384246 | 179.959 | +0.00% | +0.00% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 0 | 1 | HEAD2 | 8.298933 | 192.081 | +12.39% | +6.74% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 0 | 1 | HEAD3 | 8.381226 | 190.369 | +13.50% | +5.78% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 1 | 1 | DFLASH | 6.331330 | 155.673 | +0.00% | +0.00% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 1 | 1 | HEAD2 | 7.413781 | 172.242 | +17.10% | +10.64% | 已核验 |
| BLOCK16_DEPTH2/C1/formal | 1 | 1 | 1 | HEAD3 | 7.429277 | 169.187 | +17.34% | +8.68% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 0 | 1 | DFLASH | 7.560736 | 3197.726 | +0.00% | +0.00% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 0 | 1 | HEAD2 | 8.416682 | 3302.630 | +11.32% | +3.28% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 0 | 1 | HEAD3 | 8.527382 | 3340.784 | +12.79% | +4.47% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 1 | 1 | DFLASH | 6.381850 | 2698.189 | +0.00% | +0.00% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 1 | 1 | HEAD2 | 7.510430 | 2972.950 | +17.68% | +10.18% | 已核验 |
| BLOCK16_DEPTH2/C40/formal | 40 | 1 | 1 | HEAD3 | 7.666585 | 3017.960 | +20.13% | +11.85% | 已核验 |
| BLOCK8/C1/formal | 1 | 0 | 1 | DFLASH | 5.658829 | 133.804 | +0.00% | — | 污染 |
| BLOCK8/C1/formal | 1 | 0 | 1 | HEAD3 | 6.032407 | 138.666 | +6.60% | — | 已核验 |
| BLOCK8/C1/formal | 1 | 1 | 1 | DFLASH | 5.110850 | 120.790 | +0.00% | — | 污染 |
| BLOCK8/C1/formal | 1 | 1 | 1 | HEAD3 | 5.618908 | 129.616 | +9.94% | — | 已核验 |
| BLOCK8/C40/formal | 40 | 0 | 1 | DFLASH | 5.743107 | 3448.152 | +0.00% | +0.00% | 已核验 |
| BLOCK8/C40/formal | 40 | 0 | 1 | HEAD3 | 6.075916 | 3495.369 | +5.79% | +1.37% | 已核验 |
| BLOCK8/C40/formal | 40 | 1 | 1 | DFLASH | 5.162586 | 2989.431 | +0.00% | +0.00% | 已核验 |
| BLOCK8/C40/formal | 40 | 1 | 1 | HEAD3 | 5.694125 | 3253.723 | +10.30% | +8.84% | 已核验 |

**真实块长分布：**下表由正式 native 逐位置接受计数还原，记录长度1至 block 大小的块数量；不是模拟或 teacher 前缀乘积。

`K` 是实际接受的草稿数（0–15）；本项目 AL 的长度 `L=K+1`（1–16）包含 bonus。句尾 EOS／输出上限裁剪后，实际返回 token 数可能更少。原证据只有汇总计数，没有逐块请求ID和时间顺序，不补造事件日志。

令 `D` 为块数、`S_j` 为接受至少 j 个草稿的块数：`n_l=S_(l-1)-S_l`，`S_0=D, S_16=0`；`E[L]=Σ(l·n_l)/D=1+Σ(S_j/D)`。已核验数量和为 D、接受总数一致、期望与原 AL 一致。

C1／BLOCK16_DEPTH2/C1/formal／轮1：

| native 块长 L | DFLASH T0 | DFLASH T1 | HEAD2 T0 | HEAD2 T1 | HEAD3 T0 | HEAD3 T1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 769 | 1170 | 638 | 984 | 633 | 969 |
| 2 | 975 | 1438 | 698 | 963 | 692 | 978 |
| 3 | 819 | 1144 | 649 | 788 | 603 | 809 |
| 4 | 700 | 931 | 536 | 657 | 533 | 637 |
| 5 | 577 | 696 | 419 | 589 | 412 | 581 |
| 6 | 503 | 559 | 401 | 504 | 387 | 446 |
| 7 | 431 | 469 | 358 | 369 | 345 | 402 |
| 8 | 373 | 372 | 329 | 399 | 335 | 372 |
| 9 | 307 | 364 | 293 | 318 | 303 | 305 |
| 10 | 285 | 294 | 256 | 236 | 248 | 295 |
| 11 | 271 | 230 | 252 | 274 | 249 | 263 |
| 12 | 243 | 239 | 223 | 253 | 227 | 237 |
| 13 | 222 | 186 | 231 | 219 | 197 | 210 |
| 14 | 185 | 192 | 165 | 197 | 174 | 167 |
| 15 | 197 | 159 | 148 | 175 | 153 | 172 |
| 16 | 1255 | 1040 | 1623 | 1420 | 1657 | 1457 |
| 总块数 D | 8112 | 9483 | 7219 | 8345 | 7148 | 8300 |
| 数学期望 E[L] | 7.384246 | 6.331330 | 8.298933 | 7.413781 | 8.381226 | 7.429277 |
| 长度16的比例 | 15.47% | 10.97% | 22.48% | 17.02% | 23.18% | 17.55% |

C40／BLOCK16_DEPTH2/C40/formal／轮1：

| native 块长 L | DFLASH T0 | DFLASH T1 | HEAD2 T0 | HEAD2 T1 | HEAD3 T0 | HEAD3 T1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5182 | 7938 | 4556 | 6295 | 4419 | 6089 |
| 2 | 5968 | 9357 | 4476 | 6117 | 4364 | 5769 |
| 3 | 5543 | 7863 | 4163 | 5311 | 4052 | 4897 |
| 4 | 4415 | 6166 | 3293 | 4233 | 3201 | 3984 |
| 5 | 3655 | 4787 | 2865 | 3737 | 2763 | 3467 |
| 6 | 3087 | 3761 | 2496 | 3015 | 2405 | 3039 |
| 7 | 2731 | 3113 | 2239 | 2583 | 2200 | 2631 |
| 8 | 2344 | 2651 | 2133 | 2414 | 2053 | 2252 |
| 9 | 2084 | 2158 | 1966 | 2126 | 1869 | 2020 |
| 10 | 1836 | 1864 | 1601 | 1840 | 1634 | 1795 |
| 11 | 1728 | 1634 | 1546 | 1748 | 1514 | 1705 |
| 12 | 1534 | 1497 | 1405 | 1458 | 1364 | 1494 |
| 13 | 1430 | 1449 | 1361 | 1378 | 1374 | 1331 |
| 14 | 1304 | 1295 | 1178 | 1200 | 1180 | 1236 |
| 15 | 1293 | 1136 | 1171 | 1163 | 1199 | 1153 |
| 16 | 9105 | 7199 | 11374 | 9791 | 11612 | 10036 |
| 总块数 D | 53239 | 63868 | 47823 | 54409 | 47203 | 52898 |
| 数学期望 E[L] | 7.560736 | 6.381850 | 8.416682 | 7.510430 | 8.527382 | 7.666585 |
| 长度16的比例 | 17.10% | 11.27% | 23.78% | 18.00% | 24.60% | 18.97% |

**完整修复下的尾部机会：**先观察原完整修复路径上的真实到达和接受，再选择候选停止点。平均块长约8不表示每块都在第8位置拒绝。下表不是停修后的因果损失：停修以后 raw q 仍会提出 token，路径也可能变化。

| C | T | 完整修复对照 | 停修位置 c | 到达第c+1草稿 | 第c+1草稿被接受 | 尾部接受数期望 E[(K−c)+] |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | HEAD3／BLOCK16_DEPTH2/C1/formal | 10 | 37.17% | 33.69% | 1.404589 |
| 1 | 0 | HEAD3／BLOCK16_DEPTH2/C1/formal | 12 | 30.51% | 27.76% | 0.762591 |
| 1 | 1 | HEAD3／BLOCK16_DEPTH2/C1/formal | 10 | 30.19% | 27.02% | 1.100120 |
| 1 | 1 | HEAD3／BLOCK16_DEPTH2/C1/formal | 12 | 24.17% | 21.64% | 0.588193 |
| 40 | 0 | HEAD3／BLOCK16_DEPTH2/C40/formal | 10 | 38.65% | 35.44% | 1.493719 |
| 40 | 0 | HEAD3／BLOCK16_DEPTH2/C40/formal | 12 | 32.55% | 29.64% | 0.813804 |
| 40 | 1 | HEAD3／BLOCK16_DEPTH2/C40/formal | 10 | 32.05% | 28.83% | 1.184468 |
| 40 | 1 | HEAD3／BLOCK16_DEPTH2/C40/formal | 12 | 26.00% | 23.49% | 0.636130 |

分布和期望用于选择预实验候选点；最终是否停修应比较节省的头耗时与正式 rollout 的 AL/TPS 损失。q自适应停止需要额外的当前前缀条件预测，不能用全局平均长度代替。

**结果解释：**6组28正式窗口完成。block8/C1的DFLASH污染，不下TPS结论。block16/HEAD2在C1 TPS更好而C40较HEAD3低，不能宣布两层全面胜出。

诊断／原报告摘录（保持原口径）：

- 尚未完成的正式窗口不写结果；smoke不是测速。外部进程出现的窗口只标记，TPS比较留空。
- 块长度改变接受长度上限。HEAD2是已训练三层头的推理跳层消融，不能作为从头训练两层的结论。
- 本面板已用于开发；单轮结果不作为独立最终测试或多轮稳定性证据。

**原目录／权重入口：**[residualV3-nonthing/results/native_fused_block_depth_20261009_v1](residualV3-nonthing/results/native_fused_block_depth_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [BLOCK12/C1/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK12/C1/formal/summary.json)；[BLOCK12/C40/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK12/C40/formal/summary.json)；[BLOCK16_DEPTH2/C1/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK16_DEPTH2/C1/formal/summary.json)；[BLOCK16_DEPTH2/C40/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK16_DEPTH2/C40/formal/summary.json)；[BLOCK8/C1/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK8/C1/formal/summary.json)；[BLOCK8/C40/formal/summary.json](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/BLOCK8/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[RESULTS_CN.txt](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/RESULTS_CN.txt)。

<a id="n003"></a>

### N003 · 块内尾部停修：前12／前10

**类别／状态：**研究实验；failed；已记录6个正式窗口；C40 smoke校验失败，C1成绩仍保留；两层C40在最新block/depth实验已完成。

**做什么：**先还原完整修复的逐块1–16长度分布、到达概率和数学期望，再检查固定前12／前10停修的代价。

**对比什么：**FULL15、PREFIX12、PREFIX10，完全相同联合主体/三层头/basis；不重训。

**实验设置：**no-soft三层、Top256、basis512；冻结EMBEDDING输入表；GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。cut以新proposal位置计；修前12或10，其余3或5位置不执行修复头；实际前驱只用修正后的采样token；24组合成状态校验通过，不能扩张为所有真实批大小等价；C40真实状态后来失败。本次事后补真实频数，不宣称原10/12切点已按分布预先选择。

**即时日志／进度：** [status.json](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/single_card_C1_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

正式窗口设置：C1=200条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 FULL15。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | FULL15 | 8.381226 | 190.143 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | PREFIX10 | 8.122017 | 186.912 | -3.09% | -1.70% | 已核验 |
| C1/formal | 1 | 0 | 1 | PREFIX12 | 8.220499 | 188.336 | -1.92% | -0.95% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL15 | 7.429277 | 169.092 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | PREFIX10 | 7.142808 | 165.163 | -3.86% | -2.32% | 已核验 |
| C1/formal | 1 | 1 | 1 | PREFIX12 | 7.329147 | 168.423 | -1.35% | -0.40% | 已核验 |

**真实块长分布：**下表由正式 native 逐位置接受计数还原，记录长度1至 block 大小的块数量；不是模拟或 teacher 前缀乘积。

`K` 是实际接受的草稿数（0–15）；本项目 AL 的长度 `L=K+1`（1–16）包含 bonus。句尾 EOS／输出上限裁剪后，实际返回 token 数可能更少。原证据只有汇总计数，没有逐块请求ID和时间顺序，不补造事件日志。

令 `D` 为块数、`S_j` 为接受至少 j 个草稿的块数：`n_l=S_(l-1)-S_l`，`S_0=D, S_16=0`；`E[L]=Σ(l·n_l)/D=1+Σ(S_j/D)`。已核验数量和为 D、接受总数一致、期望与原 AL 一致。

C1／C1/formal／轮1：

| native 块长 L | FULL15 T0 | FULL15 T1 | PREFIX10 T0 | PREFIX10 T1 | PREFIX12 T0 | PREFIX12 T1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 633 | 969 | 619 | 1019 | 639 | 995 |
| 2 | 692 | 978 | 731 | 1002 | 709 | 957 |
| 3 | 603 | 809 | 613 | 807 | 612 | 787 |
| 4 | 533 | 637 | 531 | 696 | 515 | 670 |
| 5 | 412 | 581 | 434 | 594 | 432 | 592 |
| 6 | 387 | 446 | 397 | 490 | 406 | 452 |
| 7 | 345 | 402 | 354 | 405 | 351 | 394 |
| 8 | 335 | 372 | 357 | 403 | 355 | 378 |
| 9 | 303 | 305 | 302 | 333 | 307 | 320 |
| 10 | 248 | 295 | 261 | 308 | 245 | 298 |
| 11 | 249 | 263 | 649 | 726 | 255 | 272 |
| 12 | 227 | 237 | 280 | 314 | 232 | 246 |
| 13 | 197 | 210 | 260 | 264 | 567 | 622 |
| 14 | 174 | 167 | 232 | 187 | 261 | 280 |
| 15 | 153 | 172 | 173 | 155 | 194 | 188 |
| 16 | 1657 | 1457 | 1183 | 973 | 1208 | 989 |
| 总块数 D | 7148 | 8300 | 7376 | 8676 | 7288 | 8440 |
| 数学期望 E[L] | 8.381226 | 7.429277 | 8.122017 | 7.142808 | 8.220499 | 7.329147 |
| 长度16的比例 | 23.18% | 17.55% | 16.04% | 11.21% | 16.58% | 11.72% |

**完整修复下的尾部机会：**先观察原完整修复路径上的真实到达和接受，再选择候选停止点。平均块长约8不表示每块都在第8位置拒绝。下表不是停修后的因果损失：停修以后 raw q 仍会提出 token，路径也可能变化。

| C | T | 完整修复对照 | 停修位置 c | 到达第c+1草稿 | 第c+1草稿被接受 | 尾部接受数期望 E[(K−c)+] |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 0 | FULL15／C1/formal | 10 | 37.17% | 33.69% | 1.404589 |
| 1 | 0 | FULL15／C1/formal | 12 | 30.51% | 27.76% | 0.762591 |
| 1 | 1 | FULL15／C1/formal | 10 | 30.19% | 27.02% | 1.100120 |
| 1 | 1 | FULL15／C1/formal | 12 | 24.17% | 21.64% | 0.588193 |

分布和期望用于选择预实验候选点；最终是否停修应比较节省的头耗时与正式 rollout 的 AL/TPS 损失。q自适应停止需要额外的当前前缀条件预测，不能用全局平均长度代替。

**结果解释：**C1六个正式窗口完成。原固定停止点没有先按真实长度分布选择，只能作为固定切点对照，不能由平均AL约8推断尾部无价值。C40 smoke在08:50失败：PREFIX10批18的真实状态graph/eager logits及verifier q逐位校验不一致，采样ID和实际前驱相同；C40没有正式停修成绩。未跳过校验或覆盖失败证据。完整修复C40分布可查看同权重的最新block16/depth条目，来源独立标注。

诊断／原报告摘录（保持原口径）：

- 微基准完整头+15proposal采样（ms/块，非服务TPS）：N1/T0 FULL15 2.2498、前12 1.7943、前10 1.4482；N1/T1 2.0761/1.7270/1.4842。
- N40/T0 6.4434/5.8731/5.0491；N40/T1 7.2227/6.6159/5.7741；合成状态双seed，不能代替正式AL。

**原目录／权重入口：**[residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/C1/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[RESULTS_CN.txt](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/RESULTS_CN.txt)；[preregistration.json](residualV3-nonthing/results/repair_prefix_cutoff_20261009_v1/preregistration.json)。

<a id="n004"></a>

### N004 · q-only mismatch可预测性重新验证

**类别／状态：**研究实验；已归档。

**做什么：**重新定义当前模型的mismatch并检查q是否能识别风险／修复价值。

**对比什么：**q_max、entropy、margin等特征，MISMATCH和BENEFICIAL标签；校准集选阈值、提示留出检查。

**实验设置：**128提示、T0/T1共256请求、1536块、23040状态、10134到达状态；所有状态向量只存RAM，结束释放；没有部署q-gate正式AL/TPS。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**q_max对T0 mismatch AUROC约0.8354，T1 TV≥0.1约0.9416；可预测风险不等于该位置值得修。原whole-block跳过0%不能回答用户的尾部停修，应由前12/10实验验证。

诊断／原报告摘录（保持原口径）：

- 研究留出到达状态：T0 MISMATCH AUROC=0.8354248、AP=0.5867785；T1 TV≥0.1 AUROC≈0.9416、AP≈0.9459。
- T0 BENEFICIAL AUROC≈0.7876，T1≈0.8681；AUROC与AP只评价分类，不构成部署提速证明。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n005"></a>

### N005 · 原20k逐条训练配方

**类别／状态：**研究实验；failed。

**做什么：**建立新non-thinking数据下的三层标准头。

**对比什么：**预训练RAW和冻结主体仅训练修复头；与后续packed配方比较需核对实际训练预算。

**实验设置：**20k×3轮；no-soft3/Top256/basis512/OUTPUT；target/draft GPU0共卡，prompt+answer≤4096，block16；逐条anchor预算。

**即时日志／进度：** [status.json](residualV3-nonthing/results/nosoft3_20k_3epoch_block16_gpu0_20261006_v1/status.json)；[training.log](residualV3-nonthing/results/nosoft3_20k_3epoch_block16_gpu0_20261006_v1/training.log)。

协议字段：`train_count=20000; epochs=3; objective=OUTPUT teacher-prefix cumulative overlap proxy; max_anchors=1536; block_size=16; native_max_tokens=2096; native_max_model_len=4096; stage1_lr=0.0003; dose_lr=0.0003; weight_decay=0.01; warmup_fraction=0.03`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**原每条最多1536 anchors，预算远大于每包1536；含已完成epoch权重及恢复来源。顶层failed不能当方法无效。

**原目录／权重入口：**[residualV3-nonthing/results/nosoft3_20k_3epoch_block16_gpu0_20261006_v1](residualV3-nonthing/results/nosoft3_20k_3epoch_block16_gpu0_20261006_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[preregistration.json](residualV3-nonthing/results/nosoft3_20k_3epoch_block16_gpu0_20261006_v1/preregistration.json)。

<a id="n006"></a>

### N006 · 20k打包65536／1536短训

**类别／状态：**研究实验；complete；已记录12个正式窗口。

**做什么：**验证打包降低anchor总预算后的训练效率与接受效果。

**对比什么：**冻结主体只训头与预训练RAW；不能称主体+头联合训练。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 20k×3轮；OUTPUT，冻结DFlash只训练修复头；teacher/learner分卡、错误恢复与优化版本保留。

**即时日志／进度：** [status.json](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/status.json)；[training.log](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/training.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=2000条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | N_packed65536_a1536_20k | 3.995837 | 215.872 | +5.64% | — | 污染 |
| C1/formal | 1 | 0 | 1 | N_per_sequence_20k | 4.042507 | 218.988 | +6.87% | — | 污染 |
| C1/formal | 1 | 0 | 1 | RAW | 3.782508 | 253.085 | +0.00% | — | 污染 |
| C1/formal | 1 | 1 | 1 | N_packed65536_a1536_20k | 3.619443 | 195.642 | +11.17% | — | 污染 |
| C1/formal | 1 | 1 | 1 | N_per_sequence_20k | 3.622723 | 196.117 | +11.27% | — | 污染 |
| C1/formal | 1 | 1 | 1 | RAW | 3.255867 | 218.085 | +0.00% | — | 污染 |
| C40/formal | 40 | 0 | 1 | N_packed65536_a1536_20k | 4.088444 | 2022.920 | +4.82% | -5.55% | 已核验 |
| C40/formal | 40 | 0 | 1 | N_per_sequence_20k | 4.130342 | 2041.566 | +5.89% | -4.68% | 已核验 |
| C40/formal | 40 | 0 | 1 | RAW | 3.900470 | 2141.706 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_packed65536_a1536_20k | 3.664743 | 1795.517 | +11.80% | +0.73% | 已核验 |
| C40/formal | 40 | 1 | 1 | N_per_sequence_20k | 3.730786 | 1817.028 | +13.82% | +1.94% | 已核验 |
| C40/formal | 40 | 1 | 1 | RAW | 3.277807 | 1782.460 | +0.00% | +0.00% | 已核验 |

**结果解释：**打包训练完成；raw/修复正式接受与吞吐见表。旧逐条配方更新数／anchor总量不同，不是只变batch大小的单因素。

诊断／原报告摘录（保持原口径）：

- 采样预算、样本权重和更新频率同时变化，不能称纯打包工程消融。
- 其他进程污染的窗口有明确标记；smoke不计作性能结果。开发面板不等于独立最终测试。
- 结论范围：两卡供数/计算的流水线未完全重叠，并有learner GPU3共卡
- 竞争。现有证据不足以将全部耗时归因于vLLM或DFlash单一模块。
- KV固定12GiB。采样45秒，SM平均46.04%，有12个1秒样本为0%。
- 2. GPU3有别人的PID2975435；45秒中该进程可读取SM样本的均值
- 不能把缺失值直接当作精确0，也不能把整卡99%全归到learner。
- 3. 最近20次更新中位9.78秒/包，训练输入处理约6655token/s。

**原目录／权重入口：**[residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[RESULTS_CN.txt](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/RESULTS_CN.txt)；[recovery_20261007_v1/performance_diagnostic_20261007_v1/FINDINGS_CN.txt](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v3/recovery_20261007_v1/performance_diagnostic_20261007_v1/FINDINGS_CN.txt)。

<a id="n007"></a>

### N007 · OUTPUT／CE+D-PACE／三者融合

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**检查greedy专门监督、接受代理及融合实现。

**对比什么：**OUTPUT、CE_DPACE、CE_DPACE_OUTPUT；训练头的配方消融。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 20k×3轮短训练，冻结主体；BF16大张量、融合TV内核FP32归约；teacher DP2与流水线优化有多个恢复版本。

**即时日志／进度：** [status.json](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/single_card_C1_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=2000条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | CE_DPACE | 3.972351 | 215.224 | +5.02% | -15.98% | 已核验 |
| C1/formal | 1 | 0 | 1 | CE_DPACE_OUTPUT | 4.086954 | 221.047 | +8.05% | -13.70% | 已核验 |
| C1/formal | 1 | 0 | 1 | OUTPUT | 3.995837 | 217.432 | +5.64% | -15.11% | 已核验 |
| C1/formal | 1 | 0 | 1 | RAW | 3.782508 | 256.146 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | CE_DPACE | 3.455282 | 187.102 | +6.12% | -14.46% | 已核验 |
| C1/formal | 1 | 1 | 1 | CE_DPACE_OUTPUT | 3.580175 | 193.873 | +9.96% | -11.36% | 已核验 |
| C1/formal | 1 | 1 | 1 | OUTPUT | 3.619443 | 196.672 | +11.17% | -10.08% | 已核验 |
| C1/formal | 1 | 1 | 1 | RAW | 3.255867 | 218.719 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 0 | 1 | CE_DPACE | 4.064218 | 1981.666 | +4.89% | -5.10% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 0 | 1 | CE_DPACE_OUTPUT | 4.173263 | 2029.271 | +7.71% | -2.82% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 0 | 1 | OUTPUT | 4.088470 | 1994.712 | +5.52% | -4.48% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 0 | 1 | RAW | 3.874661 | 2088.224 | +0.00% | +0.00% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 1 | 1 | CE_DPACE | 3.515725 | 1699.393 | +7.05% | -3.49% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 1 | 1 | CE_DPACE_OUTPUT | 3.656983 | 1764.202 | +11.35% | +0.19% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 1 | 1 | OUTPUT | 3.667483 | 1772.452 | +11.67% | +0.66% | 已核验 |
| evaluation_recovery_20261007_v2/C40/formal | 40 | 1 | 1 | RAW | 3.284257 | 1760.804 | +0.00% | +0.00% | 已核验 |

**结果解释：**原报告区分C1/C40正式结果和训练实现优化。后续采用CE+D-PACE+OUTPUT；当前条目不是全数据联合长训。

诊断／原报告摘录（保持原口径）：

- OUTPUT: 损失绝对差1.79e-07，头参数梯度相对L2差0.0976%，cosine=0.999999523，通过=True
- CE_DPACE: 损失绝对差2.98e-07，头参数梯度相对L2差0.1016%，cosine=0.999999464，通过=True
- CE_DPACE_OUTPUT: 损失绝对差4.77e-07，头参数梯度相对L2差0.0691%，cosine=0.999999881，通过=True
- chunk64: 3/3通过，峰值allocated7.55GiB/reserved7.84GiB；单包秒数[1.668, 1.605, 1.652]。共享GPU时序波动，不宣称稳定速度倍数。
- chunk256: 3/3通过，峰值allocated11.66GiB/reserved12.82GiB；单包秒数[0.59, 0.653, 0.6]。共享GPU时序波动，不宣称稳定速度倍数。
- chunk1536: 3/3通过，峰值allocated39.12GiB/reserved45.67GiB；单包秒数[0.519, 1.464, 1.079]。共享GPU时序波动，不宣称稳定速度倍数。
- 15位置合并未采用：真实状态发现argmax变化，未通过严格校验。
- 两段独立连续采样：GPU6平均利用率22.65%和24.20%，均出现100%样本；短窗口不能代表全程。

**原目录／权重入口：**[residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/C1/formal/summary.json)；[evaluation_recovery_20261007_v2/C40/formal/summary.json](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/evaluation_recovery_20261007_v2/C40/formal/summary.json)；[evaluation_recovery_20261007_v2/FORMAL_RESULTS_VERIFIED_20261007.csv](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/evaluation_recovery_20261007_v2/FORMAL_RESULTS_VERIFIED_20261007.csv)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[OPTIMIZATION_RESULTS_CN.txt](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/OPTIMIZATION_RESULTS_CN.txt)；[gpu6_utilization_diagnostic_20261007_v1/RESULTS_CN.txt](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/gpu6_utilization_diagnostic_20261007_v1/RESULTS_CN.txt)；[pipeline_20261007_v8/RESULTS_CN.txt](residualV3-nonthing/results/fused_output_dpace_20k_packed1536_2gpu_20261007_v2/pipeline_20261007_v8/RESULTS_CN.txt)。

<a id="n008"></a>

### N008 · OUTPUT前缀平滑短训

**类别／状态：**研究实验；complete；已记录12个正式窗口。

**做什么：**检验D-PACE与OUTPUT位置权重联动／平滑。

**对比什么：**同起点的原OUTPUT与a→0.5+0.5a平滑OUTPUT；共同CE+D-PACE；RAW参照。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 20k×1轮=298包；从预训练DFlash与随机头同时训练主体+头，seed/输入/anchors配对；旧开发面板。

**即时日志／进度：** [status.json](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=20条，proposal=15，max_new=2096; C40=300条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 RAW。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| evaluation/C1/formal | 1 | 0 | 1 | ORIGINAL | 3.596333 | 194.596 | -0.49% | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 0 | 1 | RAW | 3.613968 | 238.309 | +0.00% | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 0 | 1 | SMOOTH_A05 | 3.564964 | 192.135 | -1.36% | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 1 | 1 | ORIGINAL | 3.266301 | 175.668 | +0.61% | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 1 | 1 | RAW | 3.246644 | 216.320 | +0.00% | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 1 | 1 | SMOOTH_A05 | 3.158147 | 170.019 | -2.73% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 0 | 1 | ORIGINAL | 3.648316 | 1664.372 | -2.13% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 0 | 1 | RAW | 3.727757 | 1926.150 | +0.00% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 0 | 1 | SMOOTH_A05 | 3.675680 | 1684.745 | -1.40% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 1 | 1 | ORIGINAL | 3.283146 | 1476.648 | +2.28% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 1 | 1 | RAW | 3.209992 | 1626.397 | +0.00% | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 1 | 1 | SMOOTH_A05 | 3.264469 | 1460.613 | +1.70% | — | 未核验／只看AL |

**结果解释：**平滑没有一致收益：C1/T0 -0.87%、C1/T1 -3.31%、C40/T0 +0.75%、C40/T1 -0.57% AL；不动长训。

诊断／原报告摘录（保持原口径）：

- 原OUTPUT与平滑前缀OUTPUT：短训练正式native结果
- 边界：单seed、小型开发面板；不能当独立最终测试。平滑同时改变梯度位置分配和幅度，本轮不能单独归因。性能窗口如有外部进程仅标记，本轮不下吞吐结论。

**原目录／权重入口：**[residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [evaluation/C1/formal/summary.json](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/evaluation/C1/formal/summary.json)；[evaluation/C40/formal/summary.json](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/evaluation/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[RESULTS_CN.txt](residualV3-nonthing/results/output_prefix_smoothing_20k_joint_short_gpu07_20261008_v1/RESULTS_CN.txt)。

<a id="n009"></a>

### N009 · 严格对齐：只训头 vs 联合训练

**类别／状态：**研究实验；complete；已记录8个正式窗口。

**做什么：**修复此前不同训练预算导致不可比的问题。

**对比什么：**共同初始化、逐包输入/anchor/头Adam配对，只改变主体是否可训练；RAW、HEAD_ONLY、JOINT_BODY、JOINT_HEAD。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 20k×1轮、298包；HEAD_ONLY冻结主体，JOINT_BODY=联合主体关头而非独立单训；RAW/JOINT_HEAD复用上一实验。

**即时日志／进度：** [status.json](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=20条，proposal=15，max_new=2096; C40=300条，proposal=15，max_new=2096。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| evaluation/C1/formal | 1 | 0 | 1 | HEAD_ONLY | 3.860802 | 208.221 | — | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 0 | 1 | JOINT_BODY | 3.415900 | 228.093 | — | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 1 | 1 | HEAD_ONLY | 3.544581 | 191.312 | — | — | 未核验／只看AL |
| evaluation/C1/formal | 1 | 1 | 1 | JOINT_BODY | 3.080178 | 206.653 | — | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 0 | 1 | HEAD_ONLY | 4.008370 | 1849.729 | — | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 0 | 1 | JOINT_BODY | 3.433277 | 1776.416 | — | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 1 | 1 | HEAD_ONLY | 3.433316 | 1535.445 | — | — | 未核验／只看AL |
| evaluation/C40/formal | 40 | 1 | 1 | JOINT_BODY | 3.032806 | 1532.746 | — | — | 未核验／只看AL |

**结果解释：**在严格对齐的20k一轮配方中只训头比RAW AL提高约6.83%~9.18%；联合开头比联合关头有效，但联合模型整体未稳定超过RAW。复用和补测不同时段，不作配对TPS结论。

完整严格对齐AL比较：RAW／JOINT_HEAD复用已完成正式证据，HEAD_ONLY／JOINT_BODY本轮补测；来源跨时段，因此不计算对照TPS比例。

| C | T | RAW AL | 只训头 AL | 联合关头 AL | 联合开头 AL |
|---|---|---|---|---|---|
| 1 | 0 | 3.613968 | 3.860802 | 3.415900 | 3.596333 |
| 1 | 1 | 3.246644 | 3.544581 | 3.080178 | 3.266301 |
| 40 | 0 | 3.727757 | 4.008370 | 3.433277 | 3.648316 |
| 40 | 1 | 3.209992 | 3.433316 | 3.032806 | 3.283146 |

诊断／原报告摘录（保持原口径）：

- 单seed开发面板；复用窗口与补测窗口不同时段，不能作吞吐比较。联合主体关头只隔离推理时头的贡献，不隔离训练损失的长期影响。

**原目录／权重入口：**[residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [evaluation/C1/formal/summary.json](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/evaluation/C1/formal/summary.json)；[evaluation/C40/formal/summary.json](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/evaluation/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[RESULTS_CN.txt](residualV3-nonthing/results/head_only_joint_matched_20k_1epoch_gpu07_20261008_v1/RESULTS_CN.txt)。

<a id="n010"></a>

### N010 · 修复前q vs 修复后q：尾部停修信号配对

**类别／状态：**native状态诊断；不部署截断；已归档。

**做什么：**检验PRE先看再修和POST先修再看哪种信号更能预测修复后残余错误及尾部接受机会。

**对比什么：**同一native状态、同一target前缀、同一残余mismatch标签；PRE/POST各45维q统计与同规模logistic，另列位置控制及联合变化输入。

**实验设置：**冻结联合长训主体/三层头/basis；GSM8K同128提示80/24/24研究分组；T0/T1；最多每请求6块、15proposal；GPU6为空卡启动；KV40GiB；只RAM采集，无重训。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**配对诊断完成并复核：128提示/1536块/23040状态。预测实际拒绝时POST maxq明显优于PRE；尾部单点POST不占优，累计前缀POST观测更好但T1区间跨0。延长2位置仍超5%机会代理预算，4位置退化成固定13。没有部署截断或正式AL/TPS成绩。

诊断／原报告摘录（保持原口径）：

- 实际到达的native拒绝AUROC，T0 PRE/POST=0.7999/0.9088，T1=0.7686/0.8688。
- 尾部第8–15位置的累计置信度AUROC，T0 PRE/POST=0.8246/0.8883，T1=0.8355/0.8694；T1配对差值区间包含0。
- POST单位置置信度算子0.05342ms；昂贵统计分类器未明显改善局部预测。只有省下的实际计算大于判断成本才可能提速。
- 连续2次确认+额外2位置的POST累计信号省约24%repair次数，但机会代理损失约5.5%；额外4位置退化成统一修13，不能宣称自适应贡献。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n011"></a>

### N011 · q尾部停修设计：双阈值与额外2/4位置窗口

**类别／状态：**研究设计；未启动GPU实验；design_complete_not_run。

**做什么：**根据用户要求，在一次mismatch预测后保留数值和位置缓冲，按接受价值设计尾部停修。

**对比什么：**SpecDec++/PABEE/LTT/CALM/HF机制与当前固定PREFIX10/12及q-only证据；不是正式新性能实验。

**实验设置：**block16=15proposal；实际硬lag-1；保留raw后缀并全量target验证；候选双阈值、连续2次确认、额外2/4位置；小型q信号单独校准。

**即时日志／进度：** [status.json](residualV3-nonthing/research/q_tail_stop_design_20261009_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**设计完成，具体阈值和延长窗口未测。先预测继续repair的接受价值，再考虑停止；存在C40真实状态逐位不一致，必须先定位。

诊断／原报告摘录（保持原口径）：

- 相关主文已核对：SpecDec++ §4.2–4.4；PABEE §3.2；LTT §1.1/2；CALM §3/4；HF官方实现说明。
- 保留95%修复收益时用同一联合主体关头作对照；不是允许整个AL下降5%。
- 修复前/修复后q配对诊断已另行登记；本条仅文献和设计。

**原目录／权重入口：**[residualV3-nonthing/research/q_tail_stop_design_20261009_v1](residualV3-nonthing/research/q_tail_stop_design_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/research/q_tail_stop_design_20261009_v1/RESULTS_CN.txt)；[SOURCE_CHECKS.json](residualV3-nonthing/research/q_tail_stop_design_20261009_v1/SOURCE_CHECKS.json)；[IMPLEMENTATION_NOTES_CN.txt](residualV3-nonthing/research/q_tail_stop_design_20261009_v1/IMPLEMENTATION_NOTES_CN.txt)。

<a id="n012"></a>

### N012 · 模块参数／时间／高层使用审计

**类别／状态：**研究实验；complete；已记录16个正式窗口。

**做什么：**核实4.35M头的时间花在哪，高层是否参与预测。

**对比什么：**完整三层头、跳第2层、跳第3层、跳第2+3层；固定实际前驱模块微测和native跳层。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。 完整头含递推与采样，不含主体与target；模块各15位置CUDA Graph微测；曾用原target数值路径。

**即时日志／进度：** [status.json](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/status.json)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 FULL。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | FULL | 8.414318 | 427.010 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | SKIP2 | 8.309314 | 430.380 | -1.25% | +0.79% | 已核验 |
| C1/formal | 1 | 0 | 1 | SKIP23 | 8.035976 | 424.873 | -4.50% | -0.50% | 已核验 |
| C1/formal | 1 | 0 | 1 | SKIP3 | 8.342557 | 432.532 | -0.85% | +1.29% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL | 7.355848 | 377.065 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | SKIP2 | 7.408226 | 385.822 | +0.71% | +2.32% | 已核验 |
| C1/formal | 1 | 1 | 1 | SKIP23 | 6.843401 | 367.476 | -6.97% | -2.54% | 已核验 |
| C1/formal | 1 | 1 | 1 | SKIP3 | 7.422148 | 387.754 | +0.90% | +2.83% | 已核验 |
| C40/formal | 40 | 0 | 1 | FULL | 8.545277 | 4160.161 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | SKIP2 | 8.424843 | 4116.491 | -1.41% | -1.05% | 已核验 |
| C40/formal | 40 | 0 | 1 | SKIP23 | 8.202493 | 4036.233 | -4.01% | -2.98% | 已核验 |
| C40/formal | 40 | 0 | 1 | SKIP3 | 8.446006 | 4116.312 | -1.16% | -1.05% | 已核验 |
| C40/formal | 40 | 1 | 1 | FULL | 7.611864 | 3677.244 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | SKIP2 | 7.501440 | 3660.305 | -1.45% | -0.46% | 已核验 |
| C40/formal | 40 | 1 | 1 | SKIP23 | 7.119045 | 3472.635 | -6.47% | -5.56% | 已核验 |
| C40/formal | 40 | 1 | 1 | SKIP3 | 7.552752 | 3663.268 | -0.78% | -0.38% | 已核验 |

**结果解释：**新增头4353537参数；高层残差和token改变证明参与计算。跳层不是重训两层。独立模块时间不能机械相加；该轮目标数值路径后续由batch-invariance诊断补充。

诊断／原报告摘录（保持原口径）：

- 各模块独立捕获、缓存条件不同，不能机械相加成完整头时间。
- 模块 | 新增参数数 | N1 T0毫秒 | N40 T0毫秒
- 完整头（不插分段标记，包含实际前驱递推和采样） | T0毫秒 | T1毫秒
- N1 T0：4.0567→3.7146毫秒（少0.3421毫秒，8.43%）；这是头微测，非服务TPS。
- N1 T1：4.1020→3.7594毫秒（少0.3426毫秒，8.35%）；这是头微测，非服务TPS。
- N40 T0：9.7144→9.2712毫秒（少0.4432毫秒，4.56%）；这是头微测，非服务TPS。
- N40 T1：10.4760→10.0368毫秒（少0.4392毫秒，4.19%）；这是头微测，非服务TPS。
- 高层确实参与计算；单状态改变不等于接受收益，必须另看闭环正式消融。

**原目录／权重入口：**[residualV3-nonthing/results/head_component_depth_audit_20261009_v1](residualV3-nonthing/results/head_component_depth_audit_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[MODULE_TIMINGS_CN.txt](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/MODULE_TIMINGS_CN.txt)；[RESULTS_CN.txt](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/RESULTS_CN.txt)。

<a id="n013"></a>

### N013 · 置信度输入配对消融：q／DFlash hidden／实际前驱

**类别／状态：**已完成／置信度输入配对诊断；已归档。

**做什么：**检验哪些推理时可得信息能够预测修复后的实际首拒绝及前缀存活，为尾部停修选择信号。

**对比什么：**14组输入，POST maxq旧小预测器为参考；PRE/POST q统计、位置控制、冻结hidden编码、实际前驱编码及组合，另测原始hidden；4组比较overlap软标签与真实拒绝标签。

**实验设置：**联合长训主体、三层no-soft头与basis冻结；GSM8K non-thinking、block16=15proposal、T0/T1；176提示80/48/48，最多4块/请求；GPU7空卡启动；原始状态仅RAM。线性探针参数2–4353；拟合及正则选择仅训练提示。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**176提示/1408块/21120状态，36小预测器完成并核验，旧maxq基准结果精确复现。真实拒绝排序仍支持POST maxq（T0/T1 AUROC=0.9059/0.8582）；hidden+前驱及其与q统计组合未胜出。更多POST q统计有概率校准改善，但未证明更早停修或净提速。T0部分留出代理损失超过校准预算。主体/头/basis冻结，AL/TPS未测，未变更卡6正式策略。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n014"></a>

### N014 · C40 greedy文本分叉溯源

**类别／状态：**研究实验；已归档。

**做什么：**核查732/1319跨运行文本差异是否由头优化引入。

**对比什么：**原native、target-only AR，各自重复C40/C1；再比较batch-invariant版本。

**实验设置：**GSM8K前40题、max512，non-thinking/T0；同步scalar trace，因此诊断不能作TPS；AR不是项目RAW投机基线。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**默认native18/40、AR22/40重复文本不同，C1均0/8；首次分叉处同前缀target argmax翻转。batch-invariant两者重复40题均一致。原732条缺历史logits，不能逐条溯源。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n015"></a>

### N015 · 优化头 vs 单训DFlash正式对照

**类别／状态：**研究实验；native_complete；已记录8个正式窗口。

**做什么：**衡量优化实现对总体AL/TPS的实际价值。

**对比什么：**独立单训DFlash、联合主体关头、联合主体+优化头。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | DFLASH | 7.384246 | 180.964 | +0.00% | — | 污染 |
| C1/formal | 1 | 0 | 1 | OPTIMIZED | 8.381226 | 184.554 | +13.50% | — | 已核验 |
| C1/formal | 1 | 1 | 1 | DFLASH | 6.331330 | 155.593 | +0.00% | — | 污染 |
| C1/formal | 1 | 1 | 1 | OPTIMIZED | 7.429277 | 164.332 | +17.34% | — | 已核验 |
| C40/formal | 40 | 0 | 1 | DFLASH | 7.560736 | 3132.552 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | OPTIMIZED | 8.527201 | 3188.169 | +12.78% | +1.78% | 已核验 |
| C40/formal | 40 | 1 | 1 | DFLASH | 6.383055 | 2648.228 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | OPTIMIZED | 7.664765 | 2880.929 | +20.08% | +8.79% | 已核验 |

**结果解释：**使用正式窗口及计数器判定；组件profile另列。完整结果和C1恢复证据在关联目录，不把微测2ms当整轮解码时间。

**原目录／权重入口：**[residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_optimized_vs_dflash_formal_20261009_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)。

<a id="n016"></a>

### N016 · 三层SwiGLU后共享latent预测头：冻结特征诊断

**类别／状态：**已完成／三层后共享特征诊断；已归档。

**做什么：**验证修复网络完成融合后的表示是否有额外拒绝与存活预测信号，作为修复价值研究的第一步。

**对比什么：**POST maxq／POST q统计／最终latent256／latent+maxq／latent+q统计；真实首拒绝与overlap标签分开。

**实验设置：**主体/三层头/basis冻结，GSM8K176提示80/48/48、T0/T1、15proposal；18小预测器；GPU7空卡启动；真实前驱重算latent必须逐元素重构native修复logits一致。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**18个预测器完成；176提示/1408块与上次诊断严格配对。三层最终latent线性预测真实首拒绝AUROC=0.7459/0.7255，latent+q统计=0.8919/0.8477，均未形成胜过POST maxq(0.9059/0.8582)的拒绝排序证据。部分Brier改善，尚无实际停修AL/TPS成绩。主体/头/basis均冻结；不是联合辅助训练。修复价值双分支目前只有设计，未执行。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[DESIGN_CN.md](residualV3-nonthing/research/repair_value_design_20261009_v1/DESIGN_CN.md)；[SOURCE_FACTCHECK.md](residualV3-nonthing/research/repair_value_design_20261009_v1/SOURCE_FACTCHECK.md)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n017"></a>

### N017 · 优化头正式C1恢复测量

**类别／状态：**研究实验；complete；已记录1个正式窗口。

**做什么：**补齐上一正式对照C1阶段，复用已完成权重。

**对比什么：**同轮单训DFlash／联合主体／联合+头。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_optimized_vs_dflash_c1_recovery_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/native_optimized_vs_dflash_c1_recovery_20261009_v1/single_card_C1_formal.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| formal | 1 | 1 | 1 | DFLASH | 6.331330 | 155.167 | +0.00% | +0.00% | 已核验 |

**结果解释：**只是评测恢复，不重复训练；与原C40配对时保留各自运行条件。

**原目录／权重入口：**[residualV3-nonthing/results/native_optimized_vs_dflash_c1_recovery_20261009_v1](residualV3-nonthing/results/native_optimized_vs_dflash_c1_recovery_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [formal/summary.json](residualV3-nonthing/results/native_optimized_vs_dflash_c1_recovery_20261009_v1/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)。

<a id="n018"></a>

### N018 · 停修并截短target验证前缀：v1

**类别／状态：**失败准备／证据保留；已归档。

**做什么：**将少算repair与少算target分开，检验端到端截短收益。

**对比什么：**独立单训一轮DFLASH／FULL15／仅停止尾部REPAIR_STOP／停止尾部并截短target前缀VERIFY_STOP；不测BODY_ONLY。

**实验设置：**冻结全量单轮主体、三层no-soft头/basis/4参数q预测器；GSM8K nonthinking，block16/15proposal，C1=200/C40=1319，T0/T1，max2096；两并发smoke通过后才正式。v1启动校验失败，无正式成绩。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**启动dummy温度0.9时误把未缩放score与processed q比较，校验失败；原始冻结代码和日志保留。没有正式AL/TPS。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n019"></a>

### N019 · 停修并截短target验证前缀：v2

**类别／状态：**失败准备／证据保留；failed；已记录14个正式窗口。

**做什么：**将少算repair与少算target分开，检验端到端截短收益。

**对比什么：**独立单训一轮DFLASH／FULL15／仅停止尾部REPAIR_STOP／停止尾部并截短target前缀VERIFY_STOP；不测BODY_ONLY。

**实验设置：**冻结全量单轮主体、三层no-soft头/basis/4参数q预测器；GSM8K nonthinking，block16/15proposal，C1=200/C40=1319，T0/T1，max2096；两并发smoke通过后才正式。四组统一FULL_AND_PIECEWISE，独立stock sampler核验processed q及实际保留proposal。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | DFLASH | 7.382425 | 180.450 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FULL15 | 8.381226 | 195.430 | +13.53% | +8.30% | 已核验 |
| C1/formal | 1 | 0 | 1 | REPAIR_STOP | 8.362507 | 190.039 | +13.28% | +5.31% | 已核验 |
| C1/formal | 1 | 0 | 1 | VERIFY_STOP | 8.352014 | 175.833 | +13.13% | -2.56% | 已核验 |
| C1/formal | 1 | 1 | 1 | DFLASH | 6.331330 | 155.107 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL15 | 7.429277 | 173.906 | +17.34% | +12.12% | 已核验 |
| C1/formal | 1 | 1 | 1 | REPAIR_STOP | 7.439199 | 169.327 | +17.50% | +9.17% | 已核验 |
| C1/formal | 1 | 1 | 1 | VERIFY_STOP | 7.433866 | 157.419 | +17.41% | +1.49% | 已核验 |
| C40/formal | 40 | 0 | 1 | DFLASH | 7.560594 | 3161.651 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FULL15 | 8.527201 | 3317.364 | +12.78% | +4.93% | 已核验 |
| C40/formal | 40 | 0 | 1 | REPAIR_STOP | 8.503940 | 3283.776 | +12.48% | +3.86% | 已核验 |
| C40/formal | 40 | 1 | 1 | DFLASH | 6.382775 | 2668.960 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FULL15 | 7.669673 | 2992.237 | +20.16% | +12.11% | 已核验 |
| C40/formal | 40 | 1 | 1 | REPAIR_STOP | 7.657712 | 2978.277 | +19.97% | +11.59% | 已核验 |

**结果解释：**已核验14/16正式窗口。FULL15相对独立单训DFlash：C1 T0/T1 AL +13.53%/+17.34%、TPS +8.30%/+12.12%；C40 AL +12.78%/+20.16%、TPS +4.93%/+12.11%。REPAIR_STOP未进一步提速；VERIFY_STOP的C1明显更慢。C40 VERIFY_STOP因Graph批量状态引用错误未正式测得，恢复目录仅补这2窗口。

诊断／原报告摘录（保持原口径）：

- 用户要求暂缓补测和截断扩展，先完成顶层设计；C40 VERIFY_STOP缺的两窗口不计成绩。

**原目录／权重入口：**[residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/C40/formal/summary.json)；[RUN_GUIDE_CN.txt](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/RUN_GUIDE_CN.txt)；[DESIGN_CN.md](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/DESIGN_CN.md)；[preregistration.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/preregistration.json)；[CPU_GATE.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/CPU_GATE.json)；[status.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/status.json)；[controller_pid.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/controller_pid.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[RESULTS_VERIFIED_CN.txt](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v2/RESULTS_VERIFIED_CN.txt)。

<a id="n020"></a>

### N020 · target前缀截短：v3仅补C40 VERIFY_STOP

**类别／状态：**暂不启动／顶层设计优先；not_started_user_hold；已记录14个正式窗口。

**做什么：**修复正式模式Graph输出引用的批量绑定，补齐原实验缺失窗口。

**对比什么：**独立单训DFLASH／FULL15／REPAIR_STOP／VERIFY_STOP；复用v2原14正式窗口，v3只补C40 VERIFY_STOP。

**实验设置：**GSM8K nonthinking；C1=200/C40=1319；T0/T1；block16/15proposal；max2096；同权重/阈值/FULL_AND_PIECEWISE；先正式路径smoke再独立概率契约smoke；仅使用空闲GPU7。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/status.json)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1 | 1 | 0 | 1 | DFLASH | 7.382425 | 180.450 | +0.00% | +0.00% | 已核验 |
| C1 | 1 | 0 | 1 | FULL15 | 8.381226 | 195.430 | +13.53% | +8.30% | 已核验 |
| C1 | 1 | 0 | 1 | REPAIR_STOP | 8.362507 | 190.039 | +13.28% | +5.31% | 已核验 |
| C1 | 1 | 0 | 1 | VERIFY_STOP | 8.352014 | 175.833 | +13.13% | -2.56% | 已核验 |
| C1 | 1 | 1 | 1 | DFLASH | 6.331330 | 155.107 | +0.00% | +0.00% | 已核验 |
| C1 | 1 | 1 | 1 | FULL15 | 7.429277 | 173.906 | +17.34% | +12.12% | 已核验 |
| C1 | 1 | 1 | 1 | REPAIR_STOP | 7.439199 | 169.327 | +17.50% | +9.17% | 已核验 |
| C1 | 1 | 1 | 1 | VERIFY_STOP | 7.433866 | 157.419 | +17.41% | +1.49% | 已核验 |
| C40 | 40 | 0 | 1 | DFLASH | 7.560594 | 3161.651 | +0.00% | +0.00% | 已核验 |
| C40 | 40 | 0 | 1 | FULL15 | 8.527201 | 3317.364 | +12.78% | +4.93% | 已核验 |
| C40 | 40 | 0 | 1 | REPAIR_STOP | 8.503940 | 3283.776 | +12.48% | +3.86% | 已核验 |
| C40 | 40 | 1 | 1 | DFLASH | 6.382775 | 2668.960 | +0.00% | +0.00% | 已核验 |
| C40 | 40 | 1 | 1 | FULL15 | 7.669673 | 2992.237 | +20.16% | +12.11% | 已核验 |
| C40 | 40 | 1 | 1 | REPAIR_STOP | 7.657712 | 2978.277 | +19.97% | +11.59% | 已核验 |

**结果解释：**用户明确暂不补测；后台控制器未启动。复用原14个正式窗口，缺C40 VERIFY_STOP T0/T1。已完成窗口中截断未额外提速；完整修复整体方案仍有AL/TPS正收益。CPU引用修复通过，GPU校验尚未进行。

诊断／原报告摘录（保持原口径）：

- 不重跑v2的14个已核验正式窗口。新代码未改变模型参数、修复强度、停修阈值和采样算法。

**原目录／权重入口：**[residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/comparison.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/C1/comparison.json)；[C40/comparison.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/C40/comparison.json)；[RUN_GUIDE_CN.txt](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/RUN_GUIDE_CN.txt)；[preregistration.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/preregistration.json)；[CPU_GATE.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/CPU_GATE.json)；[status.json](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/status.json)；[TOP_LEVEL_DESIGN_CN.md](residualV3-nonthing/research/repair_value_design_20261009_v1/TOP_LEVEL_DESIGN_CN.md)；[RESULTS_VERIFIED_CN.txt](residualV3-nonthing/results/native_target_prefix_cut_gpu7_20261009_v3_recovery/RESULTS_VERIFIED_CN.txt)。

<a id="n021"></a>

### N021 · Small tile头正式测评

**类别／状态：**研究实验；native_complete；已记录8个正式窗口。

**做什么：**减少低并发小矩阵计算／启动开销。

**对比什么：**原实现与小tile实现，同一头、主体、basis。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_head_small_tile_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/native_head_small_tile_20261009_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/native_head_small_tile_20261009_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/native_head_small_tile_20261009_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/native_head_small_tile_20261009_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 BASE_HEAD。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | BASE_HEAD | 8.381226 | 185.540 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FAST_HEAD | 8.381226 | 189.819 | +0.00% | +2.31% | 已核验 |
| C1/formal | 1 | 1 | 1 | BASE_HEAD | 7.429277 | 164.463 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FAST_HEAD | 7.429277 | 168.971 | +0.00% | +2.74% | 已核验 |
| C40/formal | 40 | 0 | 1 | BASE_HEAD | 8.527201 | 3187.122 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FAST_HEAD | 8.527201 | 3223.233 | +0.00% | +1.13% | 已核验 |
| C40/formal | 40 | 1 | 1 | BASE_HEAD | 7.664765 | 2879.264 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FAST_HEAD | 7.664765 | 2922.644 | +0.00% | +1.51% | 已核验 |

**结果解释：**同状态校验通过后测native；只在同轮、同目标数值路径及干净窗口比较TPS；不能因参数没变直接宣称端到端全等价。

**原目录／权重入口：**[residualV3-nonthing/results/native_head_small_tile_20261009_v1](residualV3-nonthing/results/native_head_small_tile_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/native_head_small_tile_20261009_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_head_small_tile_20261009_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)。

<a id="n022"></a>

### N022 · 结构融合优化 v2正式测评

**类别／状态：**研究实验；正式窗口已完成（顶层历史状态：native_evaluation）；已记录8个正式窗口。

**做什么：**合并前驱投影、融合及SwiGLU的冗余访存／启动。

**对比什么：**小tile基准与新融合实现；模型权重及实际前驱递推保留。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/status.json)；[single_card_C40_formal.log](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 BASE_HEAD。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1_recovery_gpu5/formal | 1 | 0 | 1 | BASE_HEAD | 8.381226 | 188.131 | +0.00% | +0.00% | 已核验 |
| C1_recovery_gpu5/formal | 1 | 0 | 1 | FAST_HEAD | 8.381226 | 190.508 | +0.00% | +1.26% | 已核验 |
| C1_recovery_gpu5/formal | 1 | 1 | 1 | BASE_HEAD | 7.429277 | 166.426 | +0.00% | +0.00% | 已核验 |
| C1_recovery_gpu5/formal | 1 | 1 | 1 | FAST_HEAD | 7.429277 | 169.529 | +0.00% | +1.86% | 已核验 |
| C40/formal | 40 | 0 | 1 | BASE_HEAD | 8.527201 | 3231.555 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FAST_HEAD | 8.527201 | 3298.465 | +0.00% | +2.07% | 已核验 |
| C40/formal | 40 | 1 | 1 | BASE_HEAD | 7.664765 | 2918.941 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FAST_HEAD | 7.664765 | 2978.183 | +0.00% | +2.03% | 已核验 |

**结果解释：**有效正式及核验报告在research/head_structure_fusion；整体验证结合batch-invariant及正式窗口，不能从模块计时直接推出TPS。

**原目录／权重入口：**[residualV3-nonthing/results/native_head_structure_fusion_20261009_v2](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1_recovery_gpu5/formal/summary.json](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/C1_recovery_gpu5/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_head_structure_fusion_20261009_v2/C40/formal/summary.json)；[RESULTS_CN.txt](residualV3-nonthing/research/head_structure_fusion_20261009_v1/RESULTS_CN.txt)；[VERIFIED_SUMMARY.json](residualV3-nonthing/research/head_structure_fusion_20261009_v1/VERIFIED_SUMMARY.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n023"></a>

### N023 · 原实现block12／8正式测试

**类别／状态：**研究实验；complete；已记录32个正式窗口。

**做什么：**检查较短草稿对AL/TPS的影响。

**对比什么：**每种block的单训主体、联合主体关头及开头。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。 block12=11，block8=7 proposal。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/status.json)。

正式窗口设置：C1=200条，proposal=11，max_new=2096; C1=200条，proposal=7，max_new=2096; C40=1319条，proposal=11，max_new=2096; C40=1319条，proposal=7，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| BLOCK12/C1/formal | 1 | 0 | 1 | DFLASH | 6.828973 | 439.205 | +0.00% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 0 | 1 | JOINT_BODY | 6.803453 | 438.046 | -0.37% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 0 | 1 | JOINT_HEAD | 7.547491 | 410.887 | +10.52% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 0 | 1 | RAW | 6.295867 | 408.034 | -7.81% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 1 | 1 | DFLASH | 5.939536 | 386.695 | +0.00% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 1 | 1 | JOINT_BODY | 5.910502 | 385.252 | -0.49% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 1 | 1 | JOINT_HEAD | 6.776560 | 373.005 | +14.09% | — | 未核验／只看AL |
| BLOCK12/C1/formal | 1 | 1 | 1 | RAW | 5.486924 | 358.692 | -7.62% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 0 | 1 | DFLASH | 6.956172 | 4686.080 | +0.00% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 0 | 1 | JOINT_BODY | 6.896923 | 4638.493 | -0.85% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 0 | 1 | JOINT_HEAD | 7.629385 | 4615.391 | +9.68% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 0 | 1 | RAW | 6.389898 | 4334.486 | -8.14% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 1 | 1 | DFLASH | 6.004208 | 4047.796 | +0.00% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 1 | 1 | JOINT_BODY | 6.034283 | 4019.651 | +0.50% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 1 | 1 | JOINT_HEAD | 6.967710 | 4205.886 | +16.05% | — | 未核验／只看AL |
| BLOCK12/C40/formal | 40 | 1 | 1 | RAW | 5.656454 | 3827.322 | -5.79% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 0 | 1 | DFLASH | 5.675449 | 369.812 | +0.00% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 0 | 1 | JOINT_BODY | 5.665946 | 368.175 | -0.17% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 0 | 1 | JOINT_HEAD | 6.053387 | 357.640 | +6.66% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 0 | 1 | RAW | 5.376526 | 351.376 | -5.27% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 1 | 1 | DFLASH | 5.139865 | 335.050 | +0.00% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 1 | 1 | JOINT_BODY | 5.123640 | 335.434 | -0.32% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 1 | 1 | JOINT_HEAD | 5.638184 | 334.778 | +9.70% | — | 未核验／只看AL |
| BLOCK8/C1/formal | 1 | 1 | 1 | RAW | 4.912352 | 322.541 | -4.43% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 0 | 1 | DFLASH | 5.742706 | 5028.695 | +0.00% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 0 | 1 | JOINT_BODY | 5.729054 | 4974.597 | -0.24% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 0 | 1 | JOINT_HEAD | 6.080978 | 4899.229 | +5.89% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 0 | 1 | RAW | 5.428350 | 4769.381 | -5.47% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 1 | 1 | DFLASH | 5.167423 | 4493.918 | +0.00% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 1 | 1 | JOINT_BODY | 5.199546 | 4357.884 | +0.62% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 1 | 1 | JOINT_HEAD | 5.696731 | 4526.884 | +10.24% | — | 未核验／只看AL |
| BLOCK8/C40/formal | 40 | 1 | 1 | RAW | 4.991109 | 4362.821 | -3.41% | — | 未核验／只看AL |

**结果解释：**是最新融合版本之前的历史实现对照，保留原窗口，不与最新融合绝对TPS混用。

诊断／原报告摘录（保持原口径）：

- 单seed/单轮数学基准评测；更换面板，不能直接和旧短训数字比较长短训练收益；吞吐不作结论。
- 单seed/单轮数学基准评测；更换面板，不能直接和旧短训数字比较长短训练收益；吞吐不作结论。

**原目录／权重入口：**[residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [BLOCK12/C1/formal/summary.json](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK12/C1/formal/summary.json)；[BLOCK12/C40/formal/summary.json](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK12/C40/formal/summary.json)；[BLOCK8/C1/formal/summary.json](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK8/C1/formal/summary.json)；[BLOCK8/C40/formal/summary.json](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK8/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[BLOCK12/RESULTS_CN.txt](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK12/RESULTS_CN.txt)；[BLOCK8/RESULTS_CN.txt](residualV3-nonthing/results/native_block12_8_gsm8k_20261009_v1/BLOCK8/RESULTS_CN.txt)。

<a id="n024"></a>

### N024 · 结构融合正式尝试 v1

**类别／状态：**研究实验；已归档。

**做什么：**检查早期融合实现的native行为和性能。

**对比什么：**小tile与结构融合。

**实验设置：**GSM8K non-thinking；C1=200、C40=1319；T0/T1、max_new2096/model4096；block16含anchor，15个新proposal；CUDA Graph。单训主体与联合主体来自各自的一轮全数据训练，权重不同。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**后续有效版本v2；旧状态可能仍显示native_evaluation，以完成窗口及verified报告判断。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n025"></a>

### N025 · 1536 anchors整块显存探针

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**比较chunk64与整1536的内存／前后向时间。

**对比什么：**原chunk与整块，保持同一包和目标。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧FP32全词表中间量使整块显存高；后续融合TV减少临时张量。具体attempt02数值见报告，不是模型质量成绩。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n026"></a>

### N026 · 修复头前后向及融合loss

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**校验融合TV与15位置并行的数值／梯度。

**对比什么：**旧loss、融合TV、并行头原型。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/head_forward_backward_optimization_20261007_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**loss优化保持数学目标；此前15位置并行未通过严格梯度校验，不直接替换。TF可用已知前驱，native实际前驱仍须递推。

诊断／原报告摘录（保持原口径）：

- 联合训练头/损失前后向：0.8622 -> 0.3860秒，下降55.23%。此项包含loss与LM投影反向，不是MLP单独时间。
- 联合训练峰值allocated：57.691 -> 45.556GiB；不含单独梯度校验的额外retain_grad显存。
- 两组一起完成每包周期：4.2194 -> 4.3153秒；耗时变化+2.27%，吞吐变化-2.22%。本轮没有验证端到端提速。
- 去掉末尾4包的中间12包周期：4.5247 -> 4.8128秒。

**原目录／权重入口：**[residualV3-nonthing/results/head_forward_backward_optimization_20261007_v1](residualV3-nonthing/results/head_forward_backward_optimization_20261007_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/head_forward_backward_optimization_20261007_v1/RESULTS_CN.txt)。

<a id="n027"></a>

### N027 · Teacher1／2与头并行初测

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**实测hidden供给及头计算耗时。

**对比什么：**teacher1+learner1与teacher2+learner1。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/teacher1_vs2_head_parallel_gpu016_20261007_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**以CORRECTED报告为准；短测不能保证全数据耗时，未通过梯度门槛的并行版不采纳。

诊断／原报告摘录（保持原口径）：

- 但顺序重复有明显波动，不能把该百分比当作稳定长期惩罚。9-pack探索性配对bootstrap差值区间：[-1.212797578631176, 1.5813114231633436]秒。
- 完整1536头固定状态计时（不含target原始/teacher logits投影、Adam，也不等新hidden）：
- 完整1536三损失校验：直接TF合并 argmax 23040/23040一致，combined损失差2.384e-7，参数梯度relative L2 0.004684（0.4684%），不是bitwise等价，也未通过预设0.2%小误差门槛。
- 真实在线训练峰值torch allocated约53.16GiB；learner真实缓存预热67GiB/预算68GiB，nvidia-smi约67.7GiB。
- 但顺序重复有明显波动，不能把该百分比当作稳定长期惩罚。9-pack探索性配对bootstrap差值区间：[-1.212797578631176, 1.5813114231633436]秒。
- 完整1536头固定状态计时（不含target原始/teacher logits投影、Adam，也不等新hidden）：
- 完整1536三损失校验：直接TF合并 argmax 23040/23040一致，combined损失差2.384e-7，参数梯度relative L2 0.004684（0.4684%），不是bitwise等价，也未通过预设0.2%小误差门槛。
- 真实在线训练峰值torch allocated约53.16GiB；learner真实缓存预热67GiB/预算68GiB，nvidia-smi约67.7GiB。

**原目录／权重入口：**[residualV3-nonthing/results/teacher1_vs2_head_parallel_gpu016_20261007_v1](residualV3-nonthing/results/teacher1_vs2_head_parallel_gpu016_20261007_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/teacher1_vs2_head_parallel_gpu016_20261007_v1/RESULTS_CN.txt)；[RESULTS_CN_CORRECTED.txt](residualV3-nonthing/results/teacher1_vs2_head_parallel_gpu016_20261007_v1/RESULTS_CN_CORRECTED.txt)。

<a id="n028"></a>

### N028 · Teacher1／2联合训练计时

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**实测同时训练DFlash主体+头时供数影响。

**对比什么：**teacher2张+learner1张 vs teacher1张+learner1张。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/joint_teacher1_vs2_timing_20261007_v2/status.json)；[training.log](residualV3-nonthing/results/joint_teacher1_vs2_timing_20261007_v2/training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**角色是teacher数量对照，不是把learner拆成两张；分别报告wait与GPU计算。

诊断／原报告摘录（保持原口径）：

- 过程 | teacher1+learner1 秒 | teacher2+learner1 秒
- 双teacher相对单teacher每步耗时变化：+4.40%；同样工作量的速度变化：-4.22%。
- 按同包配对重采样，单teacher减双teacher的耗时差95%区间：[-0.4623, 0.0998]秒。
- hidden后台组包/请求耗时与训练重叠，不能与整步其他耗时直接相加。
- 分项含同步计时开销；反向包含损失与LM投影梯度，不能当成小头本身的纯算子成本。

**原目录／权重入口：**[residualV3-nonthing/results/joint_teacher1_vs2_timing_20261007_v2](residualV3-nonthing/results/joint_teacher1_vs2_timing_20261007_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/joint_teacher1_vs2_timing_20261007_v2/RESULTS_CN.txt)。

<a id="n029"></a>

### N029 · 共享teacher同时供两个learner v1

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**一份hidden供独立主体与主体+头更新。

**对比什么：**teacher1、body learner1、joint learner1。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共享一次teacher计算可减少重复供数；耗时是共同完成一包，不能把单端时间相加。

诊断／原报告摘录（保持原口径）：

- 过程 | DFlash+头 秒 | 纯DFlash 秒
- 两组共同完成1包的端到端周期：5.5888秒；后16包共同完成耗时：89.422秒。
- 共享producer后16包平均分配/填充耗时：{'allocate_shared_seconds': 1.7119923962745816, 'fetch_fill_seconds': 3.2693777893437073, 'producer_wall_seconds': 4.981370185618289}。
- 共享队列等待不能全部归因于teacher慢：两槽上限也会让较快learner等待较慢learner释放。
- 本轮纯DFlash的整步速度是共享并发条件下的速度，不能当成它独占teacher时的速度。

**原目录／权重入口：**[residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v1](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v1/RESULTS_CN.txt)。

<a id="n030"></a>

### N030 · 共享teacher供两端 v2

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**修复供数／队列安排后复测。

**对比什么：**共同完成包周期、两端wait/compute。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v2/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**按真实833176条、15847包外推仅计算时间，保存/等待/故障另计。

诊断／原报告摘录（保持原口径）：

- 过程 | DFlash+头 秒 | 纯DFlash 秒
- 两组共同完成1包的端到端周期：4.4957秒；后16包共同完成耗时：71.931秒。
- 共享producer后16包平均分配/填充耗时：{'allocate_shared_seconds': 0.23033920233137906, 'fetch_fill_seconds': 3.629966533742845, 'producer_wall_seconds': 3.860305736074224}。
- 共享队列等待不能全部归因于teacher慢：两槽上限也会让较快learner等待较慢learner释放。
- 本轮纯DFlash的整步速度是共享并发条件下的速度，不能当成它独占teacher时的速度。
- 补充校验：两组压缩层、5个主体层的BF16权重采样均发生改变，joint头权重采样也发生改变。5秒GPU监测未发现所用1/2/4卡上额外进程；不覆盖未采到的短暂活动。

**原目录／权重入口：**[residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v2](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/shared_teacher_joint_body_timing_20261007_v2/RESULTS_CN.txt)。

<a id="n031"></a>

### N031 · 传输／并发优化 v3

**类别／状态：**训练／实现计时诊断；complete_with_failed_additional_attempt。

**做什么：**降低hidden读取和CPU传输开销。

**对比什么：**旧传输与优化版，同loss/anchors/优化器。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/shared_teacher_fusion_transport_timing_20261007_v3/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**端到端4.4818→4.6556秒／共同包，没有提速；不能用单段改善掩盖总体退化。

诊断／原报告摘录（保持原口径）：

- 数据、初始化、GPU、目标、anchor、优化器、学习率、loss一致；不采用旧的未通过梯度校验的15位置头并行。
- 基准共同完成1包：4.4818秒；优化后：4.6556秒。
- 耗时下降-3.88%；吞吐提高-3.73%。
- 去掉末尾4包的中间12包周期：4.7973 -> 4.7620秒。
- 完整1536峰值allocated GiB：{'joint': 57.69137477874756, 'body': 30.540907859802246}。
- loss数值与logits梯度校验通过，绝对loss差0；不声称全backbone逐状态bitwise等价。

**原目录／权重入口：**[residualV3-nonthing/results/shared_teacher_fusion_transport_timing_20261007_v3](residualV3-nonthing/results/shared_teacher_fusion_transport_timing_20261007_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/shared_teacher_fusion_transport_timing_20261007_v3/RESULTS_CN.txt)。

<a id="n032"></a>

### N032 · 固定内存注册／共享供数 v4

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**避免每包pin_memory大复制。

**对比什么：**旧共享供数与两slot只读注册缓冲，同一包。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/shared_teacher_fusion_registered_timing_20261007_v4/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**共同包4.4818→4.0891秒，吞吐+9.60%；联合CPU固定内存0.240635→0.000729秒；仅短测，质量未评测。

诊断／原报告摘录（保持原口径）：

- 两组共同完成1包：4.4818 -> 4.0891秒。
- 耗时变化：下降8.76%；吞吐提高9.60%。
- 去掉末尾4包的中间12包周期：4.7973 -> 4.3072秒。
- 联合训练CPU固定内存拷贝/注册均值：0.240635 -> 0.000729秒。
- 联合训练共享供数队列等待：0.8820 -> 0.7689秒。
- 峰值allocated显存GiB：{'joint': 57.69137477874756, 'body': 30.540907859802246}。
- 本轮不替换未通过严格梯度校验的15位置修复头并行实现。

**原目录／权重入口：**[residualV3-nonthing/results/shared_teacher_fusion_registered_timing_20261007_v4](residualV3-nonthing/results/shared_teacher_fusion_registered_timing_20261007_v4)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/shared_teacher_fusion_registered_timing_20261007_v4/RESULTS_CN.txt)。

<a id="n033"></a>

### N033 · Hidden供给专项计时

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**测单／双teacher及供数瓶颈。

**对比什么：**teacher复制数、wait、anchor计算阶段。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/hidden_supply_timing_20261008_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**等待时间取决于供给与消费重叠，不能把GPU空闲直接归因为DFlash慢；时长外推需按实际包数。

诊断／原报告摘录（保持原口径）：

- 本轮以实际整包耗时为验收标准；长训队列仍暂停；GPU1、2排除。
- 整包周期DP1 -> DP2：3.8154 -> 3.2517秒。
- DP2耗时变化：-14.77%；吞吐变化：+17.34%。
- 去掉末尾4包的中间20包周期：3.8769 -> 3.3363秒。
- 联合训练共享队列等待：0.9321 -> 0.3645秒。
- 联合训练计算（不含队列/H2D）：2.6643 -> 2.6726秒。
- 供数获取和拼包周期：3.6436 -> 2.9588秒。
- 历史DP1参考：旧4.3153秒，本轮DP1相同前24包3.8621秒。运行窗口及预取尾部不同，不能拆成各CPU改动的因果贡献。

**原目录／权重入口：**[residualV3-nonthing/results/hidden_supply_timing_20261008_v1](residualV3-nonthing/results/hidden_supply_timing_20261008_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/hidden_supply_timing_20261008_v1/RESULTS_CN.txt)。

<a id="n034"></a>

### N034 · 用显存换训练时间

**类别／状态：**训练／实现计时诊断；complete。

**做什么：**检查activation/checkpoint等实现取舍。

**对比什么：**同包、同目标的内存／重算策略。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**即时日志／进度：** [status.json](residualV3-nonthing/results/activation_memory_timing_20261008_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**优先以共同包时间判断是否加速；更多显存占用本身不构成GPU隔离保证。

诊断／原报告摘录（保持原口径）：

- 同一状态核心计算：2.6528 -> 2.4184秒，下降8.83%；allocated峰值45.566 -> 67.795GiB。
- 同一状态核心计算：2.3506 -> 2.0551秒，下降12.57%；allocated峰值30.541 -> 58.326GiB。
- 实际nvidia采样峰值GiB：{'0': 69.658203125, '3': 71.9072265625, '4': 74.6923828125, '5': 74.5322265625}。
- 24包fresh在线固定权重探针，排除前8包：整包3.0321秒，中间12包（去掉末尾4包）3.1787秒。
- 上轮真实参数更新的DP2整包为3.2517秒；学习率与24/32包预取尾部不同，不能把差值直接当纯显存策略造成的全局提速。

**原目录／权重入口：**[residualV3-nonthing/results/activation_memory_timing_20261008_v1](residualV3-nonthing/results/activation_memory_timing_20261008_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/activation_memory_timing_20261008_v1/RESULTS_CN.txt)。

<a id="n035"></a>

### N035 · DP2／工作区／流水线训练吞吐

**类别／状态：**训练／实现计时诊断；未提供顶层状态。

**做什么：**检查teacher供给、读取线程及训练pipeline。

**对比什么：**单teacher、双独立teacher、DP2及恢复实现。

**实验设置：**non-thinking重生成数据，prompt+answer≤4096；no-soft三层、Top256、basis512；每包最多65536输入token、1536 anchors；一包一次更新；不是每条1536。 参数、anchor与损失一致性看各自协议；时间为短测／微基准，不替代正式AL/TPS；hidden在线读取即删，无持久缓存。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**DP2是两个完整模型处理不同请求，TP1；不代表拆一个模型。保留各版本有效短测与失败原因，不把未完成attempt当成绩。

诊断／原报告摘录（保持原口径）：

- 每组12包，前3包预热，计时9包；相同输入条目及anchor身份逐包核对通过。
- 请求并发 | 固定池秒/包 | 原生秒/包 | 原生输入token/秒
- CUDA已持有的页面不能由别人抢走；仍不能保证计算资源独占，也不能直接推广到joint backbone/multi-stream。
- 数值门槛：一真实整包1536状态逐位一致，三损失32-anchor梯度逐位一致，完整1536反传及下一包驻留通过。
- 性能局限：每组仅9个计时包、时段不同，跨卡任务共享CPU/RAM/PCIe；这些是训练工程观察，不能当正式native推理吞吐结果。
- allocator requests 秒/包 wait DFlash head input_tokens/s
- 这是训练工程观察，不是native推理吞吐结果；样本小且分组顺序不同。

**原目录／权重入口：**[residualV3-nonthing/results/training_throughput_dp2_workspace_gpu027_20261007_v1](residualV3-nonthing/results/training_throughput_dp2_workspace_gpu027_20261007_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/training_throughput_dp2_workspace_gpu027_20261007_v1/RESULTS_CN.txt)；[native_gpu4_v14/RESULTS_CN.txt](residualV3-nonthing/results/training_throughput_dp2_workspace_gpu027_20261007_v1/native_gpu4_v14/RESULTS_CN.txt)。

<a id="n036"></a>

### N036 · Target／body／head整轮耗时

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**区分验证、主体、头与草稿总体成本。

**对比什么：**独立单训草稿、联合草稿关头／开头。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**组件profile不等于正式TPS，插计时标记会增开销；需优先看完整无标记计时。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n037"></a>

### N037 · Small tile算子profile

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**检查C1/C40小矩阵与采样成本。

**对比什么：**优化前后同状态完整头及分段。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**算子微测用于筛选实现，正式性能在small_tile评测条目。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n038"></a>

### N038 · Small tile同状态校验

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**校验不同tile前向／采样是否一致。

**对比什么：**原实现、候选小tile实现。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**只有通过实际q、采样、Graph/eager门槛的实现进入正式测评。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n039"></a>

### N039 · 结构融合微探针 v1

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**定位前驱/fusion/SwiGLU融合的数值差异。

**对比什么：**分段原实现与融合候选。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**保留原始梯度／前向检查，未通过候选不宣称正式性能。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n040"></a>

### N040 · 结构融合微探针 v2

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**复核融合版本及完整头时间。

**对比什么：**候选融合与小tile基线。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**模块校验和native评测分别报告，推理递推不能用teacher前驱替代。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n041"></a>

### N041 · 硬前驱投影查表

**类别／状态：**训练／实现计时诊断；native_complete；已记录8个正式窗口。

**做什么：**减少逐token大维度硬前驱编码。

**对比什么：**实际前驱embedding投影与预计算查表。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 BASE_HEAD。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | BASE_HEAD | 8.381226 | 178.133 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FAST_HEAD | 8.381226 | 185.847 | +0.00% | — | 污染 |
| C1/formal | 1 | 1 | 1 | BASE_HEAD | 7.429277 | 158.978 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FAST_HEAD | 7.429277 | 154.519 | +0.00% | — | 污染 |
| C40/formal | 40 | 0 | 1 | BASE_HEAD | 8.527201 | 3135.172 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FAST_HEAD | 8.527201 | 3184.133 | +0.00% | +1.56% | 已核验 |
| C40/formal | 40 | 1 | 1 | BASE_HEAD | 7.664765 | 2823.957 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FAST_HEAD | 7.664765 | 2876.757 | +0.00% | +1.87% | 已核验 |

**结果解释：**推理冻结权重允许预计算；按同状态q/采样校验，正式差异另列。

诊断／原报告摘录（保持原口径）：

- 所有正式窗口均通过proposal计数、请求数量、Graph覆盖及零eager/fallback检查；隔离状态见JSON，受污染窗口不作为吞吐结论。

**原目录／权重入口：**[residualV3-nonthing/results/native_predecessor_lookup_20261009_v1](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[VERIFIED_RESULTS_CN.txt](residualV3-nonthing/results/native_predecessor_lookup_20261009_v1/VERIFIED_RESULTS_CN.txt)。

<a id="n042"></a>

### N042 · 前驱＋SwiGLU融合smoke

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**降低硬前驱及逐位置启动成本。

**对比什么：**原头与融合kernel。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**此条主要为micro/smoke，不能拿smoke吞吐作正式成绩。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n043"></a>

### N043 · 候选logit写回融合

**类别／状态：**训练／实现计时诊断；complete；已记录8个正式窗口。

**做什么：**减少全词表scatter与重复写回。

**对比什么：**原候选更新与Triton融合。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**即时日志／进度：** [status.json](residualV3-nonthing/results/native_candidate_write_triton_20261009_v1/status.json)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 BASE_HEAD。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | BASE_HEAD | 8.414318 | 427.456 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FAST_HEAD | 8.414318 | 450.186 | +0.00% | +5.32% | 已核验 |
| C1/formal | 1 | 1 | 1 | BASE_HEAD | 7.355848 | 376.633 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FAST_HEAD | 7.355848 | 397.847 | +0.00% | +5.63% | 已核验 |
| C40/formal | 40 | 0 | 1 | BASE_HEAD | 8.582962 | 4176.414 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FAST_HEAD | 8.543360 | 4242.315 | -0.46% | +1.58% | 已核验 |
| C40/formal | 40 | 1 | 1 | BASE_HEAD | 7.681713 | 3724.965 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FAST_HEAD | 7.611864 | 3756.426 | -0.91% | +0.84% | 已核验 |

**结果解释：**候选外logits不变仍会影响概率归一化；必须把实际修正q送verifier。

**原目录／权重入口：**[residualV3-nonthing/results/native_candidate_write_triton_20261009_v1](residualV3-nonthing/results/native_candidate_write_triton_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/native_candidate_write_triton_20261009_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/native_candidate_write_triton_20261009_v1/C40/formal/summary.json)；[RESULTS_CN.txt](residualV3-nonthing/results/native_candidate_write_triton_20261009_v1/RESULTS_CN.txt)。

<a id="n044"></a>

### N044 · 完整头低延迟初试（未采纳）

**类别／状态：**训练／实现计时诊断；已归档。

**做什么：**压低完整形状头耗时且保持bitwise。

**对比什么：**多个优化候选与原实现。

**实验设置：**A100/BF16；同已完成主体、修复头和basis；实际硬lag-1递推与原生采样；block16=anchor+15proposal，N1/N40、T0/T1。微测只报告算子毫秒；若有正式窗口，逐窗口设置另列。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**未同时满足一致性和四种设置≥3%提速门槛；未替换原头，负结果保留。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n045"></a>

### N045 · 修复后q风险驱动的一位置延迟尾部停修

**类别／状态：**已完成／仅停止尾部修复；complete；已记录16个正式窗口。

**做什么：**判断已修复前缀的拒绝风险，与下一位置修复流水线重叠，减少无用尾部修复。

**对比什么：**同一联合主体及三层头：关闭头、FULL15、顺序预测停修、流水线预测停修；不混用不同主体。

**实验设置：**nonthinking，GSM8K，block16/15proposal；先RAM-only提示分组80/48/48信号拟合及校准，后C1=200/C40=1319、T0/T1、max2096；主体/头/basis冻结。

**即时日志／进度：** [status.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; depth=3; thinking=False`。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 FULL15。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| attempt_04/C1/formal | 1 | 0 | 1 | BODY_ONLY | 7.327339 | 178.486 | -12.57% | -5.92% | 已核验 |
| attempt_04/C1/formal | 1 | 0 | 1 | FULL15 | 8.381226 | 189.715 | +0.00% | +0.00% | 已核验 |
| attempt_04/C1/formal | 1 | 0 | 1 | PIPELINE | 8.362507 | 192.815 | -0.22% | +1.63% | 已核验 |
| attempt_04/C1/formal | 1 | 0 | 1 | SEQUENTIAL | 8.362507 | 188.721 | -0.22% | -0.52% | 已核验 |
| attempt_04/C1/formal | 1 | 1 | 1 | BODY_ONLY | 6.425752 | 157.362 | -13.51% | -6.78% | 已核验 |
| attempt_04/C1/formal | 1 | 1 | 1 | FULL15 | 7.429277 | 168.813 | +0.00% | +0.00% | 已核验 |
| attempt_04/C1/formal | 1 | 1 | 1 | PIPELINE | 7.439199 | 172.415 | +0.13% | +2.13% | 已核验 |
| attempt_04/C1/formal | 1 | 1 | 1 | SEQUENTIAL | 7.439199 | 168.680 | +0.13% | -0.08% | 已核验 |
| attempt_04/C40/formal | 40 | 0 | 1 | BODY_ONLY | 7.477374 | 3163.224 | -12.31% | -5.53% | 已核验 |
| attempt_04/C40/formal | 40 | 0 | 1 | FULL15 | 8.527382 | 3348.350 | +0.00% | +0.00% | 已核验 |
| attempt_04/C40/formal | 40 | 0 | 1 | PIPELINE | 8.504120 | 3326.624 | -0.27% | -0.65% | 已核验 |
| attempt_04/C40/formal | 40 | 0 | 1 | SEQUENTIAL | 8.504120 | 3323.506 | -0.27% | -0.74% | 已核验 |
| attempt_04/C40/formal | 40 | 1 | 1 | BODY_ONLY | 6.414621 | 2736.368 | -16.33% | -9.37% | 已核验 |
| attempt_04/C40/formal | 40 | 1 | 1 | FULL15 | 7.666585 | 3019.332 | +0.00% | +0.00% | 已核验 |
| attempt_04/C40/formal | 40 | 1 | 1 | PIPELINE | 7.653728 | 3003.466 | -0.17% | -0.53% | 已核验 |
| attempt_04/C40/formal | 40 | 1 | 1 | SEQUENTIAL | 7.653728 | 3000.173 | -0.17% | -0.63% | 已核验 |

**结果解释：**四组C1/C40正式窗口全部完成且隔离核验通过。流水线相对FULL15：C1 T0/T1 TPS +1.63%/+2.13%，AL -0.22%/+0.13%；C40 TPS -0.65%/-0.53%，AL -0.27%/-0.17%。只停repair未形成C40提速。BODY_ONLY是联合主体关闭头，不能作为独立单训DFlash。后续正式对照按用户要求改用独立单训DFlash。

诊断／原报告摘录（保持原口径）：

- 性能窗口只采正式结果；信号采集和micro不是服务TPS。额外进程只记录，不停他人任务。
- 微基准通过32组：n=1/7/18/40，T0/T1，强制15/12/10与动态决定；原始数学参考、顺序/流水线、采样和verifier q逐位一致，停止后的重算tile下降。
- 正式结果：并发 | 温度 | 组别 | AL | TPS | 相对FULL AL | 相对FULL TPS | 修复收益保留 | 平均修复位置
- 收益保留=(候选AL-同一联合主体关闭头AL)/(FULL15AL-同一联合主体关闭头AL)。95%目标是保留修复带来的增益，不是允许完整AL下降5%。
- 单轮开发面板观测不能证明普遍性能优势；真实状态的head逐位一致不能扩张为C40跨运行最终文本完全一致。

**原目录／权重入口：**[residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [attempt_04/C1/formal/summary.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/attempt_04/C1/formal/summary.json)；[attempt_04/C40/formal/summary.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/attempt_04/C40/formal/summary.json)；[RUN_GUIDE_CN.txt](RUN_GUIDE_CN.txt)；[status.json](status.json)；[RESULTS_CN.txt](RESULTS_CN.txt)；[ACTIVE_ATTEMPT.json](ACTIVE_ATTEMPT.json)；[RESULTS_CN.txt](calibration/RESULTS_CN.txt)；[predictor.json](calibration/predictor.json)；[MICRO_GATE.json](attempt_02/MICRO_GATE.json)；[native_protocol.json](attempt_02/C1/native_protocol.json)；[native_protocol.json](attempt_02/C40/native_protocol.json)；[attempt_01_failure.json](attempt_01_failure.json)；[micro_lineage.json](attempt_03/micro_lineage.json)；[native_protocol.json](attempt_03/C1/native_protocol.json)；[native_protocol.json](attempt_03/C40/native_protocol.json)；[STATS_RECOVERY_CN.txt](STATS_RECOVERY_CN.txt)；[micro_lineage.json](attempt_04/micro_lineage.json)；[native_protocol.json](attempt_04/C1/native_protocol.json)；[native_protocol.json](attempt_04/C40/native_protocol.json)；[EVIDENCE_AUDIT.json](EVIDENCE_AUDIT.json)；[C1_NATIVE_GATE_VERIFIED.json](C1_NATIVE_GATE_VERIFIED.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[RESULTS_CN.txt](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/RESULTS_CN.txt)；[calibration/RESULTS_CN.txt](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/calibration/RESULTS_CN.txt)；[preregistration.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/preregistration.json)；[attempt_02/preregistration.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/attempt_02/preregistration.json)；[attempt_03/preregistration.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/attempt_03/preregistration.json)；[attempt_04/preregistration.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/attempt_04/preregistration.json)；[calibration/preregistration.json](residualV3-nonthing/results/adaptive_repair_pipeline_20261009_v1/calibration/preregistration.json)。

<a id="n046"></a>

### N046 · q45辅助训练的初始过渡记录

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留被epoch2 q45/latent联合训练替代的准备记录。

**对比什么：**无有效模型或正式AL/TPS比较。

**实验设置：**原状态及日志见归档。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**仅初始准备文件；有效epoch2训练及辅助权重在完整E2根中保留。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n047"></a>

### N047 · epoch1与epoch2：单训DFlash及DFlash＋三层修复头正式对比

**类别／状态：**实验；complete；已记录16个正式窗口。

**做什么：**比较增加一轮训练对单训DFlash和联合修复模型的接收长度与端到端吞吐的影响

**对比什么：**E1_DFLASH、E1_JOINT_HEAD、E2_DFLASH、E2_JOINT_HEAD；各自独立单训基线，不测联合主体关闭头；同一优化实现和同一固定面板

**实验设置：**GSM8K nonthinking；C1=200/卡5，C40=1319/卡6，同时运行；T0/T1，block16含anchor+15 proposal，max_new2096/context4096，BF16，CUDA Graph，target+draft共卡；头为lean_combined优化三层no-soft，辅助预测器和截断关闭；先smoke再正式，各1轮；原权重及完整断点保留

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | E1_DFLASH | 7.384246 | 179.691 | — | — | 已核验 |
| C1/formal | 1 | 0 | 1 | E1_JOINT_HEAD | 8.381226 | 190.532 | — | — | 已核验 |
| C1/formal | 1 | 0 | 1 | E2_DFLASH | 7.478576 | 181.919 | — | — | 已核验 |
| C1/formal | 1 | 0 | 1 | E2_JOINT_HEAD | 8.515070 | 193.191 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | E1_DFLASH | 6.331330 | 154.759 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | E1_JOINT_HEAD | 7.429277 | 169.212 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | E2_DFLASH | 6.326140 | 154.769 | — | — | 已核验 |
| C1/formal | 1 | 1 | 1 | E2_JOINT_HEAD | 7.495970 | 170.657 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | E1_DFLASH | 7.560736 | 3197.935 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | E1_JOINT_HEAD | 8.527382 | 3347.124 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | E2_DFLASH | 7.609462 | 3217.680 | — | — | 已核验 |
| C40/formal | 40 | 0 | 1 | E2_JOINT_HEAD | 8.651926 | 3400.578 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | E1_DFLASH | 6.382050 | 2705.024 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | E1_JOINT_HEAD | 7.666585 | 3017.241 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | E2_DFLASH | 6.483497 | 2751.453 | — | — | 已核验 |
| C40/formal | 40 | 1 | 1 | E2_JOINT_HEAD | 7.709856 | 3032.491 | — | — | 已核验 |

**结果解释：**已完整结束，16个正式窗口均通过请求数、吞吐隔离及target/draft图回放检查，未出现eager/fallback。epoch2联合模型相对同轮独立单训DFlash：C1 T0/T1接收长度+13.86%/+18.49%，吞吐+6.20%/+10.27%；C40 T0/T1接收长度+13.70%/+18.92%，吞吐+5.68%/+10.21%。epoch2相对epoch1联合模型，四个窗口接收长度增加0.56%–1.60%，吞吐增加0.51%–1.60%，本面板上的第二轮增量较小。单训DFlash第二轮增量也较小，C1 T1接收长度-0.08%、吞吐+0.01%，基本不变。本次为单训练seed、单正式轮次，不据此宣称已完全收敛或跨任务稳定收益。

**原目录／权重入口：**[residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [summary.json](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/summary.json)；[C1/formal/summary.json](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[RESULTS_CN.txt](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/RESULTS_CN.txt)。

<a id="n048"></a>

### N048 · E2代码与对话：DSpark六套数据C1/C32重测

**类别／状态：**实验；complete；已记录48个正式窗口。

**做什么：**按用户要求补齐DSpark代码与对话的六套评测；只使用E2独立与联合三层权重。

**对比什么：**E2独立单训DFLASH vs E2联合主体＋三层FULL15_NOFILL，无专家路由、无截断。

**实验设置：**HumanEval164/MBPP256/LiveCodeBench500/MT-Bench80/Alpaca500/Arena-Hard-v2 500；作者seed980406抽样与首轮，max2048；C1/C32 T0/T1。KV固定52/44GiB，util0.92；Arena上下文12288、其余4096；两张空卡上限。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/status.json)。

正式窗口设置：C1=164条，proposal=15，max_new=2048; C1=256条，proposal=15，max_new=2048; C1=500条，proposal=15，max_new=2048; C1=80条，proposal=15，max_new=2048; C32=164条，proposal=15，max_new=2048; C32=256条，proposal=15，max_new=2048; C32=500条，proposal=15，max_new=2048; C32=80条，proposal=15，max_new=2048。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| alpaca/C1/formal | 1 | 0 | 1 | DFLASH | 2.929944 | 71.966 | +0.00% | +0.00% | 已核验 |
| alpaca/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 3.191733 | 75.870 | +8.93% | +5.43% | 已核验 |
| alpaca/C1/formal | 1 | 1 | 1 | DFLASH | 2.632960 | 64.697 | +0.00% | +0.00% | 已核验 |
| alpaca/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 2.945342 | 69.964 | +11.86% | +8.14% | 已核验 |
| alpaca/C32/formal | 32 | 0 | 1 | DFLASH | 2.930023 | 1149.874 | +0.00% | +0.00% | 已核验 |
| alpaca/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 3.192186 | 1191.603 | +8.95% | +3.63% | 已核验 |
| alpaca/C32/formal | 32 | 1 | 1 | DFLASH | 2.634266 | 1005.471 | +0.00% | +0.00% | 已核验 |
| alpaca/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 2.957605 | 1095.150 | +12.27% | +8.92% | 已核验 |
| arena-hard-v2/C1/formal | 1 | 0 | 1 | DFLASH | 3.340953 | 80.488 | +0.00% | +0.00% | 已核验 |
| arena-hard-v2/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 3.541867 | 82.787 | +6.01% | +2.86% | 已核验 |
| arena-hard-v2/C1/formal | 1 | 1 | 1 | DFLASH | 2.680992 | 64.614 | +0.00% | +0.00% | 已核验 |
| arena-hard-v2/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 2.918541 | 68.150 | +8.86% | +5.47% | 已核验 |
| arena-hard-v2/C32/formal | 32 | 0 | 1 | DFLASH | 3.340460 | 1213.686 | +0.00% | +0.00% | 已核验 |
| arena-hard-v2/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 3.542392 | 1220.883 | +6.05% | +0.59% | 已核验 |
| arena-hard-v2/C32/formal | 32 | 1 | 1 | DFLASH | 2.674198 | 957.757 | +0.00% | +0.00% | 已核验 |
| arena-hard-v2/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 2.912397 | 977.710 | +8.91% | +2.08% | 已核验 |
| humaneval/C1/formal | 1 | 0 | 1 | DFLASH | 5.541694 | 134.406 | +0.00% | +0.00% | 已核验 |
| humaneval/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 6.344182 | 148.609 | +14.48% | +10.57% | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | DFLASH | 4.839549 | 117.500 | +0.00% | +0.00% | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 5.731345 | 134.402 | +18.43% | +14.38% | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | DFLASH | 5.541621 | 1999.067 | +0.00% | +0.00% | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 6.343126 | 2175.199 | +14.46% | +8.81% | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | DFLASH | 4.810451 | 1793.421 | +0.00% | +0.00% | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 5.677878 | 1987.046 | +18.03% | +10.80% | 已核验 |
| livecodebench/C1/formal | 1 | 0 | 1 | DFLASH | 6.139662 | 143.757 | +0.00% | +0.00% | 已核验 |
| livecodebench/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 6.733483 | 152.066 | +9.67% | +5.78% | 已核验 |
| livecodebench/C1/formal | 1 | 1 | 1 | DFLASH | 4.493960 | 105.680 | +0.00% | +0.00% | 已核验 |
| livecodebench/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 5.204274 | 118.054 | +15.81% | +11.71% | 已核验 |
| livecodebench/C32/formal | 32 | 0 | 1 | DFLASH | 6.140149 | 1732.398 | +0.00% | +0.00% | 已核验 |
| livecodebench/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 6.736908 | 1768.781 | +9.72% | +2.10% | 已核验 |
| livecodebench/C32/formal | 32 | 1 | 1 | DFLASH | 4.495382 | 1181.732 | +0.00% | +0.00% | 已核验 |
| livecodebench/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 5.116651 | 1276.672 | +13.82% | +8.03% | 已核验 |
| mbpp/C1/formal | 1 | 0 | 1 | DFLASH | 4.975015 | 121.239 | +0.00% | +0.00% | 已核验 |
| mbpp/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 5.694426 | 134.211 | +14.46% | +10.70% | 已核验 |
| mbpp/C1/formal | 1 | 1 | 1 | DFLASH | 4.324796 | 105.767 | +0.00% | +0.00% | 已核验 |
| mbpp/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 5.028865 | 118.838 | +16.28% | +12.36% | 已核验 |
| mbpp/C32/formal | 32 | 0 | 1 | DFLASH | 4.980480 | 1968.256 | +0.00% | +0.00% | 已核验 |
| mbpp/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 5.693665 | 2131.298 | +14.32% | +8.28% | 已核验 |
| mbpp/C32/formal | 32 | 1 | 1 | DFLASH | 4.268837 | 1458.765 | +0.00% | +0.00% | 已核验 |
| mbpp/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 4.986037 | 1816.781 | +16.80% | +24.54% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | DFLASH | 3.144794 | 77.253 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 3.412641 | 81.210 | +8.52% | +5.12% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | DFLASH | 2.765435 | 67.968 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 3.189820 | 75.802 | +15.35% | +11.53% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | DFLASH | 3.147902 | 963.736 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 3.414774 | 987.092 | +8.48% | +2.42% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | DFLASH | 2.778786 | 824.265 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 3.184671 | 900.337 | +14.61% | +9.23% | 已核验 |

**结果解释：**complete；48/48正式窗、24/24配对、16000请求，整批最终审计通过。隔离、speed_evidence、target/draft Graph、12组T0逐题输出一致及源码/权重/依赖身份通过。E2独立单训与联合三层主体权重不同；新max2048/seed980406批次只作同窗比较，旧成绩保留。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [humaneval/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/humaneval/C1/formal/summary.json)；[humaneval/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/humaneval/C32/formal/summary.json)；[mbpp/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/mbpp/C1/formal/summary.json)；[mbpp/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/mbpp/C32/formal/summary.json)；[livecodebench/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/livecodebench/C1/formal/summary.json)；[livecodebench/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/livecodebench/C32/formal/summary.json)；[mt-bench/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/mt-bench/C1/formal/summary.json)；[mt-bench/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/mt-bench/C32/formal/summary.json)；[alpaca/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/alpaca/C1/formal/summary.json)；[alpaca/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/alpaca/C32/formal/summary.json)；[arena-hard-v2/C1/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/arena-hard-v2/C1/formal/summary.json)；[arena-hard-v2/C32/formal/summary.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/arena-hard-v2/C32/formal/summary.json)；[RESULTS_CN.md](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/RESULTS_CN.md)；[plan.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/plan.json)；[DATASET_AUDIT.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/DATASET_AUDIT.json)；[READY_AUDIT.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/READY_AUDIT.json)；[status.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/status.json)；[parallel/controller.log](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/parallel/controller.log)；[COMPARISON.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/COMPARISON.json)；[FINAL_AUDIT.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/FINAL_AUDIT.json)；[parallel/DISPATCH_MANIFEST.json](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/parallel/DISPATCH_MANIFEST.json)；[COMPLETION_AUDIT.json](residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1/COMPLETION_AUDIT.json)。

<a id="n049"></a>

### N049 · E2前两层推理 · 三领域C1/C32

**类别／状态：**实验；incomplete_failed_humaneval_groups_no_active_workers；已记录16个正式窗口。

**做什么：**验证固定E2三层权重在推理省去第三层后的同窗AC/TPS。

**对比什么：**E2独立单训DFLASH vs E2联合主体＋HEAD2_NOFILL；没有两层重训；新专家头训练仍保持三层。

**实验设置：**GSM8K C1=200/C32=1319，HumanEval164、MT-Bench80首轮；T0/T1、max2096/context4096、15proposal、BF16/batch-invariant/CUDA Graph；两层参数3761409，原三层权重保留；采集最多2卡；1vLLM服务+2头训练允许3卡，每组先烟测合同再正式测。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1/status.json)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C1=80条，proposal=15，max_new=2096; C32=1319条，proposal=15，max_new=2096; C32=80条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| gsm8k/C1/formal | 1 | 0 | 1 | DFLASH | 7.478576 | 180.318 | +0.00% | +0.00% | 已核验 |
| gsm8k/C1/formal | 1 | 0 | 1 | HEAD2_NOFILL | 8.329485 | 194.170 | +11.38% | +7.68% | 已核验 |
| gsm8k/C1/formal | 1 | 1 | 1 | DFLASH | 6.335933 | 153.155 | +0.00% | +0.00% | 已核验 |
| gsm8k/C1/formal | 1 | 1 | 1 | HEAD2_NOFILL | 7.393987 | 172.548 | +16.70% | +12.66% | 已核验 |
| gsm8k/C32/formal | 32 | 0 | 1 | DFLASH | 7.609462 | 3036.083 | +0.00% | +0.00% | 已核验 |
| gsm8k/C32/formal | 32 | 0 | 1 | HEAD2_NOFILL | 8.529896 | 3199.097 | +12.10% | +5.37% | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | DFLASH | 6.483254 | 2610.293 | +0.00% | +0.00% | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | HEAD2_NOFILL | 7.553910 | 2841.216 | +16.51% | +8.85% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | DFLASH | 3.144794 | 77.181 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | HEAD2_NOFILL | 3.366545 | 80.312 | +7.05% | +4.06% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | DFLASH | 2.820589 | 69.259 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | HEAD2_NOFILL | 3.067791 | 73.095 | +8.76% | +5.54% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | DFLASH | 3.147902 | 948.125 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | HEAD2_NOFILL | 3.366868 | 970.096 | +6.96% | +2.32% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | DFLASH | 2.776865 | 886.675 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | HEAD2_NOFILL | 3.094849 | 891.129 | +11.45% | +0.50% | 已核验 |

**结果解释：**4/6组共16正式窗已完成（GSM8K和MT-Bench C1/C32）；HumanEval C1/C32两组烟测重复被外部进程进入污染，未开有效正式配对，原始失败证据保留且不自动重试。当前无GPU worker，空轮询CPU控制器已退役；不把全局状态写成完成。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [gsm8k/C1/formal/summary.json](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1/gsm8k/C1/formal/summary.json)；[gsm8k/C32/formal/summary.json](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1/gsm8k/C32/formal/summary.json)；[mt-bench/C1/formal/summary.json](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1/mt-bench/C1/formal/summary.json)；[mt-bench/C32/formal/summary.json](residualV3-nonthing/results/epoch2_head2_three_domains_C1_C32_20261012_v1/mt-bench/C32/formal/summary.json)。

<a id="n050"></a>

### N050 · E2三领域C1/C32 · 三层头与仅推理前两层直接对照

**类别／状态：**实验；complete；已记录24个正式窗口。

**做什么：**检查省去第三层后接受长度与原生吞吐的取舍。

**对比什么：**HEAD3_NOFILL→HEAD2_NOFILL；同一个E2联合DFlash主体，保留独立单训DFlash对照在原head2根。

**实验设置：**GSM8K C1=200/C32=1319、HumanEval164、MT-Bench80首轮；T0/T1，max2096/context4096、BF16、batch-invariant、CUDA Graph、FULL15，两个头都NOFILL，无截断，C1 formal KV16GiB/C32 44GiB。只推理跳第三层，未做两层重训；独占空卡、24正式窗口。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/status.json)。

正式窗口设置：C1=164条，proposal=15，max_new=2096; C1=200条，proposal=15，max_new=2096; C1=80条，proposal=15，max_new=2096; C32=1319条，proposal=15，max_new=2096; C32=164条，proposal=15，max_new=2096; C32=80条，proposal=15，max_new=2096。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| gsm8k/C1/formal | 1 | 0 | 1 | HEAD2_NOFILL | 8.329485 | 192.787 | — | — | 已核验 |
| gsm8k/C1/formal | 1 | 0 | 1 | HEAD3_NOFILL | 8.515070 | 196.581 | — | — | 已核验 |
| gsm8k/C1/formal | 1 | 1 | 1 | HEAD2_NOFILL | 7.393987 | 171.759 | — | — | 已核验 |
| gsm8k/C1/formal | 1 | 1 | 1 | HEAD3_NOFILL | 7.488577 | 173.838 | — | — | 已核验 |
| gsm8k/C32/formal | 32 | 0 | 1 | HEAD2_NOFILL | 8.529896 | 3194.480 | — | — | 已核验 |
| gsm8k/C32/formal | 32 | 0 | 1 | HEAD3_NOFILL | 8.652069 | 3239.023 | — | — | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | HEAD2_NOFILL | 7.553910 | 2838.108 | — | — | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | HEAD3_NOFILL | 7.699185 | 2876.612 | — | — | 已核验 |
| humaneval/C1/formal | 1 | 0 | 1 | HEAD2_NOFILL | 6.260512 | 147.117 | — | — | 已核验 |
| humaneval/C1/formal | 1 | 0 | 1 | HEAD3_NOFILL | 6.344182 | 148.690 | — | — | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | HEAD2_NOFILL | 5.491384 | 129.431 | — | — | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | HEAD3_NOFILL | 5.687675 | 133.317 | — | — | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | HEAD2_NOFILL | 6.257183 | 2140.922 | — | — | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | HEAD3_NOFILL | 6.343126 | 2176.053 | — | — | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | HEAD2_NOFILL | 5.536863 | 1988.608 | — | — | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | HEAD3_NOFILL | 5.568287 | 1893.288 | — | — | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | HEAD2_NOFILL | 3.366545 | 79.973 | — | — | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | HEAD3_NOFILL | 3.412641 | 80.837 | — | — | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | HEAD2_NOFILL | 3.067791 | 72.798 | — | — | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | HEAD3_NOFILL | 3.121363 | 73.860 | — | — | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | HEAD2_NOFILL | 3.367516 | 981.401 | — | — | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | HEAD3_NOFILL | 3.414774 | 985.774 | — | — | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | HEAD2_NOFILL | 3.103634 | 892.432 | — | — | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | HEAD3_NOFILL | 3.152884 | 957.416 | — | — | 已核验 |

**结果解释：**完成三领域C1/C32、T0/T1共24正式窗/8028请求，179源码/12资产与隔离/Graph/T0逐题输出审计passed。同一个E2联合主体上三层→前两层AC12窗全降、TPS11窗下降，唯一正窗HumanEval C32 T1 +5.035%；各同窗数值见RESULTS_CN.md。没有两层重训；C1 KV16/C32 44GiB，不能与旧C1 KV52GiB报告绝对TPS相除。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [gsm8k/C1/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/gsm8k/C1/formal/summary.json)；[gsm8k/C32/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/gsm8k/C32/formal/summary.json)；[humaneval/C1/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/humaneval/C1/formal/summary.json)；[humaneval/C32/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/humaneval/C32/formal/summary.json)；[mt-bench/C1/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/mt-bench/C1/formal/summary.json)；[mt-bench/C32/formal/summary.json](residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/mt-bench/C32/formal/summary.json)。

<a id="n051"></a>

### N051 · E2三层 C1/C40 · 固定旧正式显存的模块耗时

**类别／状态：**实验；acquisition_complete_with_timing_coverage_limitations。

**做什么：**测量target验证、drafter主体、完整头以及头内模块的真实边界与耗时。

**对比什么：**仅原E2三层NOFILL；真实服务总体分段与同worker真实状态头模块重放分别报告，无模型效果或显存预算对照。

**实验设置：**固定KV C1=52GiB/C40=44GiB、utilization0.92，GSM8K C1=16/C40=80、原生EOS、max2096/context4096，T0/T1、BF16/batch-invariant/Graph。当前三领域直接对照6组审计通过且现存GPU worker退出后才启动。只两项服务工作负载，无预算扫描；头细分使用同worker真实状态重放。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**三层C1/C40 T0/T1计时采集完成，固定旧正式KV52/44GiB、0.92，无预算扫描。数值/Graph/隔离通过；覆盖复核：C40 T0满40组件仅3份，C40 T1满40主体/raw LM/头直接组件缺失，但有157份纯decode混合并发组件及同worker真实状态头模块重放。细事件约增加2.2ms扰动；重放/混合平均不得填补满40直接服务空项。未重跑，诊断不是AC/TPS。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1](residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1)。权重、basis、Adam/RNG断点均保持原位置。

<a id="n052"></a>

### N052 · E2 HumanEval/MT-Bench：独立DFlash与联合完整修复头

**类别／状态：**实验；complete；已记录16个正式窗口。

**做什么：**验证代码与聊天领域的native接受长度及端到端吞吐收益。

**对比什么：**E2独立单训DFLASH vs E2联合主体+完整FULL15_NOFILL头；仅两臂，无尾部截断/辅助预测。

**实验设置：**DSpark全量HumanEval164/MT-Bench80第一轮唯一题；C1/C40、T0/T1、max2096/context4096、BF16、batch-invariant/CUDA Graph；单轮；先实际prompt smoke；所有GPU0–7允许，仅使用宿主空卡。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/status.json)。

正式窗口设置：C1=164条，proposal=15，max_new=2096; C1=80条，proposal=15，max_new=2096; C40=164条，proposal=15，max_new=2096; C40=80条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| humaneval/C1/formal | 1 | 0 | 1 | DFLASH | 5.541694 | 134.114 | +0.00% | +0.00% | 已核验 |
| humaneval/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 6.344182 | 148.662 | +14.48% | +10.85% | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | DFLASH | 4.681263 | 113.522 | +0.00% | +0.00% | 已核验 |
| humaneval/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 5.687675 | 133.365 | +21.50% | +17.48% | 已核验 |
| humaneval/C40/formal | 40 | 0 | 1 | DFLASH | 5.541621 | 2130.519 | +0.00% | +0.00% | 已核验 |
| humaneval/C40/formal | 40 | 0 | 1 | FULL15_NOFILL | 6.343126 | 2289.712 | +14.46% | +7.47% | 已核验 |
| humaneval/C40/formal | 40 | 1 | 1 | DFLASH | 4.681261 | 1708.830 | +0.00% | +0.00% | 已核验 |
| humaneval/C40/formal | 40 | 1 | 1 | FULL15_NOFILL | 5.568287 | 1957.990 | +18.95% | +14.58% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | DFLASH | 3.144794 | 77.578 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 3.412641 | 82.838 | +8.52% | +6.78% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | DFLASH | 2.820589 | 69.598 | +0.00% | +0.00% | 已核验 |
| mt-bench/C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 3.121363 | 75.711 | +10.66% | +8.78% | 已核验 |
| mt-bench/C40/formal | 40 | 0 | 1 | DFLASH | 3.147619 | 973.845 | +0.00% | +0.00% | 已核验 |
| mt-bench/C40/formal | 40 | 0 | 1 | FULL15_NOFILL | 3.414442 | 1012.534 | +8.48% | +3.97% | 已核验 |
| mt-bench/C40/formal | 40 | 1 | 1 | DFLASH | 2.774017 | 920.795 | +0.00% | +0.00% | 已核验 |
| mt-bench/C40/formal | 40 | 1 | 1 | FULL15_NOFILL | 3.154719 | 998.453 | +13.72% | +8.43% | 已核验 |

**结果解释：**2026-10-11 E2跨域对照：HumanEval164题、MT-Bench80题，均为DSpark第一轮全量原题。状态complete，正式窗16/16。humaneval C1 T0 AL+14.48% TPS+10.85% 隔离True；humaneval C1 T1 AL+21.50% TPS+17.48% 隔离True；humaneval C40 T0 AL+14.46% TPS+7.47% 隔离True；humaneval C40 T1 AL+18.95% TPS+14.58% 隔离True；mt-bench C1 T0 AL+8.52% TPS+6.78% 隔离True；mt-bench C1 T1 AL+10.66% TPS+8.78% 隔离True；mt-bench C40 T0 AL+8.48% TPS+3.97% 隔离True；mt-bench C40 T1 AL+13.72% TPS+8.43% 隔离True。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [humaneval/C1/formal/summary.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/humaneval/C1/formal/summary.json)；[humaneval/C40/formal/summary.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/humaneval/C40/formal/summary.json)；[mt-bench/C1/formal/summary.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/mt-bench/C1/formal/summary.json)；[mt-bench/C40/formal/summary.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/mt-bench/C40/formal/summary.json)；[RESULTS_CN.md](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/RESULTS_CN.md)；[COMPARISON.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/COMPARISON.json)；[plan.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/plan.json)；[PREFLIGHT.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/PREFLIGHT.json)；[DSPARK_PROTOCOL_AUDIT.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/DSPARK_PROTOCOL_AUDIT.json)；[status.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/status.json)；[FINAL_AUDIT.json](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/FINAL_AUDIT.json)；[COMPARISON_CN.txt](residualV3_main_v1/research/epoch2_three_domains_comparison_20261012/COMPARISON_CN.txt)；[SOURCE_AUDIT.json](residualV3_main_v1/research/epoch2_three_domains_comparison_20261012/SOURCE_AUDIT.json)；[HEAD_ONLY_COMPARISON.csv](residualV3_main_v1/research/epoch2_three_domains_comparison_20261012/HEAD_ONLY_COMPARISON.csv)。

<a id="n053"></a>

### N053 · 新FULL15_NOFILL与E2独立单训DFlash正式长测

**类别／状态：**实验；complete；已记录8个正式窗口。

**做什么：**直接测量已优化FULL15联合模型对匹配epoch独立单训DFlash的接收长度及端到端吞吐。

**对比什么：**DFLASH=E2独立单训；FULL15_NOFILL=E2联合主体+三层头。不同主体权重的整体模型比较，禁止用联合主体关头替代。

**实验设置：**用户明确要求长测；C1=200/C40=1319，max2096/context4096，T0/T1，BF16、batch-invariant、CUDA Graph；GPU7顺序独占；先smoke再正式；C1 KV52GiB/C40正式44GiB，两臂一致；单轮。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C40_smoke.log)。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | DFLASH | 7.478576 | 178.410 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FULL15_NOFILL | 8.515070 | 196.152 | +13.86% | +9.94% | 已核验 |
| C1/formal | 1 | 1 | 1 | DFLASH | 6.335933 | 151.859 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL15_NOFILL | 7.488577 | 173.253 | +18.19% | +14.09% | 已核验 |
| C40/formal | 40 | 0 | 1 | DFLASH | 7.609462 | 3146.145 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FULL15_NOFILL | 8.652091 | 3365.701 | +13.70% | +6.98% | 已核验 |
| C40/formal | 40 | 1 | 1 | DFLASH | 6.482782 | 2687.595 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FULL15_NOFILL | 7.698110 | 3011.228 | +18.75% | +12.04% | 已核验 |

**结果解释：**2026-10-11新FULL15正式对照：用户明确将短测改为长测；E2独立单训DFlash对新FULL15_NOFILL联合模型。状态complete，正式窗口8/8。C1 T0 AL+13.86% TPS+9.94% 隔离True；C1 T1 AL+18.19% TPS+14.09% 隔离True；C40 T0 AL+13.70% TPS+6.98% 隔离True；C40 T1 AL+18.75% TPS+12.04% 隔离True。三层头/权重未改，仅沿用已核验的局部取消调试预填充；不测尾部截断，不重训；不同主体权重的整体模型对比。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [C1/formal/summary.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/C40/formal/summary.json)；[RESULTS_CN.md](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/RESULTS_CN.md)；[COMPARISON.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/COMPARISON.json)；[native_protocol.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_protocol.json)；[PREFLIGHT.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/PREFLIGHT.json)；[status.json](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/status.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n054"></a>

### N054 · 第二轮尾部截断实现优化：精确q45统计、设备端跳过及细化target图

**类别／状态：**实验；complete；已记录16个正式窗口。

**做什么：**修正截断未充分减少实际计算的问题，固定训练模型、预测器和阈值验证端到端吞吐

**对比什么：**同轮独立单训DFLASH、FULL15、Q45_CUT、Q45_LATENT_CUT；完整头与两种截断均复用第二轮同一联合模型；不测BODY_ONLY

**实验设置：**GSM8K nonthinking；C1=200/卡7，C40=1319/卡6等待旧评测完成且空闲；T0/T1；block16=anchor+15proposal，max_new2096/context4096，BF16；至少修8、patience2、一位置延迟；同冻结校准阈值；C1 target图1–16行，C40 ragged总行最多补7；有效前缀保留原Gumbel数学；先GPU校验、smoke再正式

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | DFLASH | 7.478576 | 179.077 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FULL15 | 8.515070 | 194.096 | +13.86% | +8.39% | 已核验 |
| C1/formal | 1 | 0 | 1 | Q45_CUT | 8.483994 | 176.843 | +13.44% | -1.25% | 已核验 |
| C1/formal | 1 | 0 | 1 | Q45_LATENT_CUT | 8.490007 | 176.611 | +13.52% | -1.38% | 已核验 |
| C1/formal | 1 | 1 | 1 | DFLASH | 6.335933 | 152.399 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL15 | 7.488577 | 171.323 | +18.19% | +12.42% | 已核验 |
| C1/formal | 1 | 1 | 1 | Q45_CUT | 7.445799 | 155.401 | +17.52% | +1.97% | 已核验 |
| C1/formal | 1 | 1 | 1 | Q45_LATENT_CUT | 7.519466 | 157.060 | +18.68% | +3.06% | 已核验 |
| C40/formal | 40 | 0 | 1 | DFLASH | 7.609462 | 3210.494 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FULL15 | 8.652091 | 3399.350 | +13.70% | +5.88% | 已核验 |
| C40/formal | 40 | 0 | 1 | Q45_CUT | 8.621040 | 3164.127 | +13.29% | -1.44% | 已核验 |
| C40/formal | 40 | 0 | 1 | Q45_LATENT_CUT | 8.623583 | 3145.501 | +13.33% | -2.02% | 已核验 |
| C40/formal | 40 | 1 | 1 | DFLASH | 6.482782 | 2734.008 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FULL15 | 7.699185 | 3027.701 | +18.76% | +10.74% | 已核验 |
| C40/formal | 40 | 1 | 1 | Q45_CUT | 7.675979 | 2851.047 | +18.41% | +4.28% | 已核验 |
| C40/formal | 40 | 1 | 1 | Q45_LATENT_CUT | 7.698767 | 2848.653 | +18.76% | +4.19% | 已核验 |

**结果解释：**GPU逐值校验和8个服务smoke通过。正式C40 8/8完成：相对FULL15，Q45_CUT T0/T1接收长度−0.359%/−0.301%，TPS−6.920%/−5.835%；Q45_LATENT_CUT为−0.330%/−0.005%，TPS−7.468%/−5.914%，当前截断无端到端净收益；C1正式继续。独立流水线探针24项一致性检查、72项计时：C40/T0完整15位置6.50ms，已知停10且无预测6.24ms，当前预测双流8.55ms；去掉早期等待几乎无收益，再删末两次无用预测8.44ms，仍慢于完整头。探针未替换正式实现；详细报告在probes/wait_path_20261011/RESULTS_CN.md。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [summary.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/summary.json)；[C1/formal/summary.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[RESULTS_CN.txt](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/RESULTS_CN.txt)；[preregistration.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/preregistration.json)；[service_startup_failed_01/preregistration.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/service_startup_failed_01/preregistration.json)；[service_startup_failed_02/preregistration.json](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/service_startup_failed_02/preregistration.json)。

<a id="n055"></a>

### N055 · 尾部截断短测迭代与CPU同步诊断

**类别／状态：**实验；structural_reaudit_complete_no_stable_tail_cut_gain。

**做什么：**修复尾部截断吞吐下降超过接受长度下降的问题。

**对比什么：**FULL15是固定联合模型控制；外部基线是独立训练epoch2 DFlash，见epoch1/epoch2正式比较。

**实验设置：**GSM8K nonthinking，C1 16题/C40 80题，max256，block16，BF16；8题事件诊断只用于定位等待，不作为吞吐证据；模型权重/阈值不变。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/status.json)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**最新完成的是iter28/29（C1=16/C40=80、max256）：共同概率准备融合/FULL15_NOFILL的TPS变化为C1 T0/T1 −0.487%/−0.065%、C40 +0.055%/+1.346%，AL不变；真正Q45截断/FULL15_PREPARED的TPS为−0.633%/−0.143%、−1.389%/−1.741%，AL为−0.699%/−0.300%、−0.375%/+0.290%，仍无截断净提速。12窗隔离与Graph通过，实际token/前驱/verifier q/后修q45/cut/bonus合同通过；KV为原生更新和发射计数等价，未独立比较KV字节，latent未核验。开发177份runtime及evaluator仍与iter21逐字节一致，选用C1 iter20/C40 iter21；NOFILL正式版与iter26–29均独立冻结。iter22–25等旧负结果保留在短测表及各报告；确定性empty调试预填充已局部取消，NOFILL真实服务头+采样C1为1.299/1.342ms、C40为5.371/6.143ms。probe33决策/熵prefix融合用于C1，probe34概率准备用于两臂，probe36精确缓存预取用于C40；probe35分片数值通过但C40慢37%–44%，未进入native。这些分段与微测不能代替TPS。DEVICE消除了旧D2H同步，但target仍执行固定补齐图；C40现有decode计数只覆盖子集，固定15槽直方图不是真实cut。尚缺iter28/29混合长度服务prepare/cache/跨流等待/target完整归因；当前不新增截断正式长测。2026-10-11本轮目录整理未产生新AL/TPS结果，完整旧追加笔记见整理前registry快照。

短测AL/TPS（不是完整正式测评；事件诊断排除）：

| 迭代 | C | T | FULL15 AL | 截断 AL | AL变化% | FULL15 TPS | 截断 TPS | TPS变化% | 隔离 |
|---|---|---|---|---|---|---|---|---|---|
| iter01 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2901.791531 | 2635.348307 | -9.182025 | True |
| iter01 | 40 | 1 | 7.543002 | 7.586493 | 0.576568 | 2649.402504 | 2477.753892 | -6.478767 | True |
| iter02_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 183.121938 | 164.437875 | -10.203072 | True |
| iter02_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 170.710089 | 154.390640 | -9.559745 | True |
| iter03 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2861.257704 | 2666.197695 | -6.817282 | True |
| iter03 | 40 | 1 | 7.564586 | 7.564687 | 0.001343 | 2627.036502 | 2439.371918 | -7.143585 | True |
| iter06_bounded | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2895.636503 | 2664.561210 | -7.980121 | True |
| iter06_bounded | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2678.651519 | 2487.440837 | -7.138319 | True |
| iter07_device_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 178.726105 | 175.030893 | -2.067528 | True |
| iter07_device_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 165.929203 | 164.357168 | -0.947413 | True |
| iter08_device_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2878.710986 | 2840.367429 | -1.331970 | True |
| iter08_device_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2659.219312 | 2658.544523 | -0.025375 | True |
| iter09_feature_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 178.037927 | 175.393316 | -1.485420 | True |
| iter09_feature_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.054074 | 163.989022 | -1.243602 | True |
| iter10_feature_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2873.301690 | 2846.860437 | -0.920239 | True |
| iter10_feature_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2673.471918 | 2632.472557 | -1.533562 | True |
| iter11_dense_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 178.964789 | 176.050873 | -1.628207 | True |
| iter11_dense_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.559147 | 165.029013 | -0.918673 | True |
| iter12_dense_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2890.810639 | 2840.173704 | -1.751652 | True |
| iter12_dense_C40 | 40 | 1 | 7.543002 | 7.586493 | 0.576568 | 2655.315451 | 2636.682587 | -0.701719 | True |
| iter13_readout_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 178.185085 | 175.954724 | -1.251710 | True |
| iter13_readout_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.248872 | 164.426836 | -1.095969 | True |
| iter14_readout_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2881.034847 | 2848.771626 | -1.119848 | True |
| iter14_readout_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2667.183454 | 2629.123891 | -1.426957 | True |
| iter15_bulk_entropy_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2902.382328 | 2847.198332 | -1.901334 | True |
| iter15_bulk_entropy_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2677.032912 | 2654.611549 | -0.837545 | True |
| iter16_epilogue_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 179.470304 | 176.819313 | -1.477120 | True |
| iter16_epilogue_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.671382 | 165.606305 | -0.639028 | True |
| iter17_epilogue_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2914.757046 | 2851.987661 | -2.153503 | True |
| iter17_epilogue_C40 | 40 | 1 | 7.543002 | 7.586493 | 0.576568 | 2678.654423 | 2650.223312 | -1.061395 | True |
| iter18_sample_max_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 178.893132 | 176.339134 | -1.427667 | True |
| iter18_sample_max_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.508703 | 165.086906 | -0.853887 | True |
| iter19_sample_max_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2908.697699 | 2867.082913 | -1.430702 | True |
| iter19_sample_max_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2683.056448 | 2653.867144 | -1.087912 | True |
| iter20_fused_input_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 179.095642 | 177.150848 | -1.085897 | True |
| iter20_fused_input_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 166.842713 | 165.553704 | -0.772589 | True |
| iter21_fused_input_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2899.820707 | 2864.693069 | -1.211373 | True |
| iter21_fused_input_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2671.802613 | 2648.496016 | -0.872317 | True |
| iter22_fused_export_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 179.323912 | 177.180507 | -1.195270 | True |
| iter22_fused_export_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 167.079633 | 165.617654 | -0.875019 | True |
| iter23_fused_export_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2903.620486 | 2842.586704 | -2.101989 | True |
| iter23_fused_export_C40 | 40 | 1 | 7.543002 | 7.586493 | 0.576568 | 2671.715369 | 2633.365789 | -1.435392 | True |
| iter24_fused_q45_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 181.761663 | 177.562052 | -2.310504 | True |
| iter24_fused_q45_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 168.660143 | 166.491236 | -1.285963 | True |
| iter25_fused_q45_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2881.878444 | 2857.028704 | -0.862276 | True |
| iter25_fused_q45_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2653.249319 | 2629.204432 | -0.906243 | True |
| iter26_no_empty_fill_C1 | 1 | 0 | 8.011261 | 7.955257 | -0.699066 | 181.926451 | 180.449233 | -0.811986 | True |
| iter26_no_empty_fill_C1 | 1 | 1 | 7.446502 | 7.424180 | -0.299761 | 169.941649 | 168.153466 | -1.052234 | True |
| iter27_no_empty_fill_C40 | 40 | 0 | 8.540297 | 8.508300 | -0.374665 | 2911.233616 | 2852.947256 | -2.002119 | True |
| iter27_no_empty_fill_C40 | 40 | 1 | 7.564586 | 7.586493 | 0.289601 | 2670.647291 | 2672.934027 | 0.085625 | True |

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/RESULTS_CN.md)；[status.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/status.json)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C1_smoke.log)；[single_card_C1_short.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C1_short.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C40_smoke.log)；[single_card_C40_short.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/single_card_C40_short.log)；[RESULTS_CN.md](residualV3_main_v1/maintenance/storage_cleanup_20261011_v2/RESULTS_CN.md)；[IMPLEMENTATION_REVIEW_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/IMPLEMENTATION_REVIEW_CN.md)；[SELECTED_IMPLEMENTATION.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/SELECTED_IMPLEMENTATION.json)；[RESTORED_RUNTIME_GATE.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/RESTORED_RUNTIME_GATE.json)；[retired_tail_probes.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_tail_probes.tar.gz)；[retired_tail_results.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_tail_results.tar.gz)；[probes/full_head_29_no_empty_fill/FULL_GATE.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_29_no_empty_fill/FULL_GATE.json)；[probes/full_head_29_no_empty_fill/SUMMARY.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_29_no_empty_fill/SUMMARY.json)；[probes/full_head_29_no_empty_fill/IMPLEMENTATION_COMPARISON.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_29_no_empty_fill/IMPLEMENTATION_COMPARISON.json)；[probes/full_head_29_no_empty_fill/audit.log](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_29_no_empty_fill/audit.log)；[probes/full_head_32_native_cost_snapshot/ANALYSIS.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_32_native_cost_snapshot/ANALYSIS.json)；[probes/full_head_32_native_cost_snapshot/INCREMENTAL_COST.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_32_native_cost_snapshot/INCREMENTAL_COST.json)；[probes/full_head_32_native_cost_snapshot/RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_32_native_cost_snapshot/RESULTS_CN.md)；[probes/full_head_33_fused_tail_decision/RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_33_fused_tail_decision/RESULTS_CN.md)；[probes/full_head_34_fused_probability_prepare/RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_34_fused_probability_prepare/RESULTS_CN.md)；[probes/full_head_35_sharded_histogram/RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_35_sharded_histogram/RESULTS_CN.md)；[probes/full_head_35_sharded_histogram/source_hashes.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_35_sharded_histogram/source_hashes.json)；[probes/PARALLEL_OPTIMIZATION_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/PARALLEL_OPTIMIZATION_CN.md)；[probes/PARALLEL_OPTIMIZATION.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/PARALLEL_OPTIMIZATION.json)；[probes/full_head_37_native_fused_followup/RESULTS_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/RESULTS_CN.md)；[probes/full_head_37_native_fused_followup/progress.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/progress.json)；[probes/full_head_37_native_fused_followup/CRITICAL_PATH_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/CRITICAL_PATH_CN.md)；[probes/full_head_37_native_fused_followup/CONTRACT_AUDIT.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/CONTRACT_AUDIT.json)；[probes/full_head_37_native_fused_followup/FINAL_REVIEW_CN.md](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/FINAL_REVIEW_CN.md)；[probes/full_head_37_native_fused_followup/HANDOFF_PROCESS_SNAPSHOT.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/HANDOFF_PROCESS_SNAPSHOT.json)；[HANDOFF_20261011_CN.md](HANDOFF_20261011_CN.md)；[RESULTS_CN.txt](residualV3_main_v1/maintenance/residualv3_takeover_20261011_native_cost_v1/RESULTS_CN.txt)；[SOURCE_REAUDIT.json](residualV3_main_v1/maintenance/residualv3_takeover_20261011_native_cost_v1/SOURCE_REAUDIT.json)；[WINDOW_ANALYSIS.json](residualV3_main_v1/maintenance/residualv3_takeover_20261011_native_cost_v1/WINDOW_ANALYSIS.json)；[RESOURCE_SNAPSHOT.json](residualV3_main_v1/maintenance/residualv3_takeover_20261011_native_cost_v1/RESOURCE_SNAPSHOT.json)；[inspection.log](residualV3_main_v1/maintenance/residualv3_takeover_20261011_native_cost_v1/inspection.log)；[START_HERE_CN.txt](residualV3-nonthing/current/START_HERE_CN.txt)；[RESULTS_CN.txt](residualV3_main_v1/maintenance/research_cleanup_20261011_v3/RESULTS_CN.txt)；[registry.json](residualV3_main_v1/maintenance/research_cleanup_20261011_v3/before/registry.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[preregistration.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/preregistration.json)。

<a id="n056"></a>

### N056 · E2三领域C32：独立DFlash vs 完整NOFILL头

**类别／状态：**实验；complete；已记录12个正式窗口。

**做什么：**按用户新并发口径补测；保留历史C40。

**对比什么：**E2独立单训DFLASH vs E2联合主体+FULL15_NOFILL；仅两臂，无截断。

**实验设置：**C32 T0/T1；GSM8K1319/HumanEval164/MT-Bench80首轮原题，同原C40面板和顺序；max2096/context4096；KV44GiB正式/38GiB smoke，BF16 batch-invariant CUDA Graph；宿主空卡，所有GPU允许。

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/status.json)。

正式窗口设置：C32=1319条，proposal=15，max_new=2096; C32=164条，proposal=15，max_new=2096; C32=80条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| gsm8k/C32/formal | 32 | 0 | 1 | DFLASH | 7.609462 | 2978.876 | +0.00% | +0.00% | 已核验 |
| gsm8k/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 8.652069 | 3183.149 | +13.70% | +6.86% | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | DFLASH | 6.483254 | 2549.938 | +0.00% | +0.00% | 已核验 |
| gsm8k/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 7.699185 | 2837.134 | +18.75% | +11.26% | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | DFLASH | 5.541621 | 1980.955 | +0.00% | +0.00% | 已核验 |
| humaneval/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 6.343126 | 2167.798 | +14.46% | +9.43% | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | DFLASH | 4.681502 | 1635.411 | +0.00% | +0.00% | 已核验 |
| humaneval/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 5.568287 | 1878.891 | +18.94% | +14.89% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | DFLASH | 3.147902 | 952.499 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 0 | 1 | FULL15_NOFILL | 3.414774 | 982.737 | +8.48% | +3.17% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | DFLASH | 2.776865 | 889.215 | +0.00% | +0.00% | 已核验 |
| mt-bench/C32/formal | 32 | 1 | 1 | FULL15_NOFILL | 3.150885 | 953.469 | +13.47% | +7.23% | 已核验 |

**结果解释：**C32新补测状态complete，同窗对照6/6、已成对报告窗口12/12；最终审计通过。gsm8k C32 T0 AC+13.70% TPS+6.86%；gsm8k C32 T1 AC+18.75% TPS+11.26%；humaneval C32 T0 AC+14.46% TPS+9.43%；humaneval C32 T1 AC+18.94% TPS+14.89%；mt-bench C32 T0 AC+8.48% TPS+3.17%；mt-bench C32 T1 AC+13.47% TPS+7.23%

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [gsm8k/C32/formal/summary.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/gsm8k/C32/formal/summary.json)；[humaneval/C32/formal/summary.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/humaneval/C32/formal/summary.json)；[mt-bench/C32/formal/summary.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/mt-bench/C32/formal/summary.json)；[RESULTS_CN.md](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/RESULTS_CN.md)；[plan.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/plan.json)；[READY_AUDIT.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/READY_AUDIT.json)；[PREFLIGHT.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/PREFLIGHT.json)；[status.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/status.json)；[parallel/controller.log](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/parallel/controller.log)；[gsm8k/C32/PARTIAL_AUDIT.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/gsm8k/C32/PARTIAL_AUDIT.json)；[FINAL_AUDIT.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/FINAL_AUDIT.json)；[parallel/FINAL_PROCESS_SNAPSHOT.json](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/parallel/FINAL_PROCESS_SNAPSHOT.json)。

<a id="n057"></a>

### N057 · 第二轮已训练q45/末层预测头的自适应尾部截断

**类别／状态：**实验；complete；已记录16个正式窗口。

**做什么：**验证第二轮内嵌训练的预测头能否在保留接收收益时减少repair和target验证工作，并获得真实端到端吞吐提升

**对比什么：**同轮独立单训DFLASH、联合模型FULL15、Q45_CUT、Q45_LATENT_CUT；两个CUT停修并截短实际target验证前缀，不测BODY_ONLY

**实验设置：**第二轮权重和两个已训练MLP均冻结；GSM8K train64题校准＋32题检查，与正式test问句不重叠；至少修8位置、连续2次低前缀生存概率、一位置延迟；仅校准选阈值，固定全修路径保留率>=99.5%；C1=200/C40=1319，T0/T1，block16=anchor+15proposal，max2096/context4096，BF16，target+draft同卡，统一FULL_AND_PIECEWISE；当前仅空卡6，串行两路；GPU参考/图等价校验和native smoke通过才进入正式；无需重导出大权重

**即时日志／进度：** [status.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/status.json)；[single_card_C1_formal.log](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/single_card_C1_formal.log)；[single_card_C40_formal.log](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/single_card_C40_formal.log)；[single_card_C1_smoke.log](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/single_card_C1_smoke.log)；[single_card_C40_smoke.log](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/single_card_C40_smoke.log)。

协议字段：`block_size=16; proposal_length=15; native_max_tokens=2096; native_max_model_len=4096`。

正式窗口设置：C1=200条，proposal=15，max_new=2096; C40=1319条，proposal=15，max_new=2096。

比例只在同一运行分组、C/T/轮次内计算；对照为 DFLASH。

| 运行分组 | C | T | 轮 | 对照／方法 | AL | TPS（token/s） | 相对对照AL | 相对对照TPS | 测速证据 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| C1/formal | 1 | 0 | 1 | DFLASH | 7.478576 | 181.034 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 0 | 1 | FULL15 | 8.515070 | 192.100 | +13.86% | +6.11% | 已核验 |
| C1/formal | 1 | 0 | 1 | Q45_CUT | 8.483994 | 172.924 | +13.44% | -4.48% | 已核验 |
| C1/formal | 1 | 0 | 1 | Q45_LATENT_CUT | 8.490007 | 172.090 | +13.52% | -4.94% | 已核验 |
| C1/formal | 1 | 1 | 1 | DFLASH | 6.335933 | 154.364 | +0.00% | +0.00% | 已核验 |
| C1/formal | 1 | 1 | 1 | FULL15 | 7.488577 | 169.527 | +18.19% | +9.82% | 已核验 |
| C1/formal | 1 | 1 | 1 | Q45_CUT | 7.445799 | 152.228 | +17.52% | -1.38% | 已核验 |
| C1/formal | 1 | 1 | 1 | Q45_LATENT_CUT | 7.519466 | 153.359 | +18.68% | -0.65% | 已核验 |
| C40/formal | 40 | 0 | 1 | DFLASH | 7.609462 | 3216.333 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 0 | 1 | FULL15 | 8.652091 | 3401.630 | +13.70% | +5.76% | 已核验 |
| C40/formal | 40 | 0 | 1 | Q45_CUT | 8.621040 | 3144.644 | +13.29% | -2.23% | 已核验 |
| C40/formal | 40 | 0 | 1 | Q45_LATENT_CUT | 8.623583 | 3148.172 | +13.33% | -2.12% | 已核验 |
| C40/formal | 40 | 1 | 1 | DFLASH | 6.482782 | 2752.019 | +0.00% | +0.00% | 已核验 |
| C40/formal | 40 | 1 | 1 | FULL15 | 7.699185 | 3031.696 | +18.76% | +10.16% | 已核验 |
| C40/formal | 40 | 1 | 1 | Q45_CUT | 7.675979 | 2827.474 | +18.41% | +2.74% | 已核验 |
| C40/formal | 40 | 1 | 1 | Q45_LATENT_CUT | 7.698913 | 2833.085 | +18.76% | +2.95% | 已核验 |

**结果解释：**C1四组正式窗口已完成；Q45_CUT T0/T1接收长度相对FULL15 -0.36%/-0.57%，吞吐-9.98%/-10.20%；Q45_LATENT_CUT T0/T1接收长度-0.29%/+0.41%，吞吐-10.42%/-9.54%。核查发现target逻辑行虽下降但C1图行仍补到16，且停止后尾部仍计算q45和采样；属于性能实现未充分省计算，不能归因为预测未训练好。C1 T0文本两个CUT相对FULL15均0/200差异，冻结源码哈希核验通过。具体成本归因待逐模块计时；C40仍运行，当前不下C40结论。 [性能核查](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/PERFORMANCE_DIAGNOSIS_CN.md)。

**原目录／权重入口：**[residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [summary.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/summary.json)；[C1/formal/summary.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/C1/formal/summary.json)；[C40/formal/summary.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/C40/formal/summary.json)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)；[RESULTS_CN.txt](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/RESULTS_CN.txt)；[calibration/RESULTS_CN.txt](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/calibration/RESULTS_CN.txt)；[preregistration.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/preregistration.json)；[calibration/preregistration.json](residualV3-nonthing/results/epoch2_trained_aux_tail_cut_20261010_v1/calibration/preregistration.json)。

<a id="n058"></a>

### N058 · 在线两种非线性辅助读出：v3计算开销校验未通过

**类别／状态：**实验；paused。

**做什么：**在第二轮长训练内公平比较q45非线性读出与q45+repair末层256维非线性读出；辅助梯度完全隔离。

**对比什么：**仅Q45_MLP与Q45_REPAIR_LATENT_MLP；共同样本、T0/T1标签、学习率和更新次数。

**实验设置：**45→32→2与301→32→2，SiLU，各1538/9730参数；65536输入token、1536 anchors×15位置；两独立AdamW .003、wd .001、clip1；输入/标签detach，无native rollout。teacher4 retained，joint5，body7。主完整断点从8847续训。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3/dflash_training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**主损失、99组梯度、CUDA RNG逐位一致，首次完整CUDA断点恢复通过；核心计算1.961815→2.213515秒（+12.83%），超过5%门槛，辅助未启用，主训练继续至8896完整断点。后续v4优化计算顺序和按需捕获表示，保留本版证据，不能据此判断预测能力。

诊断／原报告摘录（保持原口径）：

- 主/辅助完整断点CUDA恢复校验通过：cursor8848。
- 这不是预测能力或训练效果的结论。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v3/RESULTS_CN.txt)。

<a id="n059"></a>

### N059 · 第二轮内嵌两种非线性q预测器：现场校验通过、继续训练

**类别／状态：**实验；complete。

**做什么：**比较只给修复后q45与再给三层repair末层256维表示，是否提高独立置信度读出；直接嵌入当前长训练。

**对比什么：**仅Q45_MLP和Q45_REPAIR_LATENT_MLP；共同q45、样本、teacher-forcing T0/T1标签和更新次数；各自优化器，detach保证repair梯度不受影响。

**实验设置：**45→32→2和301→32→2（SiLU），1538/9730参数；833176条nonthinking、max4096；每包65536输入token，1536anchor整块，15预测位置/block16；独立AdamW .003/wd .001/clip1；均值标准差固定首个新包；仅T0 argmax一致/T1已有OUTPUT overlap标签，无native rollout。teacher4 retained、joint5、单训DFlash7。从8896/global24743完整断点续训；旧92参数线性头不再更新。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/dflash_training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**已经嵌入并启用；同真实1536-anchor包主损失、99组参数梯度和CUDA RNG逐位一致，核心计算1.933626→1.999303秒（+3.3966%）；共享特征62.129ms，两个读出更新合计4.362ms。8897共同完整CUDA断点恢复通过，两支参数/Adam/normalizer/steps一起恢复，随后连续同步更新。主训练原状态继承，teacher未重启，卡5/7各约74GiB池，显存接力已正常退役。记录运行策略与即时日志；尚无held-out AUROC/接受长度/TPS结果，BCE不是这些指标。

诊断／原报告摘录（保持原口径）：

- 原核心前后向1.933626s，加入两分支1.999303s，开销3.3966%。共享特征62.129ms，两个预测器训练合计4.362ms。6对交替，前2对warmup不计；不含主optimizer.step与teacher等待，不等于端到端吞吐或推理TPS。
- 梯度与5%耗时门槛通过，已启用两分支，第8897包后两个步骤均为1/BCE均0.693147（最后层零初始化）。
- 首次共同完整断点cursor8897/global_step24744已经提交，两个训练端CUDA保存/恢复逐位校验通过；含两预测器参数、Adam、normalizer及步数。恢复后已继续更新，两辅助步骤同步推进。后续每200更新和手动请求。
- 当前没有held-out AUROC或native接受长度/TPS结论；训练BCE不能代替这些指标。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/RESULTS_CN.txt)。

<a id="n060"></a>

### N060 · 第二轮q45独立预测器：初次现场校验OOM（已保留并续至v2）

**类别／状态：**实验；paused。

**做什么：**给正在进行的联合长训练添加repair后q45独立线性预测器，核验速度和主梯度隔离。

**对比什么：**同实际在线包、同权重与1536 anchor的原第二轮实现。

**实验设置：**完整断点cursor426；65536输入token/包，1536整块；一个teacher、joint和单训DFlash各一卡；92参数辅助分支。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1/dflash_training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**v1在首次真实包主损失反向发生OOM，未提交新共同进度，已暂停。错误源码与日志保留；v2从原cursor426以expandable_segments重做校验通过并续训。不是正式AL/TPS结果。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n061"></a>

### N061 · 第二轮续训：repair后q45独立预测器，主梯度逐位一致、计算开销+3.39%

**类别／状态：**实验；paused。

**做什么：**测量45维修复后q统计的独立预测器能否低开销加入第二轮长训练；通过后继续。

**对比什么：**同一真实在线包、同主权重/anchor/RNG的原实现；6对交替前后向，前2对warmup。

**实验设置：**从原第二轮cursor426完整续训；GPU4 teacher、GPU5 joint、GPU7单训DFlash；65536输入token/包、1536 anchor整块、block16；45维到2个标量线性读出共92参数；输入/标签detach、独立AdamW(.003,.001)、独立clip1；T0 argmax一致/T1已有OUTPUT overlap。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2/dflash_training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**现场通过：主前后向1.929856→1.995189秒，+3.3854%；辅助直接63.94ms。99个主参数张量梯度、主损失、CUDA RNG逐位一致；完整CUDA断点保存/恢复含预测器通过。已启用继续第二轮；不含主optimizer.step和teacher等待的配对计算比较，不是推理TPS/AL或预测质量结论。实时joint_training.log含AUX_Q45，详细证据见RESULTS_CN.txt与gates/auxiliary_live_gate.json。

诊断／原报告摘录（保持原口径）：

- repair后q45预测器已加入正在进行的第二轮长训练，现场校验通过，继续训练。
- 原主前后向：1.929856秒/更新
- 加入预测器：1.995189秒/更新
- 该前后向比较不含未改变的主优化器step与teacher等待；不能把3.39%直接称为整个训练流水线吞吐损失。
- 正式更新后完整CUDA保存/恢复通过：BF16权重、FP32 master、主Muon/Adam、CPU/CUDA/Python/NumPy RNG及预测器Adam/normalizer/步数均恢复。原断点保留。
- 同包测得主端峰值：对照67.752GiB、加入预测器74.267GiB；采用expandable_segments分配器，1536整块已经通过。
- 此前v1真实包校验发生OOM，没有提交新共同进度；错误原件仍在v1目录。本v2从原cursor426重做校验通过。不能用v1失败当成预测器速度成绩。
- 限制：标签是teacher-forcing训练代理，不是native首拒绝标签；预测器有无实际截断收益、AUROC、AL、推理TPS尚待训练后独立评估。这里仅加入辅助训练，没有部署推理截断。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_q45_teacher1_3gpu_20261010_v2/RESULTS_CN.txt)。

<a id="n062"></a>

### N062 · 全数据第二轮：共享teacher续训单训DFlash与主体+头

**类别／状态：**实验；paused。

**做什么：**从第一轮完成后的两组完整断点各追加全数据一轮；保存可随时暂停恢复的训练状态，并记录后续基线可对齐的训练预算。

**对比什么：**独立单训DFlash vs 主体+no-soft3头联合；分别承接各自第一轮权重/优化器，第二轮同pack与anchor。训练期间不自动测评。

**实验设置：**833176条/1019443945输入token；shuffle43共15854包；65536 token/包、1536 anchor整块、block16含15 proposal、一包一次更新；teacher TP1/DP1一张卡，两独立learner各一张；头CE+D-PACE+OUTPUT，主体raw CE/D-PACE；FP32 master/Muon/Adam全部续接，新1%warmup+cosine周期；只等空卡，每200更新及请求保存两路共同断点。策略文件：residualV3-nonthing/TRAINING_STRATEGY_NONTHINKING.md；即时入口：controller.log/joint_training.log/dflash_training.log。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1/status.json)；[training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1/training.log)；[joint_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1/joint_training.log)；[dflash_training.log](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1/dflash_training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**后台等待服务已启动；GPU实际进度以本目录status.json和current_attempt.json为准。源checkpoint校验和与CPU恢复后下一步一致性已通过；实际CUDA恢复校验在拿到卡后执行。第一轮源权重/完整状态不变。第二轮尚无AL/TPS结果；连续周期追加不等于预先规划两轮cosine。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1)。权重、basis、Adam/RNG断点均保持原位置。

<a id="n063"></a>

### N063 · 每领域一万条 · 冻结E2主体的mismatch专家头（三种子）

**类别／状态：**实验；requested_comparison_finished_timing_acquired_with_coverage_gap_baseline_incomplete。

**做什么：**验证可见mismatch线索的修复分工是否超过通用头、随机分工和同总参数对照。

**对比什么：**GENERAL_ACTIVE、SPECIALIZED2、RANDOM_SPLIT2、GENERAL_TOTAL；KEEP/A/B可见输入router；E2原头保留。

**实验设置：**三领域各10000头训练，router6144/selection3072/独立测试3072，总42288groups缓存已齐；三种子四对照，每个3遍历，每32blocks更新/4blocks microbatch，原CE/D-PACE+OUTPUT；只训新三层头/router，冻结E2body/target/embedding/LM/basis/旧aux。CPU缓存分片进GPU，头allocator cap8GiB；用户最新允许所有卡显存余量共享训练，不停止他人。原生C1/C32另等独占空卡。

**即时日志／进度：** [status.json](residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**完成：三种子四对照12模型各3遍/5613更新，三领域各10000条头训练、3072锁定留出。mismatch双专家+router对同活跃通用头T0一致率−0.212pp、T1 overlap−0.098pp，95%区间均负；T1对随机分工−0.003pp且区间跨0。当前方案未过native候选门槛；静态前缀结果不是AC/TPS。导出异常已补评恢复，训练/router SHA不变，失败log保留。

诊断／原报告摘录（保持原口径）：

- T | E2原头 | 通用头同活跃参数 | 通用头同总参数 | 随机双专家+router | mismatch双专家+router | mismatch oracle
- T | E2原头 | 通用头同活跃参数 | 通用头同总参数 | 随机双专家+router | mismatch双专家+router | mismatch oracle
- T | E2原头 | 通用头同活跃参数 | 通用头同总参数 | 随机双专家+router | mismatch双专家+router | mismatch oracle
- 局限：静态前缀，独立留出使用原来源答案；来源标签不能替代任务语义审核；稀疏分组的BF16差异见TRAIN_GATE.json。

**原目录／权重入口：**[residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2](residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2/RESULTS_CN.txt)。

<a id="n064"></a>

### N064 · mismatch专家/可见router小规模预实验

**类别／状态：**流程试跑，样本不足；complete_offline_pilot。

**做什么：**检验冻结E2主体上的修复动作互补性与可见输入选择价值

**对比什么：**原E2、KEEP、两聚类专门化suffix、激活/训练预算匹配通用suffix、学习router与离线oracle；随机和输入消融

**实验设置：**96完整短序列64–256token，48/16/8/24 group分割，192块2880状态，BF16 HF/SDPA，原repair CE/D-PACE＋OUTPUT，GPU7宿主空卡；teacher-prefix代理，无native AC/TPS

**即时日志／进度：** [status.json](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**96提示仅流程试跑，24条留出不足以判断路线；新计划按用户要求每领域>=10000头训练提示、冻结主体只训头，总量>=42288、3种子，尚未启动。原数值/权重保留，无新native AC/TPS。

诊断／原报告摘录（保持原口径）：

- 样本解释已修正：真正留出仅24提示，且来自E2已用的短序列训练数据；下面数值只描述这批流程试跑，不支持专家路由成功或失败的方向结论。大样本方案见LARGE_SAMPLE_PLAN_CN.txt，尚未启动。
- 分组：专家训练48提示，router训练16，选择超参8，留出24（720状态）；按conversation group隔离。它们均来自E2已用训练数据，不能作为E2独立泛化结果。
- 互补性：A比B的overlap高>0.005占16.81%，B比A高>0.005占20.97%。仅A/B、不含KEEP的oracle=0.587428；比通用头高0.713个百分点，prompt bootstrap 95%区间[0.413, 1.034]。这是离线局部潜力，oracle选择本身不证明router可实现它。
- KEEP贡献：oracle额外加入KEEP再增加0.428个百分点；不能把三动作oracle的全部收益归因于专家分工。
- 路由：相比同预算通用头，overlap变化-0.182个百分点，95%区间[-0.542, 0.201]，尚未胜出；相比原E2的小提升也不能排除额外小样本训练的贡献。
- 开销：只测已准备62维输入→ridge打分/argmax的CUDA Graph，C1约0.0088–0.0089ms，C32约0.0104ms，每位置/整批。排除q45/PCA准备、专家分组和native执行；不能乘15机械相加或换算TPS。
- 隔离/身份：isolation.json启动与采集/训练/收尾检查均无外来GPU进程；源代码与E2 SHA前后通过；FINAL_PROCESS_SNAPSHOT.json确认本轮PID1437038已退出GPU。
- 结论：该轮仅证明采集/小训/评分流程可运行。oracle互补和router未胜通用头都是样本内观察；24提示留出不足以支持或否定研究路线，三个关键证据尚未完成有效验证。

**原目录／权重入口：**[residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1)。权重、basis、Adam/RNG断点均保持原位置。

**原始证据：** [RESULTS_CN.txt](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/RESULTS_CN.txt)；[RESULTS.json](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/RESULTS.json)；[EVIDENCE_AUDIT.json](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/EVIDENCE_AUDIT.json)。

<a id="n065"></a>

### N065 · 退役／维护 · full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1

**类别／状态：**失败与退役记录；paused。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**即时日志／进度：** [status.json](residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/status.json)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**旧全数据计划已暂停；成功新安排为teacher1+两个learner，不恢复旧controller。含basis和权重，全部保留。

**原目录／权重入口：**[residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1](residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1)。权重、basis、Adam/RNG断点均保持原位置。

<a id="n066"></a>

### N066 · 退役／维护 · full_nonthinking_packed65536_a1536_1epoch_2gpu_20261006_v2

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**用户优先短训验证，旧全数据计划deferred，不是科学负结果。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n067"></a>

### N067 · 退役／维护 · fused_output_dpace_20k_packed1536_2gpu_20261007_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**融合训练早期流程，后续v2为主；已有权重仍保留。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n068"></a>

### N068 · 退役／维护 · joint_dflash_head_card_timing_20261007_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**用户澄清卡数定义后主动替换，不是训练方法失败。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n069"></a>

### N069 · 退役／维护 · nosoft3_20k_3epoch_packed65536_a1536_20261006_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**打包初版资源／准备失败，恢复版v3有效。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

<a id="n070"></a>

### N070 · 退役／维护 · nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2

**类别／状态：**失败与退役记录；failed。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**即时日志／进度：** [status.json](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2/status.json)；[training.log](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2/training.log)。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**打包分卡准备／资源失败，有效版v3保留。

**原目录／权重入口：**[residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2](residualV3-nonthing/results/nosoft3_20k_3epoch_packed65536_a1536_2gpu_20261007_v2)。权重、basis、Adam/RNG断点均保持原位置。

<a id="n071"></a>

### N071 · 退役／维护 · q_only_skip_probe_20261009_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**q诊断准备版本，未形成正式结果。 2026-10-09源码及失败记录已逐文件SHA验证归档，原失败目录已移除；没有模型资产。

**原始证据：** [retired_q_probe_code_and_failures.tar.gz](archives/residualV3_ledger_cleanup_20261009_v1/retired_q_probe_code_and_failures.tar.gz)；[RESTORE.json](archives/residualV3_ledger_cleanup_20261009_v1/RESTORE.json)。

<a id="n072"></a>

### N072 · 退役／维护 · q_only_skip_probe_20261009_v2

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**q观察器失败，有效统计由v4重新采集。 2026-10-09源码及失败记录已逐文件SHA验证归档，原失败目录已移除；没有模型资产。

**原始证据：** [retired_q_probe_code_and_failures.tar.gz](archives/residualV3_ledger_cleanup_20261009_v1/retired_q_probe_code_and_failures.tar.gz)；[RESTORE.json](archives/residualV3_ledger_cleanup_20261009_v1/RESTORE.json)。

<a id="n073"></a>

### N073 · 退役／维护 · q_only_skip_probe_20261009_v3

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**q观察器恢复失败，有效统计由v4重新采集。 2026-10-09源码及失败记录已逐文件SHA验证归档，原失败目录已移除；没有模型资产。

**原始证据：** [retired_q_probe_code_and_failures.tar.gz](archives/residualV3_ledger_cleanup_20261009_v1/retired_q_probe_code_and_failures.tar.gz)；[RESTORE.json](archives/residualV3_ledger_cleanup_20261009_v1/RESTORE.json)。

<a id="n074"></a>

### N074 · 退役／维护 · storage_cleanup_20261008_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**磁盘清理及保护核验记录，非模型实验；所有权重留存。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261012_v4/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261012_v4/INDEX.csv)；[RESTORE.json](archives/residualV3_research_cleanup_20261012_v4/RESTORE.json)。

<a id="n075"></a>

### N075 · 退役／维护 · teacher1_vs2_activation_timing_20261008_v1

**类别／状态：**失败与退役记录；已归档。

**做什么：**保留准备阶段与替代关系，避免误启动。

**对比什么：**不作科学方法比较。

**实验设置：**见原日志／协议；没有合格正式窗口时AL/TPS为未测。

**正式接受长度／吞吐：**未测，或未发现可核验的完整正式窗口。诊断数值不替代 AL/TPS。

**结果解释：**资源／数值门槛失败的计时尝试；不能作有效速度对照。

**原始证据：** [retired_diagnostics.tar.gz](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.tar.gz)；[INDEX.csv](archives/residualV3_research_cleanup_20261011_v3/INDEX.csv)；[retired_diagnostics.manifest.json](archives/residualV3_research_cleanup_20261011_v3/retired_diagnostics.manifest.json)；[RESTORE.json](archives/residualV3_research_cleanup_20261011_v3/RESTORE.json)。

## 自动维护与清理记录

本文件由 `residualV3_main_v1/maintenance/experiment_ledger/ledger.py` 维护；独立CPU读进程每30秒检查一次结果文件，不修改冻结训练／评测源码，不发起新训练。
新增实验应先用该脚本 `register` 登记目的、对照、设置和结论；新目录也会自动出现，缺少说明时明确标注待补充。人工解释写入 registry.json，避免下次更新被覆盖。

刷新：`python residualV3_main_v1/maintenance/experiment_ledger/ledger.py refresh`。

2026-10-09清理清单、代码归档、权重前后核验保存在 `residualV3_main_v1/maintenance/experiment_ledger/cleanup/`；没有把负结果当作无用结果删除。

- 当前主结果为E2三层FULL15_NOFILL vs 独立单训DFlash：三领域C1/C40正式与C32补测均完成；AC是接受长度，联合和独立主体权重不同。
- 同一E2联合主体两层/三层直接对照24窗完成，TPS两层11/12窗下降；该对照C1 KV16GiB，与原正式C1 KV52GiB不同，不能跨报告直接比较绝对TPS。
- 三层C1/C40模块计时采集结束，固定KV52/44GiB、util0.92。C40 T1缺满40直接主体/raw LM/头分段，只有不同活跃批大小的服务分段及同真实状态头重放；数值/Graph/隔离通过不代表计时覆盖完整。
- 每领域1万条的30k头训练/评估12模型完成，mismatch双专家相对通用头没有静态净收益，未进入native；全部新头/router/断点/缓存保留。尾部停修iter28/29也没有净TPS收益。
- 原两层vs独立DFlash仅16正式窗有效，HumanEval C1/C32隔离失败仍未完成，停止自动重试。所有本项目GPU worker及最新计时队列已退出，旧控制器不重启。
- 2026-10-12按授权删除淘汰工作目录，代码及负结果原始证据先逐文件SHA256校验归档；E1/E2完整资产、消融权重、全部正式测量行保留。日常只需RESIDUALV3_START_HERE.txt。
