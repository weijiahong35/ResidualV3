# ResidualV3 新对话交接（2026-10-13）

仓库：`/data/weijiahong/speculators-main-lora`。本交接接续2026-10-11／12的历史，重点覆盖本次六数据集E2重测、组会报告修订及最新用户约束。**动态进度快照为北京时间2026-10-13 13:01左右；接手必须重读实时状态。**

## 1. 接手先做什么

1. 读本文件、[结构化状态](HANDOFF_20261013_STATE.json)、[宿主进程快照](HANDOFF_20261013_PROCESS_SNAPSHOT.json)。PID、卡号和计数都是历史快照，需要重新核验UID、cmd和宿主GPU进程。
2. 设 `R=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1`。先读R下 `status.json`、`progress.json`、`RESULTS_CN.md`、`COMPARISON.json`、`plan.json`、`DATASET_AUDIT.json`，看 `parallel/controller.log`。这是**正在运行的新队列**。
3. 读[组会周报](GROUP_MEETING_20261013_CN.md)，其中三项文献已重新核对；来源记录在 `residualV3_main_v1/research/group_meeting_sources_20261013/`。
4. 需要完整实验沿革时读[nonthinking总账](EXPERIMENTS_NONTHINKING.md)、[thinking总账](EXPERIMENTS_THINKING.md)和[旧交接](HANDOFF_20261011_CN.md)。旧交接含后续追加记录，但其中动态状态和部分旧约束不能代替本文件。

**不要再启动另一份parallel.py，不要把当前队列误当“已完成退出”，不要重跑已完成的HumanEval或旧GSM8K／C32／C40实验。** 先核实已有控制器；只继续未完成的授权工作。不会因为换了聊天而自动需要重启实验。

## 2. 目前在跑的六数据集E2重测

用户因HumanEval／MT-Bench题数较少，要求按DSpark的代码、对话评测数据扩展，**只重测epoch2**。代码与结果分别位于同名 `residualV3-nonthing/code/` 和 `results/` 根。

- 两臂：`DFLASH`＝E2独立单训完整DFlash；`FULL15_NOFILL`＝E2联合主体＋完整三层头。两臂主体权重不同，不能用联合主体关头代替独立baseline。
- C1／C32、T0／T1，BF16、batch-invariant、CUDA Graph，nonthinking；固定15 proposal，生成上限2048。辅助预测器、专家路由与尾部截断关闭。
- **固定真实KV预算：C1 52GiB，C32 44GiB，utilization=.92；smoke与formal相同。不要为调度空卡而降低显存。** 实际已分配显存会随模型加载、Graph准备和阶段变化，不等于协议预算被修改。
- HumanEval 164、MBPP 256、LiveCodeBench 500、MT-Bench 80、Alpaca 500、Arena-Hard-v2 500。每模型／温度／并发共2000个来源行；整批48正式窗口、24配对比较、16000请求。
- 五套context4096，Arena-Hard-v2 context12288，容纳最长8431-token原始输入；无prompt截断。
- 来源固定为DeepSpec提交 `005e03b81cec38b7da6399833d609ee89a2587f2`，作者抽样数／顺序、首轮处理与Random(980406)对齐；实际请求种子为980406＋面板index。MBPP 256个来源行只有255个不同题面，是上游重复，按原样保留，未人为扩题。
- 本批只对齐作者的**数据与部分评测口径**；仍用我们的vLLM和15 proposal，DSpark原离线表为7 proposal。不是DSpark完整复现。旧测试为max2096、seed42，不能跨批绝对TPS相除。

### 最新快照

北京时间13:01左右：`valid_formal_windows=10/48`、`paired_comparisons=4/24`。

| 数据集 | C1 | C32 |
|---|---|---|
| HumanEval | 4窗完成 | 4窗完成 |
| MBPP | GPU5运行formal；独立DFlash T0/T1已出，联合头仍在跑 | 未开始 |
| LiveCodeBench | GPU6已由smoke转formal | 未开始 |
| MT-Bench、Alpaca、Arena-Hard-v2 | 排队 | 排队 |

本批HumanEval C1在12:49:50、C32在12:59:10左右完成。上一条“12:57时C32仅完成DFlash”的答复已过时。其8正式窗的隔离、speed_evidence、target／draft Graph计数均已核验；均无eager或fallback。整批最终审计尚未完成，不能写成48窗已通过。

| 数据集 | 设置 | AC：单训→三层联合头 | AC增长 | 吞吐：单训→三层联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| HumanEval | C1 T0 | 5.542→6.344 | +14.48% | 134.4→148.6 | +10.57% |
| HumanEval | C1 T1 | 4.840→5.731 | +18.43% | 117.5→134.4 | +14.38% |
| HumanEval | C32 T0 | 5.542→6.343 | +14.46% | 1999.1→2175.2 | +8.81% |
| HumanEval | C32 T1 | 4.810→5.678 | +18.03% | 1793.4→1987.0 | +10.80% |

上表取本批 `COMPARISON.json` 未舍入原值，不覆盖周报里的旧批次HumanEval成绩。AC含bonus，不是pass@1；此评测未执行代码正确率或对话judge评分。

### 控制器与日志

- 控制器快照：PID4138310、UID1024，命令是本批code根 `parallel.py`＋本批results根。启动北京时间10:07:44；实际核验仍存活。
- MBPP C1 evaluator快照PID472864；LiveCodeBench C1 formal evaluator快照PID608471。对应宿主EngineCore在GPU5／6，见进程快照。不要对照早期PID执行kill。
- 总日志：[parallel/controller.log](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/parallel/controller.log)。每数据集即时log为 `{dataset}_C{1|32}_{smoke|formal}.log`，正式分窗在 `{dataset}/C{1|32}/formal/`。
- 环境：`/data/weijiahong/miniconda3/envs/dflash/bin/python -B`；CUDA兼容设置复用 `native_runtime/runtime/utils.py` 的environment。不要用系统Python启动GPU入口。
- 队列全局与每卡使用flock，最多两张空卡。外来进程介入会归档污染尝试，再选空卡；前期多次这种重排导致延迟，失败证据在 `prior_attempts/occupation_*`，不是有效成绩。
- 控制器自动每15秒调用 `update_report.py`；结束时调用 `verify.py`。不要手工改自动生成的R/RESULTS_CN.md或覆盖正在写入的summary。
- `plan.json`中183份源码SHA本轮重新检查无差异；3份调度／报告／审计脚本身份见 `parallel/DISPATCH_MANIFEST.json`。运行期间不要编辑冻结源码。

## 3. 核心模型、训练与保护资产

epoch12＝epoch1＋epoch2，两轮都已完成，**没有epoch3授权**。两个独立learner从同一已下载nonthinking DFlash起点追加训练；不能称从随机初始化训练。

- E1：`residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1`。
- E2：`residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4`。
- 各保留 `final/body_epoch_0{1|2}.safetensors`、`joint_epoch_0{1|2}.safetensors`，以及 `checkpoints/slot_0/{body,joint}.state.bin`；本轮检查均存在。FP32 master／Muon／Adam／RNG／data cursor完整共同断点必须保留。final身份见各自manifest；E2已记录body SHA `3ec7b0b39bcbd983f7a1acd3fb0eda6317ebc6bea1a0d5d5ef4f3f09147ff23b`，joint SHA `7fad88ba97c5fc424483eb56b78ca83a2ba806e79b2541f4e617bb985be8449d`。本轮未重新计算大权重SHA，不冒充重新哈希。
- 每轮833,176有效训练会话组、约1.019B输入token；max4096，每包≤65536输入token、≤1536完整anchors，block16＝anchor＋15 proposal，一包更新一次。E1 15847更新，E2 15854更新，合计31701／模型。
- 一个共享冻结teacher供应两个独立learner，不合并梯度；独立分支raw CE/D-PACE；联合分支raw CE/D-PACE＋repair CE/D-PACE＋repair OUTPUT。teacher-prefix监督，不是native rollout。
- 主体5层；头3个256维no-soft SwiGLU残差块，inner768，basis rank512，Top256修正，剂量上限.8，新增4,353,537参数。hidden／实际前驱embedding／当前q加权basis三路输入→融合→3块→方向＋剂量→局部logit残差→完整词表采样。
- E2后段detach q45与q45+latent辅助MLP训练完；正式FULL关闭，不把它们计作本轮方法收益。

E1/E2、消融权重、旧E2 plan/order、basis、E1 panels和共享cache都是依赖，不能删。不要恢复早期on-policy昂贵监督；历史约慢8倍且无一致收益。

## 4. 已完成成绩与计时：不能重复启动

周报主表是**旧批次**三层E2的GSM8K／HumanEval／MT-Bench，C1／C32共12个配对；全部AC与TPS为正。C1 TPS +6.78%～17.48%，C32 +3.17%～14.89%。旧C40结果另保留，不把C40标成C32。

主要报告入口：

1. `residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1`：新NOFILL对独立DFlash正式8窗，C1=200/C40=1319，max2096。
2. `residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1`：旧HumanEval／MT-Bench C1/C40。
3. `residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1`：已完成旧三领域C32补测。
4. `residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1`：24窗完成。两层仅用保存的三层权重跳第三层，未重训；不是新两层训练架构。该比较C1 KV16GiB，与正式52GiB不同，只比同窗；不再重跑。
5. `residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1`：仅三层，KV52/44GiB、util.92，采集结束。没有C32分段计时。C40 T0满40直接组件仅3份，T1缺满40直接分段，见TIMING_COVERAGE_CN.md；不能把活跃批大小混合样本称满40。

C1 T0/T1的ms／批／block：主体4.460/4.461，raw LM1.544/1.545，头＋采样1.297/1.340，完整propose8.058/8.095，target forward31.936/31.955，target起点至验证终点33.511/33.545。跨度嵌套，不能相加；不是每token延迟。头内细分事件重放有约2.2ms扰动，不能当真实服务成本。

E1/E2统一实现历史16窗、RAW历史接受长度、NOFILL后续实现差异已合并进周报第2节。empty局部取消、Graph模式和KV预算／批次不同，不能把跨报告吞吐差全算作训练轮数或某一kernel优化。

## 5. 周报与三项相关工作：已经纠正的口径

用户要**我们的完整三层模型，对比对方加头版本和完整版本；各行相对各自完整已训练DFlash报告增益**。不是只比较参数量，也不是把联合主体关头冒充独立单训baseline。

周报为[GROUP_MEETING_20261013_CN.md](GROUP_MEETING_20261013_CN.md)。结构全部用文字箭头；公式改为VS Code Markdown支持的 `$...$`／`$$...$$`；正文五大节，保留RAW、E1/E2、模块时间。用户明确**专家路由先不写进周报**。

- **Domino**：[原论文v1](https://arxiv.org/html/2605.29707v1)。主表1／2用作者OPB训练的Domino对公开 `z-lab/Qwen3-8B-DFlash-b16`，不是其OPB流程独立重训DFlash。表3双方ShareGPT重训，是更匹配我们口径的主要参照，架构与主实验相同；不能命名成“头版本”对“完整版本”。表3AC未分并发，TPS由AR TPS×已舍入速度比推算，必须加≈。表2 C32温度未注明，不补成T0。表6同联合模型关／开头不能代替独立baseline。
- **DSpark**：[原论文v1](https://arxiv.org/html/2607.05147v1)。Qwen3-8B表1双方OPB重训、T1、7 proposal，**关闭验证截断**，只测固定长度草稿质量。完整结构另有confidence＋校准＋hardware-aware target前缀调度。周报新增4.4：图5静态裁剪接受率诊断；第5节动态截断生产系统以MTP-1为基线。Flash在≥80 token/s/user条件总吞吐+51%，Pro在≥35条件+52%；同总吞吐下单用户速度分别+60%～85%、+57%～78%。这是完整系统收益，不是截断单模块或DFlash收益，不能填进三领域DFlash吞吐列。
- **DFlash-2**：[作者技术报告](https://inco.ai/blog/dflash2/)＋[Muse模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2)。不要用原DFlash论文代替DFlash-2来源。表2“选择头开／卷积关”只公开Qwen3-4B GSM8K，T0 AC4.27→4.61(+7.96%)、T1 3.78→4.25(+12.43%)。**未找到HumanEval或MT-Bench对应的单独头消融**；HumanEval表3是Qwen3.5-4B卷积＋选择头，5.43→6.28(+15.65%)，不能全归因于头。Muse完整结果另有C1/C32 TPS，target／H200／reasoning设置不同。

所有来源快照、SHA、机器抽取表格、42主比较行及文档审计在 `residualV3_main_v1/research/group_meeting_sources_20261013/`。新增DSpark证据为 `DSPARK_TRUNCATION.json`。来源可直接核对，缺项写未报告，不能自行补齐并发／温度／TPS。

用户问A会可行性时的判断：当前有可信效果基础，尚缺同条件近邻头对照、q／方向剂量／OUTPUT在当前配方的贡献归因、多target与关键测量重复。单个头可以构成论文，关键是明确区别和证据；模块数量不是门槛。该讨论不是授权现在额外启动大规模训练或基线复现。

## 6. 专家路由与尾部停修的负结果

- [30k专家结果](residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2/RESULTS_CN.txt)：每领域10000头训练，router6144、选型3072、锁定留出3072；3种子×4对照12模型全部完成。冻结E2主体，只训新头/router。
- mismatch learned router对同活跃通用头T0一致率−0.212pp（95% CI [−.304,−.115]）、T1 overlap−0.098pp（[−.140,−.052]），三种子均负；没有native AC/TPS，不把oracle改善当可部署收益。
- 尾部停修iter28／29短测完成，真正CUT/FULL四窗TPS均负，未获净提速。DEVICE逻辑prefix消除同步，但target仍有图补齐及BM16 tile计算；不宣称已物理压缩所有验证计算，不重启旧run_pair/run_short_iteration。
- NOFILL、iter28/29、开发版／SELECTED身份不同；SELECTED仍C1 iter20/C40 iter21。潜在target压缩、主体mismatch改进只是研究方向，尚未形成新的稳定正结果。

## 7. 最新硬约束与后续工作

- **显存预算固定，当前重测最多同时占两张空卡。GPU0–7都允许，旧“避开1/2”已被用户撤销。** 训练阶段“teacher一张＋两个trainer可占三张”及共享卡训练曾另获授权，不适用于当前独占吞吐评测队列。宿主检查必须可靠，不能用沙箱看不见进程当空卡证据。
- 磁盘口径用户明确更正为 **`du -sh /data/weijiahong` 的600G上限（二进制）**，不是600GB十进制；本轮重新测523G。协议为644245094400 bytes，保留6GiB工作余量。
- 不停止他人、其他项目或长训；不恢复退役控制器、用户停止的旧VERIFY_STOP；不改变运行中的冻结实现／权重／阈值／显存预算。
- 2026-10-12清理v4已完成，152淘汰目录先逐文件SHA归档再删除；全部E1/E2资产、消融权重与正式行保留。档案在 `archives/residualV3_research_cleanup_20261012_v4/`；硬链接二进制修改前必须独立copy。不要再为“整理”删依赖。
- 总账由 `residualV3_main_v1/maintenance/experiment_ledger/registry.json` 在ledger.lock下维护，然后 `ledger.py refresh`；不能只改自动生成EXPERIMENTS_*.md。
- 下一步沿已有队列完成剩余数据：核对当前worker状态、同窗指标、隔离及Graph；整批完成后检查 `verify.py`最终审计，记录全部48窗／24配对与源码身份。随后在registry登记／更新本轮，refresh两总账，再将新批次结果整理进周报，保留旧批次为历史并标明协议变化。
- 用户这次只要求写交接和prompt；本轮没有重启／停止GPU进程，没有追加训练或修改runtime。

新窗口启动文本：[NEW_CHAT_PROMPT_20261013_CN.md](NEW_CHAT_PROMPT_20261013_CN.md)。
