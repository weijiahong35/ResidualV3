# ResidualV3 交接：epoch1 / epoch2、修复头优化与尾部截断

**2026-10-12最新目录整理已完成。** 日常只读 [RESIDUALV3_START_HERE.txt](RESIDUALV3_START_HERE.txt)：7项当前结果、对应代码、E1/E2与历史入口。152个淘汰工作目录先逐文件SHA256校验归档再删除；全部权重/断点/共享缓存、正式测量保留。旧prompt与重复最新状态全文已收至本轮 `maintenance/research_cleanup_20261012_v4/before/`。恢复索引 `archives/residualV3_research_cleanup_20261012_v4/INDEX.csv`。下文是历次交接，状态优先看当前入口。


**系统更新时间约束：用户要求2026-10-12北京时间14:00前停止。** 本轮专家/路由GPU预实验已12:47结束并退出，只做CPU文档收尾，本轮不再启动新GPU任务；后续接手先核对系统更新与用户新指令。停止记录见`residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/SYSTEM_UPDATE_STOP.json`。

**最新用户要求：每领域至少10000条，只训练头。** 数学、代码、对话/指令各>=10000条实际修复头训练提示，验证/留出另算；冻结E2主体、target、embedding/LM-head和基底，只更新新头内部投影/fusion/专家输出及router。方案`residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/LARGE_SAMPLE_PLAN_CN.txt`总量>=42288（头>=30000、router6144、选型3072、留出3072）、3种子，缓存上限32GiB，尚未启动。旧14336总量方案被替代；96条仅流程试跑，24条留出不足以判断路线，原始数值保留。系统更新前继续停止。

**2026-10-12最新顶层设计与预实验：可见mismatch线索选择修复动作。** [设计v2](residualV3_main_v1/research/target_draft_acceleration_20261012/TOP_LEVEL_DESIGN_CN.txt)、[96提示专家/路由预实验](residualV3_main_v1/research/mismatch_expert_pilot_20261012_v1/RESULTS_CN.txt)。12:47北京时间GPU7空卡小训完成，E2主体和原权重冻结；2880 teacher-prefix状态、24提示留出，有局部专家互补，但学习router overlap0.578482未超过同预算通用头0.580302。没有native AC/TPS，不接长测；本轮PID1437038宿主确认退出。body复用/必要重算、target物理压缩仍未实施；原E2单头正式成绩与旧设计全部保留。


**C32三领域补测已完成并审计通过（2026-10-12T02:12:23.792319+08:00）。** 12正式窗、6252请求；即时报告`residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/RESULTS_CN.md`。gsm8k C32 T0 AC+13.70% TPS+6.86%；gsm8k C32 T1 AC+18.75% TPS+11.26%；humaneval C32 T0 AC+14.46% TPS+9.43%；humaneval C32 T1 AC+18.94% TPS+14.89%；mt-bench C32 T0 AC+8.48% TPS+3.17%；mt-bench C32 T1 AC+13.47% TPS+7.23%。之后多并发默认C32，历史C40保留。 02:13宿主核验本批服务/控制器/报告器均已退出；磁盘496G。target/draft候选文献已整理，尚未测试候选机制。


**2026-10-12新授权：补测三个数据集C32，之后所有多并发默认C32，历史C40保留。** 新冻结结果根`residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1`，仅E2独立DFLASH与FULL15_NOFILL、T0/T1，题数1319/164/80；先实际smoke合同再12正式窗，max2096/context4096。状态以新root/status.json为准，不能把排队写成已完成。研究转向target验证与草稿主体时间，文献入口`residualV3_main_v1/research/target_draft_acceleration_20261012/`。


**2026-10-12 00:12北京时间跨域评测及最终审计已完成。** HumanEval/MT-Bench共16/16正式窗、1952请求，隔离及CUDA Graph通过，T0两臂输出逐题一致；与已完成GSM8K合计24窗/8028请求。全部12组同窗对照TPS为正：GSM8K C1 T0/T1 +9.94%/+14.09%、C40 +6.98%/+12.04%；HumanEval C1 +10.85%/+17.48%、C40 +7.47%/+14.58%；MT-Bench C1 +6.78%/+8.78%、C40 +3.97%/+8.43%。宿主核查两控制器已退出，不重启这批完成队列。当前用户要求我们的最新E2完全体与外部加头版本比较，见[COMPARISON_CN.txt](residualV3_main_v1/research/epoch2_three_domains_comparison_20261012/COMPARISON_CN.txt)：本项目用E2独立单训→联合头正式成绩；外部列DSpark调度关闭、Domino表6仅加头、DFlash-2无卷积选择头。不把本项目替换为E1消融；外部缺TPS明确留空，异设置不直接排名。


**跨域正式任务已完成：** E2独立单训DFlash vs E2联合主体+完整FULL15_NOFILL；数学GSM8K C1=200/C40=1319，代码HumanEval各164题，对话MT-Bench各80题（DSpark第一轮唯一题口径）。C1/C40、T0/T1、max2096/context4096、nonthinking、15proposal。跨域报告根`residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1`的status.json为complete、FINAL_AUDIT.json为passed；原183份源码及E1/E2全部权重不变。GPU7被其他用户进入导致的失败/污染预检已保存在HumanEval C40/prior_attempts，其后迁至GPU0完成。原v3与v4恢复控制器均退出，历史PID只作记录；不要重复运行已完成任务。所有GPU0–7只要宿主空卡均允许，不按编号排除1/2。实时新资源需重新核查。

**2026-10-11晚间目录整理已完成。** 当前统一入口：[RESIDUALV3_START_HERE.txt](RESIDUALV3_START_HERE.txt)，也可直接打开`residualV3-nonthing/current/`。75个旧代码、15个过渡诊断目录已校验归档后删除原目录；epoch1/2、完整断点、消融权重及正式成绩保留。恢复清单：`archives/residualV3_research_cleanup_20261011_v3/INDEX.csv`。该整理批次没有新GPU实验。

核对时间：**2026-10-11 19:29 北京时间（11:29 UTC）**。仓库：`/data/weijiahong/speculators-main-lora`。用户所说的“epoch12”在此指 **epoch1 和 epoch2**。

原19:29交接只读取证据、编写文档；随后按用户授权完成目录整理，记录见统一入口；未启动训练或GPU测试。机器状态会变化，接手时重新核查，不能把本文件中的完成时间、显存或 PID 当实时状态。


**最新状态速读：先读第13节。** 新FULL15_NOFILL/独立E2 DFlash的200/1319正式长测已完成；iter28/29组合短测与合同审计也已完成，四个CUT/FULL吞吐窗口均负。原等待队列和本轮控制器已退出，不要重启。第4节是原始E1/E2历史正式面板，第5节早期微测、第6节iter20/21保留版均不等于最新实测；最新成绩和真实服务成本在第13节。

## 1. 先读这些，再决定怎么继续

1. 本文件，了解结论、用户约束和当前选用实现。
2. [Nonthinking 实验总账](EXPERIMENTS_NONTHINKING.md)与[Thinking 实验总账](EXPERIMENTS_THINKING.md)：目的、对照、设置、AL/TPS、失败及缺测均在这里。不要把两个版本混为一个实验。
3. [epoch1/epoch2 正式比较](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/RESULTS_CN.md)。这是E1/E2训练效果的历史正式入口；最新E2 NOFILL正式比较见第13节。
4. [最新截断短测](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/RESULTS_CN.md)、[选用实现](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/SELECTED_IMPLEMENTATION.json)与[实现审查](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/IMPLEMENTATION_REVIEW_CN.md)。
5. [可复用训练策略](residualV3-nonthing/TRAINING_STRATEGY_NONTHINKING.md)，用来对齐后续基线。里面包含分阶段历史描述，当前完成状态以本文件及对应 `status.json` 为准。
6. [机器可读交接快照](HANDOFF_20261011_STATE.json)：实际路径、完整断点元数据、正式窗口和选用短测；本次检查关键路径缺失数为 0。

**当前结论：**两套模型的 epoch1、epoch2 均完成。联合主体＋修复头相对同 epoch 的独立单训 DFlash，在 GSM8K 正式窗口上有接受长度和吞吐收益。新的自适应尾部截断经过多轮实现优化，仍未实现额外净吞吐收益；不能写成成功提速方案。

## 2. 用户要求与不能混淆的口径

- 当前研究重点是压低修复头额外时间、验证尾部停修的实际价值，并完善论文的顶层设计。不要反复用“修复方向没学好”代替具体证据和解决方案。
- **DFlash 主体是 5 层 Transformer；no-soft 三层指修复头的 3 个 SwiGLU 残差块。** 当前标准仍为三层，不是三层 DFlash。
- Nonthinking 训练数据 `prompt+answer <=4096`，目标模型 Qwen3-8B；使用 nonthinking 预训练 DFlash 起点。
- **block16 = 1 个已知 anchor + 15 个新 proposal**；block12/8 对应 11/7 个新 proposal。不能把 block16 写成修复 16 个新 token。
- AL 是 native 计数 `1 + accepted_tokens / num_drafts`，包含 target bonus；不是接受概率，也不一定等于 EOS/上限裁剪后的实际返回长度。
- 正式 GSM8K **C1=200 题，C40=1319 题，max_new_tokens=2096，context=4096**；T0 greedy、T1 sampling。当前实现短测 **C1=16、C40=80，max_new=256**；smoke 和 8 题插桩诊断不能当正式成绩。
- 方法对比必须用**同 epoch 独立单训 DFlash**。`BODY_ONLY/JOINT_BODY` 是联合训练后的主体关闭头，与独立单训不是一个模型；`RAW` 也可能是未微调的预训练 DFlash。每次读真实 `body_models`、`repair_enabled_by_arm` 和 SHA。
- 截断实现对照用 `FULL15`：同一联合权重修全部 15 个位置。它回答截断是否有额外收益；不能替代独立 DFlash 方法基线。
- 尾部停修是“块内修到合适位置就停”；不是整块跳过。判断输入必须是**repair 后 q**。用户拒绝新增昂贵 native rollout 监督；不要擅自改回该训练路线。
- 辅助预测器输入/标签 detach、独立 optimizer 和 clipping；不得改变 repair 或主体梯度。新增实现先核验，再短测；保留失败证据，不绕过校验。
- BF16 大矩阵/大张量；FP32 内核归约、master、optimizer 状态的实际口径要明确。不能为了数值“接近”就放宽实际 token、前驱或 verifier q 的校验。
- 磁盘上限600G；当前容量按用户`du -sh`的显示口径报告。历史十进制GB是换算，不是用户明确规定。**epoch1/epoch2 长训权重、完整恢复断点、各种消融权重和结果绝对不能删。** 清理仅限已授权的 residualV3 预实验冗余，代码先存档；其他项目不碰。
- 最近短测控制器只在空卡、空闲显存至少 78000MiB 时启动；旧训练方案曾按当时资源使用指定卡；用户已纠正，当前所有GPU只要是空卡都可用，不按编号排除1/2。曾有针对共享卡续训的特别授权，不能把它当作任意新任务共享卡的授权。不得中断他人任务；运行中新增进程只记录，不能因此随意停止正在训练的任务。
- 用户希望多用真实工作区/KV/激活缓存换时间。大显存分配不是独占锁，不能保证他人不再分配显存。不要用无用张量填卡冒充加速。
- 用户此前明确不启动缺失 C40 VERIFY_STOP 补测；旧恢复控制器不能见空卡就重启。本次是交接，不自动开启新训练或正式长测。
- 用户要即时 log 和统一总账。每次新结果记录“做什么、比较什么、设置、完成/失败状态、AL、TPS、隔离、代码/权重身份”。

## 3. epoch1/epoch2 资产与训练履历

### 3.1 两个最终训练目录

| 项目 | epoch1 | epoch2 最终有效版本 |
|---|---|---|
| 根目录 | [full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1) | [full_nonthinking_epoch2_aux_q45_latent_20261010_v4](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4) |
| 状态 | complete，2026-10-08 完成 | complete，2026-10-10 完成 |
| 本 epoch 更新 | 15,847 | 15,854 |
| 累计 global step | 15,847 | 31,701 |
| 单训 DFlash 权重 | `final/body_epoch_01.safetensors` | `final/body_epoch_02.safetensors` |
| 联合主体＋头权重 | `final/joint_epoch_01.safetensors` | `final/joint_epoch_02.safetensors` |
| 共同完整断点指针 | `latest_checkpoint.json`，cursor15847/slot0 | `latest_checkpoint.json`，cursor15854/slot0 |
| 主协议 | `protocol.json` | `protocol.json`、`lineage.json` |
| 日志 | `joint_training.log` / `dflash_training.log` | `joint_training.log` / `dflash_training.log` |

以上 final 主体文件各 2,097,267,648 字节，联合文件各 2,105,978,602 字节。本次检查存在及文件大小，未重新读取全部大权重做 SHA；模型身份以协议/manifest/交接快照所记录 SHA 为准。

每个训练根的 `checkpoints/slot_0/{joint,body}.state.bin` 为完整 state，各预分配 8,858,370,048 字节。slot1 及对应 JSON/指针也必须保留；不得把 `.state.bin` 当无用中间文件。**safetensors final 导出不包含 optimizer，不能代替完整恢复断点。**

Epoch2 的独立辅助读出：

- [post_repair_aux_comparison.pt](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/final/post_repair_aux_comparison.pt)，147,901 字节；本次重新计算 SHA256 为 `b3609423acec1809041032b277355dba7f50eb92ebca5f05577f4dffa765fa42`。
- 包含两个非线性分支及 normalizer；完整 optimizer/训练恢复状态另外在共同 state 中。
- [现场主梯度/耗时门禁](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/gates/auxiliary_live_gate.json)。

### 3.2 训练配方，供下一批基线对齐

| 项目 | 实际配方 |
|---|---|
| 数据 | Open-PerfectBlend 重生成 nonthinking，833,176 个有效唯一 conversation group |
| 输入 token / epoch | 1,019,443,945 |
| 数据文件 | [manifest](residualV3-nonthing/data/all_nonthinking_ctx4096_20261006_v1/manifest.json)，token/mask/catalog 固定 |
| 训练输入包 | 每个模型每次更新最多 65,536 输入 token，多 document 打包 |
| anchor | **整包最多 1536**；不是每条1536；完整受监督16-token窗口，不跨 document，不重复补满 |
| supervised proposal 上限 | 1536 × 15 = 23,040 / 更新 |
| 更新 | 每包一次 optimizer.step，梯度累计1；loss按1536 anchor整块计算 |
| 两个 learner | 单训 DFlash、联合 DFlash＋头各一张卡；同一个 pack/anchor，彼此无梯度同步 |
| teacher | 一张 vLLM，TP1/DP1；同份 hidden 只获取一次，供两端 |
| teacher 参数 | BF16、KV48GiB、seq64、batched_tokens65536、model_len4097、HTTP并发128/读取16、eager、prefix cache |
| hidden | 中间层2/10/18/26/34＋最终层36；最终 norm 在 learner 处理；在线 /dev/shm 共享，不持久缓存 |
| host 重叠 | 两槽不可变在途 buffer，CPU 注册，待两端 H2D 完成才复用 |
| 激活 | 联合保留前4层、checkpoint第5层；单训保留5层；projection rows 重算 |
| learner allocator | 目标74GiB，上限76GiB；是工作空间策略，不是采样预算 |
| E1 数据顺序 | seed42、15,847 packs |
| E2 数据顺序 | seed43、15,854 packs；继承各自 E1 的参数和所有主优化器状态 |

数据 plan/order 不重新 shuffle。E2 的 order/pack 文件仍在 [epoch2_teacher1_3gpu_20261009_v1](residualV3-nonthing/results/full_nonthinking_joint_body_epoch2_teacher1_3gpu_20261009_v1)，**该旧目录即使控制器退役也仍是依赖**。

模型起点分别由最终协议指定：target 为 `/data/weijiahong/huggingface/models/Qwen3-8B`；DFlash 初始权重为 `/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash-nonthinking-20260916`。输入/输出冻结基底及 head config 仍依赖 [20261006_v1 的 basis 目录](residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/basis)，不能按旧 controller 状态删除整个根。

主损失：

- 独立 DFlash：raw proposal CE，按 D-PACE 位置权重加权。
- 联合：raw CE/D-PACE ＋ repaired CE/D-PACE ＋ repaired OUTPUT，三项系数 1/1/1；`dpace_alpha=.5`、`output_loss_weight=1`。
- D-PACE 是 CE 的停止梯度平滑前缀位置权重；不是另外加一次相同 CE。
- OUTPUT 用每位置全词表重合度 `a=1-TV(p,q)` 的原前缀乘积目标。融合 TV 内核保留 BF16 大张量，内部 FP32 归约；不应改成仅 Top256 的概率重归一化。
- Teacher forcing 前驱是监督 token；推理前驱是本次实际已修复/采样 token。OUTPUT 的训练前缀代理不是实际 rollout 存活期望。

优化器：主体二维矩阵 Muon，lr1e-3/momentum.95/wd.1/ns_steps5/`match_rms_adamw`；主体 norm/mask AdamW lr2e-4/wd.01；repair AdamW lr3e-4/wd.01，eps1e-8，fused；主参数整体 clip1。主权重 BF16，master/optimizer FP32。每 epoch 1% warmup（159更新）＋cosine到0；**E2 开启新的 LR 周期并继承 moments/step**，不是从第一步就规划两轮连续 cosine。

来源：两根 `protocol.json`、[训练策略](residualV3-nonthing/TRAINING_STRATEGY_NONTHINKING.md)、[E2 当前训练代码](residualV3-nonthing/code/full_joint_body_epoch2_aux_compare_20261010_v4)。后续基线同时对齐数据/token/anchor/更新预算，结构特有的 loss/optimizer 差异另外记录。

### 3.3 内嵌辅助预测器与恢复契约

两个只读主特征的辅助分支：

| 分支 | 输入和结构 | 参数 |
|---|---|---:|
| Q45_MLP | 修复后 q45 → 标准化 → Linear45→32 → SiLU → Linear32→2 | 1,538 |
| Q45_REPAIR_LATENT_MLP | 同一 q45＋第三层 repair 的 output_norm 表示256 → 标准化 → Linear301→32 → SiLU → Linear32→2 | 9,730 |

输出分别监督 T0 teacher-prefix target argmax 一致与 T1 已有 OUTPUT analytical overlap；BCE 输入/标签 detach、各自 AdamW lr.003/wd.001/clip1、初始化隔离主 RNG，不新增 teacher 查询。两 MLP 从 E2 第8897包开始训练，不是覆盖整轮；旧92参数线性辅助头保存在旧版本，不再更新。

v3辅助开销12.83%未通过门槛，未启用辅助但主训练继续。v4修正主图求值顺序和按需末层捕获，同一真实包99组主梯度/主 loss/CUDA RNG逐位一致；核心前后向1.933626→1.999303秒（+3.3966%），共享特征62.129ms、两读出训练4.362ms。不是完整服务吞吐，也不是推理截断收益。

共同 state 含 FP32 master、Muon/Adam、所有主 RNG、数据 SHA/order/plan、epoch/local cursor/global step，以及辅助参数/Adam/normalizer/steps。两端 fsync 后原子提交共同指针；每200更新及手动请求保存。正常暂停保存共同边界；强杀只保证从最近 committed state 重算未提交部分。

[E2 运行指南](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/RUN_GUIDE_CN.txt)里的 `control.sh status/checkpoint/pause/resume` 是该次续训的控制入口，pause只停两端、保留 adopted teacher，resume依赖原服务/卡。**现在已经 complete，不能直接 resume 旧任务假装开启 epoch3。** 将来新轮次要新目录、新 epoch plan/LR 明确记录，并从两个已提交完整 state 接上。

## 4. 原始epoch1/epoch2正式成绩（历史面板）：匹配同轮单训DFlash

以下为该轮当时的统一推理实现、GSM8K nonthinking、block16、max2096、BF16、batch-invariant target、CUDA Graph；**辅助预测和截断关闭**。16个正式窗口 complete，均有隔离证据；测量窗口 target/draft 的 captures/eager/fallback 为0，有 graph replay。

| epoch | 设置 | DFlash AL | 联合＋头 AL | AL增长 | DFlash TPS | 联合＋头 TPS | TPS增长 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | C1 T0 | 7.384246 | 8.381226 | +13.50% | 179.691 | 190.532 | +6.03% |
| 1 | C1 T1 | 6.331330 | 7.429277 | +17.34% | 154.759 | 169.212 | +9.34% |
| 1 | C40 T0 | 7.560736 | 8.527382 | +12.79% | 3197.935 | 3347.124 | +4.67% |
| 1 | C40 T1 | 6.382050 | 7.666585 | +20.13% | 2705.024 | 3017.241 | +11.54% |
| 2 | C1 T0 | 7.478576 | 8.515070 | +13.86% | 181.919 | 193.191 | +6.20% |
| 2 | C1 T1 | 6.326140 | 7.495970 | +18.49% | 154.769 | 170.657 | +10.27% |
| 2 | C40 T0 | 7.609462 | 8.651926 | +13.70% | 3217.680 | 3400.578 | +5.68% |
| 2 | C40 T1 | 6.483497 | 7.709856 | +18.92% | 2751.453 | 3032.491 | +10.21% |

E2联合相对E1联合的 AL仅增加0.56%–1.60%、TPS增加0.51%–1.60%。本面板第二轮增量较小，不能仅凭单 seed/单轮开发评测断言完全收敛。两套方法的主体权重不同，以上是整体训练方案比较，不能归因成固定主体纯加头。

正式日志：[C1](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C1_formal.log)、[C40](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/single_card_C40_formal.log)；[原始 summary](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/summary.json)与[模型/面板协议](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/native_protocol.json)。

实际评测模型路径不能凭命名猜：

- E1_DFLASH：E1 根 `evaluation_native_20261008_v1/models/body`。
- E1联合主体：E1 根 `evaluation_dspark_gsm8k_recovery_20261009_v2/models/joint_native`；头为 E1 根 `evaluation_native_20261008_v1/models/joint/head.safetensors`。
- E2_DFLASH：比较根 `models/E2_DFLASH`。
- E2联合主体：比较根 `models/E2_JOINT_NATIVE`；头 `models/E2_JOINT_HEAD/head.safetensors`。
- `native_checkpoints` 中 DFLASH 臂也登记共享 head 文件，但 `repair_enabled=false`，实际 baseline 必须由 `body_models` 判断，不能看 head 路径就认为开了头。

## 5. 原结构、已完成的头优化与层数/block实验

### 5.1 当前头的计算链

```text
DFlash 一次并行 forward → 各位置 hidden / raw logits
  hidden4096 → LN/Linear256 ──────────────────────────┐
  raw全词表q → Top256 → 冻结输入基底rank512聚合 → 256 ─┤（这两支可整块预计算）
  实际上一proposal token → 冻结派生编码表查找 → 256 ───┘
          → 三分支融合768→256
          → 3个SwiGLU残差块
          → 方向512＋dose1 → 局部Top256 logit修正
          → 完整词表T0/T1采样 → 实际token作为下一位置前驱
          → target原生验证（使用实际proposal q）
```

新增可训练参数 **4,353,537**，BF16约8.30MiB；共享冻结 embedding/LM head/basis 不计入。小参数量不等于小运行成本：全词表概率、TopK、串行15步、中间激活/采样仍有成本。Top256是修正支持，不是把生成词表缩成256。

训练 TF 并行不能直接搬到 native：推理第j步的前驱要等第j−1步修正后实际采样。不能用原始候选或 teacher token 偷换前驱来获得“并行等价”。Thinking Parallel_TF 是另一个架构实验，见后文。

### 5.2 原结构优化已经做了什么

1. 固定权重的整词表硬前驱投影预计算查表；按实际 token 取行，保留递推。
2. 小 batch GEMM tile、SwiGLU up/SiLU/down/残差及方向/dose合并，明确保留BF16舍入点。
3. 全词表分母仍保留，仅输出实际使用的Top256概率；基底取行/聚合、候选投影/写回融合，减少大临时张量。
4. stock Gumbel 第一阶段/RNG/温度保持；末端 argmax+gather 融合；完整头 CUDA Graph。

证据：[结构审查](residualV3-nonthing/research/head_structure_fusion_20261009_v1/STRUCTURE_AUDIT_CN.txt)、[融合分项计时](residualV3-nonthing/research/head_structure_fusion_20261009_v1/MODULE_TIMINGS_CN.txt)、[早期参数/高层审计](residualV3-nonthing/results/head_component_depth_audit_20261009_v1/MODULE_TIMINGS_CN.txt)。

独立人工状态、整批15proposal的完整头微测，不含DFlash主体/target/服务：

| 实现阶段 | C1 T0 ms | C1 T1 ms | C40 T0 ms | C40 T1 ms |
|---|---:|---:|---:|---:|
| 原模块审计头 | 4.0567 | 4.1020 | 9.7144 | 10.4760 |
| small-tile版（结构融合配对微测） | 2.8364 | 2.8761 | 8.5017 | 9.2408 |
| 结构融合（同窗对照） | 2.0282 | 2.0727 | 6.4775 | 7.2259 |

这些是**一个block、整批请求**的时间。不能机械相加组件，不能拿2ms微测代替正式服务中插桩“头+采样”分段，也不能按跨阶段绝对毫秒作严格配对结论。

人工分项提示：融合版C1仍突出串行MLP/输出归一化；C40全词表分母/TopK及T1采样更突出。旧硬前驱大投影已经查表，不能继续用旧成本当当前首要瓶颈。用户“仅吃掉5%接受收益”的目标未证明达成；95%相对收益保留与少5个百分点是不同要求，原预算解释在结构审查内。

target、body、头、两种完整小草稿耗时另见 [正式对照＋分段诊断](residualV3-nonthing/research/native_optimized_vs_dflash_report_20261009_v1/RESULTS_CN.txt)及 `NATIVE_COMPONENT_TIMINGS.csv`。该报告是较早实现的独立插桩测量，不是当前截断实测；不能把其中4/10ms与后来的2/6ms微测混算。

### 5.3 block12/8 与推理跳第三层

[最新融合 block/depth 结果](residualV3-nonthing/results/native_fused_block_depth_20261009_v1/RESULTS_CN.txt)，全部使用 E1。HEAD2是**同一三层联合权重只运行前两层**，没有重训两层。

| 设置 | 对比 | AL相对独立DFlash | TPS相对独立DFlash |
|---|---|---:|---:|
| block12 C1 T0/T1 | HEAD3 | +10.49% / +12.91% | +4.47% / +6.94% |
| block12 C40 T0/T1 | HEAD3 | +9.64% / +15.71% | +3.61% / +4.60% |
| block8 C1 T0/T1 | HEAD3 | +6.60% / +9.94% | baseline污染，无干净比例 |
| block8 C40 T0/T1 | HEAD3 | +5.79% / +10.30% | +1.37% / +8.84% |
| block16 C1 T0/T1 | HEAD3 | +13.50% / +17.34% | +5.78% / +8.68% |
| block16 C1 T0/T1 | HEAD2 | +12.39% / +17.10% | +6.74% / +10.64% |
| block16 C40 T0/T1 | HEAD3 | +12.79% / +20.13% | +4.47% / +11.85% |
| block16 C40 T0/T1 | HEAD2 | +11.32% / +17.68% | +3.28% / +10.18% |

高层真实有非零残差并改变输出；跳层C1略快但C40不划算，不能写成“第三层没用”。28个正式窗口均已记录，缩短block改变AL上限，跨block比较要同时看TPS；这些不是新的训练策略胜负。

## 6. 保留版iter20/21尾部截断：历史测量与代码身份

### 6.1 当前策略与实现身份

- 使用固定 E2 联合权重＋已训练辅助预测器；不改主模型/阈值/训练目标。
- q45 分支阈值 T0/T1=.1/.1；latent分支=.1/.15；至少修8位置、连续2次低前缀生存信号、一位置预测延迟，**最早停在10**。
- [阈值/校准文件](residualV3-nonthing/results/epoch2_tail_cut_optimized_20261010_v2/calibration/predictor.json)：GSM8K train64校准＋32检查，与formal test问句去重；阈值只由校准选择，目标全修路径保留率≥99.5%。该代理约束不是正式AL保证。
- 最新开发代码：[epoch2_tail_cut_short_iter_20261011_v3](residualV3-nonthing/code/epoch2_tail_cut_short_iter_20261011_v3)。每次测试代码冻结到 `iterations/<id>/native_runtime`，结果在同名结果根 `iterations/<id>`，不覆盖旧结果。
- **选用 C1=iter20_fused_input_C1、C40=iter21_fused_input_C40。** [恢复门禁](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/RESTORED_RUNTIME_GATE.json)记录开发177份运行时Python和evaluator与iter21逐字节相同。
- 最新选用优化的完整头校验/短测覆盖 **Q45_MLP**；协议模板虽列四臂，不代表独立DFLASH及latent都在每轮短测重新完成。不能替latent分支宣称相同优化已验证。

### 6.2 选用实现的同窗短测

| 设置 | FULL15 AL | Q45_CUT AL | AL变化 | FULL15 TPS | Q45_CUT TPS | TPS变化 |
|---|---:|---:|---:|---:|---:|---:|
| C1 T0 | 8.011261 | 7.955257 | −0.699% | 179.096 | 177.151 | −1.086% |
| C1 T1 | 7.446502 | 7.424180 | −0.300% | 166.843 | 165.554 | −0.773% |
| C40 T0 | 8.540297 | 8.508300 | −0.375% | 2899.821 | 2864.693 | −1.211% |
| C40 T1 | 7.564586 | 7.586493 | +0.290% | 2671.803 | 2648.496 | −0.872% |

全部观察隔离通过，但都是16/80题max256**短开发测试**。不与正式200/1319题的绝对TPS直接相除；不可把T1单次AL小涨当稳定收益。根状态 `short_iterations_complete_no_net_gain_claimed`，本次读取 `active_iterations={}`。

**选用版即时日志：**[iter20 C1](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/iterations/iter20_fused_input_C1/single_card_C1_short.log)、[iter21 C40](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/iterations/iter21_fused_input_C40/single_card_C40_short.log)。

**日志陷阱：**结果根的 `single_card_C1_short.log` / `single_card_C40_short.log` 目前链接到**未采用的iter22/23**，不是选用版；旧 `epoch2_tail_cut_optimized_20261010_v2/*formal.log` 更不是最新短测。不要删除或重标负结果日志。

### 6.3 已解决的最大实现问题

旧截断把cut长度读回CPU：`DraftTokensHandler` 中 D2H `copy_event.synchronize` 阻断CPU/GPU重叠，C1等待约8.37ms；FULL15固定占位路径约.013ms。早期短测TPS相对FULL15下降约7%–10%，明显超过AL下降。

当前 **DEVICE**：CPU保留固定15proposal占位；GPU的 `_device_prefix_cuts` 给process-local rejection sampler裁有效前缀/bonus，保留原概率比/RNG/q合同；已做实际sampler与compact参考校验。大部分同步损耗已去掉，TPS惩罚收窄至约1%，但尚未净提速。

**重要边界：target仍执行固定补齐图；没有物理压缩target的全部计算。** 修复MLP以BM16请求tile门控，只要tile中仍有active请求，整tile矩阵就执行。少修若干请求不等于按比例减少MLP成本；截断收益不能仅由repair位置数估计。

### 6.4 当前保留的新增优化

| 改动 | 校验证据 | 时间收益口径 |
|---|---|---|
| 精确BF16 Top256有界直方图缓存，边界全范围fallback；只准备前13位置 | edge、真实q45及完整头 | 减少重复扫描；非截断净TPS证明 |
| 前9个必修位置批量预测，保持逐位置原扫描布局；前10位置dense快速路径 | 完整头/真实smoke | 保持因果repair前驱，不能并行替换后续repair |
| q45非线性readout融合、Sklansky scan复现 | readout108、scan78组 | readout约.034–.044→.009ms；N1 scan仍用原CUB |
| 保留PyTorch FP32 L2范数，只融合方向/幅度后处理 | 127组件＋12完整头，含失败边界 | 同窗整头N1T0 2.029→1.873ms、N40T0 6.498→6.324ms |
| 原采样1024分块导出未加噪logit最大值，供q45复用 | 12完整头＋smoke | 整头约0.05%–0.86%缩时 |
| 前驱编码表查找＋三分支拼接一次copy | 12完整头＋smoke | C1整头约1.6%–1.9%缩时，C40接近不变 |

代码入口：`native_runtime/head_latency/{tail_pipeline,trained_tail,tail_sparse_q45,tail_sampler_max,input_assembly,output_epilogue,tail_readout_fused,tail_bounded_cache,structure_fusion,tail_graph_state}.py`；`repair/{device_prefix,device_prefix_sampler_utils,target_prefix_bridge,target_prefix_launch,native_launch}.py`。实际入口/环境/开关由各迭代 `native_protocol.json` 和启动日志决定。

同数值门禁通过的是有限样本，不是所有可能输入的数学证明；组件百分比不能相加。训练代码和梯度没有被这些推理修改改变。

### 6.5 不采用的原型，接手不要重复踩坑

- **全融合L2范数**：随机通过，真实C40 FP32归约差9.54e-7触发BF16分母舍入边界，413个方向值不同；已回到原范数，不放宽门槛。
- 活跃请求压缩MLP：数值通过，但约慢2.6%–4.2%，未接入。
- target按有效长度分组的词表投影：统一cut10微测约省18%，混合长度反而慢4.4%/12.6%；未接入。
- 调整缓存归约块：组件快但完整头慢约.05%–.35%；存档后撤回。
- 提早第5位置启动预测：完整头慢约.4%–.9%，未采用。
- **六份图外state copy融合iter22/23**：24组bytewise＋12完整头通过，拷贝11→4μs、整头微测快.1%–.4%；短测C1 TPS−1.20%/−.88%、C40−2.10%/−1.44%，未见稳定额外优势。源码保存在 `probes/full_head_26_state_export/source`；开发版恢复原六份copy。
- 初次state比对错误拿旧v2与v3全部辅助状态比较；后来增加同v3 eager/graph、有效token/q/cut及所有图状态校验。失败和修正证据保留。

全部迭代见[SHORT_RESULTS.json](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/SHORT_RESULTS.json)；探针在该根 `probes/`。勿只选iter08 T1接近0的一个窗口声称成功。

## 7. q信号与截断的研究证据

q45：repair后**单位温度**全词表q的4个标量（max、第二大、gap、归一化entropy）、9个Top1/2/4/8/16/32/64/128/256累计概率、32个排序Top32 log概率；共45维，不含token身份。它与推理sampler实际processed q的合同要分开核查，不能把统计温度直接当采样温度。

当前pre/post诊断是128提示、1536块、23040状态，按提示80/24/24划分，固定原生前缀；不是新训练MLP的独立最终测试。主要例子：

| 标签/位置集/信号 | T0 PRE→POST AUROC | T1 PRE→POST AUROC |
|---|---|---|
| 实际到达位置的NATIVE_REJECTION，confidence | .799890→.908777 | .768631→.868764 |
| 尾部8–15位置，无可接受后缀，单点confidence | .769469→.697641 | .780621→.765903 |
| 同一尾部标签，累计prefix maxq product | .824570→.888286 | .835496→.869394 |

POST对局部残余风险更强，不代表对“继续修尾部值不值”同样强；尾部累计T1的prompt配对差值区间跨0。训练BCE下降、AUROC高、少repair次数都不能代替实测AL/TPS。

旧 q-only v4：q_max对T0 mismatch AUROC约.8354，对T1 `TV>=.1`约.9416；标签不同，不能与真实拒绝AUROC拼表或套到新E2读出。用户删除的更早mismatch实验代码在存档，旧定义不能当当前验证结果。

证据：[PRE/POST核验](residualV3-nonthing/results/q_pre_post_tail_signal_20261009_v1/RESULTS_VERIFIED_CN.txt)、总账N004/N010/N013/N016。未来判断要同时报告标签、位置集合、prompt划分、校准、拒绝/尾部/修复价值，避免同块状态随机切分产生泄漏。

真实长度分布已从逐位置native累计接受计数还原到1–16，见[总账N003](EXPERIMENTS_NONTHINKING.md#n003)，数量和、接受总数及期望一致。`L=K+1`，`n_l=S_(l−1)−S_l`，`E[L]=1+Σ S_j/D`。只有汇总计数，不能声称有逐block事件日志。E1 C1 FULL15长度16占T0 23.18%、T1 17.55%；平均AL约8绝不意味着所有尾部都无价值。

## 8. 其他重要Nonthinking实验

各项完整AL/TPS和即时log见总账相应编号；旧20k面板和GSM8K正式面板不同，勿横比绝对值。

| 实验/总账 | 做什么、对照与关键设置 | 已知结果及限制 |
|---|---|---|
| N005 原20k逐条配方 | 三层标准头，每条最多1536 anchors，原GPU0训练 | 极慢；不能把它当整包1536。失败/恢复记录保留 |
| N006 20k打包65536/1536 | 验证整包预算与训练速度，三轮短训 | 完成12正式窗口；预算变化不是纯实现提速 |
| N007 三损失消融 | 冻结主体，20k×3轮；OUTPUT、CE_DPACE、CE_DPACE_OUTPUT | 融合损失T0收益较好，T1 OUTPUT仍接近/略优；选combined不等于四设置全面胜出。旧C1=200/C40=2000面板 |
| N025/N026 1536整块＋融合TV | 先loss/参数梯度校验再测chunk64/256/1536 | 1536已实验；融合版三loss均通过，峰值allocated约39.12GiB/reserved45.67GiB。是冻结主体探针，不套到联合训练显存 |
| N008 OUTPUT前缀平滑 | 20k×1epoch=298包；原前缀a与.5+.5a，共同CE/D-PACE | AL相对原版C1−.87%/−3.31%，C40+.75%/−.57%，无一致优势，未改长训 |
| N009 matched head-only/joint | 同20k一轮、起点/数据/anchor/预算对齐，补训HEAD_ONLY | 只训头比RAW AL约+6.83%–9.18%；短联合未稳胜RAW。不同时间复用窗口，不能据此下配对TPS结论，更不能否定全量联合成绩 |
| N027–N035 teacher/两learner优化 | 1teacher vs2teacher、hidden共享、H2D注册/双缓冲、激活缓存等 | [hidden供给计时](residualV3-nonthing/results/hidden_supply_timing_20261008_v1/OPTIMIZATION_SUMMARY.json)记录供数/等待；teacher卡数不改变训练端逻辑batch |
| N012/N021/N022 | 模块参数、高层、small tile、结构融合 | 已有本文件第5节证据；micro和formal分开 |
| N014 greedy分叉 | 原native与target-only AR重复C40/C1，再检查batch invariance | 默认native18/40、AR22/40分叉；同前缀target argmax翻转，batch-invariant各40题一致。旧732/1319缺logits不能逐条溯源 |
| N003 固定PREFIX10/12 | 同权重只修前10/12，后缀raw仍采样/验证 | C1：PREFIX12 AL−1.92%/−1.35%、TPS−.95%/−.40%；PREFIX10 AL−3.09%/−3.86%、TPS−1.70%/−2.32%；C40真实smoke失败，未给正式成绩 |
| N045 adaptive repair pipeline | repair后风险、一位置延迟、保守停修 | 有初始formal结果；不能把BODY_ONLY当独立基线。后续训练readout与DEVICE实现另列 |
| N019/N020 target前缀截短 | FULL15/REPAIR_STOP/VERIFY_STOP对照，E1 | v2仅14/16正式窗口，截断未额外提速；缺C40 VERIFY_STOP T0/T1，用户明确暂不补测，v3修复仅CPU证据 |
| N050 E2 trained-aux tailcut v1 | 冻结两MLP，独立train问句校准，200/1319题正式 | C1 AL损失很小但TPS约−10%；存在CPU长度同步、padding和停后冗余计算问题；后续v2/v3改实现，不归咎于“头没训练好” |
| N048/N049 截断v2/v3 | 精确q45、跳过、图桶、DEVICE、融合内核 | v3选用iter20/21，本文件第6节；仍无净TPS增益 |
| N052/N054 E2内嵌aux | 真实1536包梯度/主RNG不变，计算开销门禁 | 分别约+3.40%/+3.39%核心计算；辅助独立、完整断点可恢复 |

置信度读出不改变teacher分布，截断必须继续满足有效proposal q、accept/reject、bonus、KV/位置的原生合同；不能用独立target采样token是否等于proposal作为T1接受标签。

## 9. Thinking研究链：必须保留，不能误套到当前模型

入口：[Thinking总账](EXPERIMENTS_THINKING.md)，已登记71条。常规早期短训为冻结target/主体、128条×3轮=384更新、7proposal，C1=20/C40=300、max4096/context8192；部分长训/诊断有例外。非当前nonthinking全量联合预算。

| 总账/实验 | 主要内容 | 对交接的意义 |
|---|---|---|
| T001 原N_no_soft_2layer | 原短训基线 | 新基底必须对比原始短训N，不仅对比更弱新控制 |
| T002 10k×3轮深度长训 | 1/2/3层，C1=200/C40=2000 | 长训两层T1受益、C40T0正TPS；epoch2 BF16恢复重建master并重置Adam，非当前完整state无损恢复 |
| T003–T007 输入/深度/direction-dose | LMHEAD/EMBEDDING硬软、A–G、no-soft1/2/3层 | 形成hidden＋rawq＋真实硬lag1、去soft反馈的方向；short loss不决定最终结构 |
| T008–T023 前驱/oracle/可学习性 | 真实前缀、latent/dose/方向oracle、梯度传输 | 好动作存在不等于共享预测器学会；固定状态oracle上限不是rollout AL/TPS |
| T024/T025 TopK与rank | Top256/128/64；输出rank256/128/64 | 大多保AL但无稳定额外TPS；K是候选范围，rank是基底维度，不混用 |
| T027/T028 Parallel_TF/on-policy | 把重计算搬到h/q上下文、实际前驱轻门控；五臂统一对照 | TF小幅实现收益；T0/T1真实前缀训练约8倍慢且未一致更好。用户不愿重做昂贵alignment |
| T029 target hidden/logits输入 | hidden、logits、组合输入 | C1小收益未稳延续C40；所谓target context来自已验证前缀辅助层的压缩状态，不是未来teacher hidden |
| T030 OUTPUT＋Top16 CE | 候选竞争监督 | 不是完整DFlash-2复现；target不在候选时CE为0且不补标签 |
| T033/T034 五问题/greedy margin | 方向、幅度、finite correction、实际verifier诊断 | 有历史失败/有效版本，按总账区分，不能只看目录failed就说科学假设失败 |
| T035/T036 任务/可预测基底 | KL/overlap/margin，线性RR；h/q/硬lag1固定1537维，rank512 | C1 T1 RR_KL/RR_MARGIN小收益，C40未稳定延续且部分污染；不能把PCA压缩本身当创新 |
| T037 非线性基底 | 非线性预测梯度方向对照线性RR | 梯度误差约改善.62%/.73%，native AL相对GG变动不足.25%，无稳定额外收益 |
| T038 bottleneck probe | 动作可达与共享预测可学习性 | 不能从基底负结果认定整个结构失败，唯一原因未识别 |
| T039–T041 输入/hidden来源 | v1/v2观察器失败，v3有效 | v3仅23配对/14提示，动作冲突小但跨提示学习门槛没过；不证明输入充分 |
| T042 OUTPUT/Euclidean/Fisher | 16提示×3轮方向几何训练 | 用户第一seed C1无收益后停止；18epoch权重/10C1窗口保留，第二seed未完成/C40未测，不重启 |
| T043/T044 EAGLE-3 | 原生EAGLE的3/7proposal迁移 | 保留跨draft家族证据；RAW是原生EAGLE投机，而非target AR |

完整路径和结果以总账T编号为入口。旧代码在 [archives/residualV3_legacy_20260930/code](archives/residualV3_legacy_20260930/code)，另有结果来源/维护档案。退役记录不是缺测补跑授权；长期/消融资产不能删。

## 10. 已收集论文/作者代码与故事边界

本节只交接已有本地核验，不是本次重新联网检索，也不宣称检索穷尽。

- [DFlash/DSpark/DFlash-2与训练配方审查](residualV3-nonthing/research/training_recipe_review_20261006/TRAINING_RECIPE_REVIEW_CN.md)、[来源勘误](residualV3-nonthing/research/training_recipe_review_20261006/RECIPE_PROVENANCE_CORRECTION_CN.md)。RFC#979是RedHat贡献者在vLLM Speculators提交的长训参考，4learner×65536=262144名义token/update；RedHat模型卡2learner=131072。bestpractices示例是8192/3072的另一短管线，不能把文件名当最优成绩证明。当前两learner训练的是两个独立模型，不合并成DP2全局梯度batch。
- [Domino延迟核验](residualV3-nonthing/research/domino_latency_audit_20261009_v1/RESULTS_CN.txt)：作者报告头2.64→1.20ms；公开HF路径batch1，图1精确batch数是结合实现推断，未找到C40同延迟证据。其查表、提前hidden投影、融合GRU/输出和Graph可借鉴，greedy路径不能自动证明T1采样同成本。
- [DFlash-2启发与优先级](residualV3-nonthing/research/repair_value_design_20261009_v1/DFLASH2_PRIORITY_CN.md)：除selector外还有主体局部动态因果卷积，改善候选质量；候选边评分预计算的设计与PyTorch循环实际实现要分开。当前Top16 CE实验不等于复现这些主体改动。
- [修复价值来源核验](residualV3-nonthing/research/repair_value_design_20261009_v1/SOURCE_FACTCHECK.md)：SpecDec++、DSpark已经覆盖置信度停止/接受概率结合成本调度；OnlineSPEC有验证反馈在线更新；DeLS-Spec有冻结draft加专家。不能把“加置信度头/截断/冻结加头”单独宣布新颖。
- DSpark置信度并非凭空一个线性层：有backbone hidden和实际前驱Markov表示，sigmoid读出、overlap软标签、概率校准与硬件SPS曲线。当前q45预测器detach的设计不是完全照抄其端到端特征学习，不能用层数相同证明容量相同。
- 设计稿：[TOP_LEVEL_DESIGN_CN.md](residualV3-nonthing/research/repair_value_design_20261009_v1/TOP_LEVEL_DESIGN_CN.md)、[CORE_IDEA_CN.md](residualV3-nonthing/research/repair_value_design_20261009_v1/CORE_IDEA_CN.md)。这些是提案，尚不能当实证结论。

可用的叙事基础是“在并行draft给定状态与真实前驱条件下修正分布，并在系统成本中兑现接受收益”。mismatch可分别指logit/probability残差、T0 token不一致、T1分布距离/真实拒绝；必须在每个实验明确标签。将其进一步提升成“预测可修复价值、决定方向/幅度/算力”仍需动作反事实与成本证据。当前训练好的完整联合方案有收益，**截断创新暂无额外吞吐证据**。

## 11. 磁盘、环境、运行陷阱与维护

原交接字节快照为531,813,081,088字节，十进制换算531.813GB；用户`du -sh /data/weijiahong`显示约496G。十进制仅是当时报告采用的换算，并非用户要求。整理后继续以`du -sh`显示口径报告，并每阶段重新核查。

[20261011磁盘整理记录](residualV3_main_v1/maintenance/storage_cleanup_20261011_v2/RESULTS_CN.md)：仅对历史residualV3代码档案的54份完整SHA相同大文件做硬链接去重，释放60.58GB。路径和字节内容保留；E1/E2完整断点及长训/消融权重未删。

**硬链接档案不能原地改二进制**；需要修改时先独立copy，见[STORAGE_DEDUP_NOTICE.md](archives/residualV3_legacy_20260930/code/STORAGE_DEDUP_NOTICE.md)。关键依赖除了主权重还包括：E2旧根的order/plan、20261006旧根basis/head config、E1的GSM8K panels/模型导出，以及 `native_candidate_write_triton_20261009_v1/native_cache`共享评测缓存。

GPU评测环境：`/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026/bin/python`；CPU控制器用dflash环境。vLLM需要该环境cuda-compat、`VLLM_ENABLE_CUDA_COMPATIBILITY=1`、`VLLM_CUDA_COMPATIBILITY_PATH`和对应`LD_LIBRARY_PATH`，实际取启动记录；GPU绑定用UUID，不能随便换成逻辑index。

当前安装vLLM的async-scheduling白名单不含dflash，硬开会ValueError；当前走`--no-async-scheduling`。没有绕过validator。startup/图捕获不计入正式测速窗；eager测速不能与Graph正式结果拼接。GPU进程检查需宿主可见的nvidia-smi；沙箱空输出不是空卡证据。

旧训练指南里有固定卡4/5/7及旧PID，均为当时履历，接手不得直接杀或恢复。部分CPU旧controller有pause.request仍存活，目录显示waiting/paused不代表新训练待启动。CPU报告watcher可能继续写汇总，不会自行训练、删文件或占GPU；重启任何controller前查命令/root/lock/状态，防止重复作业。

总账生成器：[ledger.py](residualV3_main_v1/maintenance/experiment_ledger/ledger.py)，叙述登记：[registry.json](residualV3_main_v1/maintenance/experiment_ledger/registry.json)。总账是自动生成产物，改结论应在登记/解析器中维护并使用已有锁，避免直接改MD随后被watcher覆盖。

## 12. 新对话接手的执行顺序

1. 读取本文件和选用实现/恢复门禁/当前status；只先核对，不重复跑已完成实验。
2. 核对运行中进程、GPU UUID/隔离及磁盘；已完成E1/E2不要重启。确认尾部根即时log指向的版本。
3. 先核对第13节最新完成结果；选用仍iter20/21，NOFILL及iter28/29均为独立冻结实测版本。新优化必须新冻结迭代，固定E2权重/阈值，先实际数值门禁再16/80短测。已明确授权并完成的NOFILL正式长测不重跑；未有稳定截断收益，不新增截断正式长测。
4. 分开回答C1与C40剩余成本：预测/统计开销、GPU关键路径、BM16停修能否真省矩阵、target实际图行与padding、CPU等待是否复发。用同窗对照，不能只凭util或repair计数归因。
5. 不改T1 Gumbel RNG/全词表概率或真实lag1换速度；先确认有效prefix/q/bonus/KV合同。若尝试物理target截短或新latent优化，需要各自完整门禁。
6. 每次保留通过版及负结果，更新总账、版本选择和即时log。输出明确“选用/未选用、短测/正式、是否净提速”。用户目标是时间与可靠比较，不是生成更多目录或只给方案。

可直接粘贴的新对话入口在 [NEW_CHAT_PROMPT_20261011_CN.md](NEW_CHAT_PROMPT_20261011_CN.md)。旧 `HANDOFF_RESIDUALV3_NEXT.md` 是9月历史窗口，不是当前交接入口；保留其历史指令，不要套到本轮。

## 13. 最新完成结果与新窗口接续（2026-10-11 19:29 北京时间）

### 13.1 用户最近请求与当前状态

最近用户依次要求：对照Domino约1.2ms重新审查头结构；验证人工头1.854ms是否低估真实服务增量；把新FULL15/独立DFlash比较改为正式长测；长测同时继续优化块内截断。上述正式比较、真实成本诊断和新截断候选短测均已完成。当前请求是写交接与prompt，本轮只检查/更新文档，没有启动新GPU作业。

**此前聊天“等待GPU0/7”的状态已过期。** iter28 C1于北京时间2026-10-11 13:48:56完成，iter29 C40于14:02:21完成，合同审计于14:03:08通过。C1/C40都是GPU0空卡隔离观测；原GPU7协议在首个服务前归档，源码不变。当前宿主进程核验：正式控制器3636012、短测控制器4049950/4049954、报告4049951、审计4034575均已退出，没有这批任务的存活命令。PID只供查历史，**不要重启run_pair、run_short_iteration、reporter或已完成正式测试**。证据：[进程快照](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/HANDOFF_PROCESS_SNAPSHOT.json)、[最终进度](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/progress.json)、[合同审计](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/CONTRACT_AUDIT.json)。

### 13.2 新FULL15_NOFILL正式长测：本次同窗对独立单训DFlash

用户明确说“不是短测改成长测”，已据此完成正式长测。之前限制未获净收益不得长测针对截断，不能反过来否定本次明确授权；也不代表现在获准反复长测所有新原型。

E2独立单训DFlash对E2联合主体＋三层FULL15_NOFILL；GSM8K nonthinking，C1=200、C40=1319，max2096/context4096，T0/T1、BF16、batch-invariant、CUDA Graph、block16。GPU7顺序空卡；C1 KV52GiB，C40正式44GiB、smoke38GiB，各臂一致。旧E1/E2正式面板C40是46GiB，因此只作本次同窗相对比较。8窗6076请求均完成，Graph/隔离/源身份通过；T0两模型逐题消息、结束原因、长度在200/1319题全部相同。AL表示接受长度，不是GSM8K答题准确率。

| 设置 | 独立单训AL | 新FULL15 AL | AL增长 | 单训TPS | 新FULL15 TPS | TPS增长 |
|---|---:|---:|---:|---:|---:|---:|
| C1 T0 | 7.478576 | 8.515070 | +13.86% | 178.410 | 196.152 | +9.94% |
| C1 T1 | 6.335933 | 7.488577 | +18.19% | 151.859 | 173.253 | +14.09% |
| C40 T0 | 7.609462 | 8.652091 | +13.70% | 3146.145 | 3365.701 | +6.98% |
| C40 T1 | 6.482782 | 7.698110 | +18.75% | 2687.595 | 3011.228 | +12.04% |


报告：[正式结果](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/RESULTS_CN.md)；[原始比较](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/COMPARISON.json)；[最终审计](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/FINAL_AUDIT.json)；[C1正式日志](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C1_formal.log)；[C40正式日志](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/single_card_C40_formal.log)。

本表FULL15只含已核验的局部取消empty调试预填充，runtime与iter26的178份Python一致；**不含下述概率准备融合，不开截断**。它是两套不同主体权重的整体方案比较。独立body SHA `0b8cef16a7e171981111a621e2551a02c663d046ffc575d6cf6d0252c8388ce7`；联合body SHA `ad4ab70fff01f329780d1e5a245b5268e238dd1f68dd9ec51c909ca252728168`；头SHA `6635f429a363ba9928da653f3337ce991e644c379cb7ac4e159ac1413b71944f`。完整身份在各协议，不能用关头联合body替baseline。

### 13.3 已验证的真实头成本，不能继续沿用旧2ms结论

发现确定性模式把torch.empty调试预填充录入repair Graph；只在头范围关闭预填充，finally恢复，保留确定性算法、BF16算序、RNG与全词表采样。人工C1 T0/T1从1.854/1.920降至1.307/1.359ms。

用户质疑微测低估服务成本后，已用同E2联合主体开关头、真实服务内分段及同worker输入重放核验：

| C | T | 原服务头+采样ms | NOFILL服务头+采样ms | NOFILL完整propose对同主体关头增量ms |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1.8426 | 1.2989 | 1.3018 |
| 1 | 1 | 1.8850 | 1.3422 | 1.3297 |
| 40 | 0 | 6.1872 | 5.3710 | 4.9767 |
| 40 | 1 | 6.9675 | 6.1434 | 5.0601 |

单位为整批每decode block，不是每输出token。诊断包含body、raw LM、head、query Graph、context KV、完整propose、target forward独立事件；嵌套跨度禁止直接相加。C1=2/C40=40、ignore_eos256短上下文、C40 KV38GiB，插桩影响图，因此不是正式吞吐证据。关头对照context/接受轨迹不同，增量是观察估计，非严格同状态因果差；12组同worker真实状态重放token/q精确通过。历史4.116ms与2.463ms差距包含旧微测缺少batch-invariant ATen，不能机械加到当前1.854ms。Domino作者1.2ms的GPU/路径/边界不同，不能当相同条件硬目标。详见[真实服务报告](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_32_native_cost_snapshot/RESULTS_CN.md)。

### 13.4 新截断组合iter28/29已完成，仍无净收益

仅改实现，E2权重/阈值/三层结构不变；Q45=.1/.1，最少8、patience2、延迟1、最早10；C1=16/C40=80、max256/context4096、BF16/Graph。三个臂都是同一联合模型，**本表不是独立DFlash比较**：

- FULL15_NOFILL：上一轮已核验的no-fill控制。
- FULL15_PREPARED：共同概率准备融合，BF16→FP32 exp直接写，保留原分母归约/舍入。
- Q45_PREPARED_CUT：同样概率准备＋原冻结尾部策略；C1增加熵prefix/决策融合，C40增加提前精确缓存。

| 设置 | 对比 | AL变化 | TPS变化 | 隔离 |
|---|---|---:|---:|---|
| C1 T0 | FULL15_PREPARED / FULL15_NOFILL | +0.000% | -0.487% | True |
| C1 T1 | FULL15_PREPARED / FULL15_NOFILL | +0.000% | -0.065% | True |
| C1 T0 | Q45_PREPARED_CUT / FULL15_PREPARED | -0.699% | -0.633% | True |
| C1 T1 | Q45_PREPARED_CUT / FULL15_PREPARED | -0.300% | -0.143% | True |
| C40 T0 | FULL15_PREPARED / FULL15_NOFILL | +0.000% | +0.055% | True |
| C40 T1 | FULL15_PREPARED / FULL15_NOFILL | +0.000% | +1.346% | True |
| C40 T0 | Q45_PREPARED_CUT / FULL15_PREPARED | -0.375% | -1.389% | True |
| C40 T1 | Q45_PREPARED_CUT / FULL15_PREPARED | +0.290% | -1.741% | True |


结论：共同准备在C40 T1本次+1.346%，C40 T0仅+0.055%，C1略负；不能称四设置稳定提升。截断相对同样优化的FULL15四窗TPS均负，**不把候选部署为已成功停修方案**。selected仍C1 iter20/C40 iter21，开发177份runtime及evaluator本次重核仍与iter21逐字节一致；NOFILL正式版与新候选保留在独立冻结目录，不覆盖开发版。

[最终组合结果](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/RESULTS_CN.md) / [C1短测日志](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/iterations/iter28_fused_prepare_decision_C1/single_card_C1_short.log) / [C40短测日志](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/iterations/iter29_fused_prepare_prefetch_C40/single_card_C40_short.log)。各迭代native_protocol记录182份源码SHA，完整头和服务实际token/真实前驱/verifier q/后修q45/有效prefix与bonus/Graph通过；12个测速窗口都隔离，target/draft无capture/eager/fallback。新旧FULL输出在C1/C40、T0/T1均逐题相同。KV使用原生计数更新与compact-prefix参考的发射数等价校验，**没有独立KV缓存逐字节比对**；仅Q45核验，latent未测。

### 13.5 四个微测原型与剩余关键路径

- probe33：熵最大值prefix/决策状态融合，完整头18组及90组混合状态通过；C1整头微测约省0.5%–0.7%，C40不明显，只接入iter28 C1。
- probe34：共同概率准备，9组含实际anchor孔洞stride及18组完整头通过；人工C40 FULL15约5.48→4.43ms、6.25→5.19ms。两臂共同收益与截断收益必须分开；native结果如上，并非微测省1.05ms就必然对应TPS同幅增加。
- probe35：整数直方图4分片，数值通过，C1慢4%–5%、C40慢37%–44%，已拒绝进入native；失败源码/计时保留。
- probe36：在Top256 IDs就绪后提早相同缓存计算，18组完整头及Graph实际token/q/cut通过；C40微测省0.05–0.06ms，C1反而慢1.0%–1.4%，只接入iter29 C40。native未兑现截断净收益。

probe33首个启动曾因缺CUDA兼容环境失败于上下文初始化前；修正后复用runtime.utils.environment，数值通过。失败日志留在missing_compat_environment，不能混成数值反例。probe34原flag-off参考TopK/float发射顺序换过；iter28/29旧控制已恢复cast-before-TopK，不能说所有旧参考源码字节相同。

精确缓存/entropy/q45、跨流事件和BM16整tile门控仍有成本；target依然固定补齐图。probe34强制cut10时C40 T0整头仍比相同PREPARED FULL慢0.274ms；真实策略平均修约13个位置，省得更少，且AL下降时每轮还须更快才能抵消。现有native仅有整个窗口计时，**尚无iter28/29真实混合长度服务的分段成本归因**。下一步优先用最新冻结配置定位prepare/cache、预测流与主流重叠/等待、target padding关键路径，再提出单一新实现并做短测；不要把人工/单个kernel加总当端到端证据。

原始微测：[汇总](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/PARALLEL_OPTIMIZATION_CN.md)；[关键路径说明](residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_37_native_fused_followup/CRITICAL_PATH_CN.md)。没有epoch3、重训、新native监督或新阈值；不重复包装已失败的on-policy路线。

### 13.6 新窗口第一步

先读本节、机器状态JSON、probe37结果/审计与两总账，确认全部complete；接着只读检查GPU/磁盘/源码，不重复跑本批。若继续实现优化，先建立新的冻结迭代并保持权重/策略，完成实际数值合同后做16/80题短测；未有稳定净收益，不新增截断200/1319正式长测。最新结果源是iter28/29和NOFILL正式根；根目录旧日志链接和v2 formal日志均不能冒充最新结果。每轮更新registry再refresh两总账；记录同窗AL/TPS与隔离、源身份。资源受限时给证据，不能动别人或已完成/暂停训练。


## 14. 2026-10-12下午：每领域一万条头训练已启动

用户恢复系统更新后的工作，并要求每领域至少10000独立训练提示、只训练头。新冻结执行根为 `residualV3_main_v1/research/mismatch_expert_headonly_30k_20261012_v2`；实时读 `status.json`、当前调度日志 `controller_attempt06.log`。

已准备42288独立groups：头训练30000（每领域10000），router6144、选型3072、独立测试3072；三种子复用同一数据。锁定测试直接与E2实际833176训练groups核验交集0；原prompt池843028为行数、841050为首用户group数。训练用E2重生成前缀，独立留出为未见group原来源第一轮答案，属于静态teacher-prefix诊断。来源族为领域代理，不能称完美任务标签。

只训练新头全部输入投影/fusion、专家suffix、direction/dose/router，冻结E2联合body、target、输入/输出表、basis及旧q45/latent。四对照4353537参数通用头、6262274双专家、同预算随机分工、6262365参数通用头；每组每种子3次完整遍历，loss保持CE/D-PACE+OUTPUT，raw损失冻结为常数。12提示采集检查及四组训练/恢复检查通过；BF16双专家按相同分组重建E2逐位一致，分组与整批数值差记录在TRAIN_GATE.json，不能称native batch-invariant/Graph已通过。

初始误启动六卡，用户纠正后已核UID/cmd/starttime并停止四个本人采集进程，显卡释放已审计；一般本任务最多两卡，用户最新允许1vLLM服务+2张训练时三卡。当前离线缓存采集/训练只用GPU0/1两槽，无常驻vLLM。六数据分区按原身份排队续跑，缓存<=32GiB；训练保存FP32masters/Adam/RNG/cursor。不要再恢复旧96提示或已完成原生队列。

当前大样本在采集，头训练与三种子结果由调度器自动衔接。报告入口RESULTS_CN.txt，源码/权重身份SOURCE_IDENTITY.json。没有新native AC/AL/TPS；完成静态验证后再按证据决定native合同/C1/C32短测。两总账已注册，结果完成时自动刷新。

### 14.1 E2前两层推理补测

用户确认仅C1/C32推理使用前两层，新专家头大样本训练保持三层。新code/results根 `epoch2_head2_three_domains_C1_C32_20261012_v1`；原E2三层未改，独立derived_head2移除blocks.2六个tensor，保留的3761409参数逐位一致。GSM8K C1=200/C32=1319，HumanEval164、MT-Bench80首轮，T0/T1、max2096/context4096，24正式窗计划。GPU1 GSM8K C1 smoke已启动，GPU0大样本采集继续，采集通常最多两卡；当1vLLM服务与另两张头训练并行时才可三卡。实时读新root/status.json与gsm8k_C1_smoke.log，当前共享控制器controller_attempt06.log。每组先合同烟测、正式测、PAIR_AUDIT；两层未经重训，不用旧E1跳层结果代替本轮E2。

### 14.2 2026-10-12 16:38之后：用户放开卡数，开始共享余量分片头训练

用户明确“那开始训练吧，这回就是有卡就占，因为每张卡都被占了，可以分片”。旧2卡/3卡组合上限已被本次训练授权替代；全部卡有他人进程时可在显存余量内训练，禁止停止他人或其它项目。新源码快照在头实验根implementations/shared_cpu_cache_v3；原exclusive v2源码/检查未改。CPU保留缓存，每loss microbatch4blocks送GPU；每32blocks更新一次，参数/损失/数据预算不变，头PyTorch allocator cap8GiB。新gate确认峰值3.112GiB、四种头及恢复下一更新逐位一致；当前六个训练NVML各约3.5GiB。CUDA兼容环境沿用。

42288缓存已全部完成，CACHE_COMPLETE_AUDIT.json核验所有SHA、shape、独立group和三领域各10000头训练；含剩余1536heldout的共享补采完成，采集进程退出。GPU0/1/4/5/6/7分别分派三种子通用头与mismatch专家头，已有实际optimizer更新和master/Adam/RNG/cursor断点。12个seed/arm独立任务按卡余量续接，禁止把不同模型梯度合并。新专家训练仍三层，仅原E2 native测试前两层。真实进度读status.json，当前CPU调度shared_controller_live_attempt00.log；新源码controller_live.py，训练源train.py不能在运行中改。

原native GSM8K C1 raw T0/T1完成后，GPU1被chenkui新进程进入，head2 vLLM启动时free62.08GiB < .92要求72.91GiB，服务退出。整组未完成，没有配对TPS增长；失败smoke/formal/protocol/log以SHA核验移动到测试root/gsm8k/C1/prior_attempts/foreign_startup_20261012_0829。C1/C32三领域任务已带入新调度，只在空卡跑，不将共享训练耗时用于论文TPS。旧原生控制器不恢复，完成的其它正式报告不重复。

### 14.3 2026-10-12晚间：全部头训练及静态评估完成

最新（2026-10-12 18:30北京时间）：每领域10000条的头训练和静态评估全部完成，3种子×4对照共12模型，每模型3遍/5613更新。mismatch双专家+router相对同活跃参数通用头T0一致率−0.212百分点、T1 overlap−0.098百分点；T1对随机分工−0.003百分点且区间跨0，未通过下一阶段native候选门槛。此为static teacher-prefix诊断，不是AC/TPS。详读头root/RESULTS_CN.txt、RESULTS.json、ANALYSIS_RECOVERY_AUDIT.json。汇总导出错误已修复，训练源/12个best/6份router SHA不变；只补评未重训，失败log保留。另一个原E2前两层C1/C32测试仍未完成：GPU1 GSM8K C1 formal运行，其余组队列继续；调度shared_controller_recovery_v2_attempt00.log。

恢复证据：ANALYSIS_RECOVERY_PLAN.json固定旧训练源与所有已训练头/router文件SHA；ANALYSIS_RECOVERY_AUDIT.json在三种子全部完成后核验通过。原analysis日志保留，追加analysis_finalize_v2.py只装载3072留出组调用原score、完整词表LM投影，分别使用CPU metadata索引与CUDA评分索引；只补评、不训练，CPU调度替换没有中断原生服务。原E1/E2和完整恢复断点未改。静态负结果不推进新专家原生长测；原E2推理前两层补测是独立授权任务，继续队列。

### 14.4 原E2两层与三层直接对照

2026-10-12 19:02北京时间追加：用户明确两层必须与三层直接比。新冻结code/results根epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1；同一个E2联合主体，HEAD3_NOFILL→HEAD2_NOFILL，二者NOFILL/FULL15，三领域C1/C32、T0/T1，200/1319、164、80首轮，max2096/context4096、BF16/batch-invariant/Graph；两层仍仅推理跳第三层，未重训。24正式窗已进入空卡队列，预检通过，直接GPU测试尚未启动。当前GPU1原head2 vs DFlash的GSM8K C32 formal继续；C1已通过4正式窗，已完成和失败历史全部保留。调度controller_depth_comparison_attempt00.log。新报告列三层→两层AC/TPS变化，只用同卡同组窗口。

新增入口：residualV3-nonthing/current/01_results/epoch2_head2_vs_head3_C1_C32。直接对照预检再次核验两层保留tensor逐位相同、源码冻结、相同联合主体及两个NOFILL设置；actual sampling/前驱/verifier q/Graph需排队smoke，尚未通过。原head2 DFlash对照队列继续，原独立训练baseline不换成联合主体关头。只是补层数消融，30k新专家训练静态负结果及所有12权重不动，不启动新训练。controller_depth_comparison.py仅代替本人CPU调度，现存GPU worker保留；独占空卡才有TPS证据，同结果root每次单worker防并发报告写入。registry已登记并refresh两总账。

### 14.5 两层/三层GSM8K完成与HumanEval隔离失败恢复

2026-10-12 19:55北京时间核查：两层/三层直接对照完成GSM8K C1/C32两组、8/24正式窗，PAIR_AUDIT均passed、T0输出逐题一致、同联合主体/隔离/Graph通过。两层对三层AC四窗−2.179%/−1.263%/−1.412%/−1.887%，TPS−1.930%/−1.196%/−1.375%/−1.339%（顺序C1 T0/T1、C32 T0/T1）；这是已测GSM8K结果，另两领域未完成。GPU6 HumanEval C1两层/三层已进入formal；GPU5原两层vs独立DFlash的MT-Bench C1 formal继续。原HumanEval C1/C32烟测被外部进程进入污染，未开formal，全部失败文件SHA存档到各自prior_attempts/foreign_smoke_20261012_1154，并仅允许审核后的attempt00各重试一次；不碰外部进程，不放宽隔离。当前CPU调度controller_depth_comparison_retry_attempt00.log，native/source权重不变，现存GPU worker未中断。

### 14.6 固定旧正式显存的三层模块计时后续队列

2026-10-12晚间用户授权：当前两层/三层对照结束后接三层C1/C40模块计时。用户明确禁止显存改来改去和为改预算重复实验；固定旧三层正式KV C1=52GiB/C40=44GiB、utilization0.92。只有两项真实服务诊断，头各模块用同worker真实状态重放，无第二套整段工作负载、无预算扫描。新root epoch2_head3_module_latency_C1_C40_20261012_v1；CPU排队module_profile_queue_attempt00.log已运行，尚未开GPU，新数值合同pending。当前C1/C32运行和已有结果不改、不重跑；本轮C1曾用16GiB的配置差异必须标注。取消的八条件规划尚未启动GPU，配置存档planning_archive。

代码/测量边界：新runtime仅新增CUDA事件和target LM/rejection包装；头函数剥除事件context后的AST与原数学逐项相同，源码独立冻结，原在跑两层/三层源码SHA核验不变。服务总体分段与头内同状态重放分表，不把细事件扰动或嵌套span总和作为原生吞吐。新结果的NVML/allocator峰值均留证；资源不足应等待或保留失败，不静默改KV/关闭Graph。先实际token/前驱/verifier q与Graph合同再计时，绝不声称已通过尚未执行的GPU合同。

### 14.7 全部直接层数对照完成，模块计时采集完成但覆盖有限

2026-10-12晚间完成核验：两层/三层三领域C1/C32共24正式窗、8028请求全部完成，21:33审计passed，T0逐题同输出，179源码/12资产核验。两层AC12窗全降，TPS11窗下降；唯一正窗HumanEval C32 T1 +5.035%，不得宣称两层稳定更好。三层C1/C40计时21:43采集结束，固定KV52/44GiB、util0.92，无预算扫描，数值/Graph/隔离通过。覆盖复核发现C40 T0满并发组件仅3份、C40 T1满40主体/raw LM/头直接组件缺失；后者有157份纯decode混合并发组件及完整头/模块同真实状态重放，不能补成满40直接服务时长。此轮仅复核已有数据，无重跑。全部本人GPU worker和计时队列已退出，空轮询CPU控制器已核身份退役；原两层vs单训DFlash仍4/6组完成，HumanEval C1/C32两组隔离失败保留未完成、停止自动重试。

入口：直接对照 root/RESULTS_CN.md、FINAL_AUDIT.json；计时 root/RESULTS_CN.md、ANALYSIS.json、TIMING_COVERAGE_CN.md、TIMING_COVERAGE_AUDIT.json。计时FINAL_AUDIT的passed是数值/Graph/隔离及记录完整性，其full_concurrency_timing_coverage_complete=false；status明确acquisition_complete_with_timing_coverage_limitations。源与原始样本不变；旧自动汇总写入analysis_history，追加覆盖复核代码没有GPU调用。不要重复已完成24窗或因为显存重跑；用户明确没时间，未来只在明确新请求下处理剩余证据/失败基线。RETIRE记录见head实验根RETIRED_NATIVE_CONTROLLER_20261012.json；PID均历史，恢复前重新查身份。
