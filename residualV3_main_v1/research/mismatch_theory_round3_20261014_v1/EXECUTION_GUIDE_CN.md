# ResidualV3 第三轮执行指引：核清概率边界，检验修正是否过强

版本：2026-10-14 v1。仓库：`/data/weijiahong/speculators-main-lora`。配套：[启动 prompt](START_PROMPT_CN.md)。

## 先用一分钟读懂任务

**已经知道：** 冻结修复头平均能缩小大小模型的分布差异，在真实解码中也有每轮接受收益。**仍不知道：** 它为什么在一部分状态上失败，已有结果能否支撑超出“又一个有效 AR 修复头”的解释。第二轮新增几何量没有提供实质解释增量，不能继续当成已成立的理论。

本轮只做两项工作：

1. **随机源诊断：** 检查接受采样的随机源和概率记录，判断原概率残差警报能否归因于特定实现；这是测量可信度问题。
2. **固定半强度对照：** 保持模型、方向和输入不变，把已有修正强度固定乘 0.5，与原强度及同主体不修正比较；这是对“一部分失败可由减小现有修正缓解”的直接干预。

两项都不预设成功。半强度即使有效，也只提供当前模型的局部机制证据，**不会自动产生一般 mismatch 理论或证明区别于全部近邻的方法创新**。本轮结尾必须说明理论推进了多少，不能用完成工程或更多正数代替回答。

当前交付 chat 仅制作任务书。用户将启动 prompt 交给执行 chat 后，执行 chat 可在下述范围内自主编写隔离代码、采集、分析和收尾；无需重复申请“重开第二轮”。旧轮次已结束，不向旧 run 追加数据或改结论。

## 1. 必读资料：按问题读取

以下缩写均为仓库相对路径；执行时展开为真实路径，不将缩写当作已有环境变量。

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
MAINT=residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1
```

| 顺序 | 资料 | 必须弄清的问题 |
|---|---|---|
| 1 | R2 `results/RESULTS_CN.md`、`THEORY_CARD_CN.md`、`CLAIM_EVIDENCE.json`、`RUN_AUDIT.json`，以及 `COMPLETION.json`、`NEXT_CHAT.md` | 哪些结果支持、削弱或没有回答原主张？ |
| 2 | R2 `results/NATIVE_PROBABILITY_BOUNDARY_CN.md`、`MEASURES_AND_IDENTITIES_CN.md`；`analysis/NATIVE_BRIDGE_AUDIT.json`、`NATIVE_BRIDGE_REVIEWED_AUDIT.json`、`NATIVE_RNG_SOURCE_REVIEW_CN.md` | 机械身份核验通过与概率规律未建立如何并存？ |
| 3 | R2 `analysis/NATIVE_RNG_SOURCE_AUDIT.json` 的 source 元数据、`analysis/audit_native_rng.py`，及它们指向的实际安装版源码 | proposal、接受 U、bonus/correction 各用什么随机输入？不要整份打印大型逐轮 JSON。 |
| 4 | R2 `code/native_scheduler_capture.py`、`native_verifier_capture.py`、`native_accounting.py`、`run_native_stage.py`、`native_diagnostic_launch.py` | 怎样取得真实 GPU token、实际生成 q、验证 p/q、原始接受与已发输出？ |
| 5 | R2 `code/hf_frozen_adapter.py`、`repair_replay.py`、`prepare_round2_samples.py`；FROZEN `repair/model.py`、`head_latency/structure_fusion.py` 及其 BF16 更新依赖 | 强度乘数应插在哪里？怎样维持同状态、同主体、同原始 q？ |
| 6 | R1/R2 `protocol/SAMPLE_MANIFEST.json`、R2 `protocol/PROTOCOL.json`、`LOCK.json`，以及 R2 `results/RUN_FILE_MANIFEST.json` | 排除哪些已暴露题？哪些源码、资产和测度已锁定？ |
| 7 | R1 `sources/LITERATURE_MATRIX.md`；R2 `sources/ROUND2_LITERATURE_MATRIX.md`、`analysis/FINAL_THEORY_INTERPRETATION_REVIEW_CN.md` | 哪些是标准数学、已有修复思路，哪些才可能是本轮新增证据？ |

文献只做与本轮结论直接相关的定向回查：单状态拒绝采样/连续前缀条件读 Leviathan、SpecTr、Block Verification；已有条件失配与轻量修复读 Domino、DSpark、DFlash-2、DFly、xPress、DSpine。先用已存一手版本和阅读记录，必要时访问原文并记录版本差异。不把文献扩搜变成新的主任务，不用“没搜到同名方法”证明创新。

原六数据集评测在本任务书核验时 `status=complete`、FINAL 和 CPU COMPLETION 均通过，24 个配对完成。只读核验 FORMAL/MAINT 的状态，不启动原 controller、`parallel.py` 或收尾程序。

## 2. 事实、假设与论证顺序

| 现有证据 | 允许的结论 | 不允许的升级 |
|---|---|---|
| R1 120 题：正确前驱相对置换前驱 TV 改善约 0.179 | 当前修复利用了前驱对应信息 | 条件失配必然主导全部大小模型差异 |
| R2 新 48 题：native 局部 TV 改善约 0.128；同联合主体开头相对关头每验证轮已发 accepted proposal 增加约 0.892 | 冻结修复头的局部效果与真实每轮接受收益均存在 | 全部性能收益由单一 mismatch 来源解释；插桩结果等于 TPS |
| R2 新几何特征相对 MSE 改善仅 0.02015%，区间跨 0，远低实质阈值 | 该固定解释假设被削弱 | 换特征/删覆盖基线后沿用旧 test 宣称确认 |
| R2 HF 约 14.27% 加权状态修复后变差 | 有值得解释的失败状态 | 失败已经被识别为剂量过大 |
| 原工程概率审计 false；源码显示共享第一层 Philox 来源；1758/1758 first-reject 重构吻合 | 机械行为有证据，概率解释保留疑点 | 共享来源已证明有偏，或计数吻合已证明独立无偏 |

本轮论证顺序：**大小模型分布差异 → 修正具有收益也有失败 → 固定方向下改变强度能否改变这种取舍 → 该变化是否伴随真实接受后果。** 随机源诊断服务于解释可信度，不取代上述科学问题。

## 3. 工作区、预算与自主边界

- 只新建 `R3/runs/<UTC时间>_mismatch_r3/`。复制必要代码到新 run，审计硬编码路径和 import，禁止写回 R1/R2、冻结导出/权重/basis、安装环境、原正式报告。旧 run 不因新证据改成 passed。
- **本轮方案的默认硬上限为 3 GPU小时，同时最多 1 张 GPU。** 随机源支路最多 1 GPU小时，半强度支路最多 2 GPU小时；启动、编译、warmup、失败、重试、采集及清理均计入。该数额是本任务书的新轮上限，不是历史剩余额度，也不是完成时间保证；不能互相借额后扩大工作量。
- 随机源 CPU 诊断最多累计 1 CPU小时（各子进程 CPU 时间求和）；全 run 最多 4 个 worker、8 个计算线程，新增磁盘最多 20 GiB。保留模型路径引用，不复制大权重；全词表概率以流式统计和小规模工程快照为主，避免保存所有轮的完整向量。
- 实时检查宿主 GPU UUID、UID、进程和预约；旧 GPU7 空闲记录不是现在的准入依据。不干预他人任务；只管理匹配 UID/PID/PGID/startticks/CMD/run 路径的自有进程。
- 父进程监督累计时间，预留落盘/退出时间；每次 GPU 启动记录目的、验收点和预计剩余成本。新协议允许限定重试，不继承旧轮终止计数；同一缺陷没有新定位或代码变化不重启。连续两次修订仍卡在同一验收点且无定位进展，结束该支路。
- 范围内编码、CPU 验证、dev 工程修复、锁定采集、负结果分析和收尾均自主进行。不训练头/主体，不搜索其他剂量、不增加 gate/router/预测器，不做近邻全量复现或服务吞吐评测。
- 若资源不足或工程受阻，先完成独立可做的支路，保存部分结果并说明未完成项；不得缩短到几条有利结果后冒充预定实验完成。超上限或新增结构需另立任务，不在本轮悄悄追加。

## 4. 实验 A：随机源诊断，固定 6 题 × 8 seed × 2 设置

### 4.1 检验对象和边界

仅使用 R2 清单 `engineering=true` 的原 6 条 dev（每域 2 题）和冻结联合主体＋完整头 B2；全部属于已暴露工程数据。每题固定 8 个重复，每个重复两设置，合计 **96 条真实请求**。这些是新的随机源干预诊断，不是重做旧主实验、不能称新留出。

- `R_original`：原实现，实际安装源码身份与旧记录相核验。
- `R_split`：仅在新 run 的隔离 overlay 中，把 **first-reject 接受 U** 改为与 proposal 分流的随机输入。proposal Gumbel、头、target、bonus/correction 算法及随机调用保持原身份。禁止简单替换共用 seed 参数而顺带改掉其他随机调用。

必须记录真实 GPU position/counter、请求、验证轮、slot、proposal token、生成 q 与 verifier 所用 p/q、接受 U、实际判决用 logp/logq、必要 LSE、接受布尔值和 first-reject。采样时实际使用的量与之后 torch FP32 重构的量分字段保存。必须从实际操作数复现原模式全部 first-reject；不能仅靠重新推导位置代替 buffer 绑定。拿不到这些量时，A 的概率/数值身份验收记未完成，不把重构叫作 bitwise 审计；B 可按 §4.3 单独判断机械准入。

分流合同在看本轮结果前写定：只给 reject 分配独立 role key，**保留原 position/counter 映射**，不从已经生成 proposal 的 Philox word 派生 reject U。固定并保存两角色的种子表和生成方式，不能试多个 salt 挑结果。按 request、重复、验证轮、块内位置记录实际随机数地址，检查整数范围及“已经影响过接受决策”的地址复用；生成过但未 reached 的 U 后来再用，不自动算依赖缺陷。若发现真正的决策地址复用，单独报告，不能同时悄悄改 round counter 后把差异全归于 role 分流。**PRNG 分流按伪随机假设解释，不因用了另一个 hash 就宣称数学独立。**

### 4.2 先建立能辨识问题的检查，再采集

1. **源码合同与 CPU 小词表：** 固定玩具 p/q（相同、不同、接近拒绝边界及零质量边界）、first-reject、EOS、预算、bonus/correction 和输出对账。独立理想 U 下的解析/精确枚举值与模拟相容；测试代码和预定容差存新 run。玩具正确不能证明真实模型采样正确。
2. **同路径随机源核验：** 在限定 dev 快照上核查原/overlay 原模式同输入行为；分流模式保持 proposal 生成侧不变、接受 U 的 role key 确实改变。原模式 replay 保留捕获的 proposal 和原 U，只验实现身份；禁止冻结旧 proposal 后重抽原 seed，再称为原实现的条件期望。对固定 proposal 路径重采独立接受 U，只验证条件计算链路，不估计原 U 给定 proposal 的真实条件分布。必要的 kernel 小测计入实验 A 预算。
3. **完整请求诊断：** 原 6 题，重复编号 0…7。proposal seed 默认为 `2026101400 + 100*题目序号 + 重复编号`，题目按 group SHA 排序；split 的独立 role 种子表另按固定规则生成并入锁。C1/T1、15 proposal、max512、context4096、eager、同步、中性采样参数。两设置相同题目和重复索引配对，**不假设真实轨迹相同**。先在其中一对完成工程验收；通过且字段/协议一致就计入 96，不重新跑一套相同请求。

对每请求计算实测已发 accepted proposal 总数与工作估计的差：

```text
r = N_emitted_accepted − Σ_round Σ_k G_k Π_{i≤k} b_i
b_i = min(1, p(actual_token_i) / q(actual_token_i))
```

G 仅含 proposal 长度、先前 EOS 和轮初预算，不含实现接受事件。按完整预生成路径定义，不能把拒绝后的 G 改成 0 来让公式贴合计数。区分 sampler 原始接受长度和裁剪后的 emitted 数；bonus/correction 不计入 accepted proposal。

此式在相应独立性等假设下才有期望解释，沿用时注明算术精度、停止规则和工作模型；有限精度 U 的实际判决与连续均匀假设分开。**不能以两个设置各自的区间是否跨 0 作为它们有差别的检验。** 报告全部 6×8 配对残差、每题 seed 均值/离散度、`r_split − r_original`；先每题平均8个重复，再6题等权。用固定2,000次分层配对bootstrap、seed=2026101402，题目为外层、重复为内层，重采样时两设置保持配对。仅 6 题，范围限于工程诊断；残差总量受请求长度/路径变化影响，另报告轮数和输出长度作解释，不把除以随机轮数后的量偷换为主量。

“共享随机源解释了原警报”需要源码干预身份、条件链路以及完整请求结果共同支持。残差变小、区间含 0 或小词表通过，均不能单独证明真实输出无偏，也不能认证 bonus/correction 或完整输出分布。若没有明确干预效应，记为“不支持当前归因/精度不足”，不能改 seed 或增加重复直到消警。预算不足时按预定题序在完整两设置配对块后结束，保留失败与未完成项；不得删除异常seed。

### 4.3 给下一实验选择实现，不按效果挑

在开启半强度留出前记录 `RNG_DECISION.json`：

- 若 split 的调用范围、buffer/计数绑定和工程合同通过，半强度 native 三臂统一使用 `R_split`。这是预定的工程选择，**不要求残差变得好看**，也不构成无偏证明。
- split 未通过、原模式机械身份仍通过时，三臂统一用原模式，仅报告真实经验计数并保留概率边界。
- 原模式机械身份也无法核实，半强度支路只做 HF；native 标未完成。概率理论未证本身不阻断可靠的经验计数，token/q 错配则必须阻断。

不新增“两种 RNG × 三种剂量”的完整交叉试验，不追补旧六数据集正式评测。

## 5. 实验 B：只改变强度，不改变模型

### 5.1 三臂和数值合同

三臂全部使用同一冻结 E2 联合主体和头权重：`η=0`（joint raw）、`η=0.5`（半强度）、`η=1`（原完整头）。本轮不加入独立单训 DFlash；它不是识别此次强度干预所必需的第四臂。

在同 C/H/前驱状态下，用头已经预测的 direction 与 dose，把 **sigmoid/cap 之后的 dose 乘 η，再送入原 BF16 logit 更新**。保留原 raw Top256 集合、hidden norm、输入/输出 basis 和完整词表归一化/采样；不是温度 0.5、不是概率插值、不是 dose_head 参数减半。η=0 可直接走同主体 raw。BF16 舍入使最终 logits 未必严格位于两臂的算术中点，记录真实更新顺序而非伪造精确线性。

在 dev 上检查：η=0 与同状态 raw 的 TV ≤2e−6；η=1 与冻结原/优化实现的 TV ≤2e−5；η=.5 与独立按上述顺序实现的参考更新 TV ≤2e−5。不按 fresh 结果放宽。保存三臂实际 logits/q、方向、dose_before/dose_after、候选身份的小规模快照；原/半强度对照除这个乘数外应一致。

native 各臂反馈自己实际采样的前驱，实际生成 token 必须从该臂 qη 采样，verifier 使用同一 qη。禁止生成完整强度 token 却拿半强度 q 验证。η=1 dev parity 的数量按实测报告，不继承或虚构 fresh parity 次数。

### 5.2 样本和开发：新留出不再使用 MT-Bench

本次只读按 R2 规范化规则核验：GSM8K 面板 200 题排除前两轮 80 后剩 120；HumanEval 164 排除 80 后剩 84；MT-Bench 80 已全部暴露；Alpaca 正式选定面板 500 题与前两轮 group 无重合。该核验仅覆盖 R1/R2 清单；执行时还需合并其他已知机制暴露记录。

固定来源：GSM8K 使用 R2清单指向的旧 `final_C1.jsonl`；HumanEval/Alpaca 使用 FORMAL 下各自 `panels/dspark_selected.jsonl`。**Alpaca 不从 `sources/alpaca.jsonl` 的 52,002 行重新抽样。** 保留原第一轮 messages、tokenizer/chat template、source_line 和题面；不是将 Alpaca 当成 MT-Bench 重复试验。

| 用途 | 样本 | 规则 |
|---|---|---|
| 半强度 dev | GSM8K、HumanEval 各取 R2 dev 按 group SHA 排序前 6 题；Alpaca 新取 6 题；共18 | 只用于实现、精度、计数和耗时检查。旧 C/H 只读复用，新增 η=.5 字段；不重做旧主效应。 |
| 半强度 fresh | GSM8K、HumanEval、Alpaca 各16题，共48 | 排除 R1全部192、R2全部84中不同组、所有本轮dev及其他已暴露机制题。三臂与HF全部同组。 |

算法固定：组键沿用 NFKC＋空白合并＋GSM 固定后缀剥离；同时保存完整 messages 哈希，避免把不同 Alpaca 输入合并或遗漏。各域合格 group 先排序，再以 `2026101403 + sum(ord(c) for c in domain)` 的 Python Random 洗牌。Alpaca 先取 6 dev，再取16 fresh；其他域旧dev与剩余新题分开。保存实际实现/版本与抽样清单。锁前按题面身份、context≤4096（含max512）及去重规则排除；不能看模型效果后换题。来源/已知训练精确重合按元数据记录，不宣称排除了语义或预训练重合。

若真实剩余不足，优先缩为三域相同题数并锁前记录原因；不得用已暴露题补足。不把48题称为从未参加过项目评测，它们是**新的机制分析留出**。一个题的所有branch、anchor、seed、arm始终同组。

### 5.3 同状态 HF 与真实 native 分开回答

- **HF 主分析：** 沿用已核 HF 参考的 target-branch 状态生成，最多 anchors 1/65、每 anchor K=8、15 proposal，T1。对每一个完全相同 C/H/target p 计算 raw/half/full；方向和 dose 的输入不随臂变化。i=2…15 为主，首位置单报；EOS/缺 anchor 原样保留，不补题。每状态先配对，再按 branch→position→anchor→prompt 汇总，最后三域等权。
- **native 后果：** 每题三臂真实生成，默认 C1/T1、15 proposal、max512、context4096、eager/同步，明确 temperature=1、top_p=1、top_k关闭及无额外惩罚，EOS集合从各实现读取并报告。三臂统一 §4.3 已冻结 RNG 设置、请求 seed 和中性采样参数。每臂使用自己的真实轨迹，同 seed 不是同轨迹。
- **新增局部 native 诊断保持小规模：** 在实际 reached、独立路径 G 有效的状态上，可同 H 重放三臂分布，必须注明来自哪个生成臂；含首次拒绝位置，不再乘一次实际接受指示。不得把两个生成臂各自的 visited 平均 TV 当作同状态干预效果。无需另建新探针或拟合解释模型。
- 不把 HF 与 native 混成一个平均效果，不用 teacher-prefix overlap 乘积冒充真实连续存活。本轮 native 计数是每轮后果，不是 TPS、任务正确率或总生成 token 增长。

### 5.4 锁定的主量、辅助量和判据

令同状态 `Dη=TV(p,qη)`，`Δη=D0−Dη`。**HF 主效应为 `D1−D0.5`，正值表示半强度更接近 target。** 同时报 `Δ1`、`Δ0.5`，确保“靠近 raw”没有被误写为整体修复更好。

**native 关键后果为半强度减完整强度的每请求比例差：** `已发 accepted proposal / 实际 proposal 验证轮数`。先每请求求比例再配对，不把全局总数之比代替它；分母零记缺失并报告。`实际输出 / 全部调度轮数`为辅助量；完整头和半强度分别相对 joint raw 的差也需完整展示。

辅助表固定展示两类取舍，并按三域及位置报告：

1. 原本 full 比 raw 更差的状态中，half 相比 full 改善多少，其中多少恢复至不差于 raw。
2. 原本 full 有益的状态中，half 丢失多少收益、多少变成有害；另报三臂 TV、负状态率、效果分布及全部prompt差值图。

分类采用 dev 精度核验前即指定的 TV 数值带 `±1e−5`（是数值带，不是实质阈值）；若实测精度达不到该带，先修测量，或锁前放弃该分类而保留连续量，不能在 fresh 看结果后调带。**负状态率下降本身不证明方法更好：η=0 就没有任何修复损伤，也没有修复收益。** 按 full 好坏分组的表是预定诊断，不作为筛选后的新总体确认。

统计：固定 2,000 次按域分层的 prompt 配对 bootstrap，seed=2026101404；HF所有状态随prompt整体重采样，native三臂随prompt配对。报告效应量、95%区间、逐域方向和有效样本数；不把token/验证轮当独立样本。若宣称“平均修复与真实接受均改善”，对两个主效应用各97.5%区间且同时下界>0的保守规则。

本轮没有可辩护的最小实质增益门槛，因此只按以上规则判断方向，并报告绝对大小与区间；不得将任意小的正数包装成“实质提升”。区间跨0记不确定，区间全负记当前干预变差，不把“不显著”一律写成无效或等效。单seed的prompt区间不覆盖训练随机性或完整随机流重复的不确定性。样本小，不承诺检出所有真实变化，不按结果扩样。

## 6. 锁定、执行和失败分支

执行顺序：`旧证据核验 → 新run/账本 → CPU/源码合同 → A限定诊断 → B dev → RNG与B协议锁定 → B fresh一次采集/主分析 → 报告/释放资源`。A与B各有自己的预定清单/源码锁，不能把A看到的效果用于选择更好看的B设置；B只按机械准入规则选择实现。

RNG和半强度记录独立的机械 gate、数值 gate、概率解释状态及科学判定，禁止一个笼统 `passed` 抹平全部。模型路径、SHA/stat身份、dose插入点、RNG角色表、输入模板、采集/统计源码、阈值、seed、样本、测度、缺失规则、预算全部写入新 `protocol/LOCK.json`，锁后才开启 fresh 效果。

锁后发现真实实现错误：停止受影响支路，保留原数据与缺陷证据，记录是否已看见效应。纯采集失败只恢复未完成且协议同一的请求；已完成合格请求不重复。已经暴露的效果不能在改算术、改假设后再叫未见确认；若无法在范围内保持确认性，转为明确标注的探索性报告或结束，不自动另抽一批。

如果 A 耗尽预算/不支持 RNG 归因，仍推进 B 的可靠部分；如果 B 半强度不如 full，按预定规则完成报告并停止，不搜索η=.25/.75，不训练新头，不把坏状态率好看作为推翻主量的理由。

## 7. 结果怎样改变论文论点

| 观察 | 允许推进的主张 | 必须保留的边界 |
|---|---|---|
| half 的同状态平均 TV 与真实每轮接受都优于 full | 当前冻结头在固定0.5收缩下可改善；修正强度是值得继续研究的因素 | 未得最优强度、完整成败理论或跨方法普遍规律；不直接覆盖旧正式主表 |
| half 缓解部分原坏状态，却损害更多好状态/平均不增 | 存在局部可缓解的失败，但统一收缩有收益代价 | 不能推荐统一半强度，更不能据此声称自适应gate已可行 |
| HF 改善、native 后果不确定或反向 | 同状态距离改善未获得相应真实后果支持 | 不宣称更长连续接受；路径、位置权重等只是竞争解释，不能挑一个当结论 |
| native 后果改善、HF 主量不确定或反向 | 当前干预有实际接受收益，但预定局部TV解释未获支持 | 不把接受正结果补写成局部匹配机制已确认 |
| half 没有改善，或 full 明确更好 | 本轮不支持固定0.5干预，保留完整头 | 不证明所有强度问题不存在；本轮也不继续搜索 |
| split 仅改善机械合同，概率残差/归因仍不清楚 | 测量更可审计，相关概率边界仍在 | 工程修复不算论文理论贡献；小样本相容不等于 lossless 已证 |

**即使两项全正，也还缺跨条件可复现的解释规律和相对已有工作的实质增量；本轮只决定是否值得沿“修正强度与收益代价”继续走。** 若全负，明确建议收缩 mismatch 强理论叙事，以已有方法和系统收益为证据主体；不要为了写出新理论而继续无界试验。

## 8. 必须交付的文件与用户摘要

新 run 至少保存：

```text
protocol/{PROTOCOL,LOCK,SAMPLE_MANIFEST,EXPOSURE_LEDGER,RNG_DECISION}.json
code/  sources/  logs/  data/  analysis/
analysis/{RNG_CONTRACT_AUDIT,RNG_REPORT,DOSE_PARITY_AUDIT,DOSE_REPORT}.json
results/RESULTS_CN.md
results/CLAIM_EVIDENCE.json
results/RUN_AUDIT.json
results/RUN_FILE_MANIFEST.json
COMPLETION.json
NEXT_CHAT.md
```

文件名可按阶段细分，但主索引须能直接定位。审计记录完成/失败/缺失请求、RNG与概率算术身份、所有暴露题、实际预算/磁盘、冻结资产复核、自有进程退出及预约释放；大权重只有历史stat身份时不伪称历史全SHA核验。保留 R2 false 审计和 `logs/FINAL_TECHNICAL_AUDIT_CORRECTION.json` 的原始事实。

`RESULTS_CN.md` 的第一页用通俗中文回答：

1. 现在新增证明/支持了什么？用一句话加效应量说明。
2. 没证明什么？随机源、半强度和一般理论分别判定。
3. 减小修正救回了哪些状态，又丢掉什么？平均取舍怎样？
4. 真实接受有没有相应变化？哪些量不能叫TPS或质量？
5. 论文主张应该保留、收缩还是调整？至多提出一个有证据驱动的后续方向，**本轮不追加执行**。

正文附完整的配对结果、区间、真实图表和 claim–evidence。缺失的native或概率证明明确写缺失，允许以合格负结果或有边界的部分完成收束；不以预算尚余或论文创新尚不足为理由自动开第四轮。
