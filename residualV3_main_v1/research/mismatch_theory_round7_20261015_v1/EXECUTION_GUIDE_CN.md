# 第七轮执行指南

2026-10-15 v1。先读 [研究设计](THEORY_PLAN_CN.md)，随后按 [启动prompt](START_PROMPT_CN.md) 的授权自主执行。本文编写chat只准备文档和CPU合同。

## 1. 范围与路径

本轮允许：新run内实现固定概率诊断、只读复用R6数据、必要的缺字段冻结重放、新36题采集、固定统计与案例、审计和收尾。主体、原头、target、basis及安装环境冻结。**没有预测器拟合、头训练、新LP/见证优化、native/RNG/TPS或剂量搜索。** 不因旧负结果、覆盖不足或预算余额扩支路。

仓库 `/data/weijiahong/speculators-main-lora`，输出只写：

```text
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/<UTC>_mismatch_r7/
```

旧run、归档、正式源码/权重/协议、自动生成报告均只读。新缓存也放新run，import前设 `PYTHONDONTWRITEBYTECODE=1`。复制必要旧代码并修改硬编码ROOT/入口，禁止直接启动旧run脚本。R6“不自动第七轮”保持旧终态；用户交付本prompt后，新的有限范围生效。

## 2. 必读与代码复用

精确路径见 [sources/READING_GUIDE_CN.md](sources/READING_GUIDE_CN.md)。先读R6完整报告、固定案例、理论卡片、COMPLETION/NEXT_CHAT与概率身份；再读 `read_capture.py`、`mismatch_math.py`、`collect_mechanism.py`、`prepare_inputs.py`、`novel_stability.py`。

R6已有48题、92个完成anchor，默认CPU流式复用。缺字段才固定原C/H重放，记录理由，不能整批重采或把旧题重新称未见数据。R6原有预测器/LP代码不在运行范围。

关键复用边界：

- `common_direction()` 的u/o可复核使用，旧 `action()` 是M_common/2探针，不能拿来计算本轮c。
- qh由实际保存的raw BF16 logits复制后，在candidate IDs上替换 `HEAD_ARRAYS.selected_updated_logits_bits[group_id]`，再做全词表FP64 softmax；不用basis/dose的CPU重投影代替。
- raw/hidden按 `position−1`，prepared张量按 `prepared_arrays_position_indices` 映射，注意其实际零基索引；不能把过滤后的排列当位置。
- `GROUPS.json` 与INPUT/HEAD_BRANCH资产核验完整Z、raw和qh一致。不能仅比前驱token或压缩特征hash。
- `TARGET_BITS.npy` 的uint16是BF16位；行号必须与row_id一致，不能数值强转成logits。
- 旧 `load_context()`/`position_data()` 会接触全部target和效果字段；新题动作构造阶段不得直接使用这些全量入口。

## 3. 先做什么，再做什么

**A：旧数据开发。** R6原fresh48现在全部已暴露，用于固定新诊断的工程和描述。核输入SHA、重构概率、数学合同、动作大小与匹配novel覆盖。允许计算新量，不重跑R6 HN/HP/HC、预测器或oracle主分析。开发结果不能用于换公式、阈值、K、域、位置或对照。

**B：锁定新轮。** 写 `PROTOCOL/LOCK/DIAGNOSTIC_RULES/STATISTICS_RULES/RESOURCE_POLICY`，保存代码、源资产、样本、种子、数值阈值和输入访问规则。确认数学合同及工程测量正确后才采新题。旧数据效果负不构成停止新确认或调整方案的条件；确有不可测/资源阻碍可partial，但要明确技术原因。

**C：新36题一次采集。** 固定三域各12题、K64，遵守下一节。采集器只按锁定流程保存数据与身份，关闭给动作开发提供fresh效应反馈的摘要。中断时核实已完成分片SHA，续缺失分片，不重跑已完成题或更换seed；中断时间计预算。

**D：fit动作先封存，eval后评估。** 独立导出器把fit/eval target分开，生成无效果字段元数据；动作构造入口只读fit bank及q0/qh，保存全部合格fit组动作后签名，再允许评估入口读取eval bank。见第5节。

**E：报告与收尾。** 固定两主量、失败描述、有限真实案例和图表；给出支持/未支持/覆盖不足及边界。完成审计后退出自有作业、释放预约，不自动R8。

## 4. 新题、历史与科学人群

新题：GSM8K/HumanEval/Alpaca各12，共36。沿R6固定面板及DeepSpec提交，NFKC/空白/GSM后缀规范化、完整messages/token核对、长度资格均继承。机制未暴露不等于未参加正式评测，未知训练/预训练语义重合继续披露。

准备时只读当前机制manifest与panel排除后余量为GSM60、HumanEval24、Alpaca434；执行时合并最新全部 `mismatch*/runs/*/protocol/SAMPLE_MANIFEST.json`、专家/headonly及其他实际panel重新算，不能将R6选前76/40/450直接继承。完整题面组去重，所有anchor/历史随题面归属。

按合格group SHA排序，再用 `Random(2026101507 + sum(ord(c) for c in domain))` 洗牌取前12。固定每域12，不沿旧脚本自动改为16或按效果补题。若执行前资格不足12，先完成可做的旧数据部分，报告新确认受阻；不静默换域、缩规模或用已暴露题补齐。

固定anchor1/65、15 proposal、自然target T1全词表采样、原EOS、context4096与R6原HF anchor/生成预算；不截prompt。K64=fit16/eval48，主位置3/8/15，控制1/2。中间历史连续生成，仅保存所需位置的target及头状态。缺anchor、提前EOS、未匹配Z保留，不能换位置或强制碰撞。

master seed `2026101526`，bootstrap `2026101527`；trajectory/fit/eval各独立命名空间，沿R6 SHA256稳定生成规则，保存显式整数表。不得Python内置hash，不新增剂量、K或seed网格。

主集合：**同一C/i/精确Z的fit子组内部**至少2个不同完整H，不能跨Z计数；eval匹配该组Z；完整eval H未出现在同C/i全部fit历史中（含其他Z组）。所有自然重复采样槽保留频数；复制的同一观测去重。独立seed不是novel。没有匹配为无定义，不填0。m=0保留原头与零主差，不筛走“没用”状态。

seen、所有匹配、所有eval主状态覆盖分别报。只有匹配组可定义本诊断；不得将全部eval的原头平均收益与主子集结果相除。i1/2仅用于身份控制，不强行要求i2跨前驱变化为零；同Z的i2不得有不同更早H。

## 5. 新题fit/eval的访问隔离

采集输出可以包含完整target。独立导出器按pool和row_id拆分 `FIT_TARGET_BITS`、`EVAL_TARGET_BITS`，并输出元数据白名单：C/anchor/i、row_id、group_id、pool、H、前驱、来源身份和数值重构所需索引。去掉 `D_raw64/D_head64/gain64`、所有效果统计及eval target值。

动作入口仅接受fit bank、白名单元数据、实际q0/qh及完整Z身份。评估H可用于预定novel/matching标记，但不能选择方向或幅度。对所有合格fit组构造动作，无论最终是否有eval匹配，不按eval表现删组。

每组记录：fit row IDs、不同H数、q0/qh身份、u/o构造来源、A/B、m_raw/m_effective、t、ρ、c与qc/qr/qu的数组SHA。可只存c及可复算来源减少磁盘，但五分布的哈希与复算误差必须可核验。写 `ACTION_BANK_SEALED.json` 后，独立评估入口才读取eval target并计算η/κ。

全文件SHA读取原始字节用于完整性校验可以由独立审计入口完成；不能把这种读取写成“没有任何进程读过eval”。准确声明是**eval数值与效果未进入动作构造或选择**。访问日志、入口源码、实际路径白名单共同核验，不声称同UID目录具有不可绕过的操作系统权限隔离。

旧R6数据已暴露，只需复现同样数据流，不伪称旧数据盲测。新题封存后发现代码错误，保留原产物和暴露记录；受影响重分析标探索，不重新包装未见确认。

## 6. 动作与数值合同

严格使用理论计划公式，不优化c、不新增方向。u/o定义跨全部fit分支，重复H保持自然记录但不改变最小值；只在资格中按不同H计数。

`m_raw=min(sum a,sum b)`；若≤1e−12，effective m=0、c=0；否则m=m_raw。原 `M_common=min(sum u,sum o)` 可描述但不能用于ρ。若m=0直接ρ=0，不做0/0；非零时t必须大于0且m≤t。

BF16位按原实际GPU资产读取，稳定FP64 softmax重构完整词表，不截尾/合桶/额外epsilon。五分布均核非负、和为1；保存数值残差。默认绝对残差容差1e−10，包括：

```text
sum c = sum r = 0
c与r逐坐标同向；|c|≤|d|，|r|≤|d|
TV(qh,qr) = TV(qh,qu) = m
TV(q0,qc) = m
TV(q0,qr) = TV(q0,qu) = t−m
fit每支的Gc = m
Gh − Gc − Gr ≤ 0
η = Gc + (Gh−Gc−Gr)
```

不得clip、重归一化、扩大容差掩盖失败。原q0/qh与旧保存标量重构差默认≤1e−10，任何失败保存分片并定位；不要换CPU头实现。损伤/正收益符号带固定1e−5，连续均值使用未阈值化值。qr/qu非零差覆盖用 `TV(qr,qu)>1e−12`，不据效果调阈值。

数学合同含两主量四种符号组合、c0但仍能共同修、单独有益却完整边际有害等反例。通过仅说明实现和逻辑一致，不是模型证据。

## 7. 统计和失败分析

两主量使用完全相同的novel主集合：

```text
H_remove   = mean[TV(p,qr)−TV(p,qh)]
H_specific = mean[TV(p,qr)−TV(p,qu)]
```

每C/i内自然频数平均，题内有定义位置等权、有效anchor等权，域内题面等权，三域等权。从每域该主集合有定义的题面分层做2000次整题bootstrap，保留题内所有配对。主区间双侧97.5%（分位0.0125、0.9875，两项Bonferroni），另报95%描述。CI条件于本次固定分支与规则，不覆盖全部采样/模型不确定性。

覆盖门槛：每域至少6题有定义；H_remove另需每域至少6题含非零m；H_specific另需每域至少6题含 `TV(qr,qu)>1e−12`。门槛只影响支持判定，不按非零组重选点估计或bootstrap人群。缺域为主量未定义，不偷偷改两域均值。主CI下界>0且本项覆盖满足才称支持；两项独立判定，不互相补证。

必报五分布TV、Gc/Gr/Gh、η/κ、m与t、m/t（t=0为NA）、零动作率、novel/seen/未匹配覆盖、分域/位置及受损比例。m/t只是移动质量比例，不是收益解释率。I只报恒等式残差与描述，不做主假说、不把其非正包装成新发现。

失败四类严格按理论计划，主频率在novel中计算；seen/全部匹配另表。每类同时给占全部主状态与占原头受损状态的加权比例，保留零分母NA；按上述自然层级权重计算加权事件质量的比值，不平均“每题比例”去掉无坏状态题。分类反映固定分解下状态现象，不称唯一原因。

实际token账本用 `0.5*(|p−q0|−|p−q|)`，五分布完整词表求和与总TV对齐；正、负贡献分别保存。朝target、反向、越界但仍改善、越界致损按实际概率判定，不能把所有越界都当受损。

案例最多8个主集合状态：每个失败类按domain/prompt SHA/anchor/i/Z/H字典序取前2个不同完整H状态，不按损伤大小挑；同一自然H重复槽只展示一次、总体频数仍保留。无类别就写未观察到，不跨类凑满。每例附同C/Z另一个按字典序选出的不同H作背景对照；没有则写缺失，不额外采集。背景对照不增加总体样本数。每例最多展示正负各8个token，并链接完整账本。

不做收益回归、Shapley百分比、随机删坐标网格或新oracle来救结论。两主量不支持也是本轮有效答案。

## 8. 共卡、预算和恢复

**允许共卡，显存能容纳工作集就用，不等空卡、不要求utilization=0、不固定卡号。** 实时查看全部可用GPU UUID、free memory及预约；已有独占预约只跳过该卡。禁止旧空卡gate、终止他人进程或重启原controller/parallel/收尾。

R6采集观测自有峰值23,512 MiB，仅供本阶段估计；实际准入按阶段预计峰值加4 GiB余量，自有峰值默认≤32 GiB，同时1卡。优先流式C1、小微批。没有卡能容纳最低工作集时推进CPU工作并短周期复查；不擅自改变科学样本、K或位置。有满足条件的卡直接执行，无需再问共卡许可。

全轮所有run合计上限：8 GPU小时、24 CPU小时、700 GiB新增磁盘。失败、编译、重试、等待及共卡变慢均计GPU进程存活墙钟；CPU进程树user+system累计，临时文件/副本/失败分片计磁盘，不换run重置。预算是上限，不是必须用满。

CPU worker≤4、所有自有计算线程合计≤8。import前OMP/MKL/OPENBLAS/NUMEXPR均1，Torch intra/inter-op各1；驱动/IO线程单列监测，不能隐藏数值线程超限。R4/R5旧资源披露和R6非全时刻线程生命周期证明继续保留。

记录UID/PID/PGID/startticks/CMD、run路径、GPU UUID及共享预约。只管理核实身份的自有作业。阶段最多2次有新定位的技术修复重试，无新诊断不反复启动；科学负结果不是重试理由。到限保存并退出，独立完成结果保留。

## 9. 必交结果

```text
protocol/{PROTOCOL,LOCK,DIAGNOSTIC_RULES,SAMPLE_MANIFEST,EXPOSURE_LEDGER,
          SEED_TABLE,NUMERIC_CONTRACT,STATISTICS_RULES,RESOURCE_POLICY}.json
analysis/{IDENTITY_AUDIT,INPUT_ACCESS_AUDIT,ACTION_BANK_SEALED,
          HEAD_COMMON_COMPONENTS,REMOVAL_COMPARISON,COVERAGE,
          FAILURE_ACCOUNTING,NUMERIC_AUDIT}.json
results/{RESULTS_CN.md,METHODS_CN.md,MECHANISM_CASES_CN.md,
         CLAIM_EVIDENCE.json,RUN_AUDIT.json,RUN_FILE_MANIFEST.json}
COMPLETION.json
NEXT_CHAT.md
```

第一页简单回答：①原头实际概率移动中是否有具正边际价值的共同部分；②是否超过同质量均匀移除对照；③共同部分何时也救不了整体失败；④哪些仍不知道。旧开发与新确认分表，不把所有状态当独立题数。

图最多四类：覆盖及m/t；η/κ配对区间；五分布损失与失败分类；真实同Z概率移动和token补偿。I非正和fit Gc=m放数学附录，不当结果亮点。最后给一段当前可写论文的有限论点及尚未闭合的证据，不自动提出必须开启下一轮。

科学负结果可complete；必须采集/主分析没完成则partial并列剩余项。覆盖不足可完成但不支持；没有某类案例可not_observed，不补采。科学状态、测量身份与资源状态分别报告。

收尾核自有进程退出、预约释放、预算、静态SHA与旧资产身份。R2原native概率警报passed=false及历轮负结果/资源/证书披露继续保留。不据本轮HF分布诊断消除旧警报，不自动第八轮。
