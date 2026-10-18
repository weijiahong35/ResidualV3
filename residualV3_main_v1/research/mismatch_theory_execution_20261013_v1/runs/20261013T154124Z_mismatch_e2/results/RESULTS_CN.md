# ResidualV3 mismatch：本轮结果与反证边界

Run `20261013T154124Z_mismatch_e2`，收束 UTC 2026-10-14T02:25:39.254519+00:00。**本轮限定执行已收束：A/B完成；native C按三次尝试／两次技术恢复的退出条件归档为不确定。** 本报告由AI协助源码、原文与统计审计；没有训练新头或主体，没有发布新插桩TPS。

## 主要发现

**已支持，限声明的HF参考测度：** 冻结E2头确实利用前驱与完整目标前缀的对应信息，并改善同主体同状态的分布。120题留出主效应TV(置换)−TV(正确)为 **0.179090 [0.164850, 0.193114]**，必须同时报告的TV(联合raw)−TV(正确)为 **0.250882 [0.234390, 0.267329]**；两区间下端均超过锁定阈值0.005，首位置置换效应严格为0。三个域均为正，保留无可区分前驱组。[唯一固定主分析](../analysis/EXPERIMENT_REPORT_test.json)

**削弱“条件mismatch必须主导”的预设；总体占比仍不确定：** K8经验条件KL为0.723250 [0.665094, 0.782219] nats，联合raw平均拟合项为2.472762 [2.229292, 2.742054] nats。当前有限K与测度下拟合项点估计更大；dev K4/8/16条件项仍上升，不能把经验占比称为总体精确分解。拟合项混有模型类、优化、目标、数据和校准，不能叫“参数容量鸿沟”。

**已支持有限竞争解释；并非所有状态都优于简单校准：** 联合raw冻结单温度仅取得0.014296 [0.011207, 0.017588]的TV改善，正确头仍比它好0.236586 [0.219424, 0.253574]。这排除了“该单温度参照足以解释全部观测”，没有排除所有通用再拟合。正确头对联合raw改善／恶化的状态比例，按prompt层次聚合，分别为 **87.1555%／12.6870%**（其余为相等，不是prompt变差比例）；低raw-TV的预定子组中校准优于头。

**不确定：** 这些局部分布收益能否在native实际到达前缀上转为连续接受／实际发出收益。本轮成功native请求与verifier记录均为0；实现失败不构成H_C反证。[C退出审计](../analysis/NATIVE_BRANCH_EXIT_CN.md)

## 模型、状态与覆盖

目标为冻结Qwen3-8B；B0为E2独立DFlash主体；B1为E2联合主体关头；B2/B3共用同一联合主体、同hidden/raw/basis及冻结depth3、use_soft=false头，只有前驱对应关系不同。两种basis分别由target embedding与LM head构造，不能互换。Top256只限定logit残差，采样与距离仍为151936完整词表。[资产身份](../sources/ASSET_AUDIT.json)、[层序/anchor语义](../analysis/LAYER_ALIGNMENT_SOURCE_AUDIT.json)

三域各64个唯一prompt，dev24/test40，pilot8包含于dev；总192题。pilot24与主dev72复用标签、已完成shard不重跑；正式test120题均完成并仅做一次指定主分析。test是**本次分析留出**，不是项目从未使用的数据。训练group元数据精确规范化匹配为0，不能排除语义近重复或预训练重合。[样本清单](../protocol/SAMPLE_MANIFEST.json)、[训练匹配审计](../analysis/TRAINING_OVERLAP_AUDIT.json)

本轮没有另采历史teacher-cache前缀作配对测度，native all-proposal／reached-prefix没有有效新记录；不声称已测得训练—部署gap。

A/B测度是 `target-branch/alive-at-i`：固定target轨迹C、anchor1/65，K8 target分支H；raw q在同C/i不随H变。T1/top_p1/top_k关闭，模型BF16、归一化及距离FP32；target轨迹只生成到65（不超过原256上限），不替换终止anchor。test有235/240个可用C、3500个C/i、27995条分支观测；5个anchor65因target终止缺失。5个C/i为n7，其余3495为n8；n1为0。1134/3500个C/i无可区分前驱，仍进入主结果。EOS预测状态纳入，EOS之后不补分支；test轨迹有5次EOS、分支有33次EOS，额外151643没有被采到。

统计为branch→位置→C→prompt，主效应位置2..15、域均权；2000次prompt配对bootstrap，seed781013。区间不把token/分支当独立重复，也不消除有限K偏差；分层是探索性，未扩搜显著。[锁定协议与源码](../protocol/LOCK.json)

## B：留出干预与校准

| 臂（同测度，i2..15） | 平均完整词表TV |
|---|---:|
| B0 独立主体 raw | 0.669346 |
| B1 联合主体 raw | 0.663866 |
| B2 正确前驱修复 | 0.412984 |
| B3 分支内置换前驱修复 | 0.592074 |
| B4 联合 raw 温度校准 | 0.649571 |
| B4 独立 raw 温度校准 | 0.653553 |

| 配对效应 | 点估计 [95%CI] |
|---|---|
| 置换−正确（主要指标） | 0.179090 [0.164850, 0.193114] |
| 联合raw−正确（必要同时满足） | 0.250882 [0.234390, 0.267329] |
| 独立raw−正确（主体不同参照） | 0.256362 [0.237381, 0.274110] |
| 联合校准−正确 | 0.236586 [0.219424, 0.253574] |
| 联合raw−联合校准 | 0.014296 [0.011207, 0.017588] |

校准只用完整dev72，在固定logT∈[-2,2]、步长0.1的grid上拟合各raw一个标量：joint T=0.4965853、independent T=0.4065697；test没有参与拟合。dev分支B4未缓存，未伪造其配对区间。[冻结校准](../protocol/CALIBRATION.json)

| 域（test各40题） | 置换−正确TV [95%CI] | 联合raw−正确TV [95%CI] |
|---|---|---|
| gsm8k | 0.255631 [0.229423, 0.281057] | 0.315068 [0.282825, 0.346808] |
| humaneval | 0.169556 [0.150101, 0.190398] | 0.241827 [0.215814, 0.267945] |
| mt-bench | 0.112082 [0.085828, 0.138805] | 0.195750 [0.164177, 0.225172] |

![留出同状态TV](../analysis/figures_test/B_TV_positions.png)

![留出配对效应](../analysis/figures_test/B_paired_effects.png)

首位置的置换−正确为0 [0,0]；联合raw−正确为0.002489，区间[-0.001622,0.006252]，不把首位置强说成有明显修复。后部位置人数有所下降（i15可用117题），位置变化不全是信息丢失。

**实际失效／竞争边界（预定探索性分层）：**

- raw TV∈[0,.25)的6638个状态、109题中，正确头较联合raw仍有小改善0.017163 [0.006728,0.026882]，但“校准−正确”=-0.022882 [-0.034481,-0.012635]：校准更好。不能宣传每个兼容状态都需要头。
- target在raw Top256内质量<.5的1438个状态、95题，raw−正确=0.030854 [0.022451,0.039928]；质量≥.99的22943个状态、120题为0.271039 [0.253646,0.288019]。这是人群关联，不是支持约束的因果百分比；低质量组仍有改善，不能说Top256之外完全不可修复。
- target熵≥4 nats只有1状态／1题且缺域，结论不确定。这些target TV/熵/质量为oracle离线分层，不是部署可见的gate；没有拟合预测探针。

任务书另要求的KL/overlap做了一次描述性补充，使用同一已锁层次估计器：联合raw KL=3.408841、正确=1.821528、置换=4.910945 nats（i2..15）。**置换的TV小于raw，KL却大于raw，损失排序不能互换。** 正确overlap=0.587016，与1−TV的数值闭合相符。补充报告细节在collection已开始、formal效应未打开前登记，不能冒称锁前单独预注册；不改主分析、不调参、不挑案例，B4 KL未采集。[补充及区间](../analysis/SECONDARY_B_KL_TEST.json)、[声明时间](../protocol/SECONDARY_KL_REPORT_SPEC.json)

## A：条件变化与剩余平均拟合

A按i1..15聚合，和B主效应的i2..15口径不同。

| raw主体 | 条件KL nats [95%CI] | 平均拟合KL nats [95%CI] | 总KL nats [95%CI] |
|---|---|---|---|
| 独立 | 0.723250 [0.665094, 0.782219] | 2.557609 [2.321828, 2.802767] | 3.280859 [3.023127, 3.540163] |
| 联合 | 0.723250 [0.665094, 0.782219] | 2.472762 [2.229292, 2.742054] | 3.196011 [2.937594, 3.478930] |

两臂条件项相同是共用target p与分支的必然结果，不是联合训练的新发现。最大KL闭合误差3.815e-6，低于5e-5容差。半pairwise-TV U-stat为0.220553 [0.204160, 0.236933]：它诊断同可见信息下的不可区分限制；点估计不是总体严格风险下界或参数容量因果下限。

dev预选12题共同360个C/i中，条件KL K4/8/16为0.572204/0.689352/0.806584；联合拟合为2.310592/2.238946/2.202253。K16仍未饱和，不估“总体有多少失配是条件缺口”，不强加位置单调。原值、n1和成对缺失规则均保留。[dev固定报告](../analysis/EXPERIMENT_REPORT_dev.json)

![留出经验分解](../analysis/figures_test/A_KL_positions.png)

![dev限定K敏感性](../analysis/figures_test/A_dev_K_sensitivity.png)

## C、原批次及计数边界

原批次已最终完成48有效窗／24配对，FINAL_AUDIT与CPU COMPLETION_AUDIT通过，链接及12份formal summary SHA独立复核；没有补回归档attempt。全配方AC增幅6.0137%–18.4273%、TPS增幅0.5930%–24.5424%；主体不同，不能分摊训练／推理贡献。历史E1同主体开头AC增益13.73%–18.78%已存在，不能说以前从未做开关头；不是本轮E2 TPS。[原最终结果复核](../analysis/EXISTING_RESULT_AUDIT_FINAL.json)、[历史负结果与开关头](../analysis/HISTORICAL_EVIDENCE_CN.md)

C三次尝试依次失败于IPC路径、未armed的warmup、错误拒绝正常CPU占位token。第三次只有普通anchor输出[10061]，没有proposal验证；不当作C数据。原生预算／EOS来自scheduler，但真实draft token驻GPU，CPU[-1]仅表示长度，当前新插桩未完成二者联结。技术恢复额度耗尽，本轮C结束为**不确定**，不按stage/arm重置。[完整退出与宿主清理](../analysis/NATIVE_BRANCH_EXIT.json)

HF参考EOS=[151645]；原生实际首轮EOS=[151643,151645]，因为安装版`--generation-config vllm`保留generation config的特殊token。cached/full-prefix在48个dev状态的最大TV差0.060815，fresh cached重复均bitwise相同、top1相同；此为不同计算模式观察，不是head replay误差，也不是HF-vLLM总体上界。A/B结论不自动迁移native。[EOS审计](../analysis/EOS_RUNTIME_IDENTITY_CN.md)、[cache诊断](../analysis/DEV_TARGET_CACHE_PROBE.json)

T1一步接受=1−TV；连续可发出proposal存活要在实际Q上用G_k∏b_i，b_i=min(1,p_i(x_i)/q_i(x_i))。G只包含proposal路径EOS、块长与轮开始预算，不包含真实接受随机事件；actual reached另列。proposal接受、correction、bonus与实际发出分别记。普通teacher/draft路径的∏overlap不能代替该期望；T0用贪心一致性。本轮未得到native存活／发出曲线，不伪造数据或插桩TPS。[理论卡片与CPU反例](../analysis/THEORY_CARDS_CN.md)

## Q1–Q7与三层主张

| 问题 | 本轮回答／状态 | 不能推出什么 |
|---|---|---|
| Q1 比较什么？ | **已支持**：同C/H/i上的full-vocab p与各臂q，明确T1、信息集与HF target-branch测度。 | 拒绝不是语义错误；teacher-prefix／native all／reached测度不能互换。 |
| Q2 条件或平均拟合？ | **已支持经验分解；削弱必须条件主导；总体占比不确定**。有限K下拟合项更大，K敏感性未消失。 | 拟合项不等于不可约容量，两个分量不是TV/AC增益的可加贡献。 |
| Q3 头利用什么？ | **已支持（本参考测度）**：留出正确对应信息同时胜置换、同主体raw和冻结单温度；负状态、低raw-TV及tail边界保留。 | 不称恢复完整H、不归因总体增益百分比、不排除所有通用再拟合。 |
| Q4 哪些归于推理修复？ | **已支持局部同主体头效应；整体AC/TPS分摊不确定**。 | 不把独立与联合主体、各自native轨迹的结果机械相加。 |
| Q5 怎样到连续前缀？ | **已支持公式及计数适用边界；native迁移不确定**，C技术退出。 | 不拿target-branch overlap乘积当真实AC；不删除终止轮套固定长度式。 |
| Q6 能迁移留出吗？ | **已支持本次120题可用状态**，各域40题，CI按prompt；未做预测探针。 | 不称从未被项目／预训练见过，不承诺新域、长native路径或可部署收益预测。 |
| Q7 比近邻多知道什么？ | **已支持本项目经验结构与反证边界；方法／理论新颖性仍不确定**。 | 标准KL/TV、前缀目标、前驱修正及三路结构不能自动记为创新；跨论文数值不排名。 |

**文献已知：** 最大耦合／TV接受、条件依赖、蒸馏保真与容量／优化竞争解释、前缀接受目标及前驱修正已存在。**项目假设：** H_A条件结构、H_B对应信息、H_C native迁移、H_G预测可泛化各有独立出口。**实际结论：** H_B及局部留出迁移支持；H_A条件存在有证据但主导／精确占比未建立；H_C不确定；H_G未执行不作主张。

一手矩阵指定21项＋1竞争来源，共22项，20份完整原文保存；按记录章节阅读，L05/L06只有一手browser节选。核查DistillSpec、容量／保真、前缀目标、Domino、DSpark、DFlash-2、DFly、xPress、DSpine及版本／范围，不用跨论文数字证明本项目优于它们。[文献矩阵](../sources/LITERATURE_MATRIX.md)、[来源审计](../sources/SOURCE_AUDIT.json)

## 成本、交付和下一轮

本轮GPU累计 **0.751989小时**，包含pilot、cache检查、三次C失败、dev和test；全部限GPU6 UUID `GPU-16d48373-c956-39a9-3989-4676fb6aa22c`，没有新训练，只有dev两个温度标量。源码／权重／完整断点／原协议／原自动结果未覆盖。CPU数学与接口验收不冒充E2证据。[运行审计](RUN_AUDIT.json)、[claim–evidence](CLAIM_EVIDENCE.json)、[最终宿主审计](../analysis/FINAL_RUNTIME_AUDIT.json)、[解释复核](../analysis/FINAL_INTERPRETATION_REVIEW_CN.md)

仅建议**一项**后续工作：在新授权轮次中修正native测量桥接，scheduler提供request/预算/长度，GPU worker提供实际token/p/q；先每域2个dev prompt、B0/B1/B2，max512、C1T1、单卡，CPU约2–4小时、GPU硬上限2小时。只有actual q/token对齐、普通／零输出prefill及EOS/bonus/修正／HTTP发出全部闭合，才研究迁移；任何身份或计数不一致就停止、不作机制结论。若native reached中局部效果消失，或存活／真实发出不改善，则削弱H_C。**本轮不自行恢复C，不训练新结构。**低raw-TV分层是oracle标签，不能直接拿来部署gate。

恢复入口、不可重跑清单、预算和明确缺口见[NEXT_CHAT](../NEXT_CHAT.md)。可信的结果是失配结构与有限适用范围，不是把mismatch故事包装成普遍成立。
