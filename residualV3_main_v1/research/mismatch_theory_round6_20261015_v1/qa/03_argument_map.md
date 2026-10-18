# 论证架构：基础审查后定稿

主链：新历史是否破坏共同方向 → 不可见历史差异能否被离线预测器利用 → 方向翻转后是否仍有兼顾空间。原头保持冻结；预测的是原头实际Δ=TV(p,q0)−TV(p,qh)，不是R5探针结果。

T1：新48题(三域各16)，K64=fit16/eval48，anchors1/65，主位置3/8/15、负控1/2。沿R5固定M/2探针，eval的完整块内H若与同C/i任何fit H完全相同则标seen，否则novel；不同于跨题测试。主量仅novel且真实fit≥2不同H、eval匹配Z子集。自然频数主权重，去重H等权仅描述；不补采到新H出现。

T2：旧R4新48题为训练，R5新36题为开发；R4旧18仅工程，不混入正式train/dev。清除跨源复制的同一观测，保留自然重复的不同采样槽。新48题所有eval状态用于真正跨题测试，不用新题fit标签训练、标定、调阈值。新C fit标签只能给T1诊断。

基线A使用完整raw hidden4096、previous embedding4096、实际qp256、fusion_concat768及6个共同标量，预计9222维；B再加更早H序列（去紧邻前驱），静态embedding RMS归一化后固定32维随机投影，13个有序槽+mask，共429维。qp/fusion是已有信息的确定性变换。完整A替代初稿压缩A/A_full四模型方案。

固定width32单隐藏层GELU MLP，可表达Z×H交互。AdamW lr1e−3、batch512、50epoch、单seed；仅三个decay {0,1e−4,1e−2}在旧dev按Brier选取。只有最终epoch，不搜索结构/表示/checkpoint。有限模型失败不等于历史无信息。自然层级权重、不做类别重加权。

同一B置乱对照不重新拟合：冻结B，同C/i/完整Z内将eval更早H对应打乱8次（保持H多重集、组大小和所有Z），先算每次Brier再平均，不先平均概率；同Z核验后可等价置乱B输出。单一H/单状态零差保留，未定义不填0。该对照用于识别历史与结果的匹配价值，不是随机干预因果或置乱显著性检验。

y_bad=1[Δ<−1e−5]，其余含中性为0。主评价weighted Brier，另给损伤prevalence、常数训练先验预测器、PR-AUC、开发集固定10%FPR阈值下的真实test FPR/recall及校准；不以准确率为主。阈值不在test重设；自然类别权重，无欠采样/测试平衡。

T3：同样旧train/dev拟合固定ridge（λ1e−3，LSMR上限1000）预测Δ，作为收益幅度次要量；方向只预测一个v*=argmax_v|qh−q0|（仅用raw/head、ties按ID），标签p(v*)相对q0(v*)的上/下/中性。两臂共同加v*静态embedding及q0(v*)/qh(v*)，不以真实当前采样token代替。只在旧train/dev拟合线性3类预测器，固定上述历史表示，logloss/方向命中率描述；不搜索token、不预测整词表、不把该标签与Δ好坏等同。

T4：自然同C/Z历史对照，优先按无结果的token汉明距离（同长、同前驱、恰好1处更早差异）列案例，不把词面差异等同语义类别。受控的是C/i/Z，不能把多处H差异归给单词。有限新状态oracle最多12对（每域按预定排序前2异号+2非异号）；只复用R4 R_ind/R_shared/R_safe松弛概率操作求解器、固定100次/间隙1e−6，未证保留区间。不新优化actual basis动作，松弛共同动作不能叫实际头可达，找不到不能判不可修。分类选例为描述，不估计人群成因比例。

三项主量：HN novel probe gain；HP Brier(A)−Brier(B)；HC Brier(B-shuffled)−Brier(B)。各98.333%CI预控三项，另95%描述。HN每域≥6题有定义；HP每域至少6题含坏状态且该域同时有两类；HC至少10题含同Z坏/非坏不同标签且每域≥2题，否则增量解释覆盖不足。条件不满足不增加题/改门槛。收益/单token方向/12对oracle均次要，不拿它们补主检验。

本轮先CPU旧资产训练开发和锁定模型/特征源码/scaler/阈值，再新GPU采集。预测worker无TARGET_BITS/标签/fit统计访问，先签predictions再评估。完整题面簇bootstrap 2000，域等权。模型正则、embedding投影seed、三套统计和新增48题seeds在fresh前锁。

基础审查：机制执行8.35/10，预测设计8.56/10；均超过7.5。已落实完整A/B、交互MLP、R4同位置测度、特征/标签隔离、三套人群、T3 token身份、12对全局总额。详细实施以EXECUTION_GUIDE_CN.md为准。
