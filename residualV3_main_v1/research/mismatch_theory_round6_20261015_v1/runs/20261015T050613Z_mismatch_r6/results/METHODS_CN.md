# R6固定测度与解释边界

本轮研究失配的历史稳定性和原头损伤的可预测程度。主体、target、原修复头与basis冻结；小型预测器只输出风险、收益或单token方向，不生成修复分布。

## 人群与概率身份

R4原fresh48题训练、R5原fresh36题开发，R6三域各16条新机制题测试。完整messages、token和继承group SHA隔离，所有anchor与分支随题面分配。新题曾可参加项目正式评测，不声称排除未知训练或预训练语义重合。

每有效anchor1/65自然采样K64，fit16/eval48；原T1、EOS和存活条件保留。中间历史连续生成，只保存位置1/2/3/8/15，主位置3/8/15。自然采样槽中的重复H保留频数，复制的同一观测去重。EOS、缺anchor和无匹配Z不补采、不换题。

读取实际BF16 logits位，稳定FP64 softmax重构完整词表。原头q来自实际GPU保存的selected_updated_logits替换raw Top256，未用CPU重投影。完整Z由实际全部输入张量确认，组内raw与完整qh必须一致。原始标签与重构误差另记身份审计。

## T1：标签参与的共同方向在新历史上是否稳定

沿R5固定共同流u/o/M与M/2。每个同C/i/Z动作仅用fit target，先保存动作再评估eval。M≤1e−12保留零动作，不修改p/q；fit改善是构造保证，不作为新发现。

novel指完整块内H不在同C/i的任何fit历史中。HN只用fit含至少两种不同H且eval匹配完整Z的novel状态。seen、novel、未匹配和零动作分别呈现；不同seed不代表未见H。该全词表概率动作不是原头合法动作或在线方法。

## T2：在读标签前预测原头损伤

标签是实际Δ=TV(p,q0)−TV(p,qh)，坏状态为Δ<−1e−5。A读取完整原头可见输入及固定标量（9222维）；B加排除紧邻前驱的最多13个有序历史token，每槽固定32维Rademacher静态embedding投影与mask（额外429维）。完整列定义见FEATURE_SCHEMA；ID只供join/分组。

特征和预测worker拒读当前位置target、TV/Δ、oracle、fit统计、历史target likelihood、当前/未来token。train-only scaler；固定单层width32 GELU MLP，50epoch、AdamW lr1e−3、batch512。A/B各仅3个weight decay，由旧dev最终epoch Brier选定，不按fresh选择checkpoint或seed。

同一个冻结B在精确同C/i/Z内做8次固定历史块置乱。先逐次算Brier再平均损失；单历史组贡献真实零差，仍留在全部eval主状态人群。输出预测及实际置乱索引并签名后，独立标签worker才评估。新题fit target只供T1，不参与模型/scaler/阈值/校准。

训练加权目标为全局Σwℓ；均匀访问行时batch损失固定(N/b)Σbatch wℓ。无类别重加权。dev阈值是全部唯一分数与+∞中满足加权FPR≤10%的最小值，整组处理ties；test不重调。常数基线是训练加权坏状态率，另报全预测非坏。

## T3：次要预测

收益用固定λ=1e−3 ridge、加权LSMR，截距不罚、atol/btol1e−8、maxiter1000。固定有限解不自动等于收敛，stop code与残差披露。

方向token在看p前由v*=argmax|qh−q0|选择，同值取最小ID。A/B共同增加完整静态embedding及q0(v*)、qh(v*)；固定线性三分类预测p(v*)−q0(v*)的下/中性/上，带宽1e−5。固定50epoch，不搜索；常数类别先验依训练权重计数并加0.5伪计数。logloss统一clip至[1e−12,1−1e−12]并报告次数。

T3不替代主检验。方向可预测不等于整体修好；该预测发生在qh可计算后，不声称运行原头前已能决策。

## T4：至多12对有限松弛诊断

只选新eval不同完整H、同C/i/Z自然分支，每域异号与非异号各前2对；按更早H汉明距1优先再固定字典序，排除中性、不按效应大小排序。不足不跨类补齐。这是按结果选择的解释案例，不估计失败比例或单词语义因果效应。

复用R4完整尾部S/U松弛：S为raw Top256，S内概率可变，U内比例保持raw、总质量可变。固定100轮、gap1e−6、feasibility1e−7、每LP30秒。保存实际返回primal/dual与输入，不后造证书。逐支严格改善必须由候选实际TV核对；均值最优或求解失败不能推出共同严格改善不存在。这里的可行候选是松弛概率动作，未验证原头basis/dose可达。

## 统计与结论

每C/i状态按自然频数平均，位置等权、anchor等权、题面等权、三域等权。HN采用预定novel子集；HP=Brier(A)−Brier(B)、HC=均值置乱Brier−真实Brier均用全部合格eval主状态。覆盖门槛只影响支持判定，不改点估计人群。

2000次域分层整题bootstrap，题内全部anchor/分支/8次置乱一起抽取。HN从各域有定义HN题抽取；HP/HC从全部合格测试题抽取。三项主CI各双侧98⅓%（尾部分位1/120与119/120），并给95%描述CI；支持要求预定覆盖满足且主CI下界>0。分类bootstrap缺类记NA及比例，不补抽。

区间条件于本轮固定模型seed、分支和置乱，不覆盖训练seed不确定性。不把状态或token当独立样本，不合成一般理论总passed。覆盖不足、负结果和有限预测器失败均不能证明历史无信息。
