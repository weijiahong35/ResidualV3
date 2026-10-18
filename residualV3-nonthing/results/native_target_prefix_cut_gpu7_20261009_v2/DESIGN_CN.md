# target验证前缀截短小实验

目的：把“少做尾部repair”和“target少验证尾部”分开，检验端到端省算。

对照使用相同全量数据、单轮独立训练DFlash；联合主体关闭头的BODY_ONLY不再测。既有所有权重及预测器保持冻结。

FULL15全部修；REPAIR_STOP按已校准的q门控停止尾部repair，仍验证15proposal；VERIFY_STOP使用同一门控并将真实更短proposal列表提交原生scheduler，让target只验证保留前缀加bonus位置。修复长度与验证长度暂相同，便于归因。

按生成顺序决策，当前候选之前的信息才能决定是否保留该候选；未来候选不能反向参与长度选择。原生实际q、位置RNG、lag-1、KV/bonus处理与变长前缀一致性必须在C1/C40 smoke检查。

FlashAttention只支持均匀decode的完整图，变长batch用原生piecewise图。四组配置统一为FULL_AND_PIECEWISE并记录实际执行，不声称所有target运算都进完整图。没有decode eager fallback才进入正式窗口。

上轮v1在dummy T0.9把未缩放score当processed q，校验错误，无正式成绩。本版用独立stock sampler重建processed q和proposal ID。

当前实验不训练新的价值头。后续价值标签应比较同一已有实际前缀下“继续修”和“停止修”的收益，每条分支重新计算自身target前缀；拒绝概率不能代替修复收益。
