# 只读证据入口

本包不重复一轮广泛检索。先读已完成实验与此前来源核查，再对照本包声明边界。文献提供竞争解释和方法近邻，不替本模型证明成因。

## 本地证据（执行前必读）

1. [旧完整结果](../../mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse/results/RESULTS_CN.md)：区分规定成本优势与学习未确认。
2. [旧覆盖表](../../mismatch_basis_reuse_20261016_v1/runs/20261016T143529Z_basis_reuse/results/COST_COVERAGE.csv)：r(.5)在.8预算内E/G都接近全覆盖；不足以预设基底路由必要。
3. [原方法来源与负结果](../../mismatch_research_decisions_20261016_v1/ORIGIN_REVISION_CN.md)：跨域聚类与早期双专家负结果不丢弃。双专家共用基底，不能充当不同基底路由的直接结论。
4. [此前收缩决定](../../mismatch_focus_review_20261017_v1/DECISION_CN.md)及[来源设计草案](../../mismatch_focus_review_20261017_v1/NEXT_EXPERIMENT_CN.md)。本包补全了执行合同，修订N共同支持集、未领先不等价、尺度单位、尾部资产依赖和bootstrap。执行以本包锁定合同为准，不静默混用旧建议。

## 三个相关一手入口

|来源与核查范围|作者方法/主张|对本包的推断与边界|
|---|---|---|
|[CorDA](https://arxiv.org/abs/2406.05223)，本轮核摘要方法描述|利用样本输入激活的统计信息影响权重分解和适配方向|数据本身能影响有用方向，因此旧E/G比较需要数据相关对照；本包F不是CorDA复现|
|[LoRA-GA](https://arxiv.org/abs/2407.05000)，本轮核摘要方法描述|通过初始化使低秩乘积的首步梯度贴近完整微调梯度|从误差/梯度定方向不宜宽泛声称首次；本包不检验训练收敛|
|[Arrow / Towards Modular LLMs](https://arxiv.org/abs/2405.11157)，本轮核摘要；前次已读路由章节|根据新输入选择可复用LoRA适配器|上下文选模块已有近邻；本包不构造或评价路由器|

更广的原文核查记录：[草稿修正近邻](../../mismatch_focus_review_20261017_v1/sources/SPEC_SOURCES_CN.md)、[残差/方向近邻](../../mismatch_focus_review_20261017_v1/sources/RESIDUAL_SOURCES_CN.md)、[目标与几何](../../mismatch_focus_review_20261017_v1/sources/OBJECTIVES_AND_GEOMETRY_CN.md)。这些记录不是外部实验复现。

新结论只能来自本包实测E/F/N比较。Hessian、最小范数与正交不变性是工具，不作为新增理论贡献；通过本包不等于首次性或论文录用已确立。
