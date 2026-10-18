# 证据阅读指南

先读本地边界，再看近邻，禁止因看到新文献临时扩实验面板。

## 本地

- [基底来源决策与旧实验核查](../../mismatch_research_decisions_20261016_v1/NEXT_BASIS_CHECK_CN.md)：E/G对照尚缺；T036/T037、T038、R4各自回答什么。
- [原始聚类及负结果](../../mismatch_research_decisions_20261016_v1/ORIGIN_REVISION_CN.md)：低领域NMI不是可共享修复；双专家阴性保留。
- [方法源码与身份](../../mismatch_research_decisions_20261016_v1/METHOD_DRIVEN_IDEA_CN.md)：当前B及输入/输出映射；旧理论与当前配方的区别。
- [机制发现零候选报告](../../mismatch_mechanism_discovery_20261016_v1/runs/20261016T061522Z_mechanism_discovery/results/RESULTS_CN.md)：无新的来源机制；本包不把零候选撤销。

## 一手文献及本包用途

|文献|核查位置|作者工作|本包借鉴/不能借用的结论|
|---|---|---|---|
|[Caprese](https://arxiv.org/html/2505.07861v2)|图2、§3.1、§3.5|稀疏FF残差低秩补偿|借鉴“明确误差对象—补偿—替代资源投入”；不证明当前p−q低秩或来源|
|[SAD-LoRA](https://arxiv.org/html/2607.04306v1)|§4.1、§5.2、§6|单层数据加权更新误差的方向/系数分解|借鉴分开成本与学习；不能直接移植为TV定理或当前来源|
|[LoRA-GA](https://arxiv.org/html/2407.05000)|§3、§4.3表3|初始梯度近似与尺度消融|借鉴控制方向/尺度；初始分析不证明长期机制|
|[VeRA](https://arxiv.org/html/2310.11454)|§3.1|固定随机矩阵配可学习缩放|要求随机/几何对照；其有效动作空间不等价于本包|
|[ResKD](https://arxiv.org/html/2006.04719)|§III-B、§IV|残差学生及集成/联合训练对照|“学误差”本身已有先例，不作首创|

本包的潜在认识是限定代理设定中误差来源方向的额外修复价值；文献存在不等于假说成立或被否定。定向阅读不构成穷尽新颖性审查，完整概率模型的修复成本不是上述论文已验证的原模型因果。
