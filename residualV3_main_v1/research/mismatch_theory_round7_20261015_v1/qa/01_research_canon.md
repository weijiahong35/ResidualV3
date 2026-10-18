# 事实与边界

- R6 run：`residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6`，48题完成。HN在44题868匹配novel状态改善TV0.259371，98⅓%CI[0.209580,0.310423]，损伤约2.77%。限该人群、fit标签探针。
- R6 HP/HC不确定；风险Brier未胜常数，方向logloss也未胜常数。训练/开发差距大，不证明历史无信息，不恢复预测器搜索。
- R6固定12对中11对有S/U松弛共同改善候选；不证明实际basis/dose可达。pair01给出跨token补偿，pair02需求同向但幅度不同。
- R4合法共同动作、R5共享偏差与条件对应以及R1–R3未支持结果保留原测度。本轮不重新归因独立DFlash→联合模型的全部AC/TPS增益。
- 文献事实1：Banerjee等2005统一研究包括relative entropy在内的Bregman聚类；均值分组不应包装成新理论。原文书目/摘要：https://jmlr.org/papers/v6/banerjee05b.html 。
- 文献事实2：Leviathan等2023提出保持目标输出分布的标准投机解码算法；其结论不替本地旧runtime随机源问题背书。https://proceedings.mlr.press/v202/leviathan23a.html 。
- 文献事实3：Hewitt与Liang2019用控制任务约束probe能力与记忆解释；本轮概率移除对照不是其word-type控制任务复现。https://aclanthology.org/D19-1275/ 。
- 上述三原文页面本chat已核对；不声称全文重读或新颖性全面检索。
- 新诊断c是原头d内、fit逐token一致且不越界的保守平衡子动作。c=0不证明没有共同TV改善；c不是全部可修空间或最优动作。
- qc/qr/qu是概率空间诊断分布，归一化不等于原basis/dose可达。均匀概率移除不是R3剂量减半，也不是新部署方法。
- I=Ghead−Gc−Gr≤0来自同向逐坐标移动与绝对值几何；只作合同，不是新机制发现。任何贡献比例/完整因果归因均未建立。
