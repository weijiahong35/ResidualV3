# 第六轮启动prompt

复制分隔线以下全文给执行chat。它授权执行本轮限定研究，不是仅要求再交一份计划。

---

你现在接手ResidualV3第六轮失配机制研究。仓库是 `/data/weijiahong/speculators-main-lora`。

先读以下文档并落实其中固定协议：

```text
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/README_CN.md
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/THEORY_PLAN_CN.md
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/sources/READING_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/qa/INPUT_AUDIT.json
```

第五轮已完成归档：

```text
residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5
```

它支持匹配子集上的共同偏差修正及等幅度条件对应价值，也给出了早期历史让target需求反向的案例。但独立抽样分支大多重复了fit中的完整历史；标签参与的共同流探针不能证明能事前预测修复。

**本轮从失配解释出发：历史怎样改变target与草稿之间的偏差，什么共同方向在新历史上仍成立，以及已有历史能否预判原头何时修好、何时修坏。** 不要把工作转成训练更好的AR修复头，也不要用工程审计代替科学答案。

在新run自主完成四项限定任务：

1. **T1 未见历史稳定性。** 新48题，GSM8K/HumanEval/Alpaca各16；每anchor K64=fit16/eval48。沿R5固定M/2共同流探针。novel须完整块内H未出现在同C/i的任何fit分支中，不能用不同seed冒充；不补采到有冲突或通过覆盖。
2. **T2 历史预测损伤。** R4原fresh48训练、R5原fresh36开发，新48题全部合格eval主状态测试。以冻结原头实际TV改善Δ定义坏状态标签，预测其概率。A读取完整现有可见信息，B加更早有序历史，同一B在精确同C/i/Z内置乱历史作对照。固定width32 MLP及旧dev正则选择，严格按指南，不搜索新结构。
3. **T3 次要收益/方向预测。** 只按固定ridge与线性三分类方案，分别预测Δ和由q0/qh预选token的需求方向。它们不能替代T2，也不输出新修复分布。
4. **T4 有限失败解释。** 全轮最多12对固定规则选择的自然分支，给真实概率账本及已有S/U松弛oracle界。松弛可行不等于原头basis/剂量可达，未找到也不等于不可修。

这里明确允许拟合**小型离线诊断预测器**，以及指南限定的12对松弛诊断；这是R5旧范围之后的新授权。主体、原修复头、target、basis、安装环境冻结。不训练新头，不改输入/损失，不上线gate，不开native、RNG、TPS、剂量网格或其他支路。旧负结果和概率警报保留，不自动开启第七轮。

先完成旧数据提取和模型锁，再打开新题。新题fit target标签只供T1，绝不能用于预测器训练、标准化、校准、阈值或特征选择。预测worker只读推理可得特征，不读当前位置target、TV/oracle、历史target likelihood、当前/未来token、来源ID或fit统计。先保存并签名预测，再读取标签评估。

训练/开发/测试按题面完整隔离，旧数据统一到主位置3/8/15及相同自然target存活测度。重复自然采样保留频数，复制观测去重。HN、HP、HC的条件人群分别报告；HP/HC用全部合格eval状态，不能只挑异号组。Brier、常数基线、损伤比例、PR-AUC、固定阈值recall/FPR和校准一起给。负结果或覆盖不足诚实报告，不换特征、换域、扩大样本救结论。

**有能容纳本阶段工作集的卡就用，允许共卡，不等空卡，不要求GPU利用率为零，不固定GPU0/7。** 实时查显存和预约，满足预计峰值+4 GiB余量就推进；不终止他人任务。无法容纳最低工作集时继续CPU工作并复查。无需重复询问共卡许可。

总上限：8 GPU小时、24 CPU小时、新增磁盘700 GiB，同时1卡；CPU worker≤4、计算线程合计≤8，import前限制数值线程。失败、编译、等待、临时文件和重试均计入，不换run重置；同阶段最多2次有新定位的修复重试。只管理核实身份的自有进程与预约，不动原controller、parallel、收尾或历史资产。

结果写到：

```text
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/<UTC>_mismatch_r6/
```

不要覆盖本目录方案或旧结果。先简报你读到的R5实际终态、数据与执行范围，然后按指南自主推进，不停在“建议以后实验”。报告前页用简单中文回答：未见历史能否共同修；历史能否提前预测修坏；能否预测方向；失败后是否还有兼顾空间；还有什么不知道。

交付完整结果、图表、claim–evidence、模型/特征/样本锁、预算及身份审计、COMPLETION和NEXT_CHAT。科学负结果可以完成；必要任务没做完就partial。收尾退出自有作业并释放预约，不启动下一轮。
