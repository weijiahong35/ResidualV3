# 事实

R5=`../mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5`。

- R5共同偏差探针匹配eval改善TV .307773；同幅度真实分组对置乱增量 .018342。限条件子集、全词表标签探针。
- 1034/3255主eval状态合格，862条H在fit出现过、172未出现；不能称所有新历史泛化。真实原头同Z不同H可一好一坏。
- 原头全部主状态损伤率约13.328%，与探针人群不同；不能把高准确率或两个损伤率差当方法收益。
- R4 153/163旧异号组存在实际合法共同改善raw动作。方向冲突不等于所有共同动作都失败。
- R1条件主导未证、R2几何预测增量未支持、R3减半不优、R4一般指标增量不确定均保留。
- R5科学完成；线程观测缺口、R2概率警报、R4资源披露保留。完整源路径和SHA需输入审计。
- Hewitt & Liang 2019《Designing and Interpreting Probes with Control Tasks》https://aclanthology.org/D19-1275/ ：探针能力/记忆会影响解释，需要对照；本轮同Z置乱不是直接复现其word-type control task。
- Voita & Titov 2020《Information-Theoretic Probing with Minimum Description Length》https://aclanthology.org/2020.emnlp-main.14/ ：评价探针不应只看最终准确率；本轮不额外实现MDL支路。
- Banerjee等2005，https://jmlr.org/papers/v6/banerjee05b.html ：均值分组工具已知，沿R5来源；不把新预测分数称信息论上限。
- 静态冻结token embedding lookup允许；目标在真实H上的logits/hidden/likelihood、p−q、TV、oracle量不能作预测输入。
