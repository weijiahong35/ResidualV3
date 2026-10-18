# Research canon

文献事实及原文核查入口：

1. EAGLE-3以去feature回归后的分步收益/损失及training-time test联系机制与方法：[v3](https://arxiv.org/html/2503.01840v3)。
2. DFlare具体指向逐层条件共用及容量扩展趋缓：[v1](https://arxiv.org/html/2606.02091v1)。
3. DCOCO已研究候选存在但选择不对：[v1](https://arxiv.org/html/2610.12570v1)。不能直接把候选覆盖认领成创新。
4. 硬/软概率监督、接受导向训练已有DistillSpec、LK、VSD背景，见旧followup资料。

本地事实：R1前驱干预有效但非独立成分分解；全局温度不能复现完整增益；首位置收益近零。R2候选内改善大但几何预测未确认。R3半剂量未推荐。R6新增历史/方向预测未确认。R8共同动作在两个主体各自目标与草稿来源匹配子集有效，但reached覆盖不足。R9有限族共同安全余量和高收益取舍并存，不识别训练/规模原因。路径见包的资料索引及既有MECHANISM_SOURCE_SCAN。

已读R8 collect_r8.py：每C有raw.npz(D/J BF16位模式、J_hidden)，target.npz、F.npz、META.json；target完整概率仅位置1/2/3/8/15保存，不能凭sampled-token标量补齐其他位置。F记录完整头logits，可在F-source记录上配对q_J与q_F。无D_hidden归档承诺。

R8数据已全部暴露，文件名fresh_test不改变本轮回顾性身份。保留自然重复H。D/J不是不同target家族；F来源是J+头而非J raw。

禁止：因模型有效断定存在新机制；把误差指标当成误差来源；先定输出层问题再找阳性；换指标/挑坏状态救候选；把旧thinking归因结果混入E2；重开任何旧stage。

