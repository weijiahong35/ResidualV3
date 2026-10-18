# 审核记录

范围审核（/root/discovery_scope_review）：7.875/10，通过但要求明确绝对修复、bootstrap和固定组、随机性和真实位移门禁。

初次可行性审核（/root/discovery_feasibility_review）：7.88/10，尚有P0。要求teacher argmax及索引/C内aux、训练参数名单和步长、评估固定分组与CI、实测资源门禁。模型头部核算训练基础显存约12.45GiB。

执行合同v2落实上述修改。复核8.50/10，允许进入工程门禁，正式训练条件为数值及资源门禁通过。要求实际batch4含最长C、AMP/TF32关闭、CUDA同步计时。

代码审核：未发现P0；补充U step0与共同基线的TV/CE/weight一致断言及每次重置的原参数逐元素核对，已落实。TRAINING_COMPLETE只表示训练完成，不代表科学阳性。

资源准入细化：target阶段预计20GiB；训练基础权重/梯度/冻结词表12.45GiB，加5.55GiB激活和逐参数临时空间，预计18GiB。准入保留额外4GiB；硬峰值20GiB不变。无合适卡时不启动worker。实际性能和峰值另在qa与resource日志。

数值工程两次修复及失败保留于TECHNICAL_REPAIRS.json、TARGET_ALIGNMENT_attempt1/2.json。增量KV BF16形状差异未放宽阈值；最终独立完整前缀逐位置参考误差为0。没有根据科学效果改样本或步数。
