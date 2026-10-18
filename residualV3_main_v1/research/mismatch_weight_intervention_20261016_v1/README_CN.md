# 实际权重干预

用户2026-10-16要求直接确认具体环节，不能反复以“缺因果证据”结束。本包在新run进行有界的完整主体继续训练干预，旧资产保持只读。

具体假说：同一位置不同训练块获得不同D-PACE权重，是否妨碍原来低权重状态的分布拟合？W保持原样本权重；U取消同位置内样本差别，保留W的位置总权重。相同E2独立主体起点、数据、32步、参数步幅。此测试不证明历史训练主因；阳性和阴性均按预定口径报告。

当前run：[20261016T071044Z_weight_intervention](runs/20261016T071044Z_weight_intervention/)。具体样本、参数、数值和因果判别合同：[EXECUTION_CONTRACT.md](runs/20261016T071044Z_weight_intervention/protocol/EXECUTION_CONTRACT.md)。

预算：单卡共享，累计2GPU小时/6CPU小时，新增16GiB，RSS32GiB，自有GPU峰值20GiB。原权重和旧run不写入。没有native/TPS任务。
