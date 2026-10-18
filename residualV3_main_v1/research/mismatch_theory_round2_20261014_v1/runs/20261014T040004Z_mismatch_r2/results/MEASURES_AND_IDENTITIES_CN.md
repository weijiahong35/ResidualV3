# 第二轮模型、样本与测度身份

本文件说明固定实验的比较对象。效果与最终状态见 [RESULTS_CN.md](RESULTS_CN.md)，所有资产身份以 [ASSET_AUDIT.json](../sources/ASSET_AUDIT.json) 和 [LOCK.json](../protocol/LOCK.json) 为准；manifest 内首轮检查快照不是最终复核。

| 身份 | 主体 / 修复 | 合法比较 |
|---|---|---|
| B0 | 冻结 E2_DFLASH 独立主体；无修复头 | native 配方参照。其差异同时包含主体训练身份，不能分摊成头的因果百分比 |
| B1 | 冻结 E2_JOINT_NATIVE 联合主体；实际采样 joint raw q | 与 B2 同主体，分别生成各自真实轨迹 |
| B2 | 同一联合主体＋冻结 E2_JOINT_HEAD；同次 raw Top256 候选上的残差 | native 同 H 的 shadow raw 与实际 repair 比较局部 TV；与 B1 完整请求比较实测接受后果 |

联合主体导出 SHA `ad4ab70fff01f329780d1e5a245b5268e238dd1f68dd9ec51c909ca252728168`，独立主体导出 SHA `0b8cef16a7e171981111a621e2551a02c663d046ffc575d6cf6d0252c8388ce7`，头 SHA `6635f429a363ba9928da653f3337ce991e644c379cb7ac4e159ac1413b71944f`。所有导出及头/basis在本轮重新校验；target 大分片按已归档 stat 身份核对，缺历史可信全文件 SHA 的限制保留。basis 是固定 embedding basis，不能混称为 target LM head。

| 测度 / 算术 | 状态如何产生 | 可回答的问题 |
|---|---|---|
| teacher-prefix | 目标模型生成参考前缀；本轮主要用其确定 C | 参考轨迹与身份；不代替 native reached 的访问分布 |
| HF target-branch | 固定 C、冻结 H、K=8 条 target 完整前缀分支；最多两个 anchor（1/65）与15位置 | 同状态 raw/repair 与受限集合几何；主检验 H_R 的离线预测损失 |
| native all-proposal | 每臂自身运行；worker 真实 token 与实际生成 q 绑定，scheduler 给轮次与预算 | 完整 proposal 路径诊断，不能以普通路径 overlap 乘积替换真实存活 |
| native reached 且 G 有效 | 实际 first-reject 到达位置，含拒绝位置；另用 proposal 路径 EOS、长度、轮初预算约束 G | 主检验 H_N：B2 同 H 的 shadow raw TV−actual repair TV；不再乘实现接受事件 |
| native 已发出 | scheduler 停止处理后的实际片段，并与 HTTP completion_tokens 对账 | accepted proposal、bonus、correction、ordinary 分项；B2−B1 每请求的实测后果 |

HF 几何来自同 BF16 全词表 logits 的稳定 FP64 归一化，独立核验 FP32 sampler 漂移。native p/q 来自实际采样处理后的 logits，保存的是 torch FP32 数值重构；并未声称逐位等于 Triton LSE buffer。两者不混合为单一指标。词表151936，T1/top-p=1/top-k关闭且无额外惩罚；HF EOS 为151645，native EOS 为151643/151645。EOS/缺 anchor 不补采，不事后只挑无终止轮。

`b_i=min(1,p(token_i)/q(token_i))` 使用同轮真实 proposal token。`G_k∏_{i≤k}b_i` 的 G 不含实现接受事件。该乘积作为理想独立 uniform 模型下的工作估计；当前 runtime 的对应条件独立规律未建立。原工程6题 B2 的概率残差警告与 failed 审计保留，机械 reviewed gate 只批准实际身份、局部距离与计数测量。新样本区间是否覆盖0不撤销这项边界。

开发集每域12题共36题，工程6题为其子集；新确认性留出每域16题共48题，排除前轮全部192题及本轮调试题。MT-Bench 使用原80题面板中剩余16题。训练元数据834998条的精确规范化匹配为0；语义重复、项目历史评测和预训练暴露不能排除。这里只称本轮机制分析留出。

HF 统计按 branch→position→C→prompt，native 局部按请求内合格状态→prompt，后果先计算完整请求比例再作 prompt 配对；三域等权、prompt bootstrap。缺失合格状态及零分母显式记录，不填0。数据、特征、scaler、ridge、阈值、统计和源码均在新留出主分析前锁定；完整规则见 [PROTOCOL.json](../protocol/PROTOCOL.json)。
