# ResidualV3 mismatch：CPU 阶段证据与未完成项

Run：`20261013T154124Z_mismatch_e2`。本报告由 AI 辅助原文核验、源码审计、数学验证及独立代码实现形成。**这是可恢复的阶段交付，整体研究未完成；没有新的冻结 E2 pilot、正式 A/B/C 或留出机制结果。**

## 当前结论

1. **已支持，限现有有效产物：** 新六域队列已完成的 40 正式窗/20 配对显示整体联合配方 AC 与 TPS 增幅均正；分别为 8.48%–18.43% 与 2.10%–24.54%。Arena 旧 attempt 产物全部排除。两臂主体不同，这些范围不识别推理修复的因果贡献，也不是最终整批审计结论。[有效窗复核](../analysis/EXISTING_RESULT_AUDIT.json)
2. **已支持，源码和计数定义：** 冻结头读取当前位置 hidden、实际前驱 embedding 和原 q 的特征；输入/输出 basis 数值不同；Top256 是 logit 残差支持，采样仍为完整词表。候选外 token 的总概率会重归一化，但其未修改 token 之间的 odds 保持。这提供一种具体的表达边界，不证明自然 E2 残差具有何种主导结构。[源码审计](../analysis/RUNTIME_AUDIT_CN.md)
3. **已支持，数学和计数边界：** 一步接受是 1−TV；连续存活需在实际 proposal Q 上用 token 验证概率 b 的乘积。CPU 非平凡反例 S₂=.24，而普通 draft 路径的 overlap 乘积期望为 .144。formal `acceptance_at_pos` 在 EOS/预算截断前计数；其 AC 恒等式不是终止后实际发出长度的证明。[理论卡片](../analysis/THEORY_CARDS_CN.md)
4. **不确定，E2 机制：** 条件变化是否重要、正确前驱是否胜置换和同主体 raw、校准能否解释大部分改善、机制是否迁移到 native reached 前缀，目前都缺新模型数据。不能将资源等待、代码尚未通过 E2 pilot 或实验未执行写成科学反证。

## 为什么尚不能运行模型诊断

`logs/RESOURCE_LATEST.json` 保存实时宿主证据，而非沿用启动 prompt 快照。核验时原控制器 PID4138310、Arena C1 attempt2 evaluator PID3598094、CPU watcher PID664385 均仍在运行；控制器状态 running、收尾 waiting_existing_controller。原 GPU6 的预约仍持有，FINAL_AUDIT/COMPLETION_AUDIT 均缺失。READY_AUDIT 的 require_complete=false。

这是用户规定的执行前提，不是追加权限请求。本轮未启动、停止、重启或接管原作业；未占其卡；无自有 GPU 作业，也未部署一个未来自动启动的研究 watcher。所有新写入限本 run。资源 gate 未释放时继续 CPU 可执行工作；CPU 产物交付不等于 A/B/C 已结束。

## Q1–Q7

| 问题 | 本阶段答案和状态 | 证据与边界 |
|---|---|---|
| Q1 比较什么 mismatch？ | **已支持（定义/源码）**：同 s=(C,H,i) 上、同词表与采样变换的 target p 和实际 proposal q；T1 全词表 TV/overlap/KL，T0 argmax；独立 raw、联合 raw、联合＋头三种身份分开。 | [协议草案](../protocol/PROTOCOL.json)、[资产身份](../sources/ASSET_AUDIT.json)。teacher-prefix、target-branch/alive-at-i、native/all-proposal 和 reached-prefix 不互换；拒绝事件不等于语义错误。 |
| Q2 条件变化还是平均拟合偏差？ | **已支持（恒等式/有限样本限制）**：固定 C/i 且 q 不随 H 变，经验 KL 可精确分为条件变化项与平均拟合项；**E2 占比不确定，待 A**。 | [CPU数学审计](../analysis/CPU_MATH_AUDIT.json)。K 有限会偏低估条件项、偏高估拟合项；n=1 退化零项不证明总体变化小。不能把拟合偏差直接叫不可约容量。 |
| Q3 头利用什么结构？ | **已支持（可见输入和表达边界）；E2 干预效应不确定**。B 正确/置换只变前驱，保同 C/i 边缘、固定 hidden/raw/H/p；必须同时胜 raw，i=1 为负对照。 | [独立重放代码](../code/repair_replay.py)、[位置采集代码](../code/ab_position.py)。CPU 随机小头 parity 检查不是 E2 pilot；B4 dev 单温度参照尚未拟合。 |
| Q4 哪些收益能归于推理修复？ | **历史 E1 已支持开关头 AC；当前 E2 归因不确定**。不能说过去从未做开关头；不能把主体不同的整体收益拆成训练/推理百分比。 | [历史证据](../analysis/HISTORICAL_EVIDENCE_CN.md)。E1 开头比同主体关头 AC +13.73%–18.78%，但 acceptance-only、非 E2 TPS。新 B/C 尚缺。 |
| Q5 局部改善怎样到连续前缀？ | **已支持（公式/原计数边界）；E2 迁移不确定**。实际 Q 上估计 G_k∏b_i，G 只依赖 proposal EOS、轮开始预算和长度；实际 reached 单列；accepted proposal、bonus、correction、实际发出及截断分开。 | [理论卡片6–7](../analysis/THEORY_CARDS_CN.md)、[native静态审计](../analysis/RUNTIME_AUDIT_CN.md)。新 verifier capture 尚未 native 验证，权威 round-start metadata 和 scheduler emission companion 尚需接通；不发布插桩 TPS。 |
| Q6 能否迁移到留出题/其他状态？ | **不确定**。192 唯一题已按 prompt 分组固定：三域各 dev24/test40，pilot8 包含于 dev；新 test 标签读取0次。 | [样本清单](../protocol/SAMPLE_MANIFEST.json)、[训练重合审计](../analysis/TRAINING_OVERLAP_AUDIT.json)。已有项目评测使用；没有在训练 group 元数据中发现精确匹配，不排除语义近重复/预训练。尚无探针，不作可部署泛化预测主张。 |
| Q7 较近邻多知道什么？ | **文献重合已支持；新增 E2 机制贡献不确定**。条件失配、前驱修正、前缀目标和最大耦合已存在。当前新增的是有待落地的同状态可反驳测量，不把标准理论工具记作创新。 | [原文矩阵](../sources/LITERATURE_MATRIX.md)、[来源审计](../sources/SOURCE_AUDIT.json)。DSpine T1 各臂 proposal/验证口径不同；跨论文 AC/TPS 不排名。 |

## 已完成的实际工作

- 从当前有效 formal summaries 独立复核 40 窗/20 配对。位置2–15的280个设置×位置单元均提高，首位置9个配对降低；保留各臂自己的轨迹身份。`AC−1−ΣS` 最大绝对残差约 1.8e−15，是原 sampler 指标定义的一致性。[数据表](../analysis/EXISTING_SURVIVAL.csv)、[图](../analysis/EXISTING_SURVIVAL.png)
- 原 protocol 所列183个小源码 SHA 全匹配。E2 export/训练 final、mask、head、basis、target/tokenizer 小配置身份分别记录。target 大 shard 未在有限范围旧 manifest 中找到可信 SHA；只记 stat，没有大 I/O 重散列或虚构哈希。
- CPU检查共71项通过（math30＋AB28＋native accounting13）：KL闭合/下溢、TV界、有限K偏差、前驱边缘置换、Markov缺更早历史、b/a反例、EOS/预算/bonus及不等组数聚合。新增 A/B 联检见 [AB审计](../analysis/AB_CPU_AUDIT.json)，终止计数见 [native accounting审计](../analysis/NATIVE_ACCOUNTING_CPU_AUDIT.json)。这些均为数学或随机小模型验收，不是冻结E2实验。
- 独立 `ab_position.py` 实现完整词表 A/B 标量采集、正确/置换前驱、i1负对照、K/n退化记录及dev温度grid；`aggregate_ab.py` 按 branch→position→C→prompt 聚合、prompt bootstrap、域均权与缺失覆盖；`fit_temperature.py` 拒绝test拟合。协议锁检查真实冻结E2/dev-GPU pilot，CPU合成审计不能满足该条件。
- 实现只读 HF/SDPA A/B 后端，明确独立 runtime 身份；未构造模型对象，未执行 E2。新 native verifier hook 仅在新自有 server 中安装，读取实际 processed p/q；当前不能凭此声称 scheduler 终止插桩已完成。资源执行器/租约接口仍待资源释放后接通和 dev 验收。
- 原文核查和历史证据保留容量/优化、简单校准、主体改善、测度迁移等竞争解释。三种子专家负结果、尾部停修无净TPS、旧on-policy约8倍成本不稳定胜TF均不重启。

## 可被推翻的预测及退出

H_A：预定 target 分支测度上的条件量若稳定且与修复量对应，才支持它作为重要解释；条件量小或 K 不稳就降级。不预测随位置单调，不把它变成 TV/AC 加法分解。

H_B：留出正确前驱应同时胜置换和同主体 raw，且校准不能取得近似改善；否则降低“对应信息带来有效修复”的解释。置换可能改变联合输入分布，结论不超过冻结映射对对应信息的依赖，不给总收益分摊比例。

H_C：同状态局部改善应在 native reached 状态仍可见，并与终止规则下的真实发出/存活变化相符；若只在少到达后部或 target 分支有效则降级。概率/计数合同不通过、覆盖不足或区间宽时为不确定。

当前阈值尚未用 dev pilot 冻结，协议仍为草案，正式test尚未打开。负结果不增加K或扩搜到显著；技术重试最多两次；累计24 GPU小时/1卡/新增50GiB/CPU≤4不按分支重置。

## 一页决策

**当前能站住的部分：** 原系统有效配对的整体增益、头的真实输入/候选外表达边界、原计数定义、理论估计量及其反例。**当前不能主张：** 条件 mismatch 主导、大小参数差造成不可约误差比例、所有收益来自推理头、恢复完整历史、OUTPUT首次提出、跨论文优于近邻、E2机制已泛化。

没有新 E2 核心证据，因此本阶段不提出新结构训练或方法改造候选。下一步是恢复已授权本轮的 dev pilot、锁定 A/B/C，不是申请扩大任务；精确阻塞和未实现接口见 [NEXT_CHAT](../NEXT_CHAT.md)。本阶段不宣称研究成功或失败，只保留可信的事实、边界与可继续执行的位置。
