# 第八轮：共同失配的跨主体与跨历史分布研究

2026-10-15。**这是执行交接包，实验尚未开始。** 用户已选择把两个问题放在同一轮；把 [START_PROMPT_CN.md](START_PROMPT_CN.md) 分隔线以下全文交给执行 chat，即按该有限范围自主执行。

本轮回答：

1. 独立 DFlash 与联合主体，是否都存在能改善未见历史的共同偏差？
2. 从 target 历史找出的共同修正，换到草稿自己产生的历史后是否仍有效，什么时候失效？

两项共用新36题、相同块前上下文 C、两主体 raw 预测和 target 标签。**两种被诊断主体 × 三种历史来源**组成六个对照单元；两个研究同时采集、独立判定，不等第一项阳性才做第二项。

|文件|用途|
|---|---|
|[THEORY_PLAN_CN.md](THEORY_PLAN_CN.md)|问题、共同修正定义、四个主检验、失败解释和反证条件|
|[EXECUTION_GUIDE_CN.md](EXECUTION_GUIDE_CN.md)|执行顺序、数据身份、采样/验证、预算、恢复与交付|
|[START_PROMPT_CN.md](START_PROMPT_CN.md)|给新执行 chat 的完整启动指令|
|[sources/READING_GUIDE_CN.md](sources/READING_GUIDE_CN.md)|先读什么，以及每份资料要回答的问题|
|[protocol/DESIGN_SPEC.json](protocol/DESIGN_SPEC.json)|机器可读计划；执行时另建实际代码/权重/样本 LOCK|
|[qa/INPUT_AUDIT.json](qa/INPUT_AUDIT.json)|编写时的材料哈希及题面余量快照，执行前须重核|
|[qa/DOCUMENT_AUDIT.json](qa/DOCUMENT_AUDIT.json)|文档一致性与交接检查，**不是实验审计**|

核心区别：本轮直接研究两个主体的失配，用不依赖原头动作的共同概率探针；不把 joint 头移植到独立主体，不训练预测器，不新增模块。

历史来源包括 target、独立草稿、联合主体加完整原头。均采用新锁定的 **HF参考采样口径**；参考 first-reject 只说明原草稿路径哪些位置被到达。它不等价于生产 vLLM、不证明探针可部署，也不解除旧 native 概率警报。

**有显存足够的卡就用，允许共卡，不等空卡。** 同时最多1卡，全轮上限8 GPU小时、24 CPU小时、700 GiB新增磁盘。旧实验、权重、安装环境和正式评测保持只读。执行完本轮收束，不自动第九轮。

R7 已完成：η=0.2310、κ=0.03466，支持有限共同部分与原头收益的联系；不能由此推出跨主体或跨历史来源稳定。当前新题余量快照为48/12/422，分别对应GSM8K/HumanEval/Alpaca；正式评测已见、机制未见与训练未见必须分别描述。
