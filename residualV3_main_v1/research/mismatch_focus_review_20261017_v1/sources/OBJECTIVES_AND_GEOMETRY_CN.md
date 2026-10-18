# 训练目标与几何补充核查

本文件是根代理实际打开一手资料后的补充，非穷尽综述。只用下列已读部分支持论断；没有复现实验或将预印本默认为已录用。

|来源|读范围|作者主张与证据|本项目应如何使用|
|---|---|---|---|
|[Acceptance-Aware Draft Model Training for Speculative Decoding](https://arxiv.org/html/2609.24150v1)，Tianhua Xia等，2026-09-21预印本|§1–3，特别是§3.1–3.3公式|分别给出贪心目标token概率及采样TV重叠构造的窗口目标，并有GRPO扩展|“KL代理不等于接受长度”已有直接近邻。其窗口乘积写出了给定前缀下条件独立假设，不能直接套成所有在线轨迹的精确无条件期望；此处未复核其全部实验表|
|[Variational Speculative Decoding](https://arxiv.org/html/2602.05774v1)，2026年预印本|§4.1的路径概率/变分目标、§5主要表和消融文字|用路径可接受性构造变分目标，并有重加权、正则和候选数消融|路径目标是已有方向；变分模型里的参考路径分布不等于真实草稿采样分布。没有据作者推导确认我们的KL代理头失败原因|
|[AdaSPEC](https://arxiv.org/html/2510.19779v1)，2025年预印本版本|§2、§3完整选择步骤、§4.1–4.4|先蒸馏参考模型，以草稿相对参考的token损失差选训练位置，多个任务上比较接受率|“选择更值得学的误差”已有方法；作者的可学习性判断是其选择准则，不是任意错误可学习的保证。不同训练预算/参考模型成本需一起看|
|[TaRA](https://arxiv.org/html/2609.02639v1)，Taehyeon Kim、Eunhyeok Park，2026-09-02预印本|§1–3、§4.1及Table1|Taylor近似、Fisher/K-FAC替代，结合激活与梯度协方差分解W，接回低秩初始化；多任务微调|“方向应适应训练信息”非常接近的理论邻居。其近似链与参数矩阵低秩目标，不是本项目有限概率修正的最优性定理|
|[How Divergence Becomes Decision Flips in Compressed Language Models](https://arxiv.org/html/2610.00694v1)，Beatriz Almeida Felicio，2026-09-30预印本|摘要、§1–4前半段，定义/统计口径/局部推导|研究压缩扰动、分布变化与argmax改变；在多模型/语料上测经验比例，并报告预注册失败边界；小扰动关系需对称、独立等假设|用于提醒“分布变化不自动等于最终决策收益”。TV与贪心翻转的经验接近不是恒等式；不能用来把本地TV差直接改名接受率。未逐项复核其后半篇证明与部署实验|
|[RepSpec](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d70ea003729b440b89a2f958a5554c1f-Abstract-Conference.html)，Feiye Huo等，ICLR2026|正式会议记录及摘要，正文未读|训练时增加可合并线性分支，推理并回骨干，摘要报告接受长度改善|只用于定位训练重参数化已有正式论文；不基于摘要判断其全部理论/消融充分性|

额外召回但未作为决定依据：SpecKD（2510.24021）、Test-Time Speculation（2605.09329）、BALANCER（IJCAI2026/475）。本次只有检索摘要，不能写成已读方法证据。

根代理实际查询：

1. `speculative decoding draft model residual correction subspace alignment low rank error 2026`
2. `speculative decoding token importance loss total variation distillation token-level acceptance 2025 2026`
3. `language model residual error subspace vocabulary frequency covariance task aware low rank correction`
4. `"error" "subspace" "residual" "speculative decoding" correction basis`
5. `"gradient covariance" "Fisher" low rank adaptation initialization mismatch`
6. `"teacher" "student" "residual" "cross-domain" correction subspace`

搜索中出现的聚合解读、博客转述和镜像仅用于发现题名，支撑结论的链接均转到一手正文或正式记录。其余两份来源笔记分别列出实际查询与读范围；没有计算伪精确的检索覆盖率。没有调用Semantic Scholar做独立交叉元数据校验，不把它报告为完成。
