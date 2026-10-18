# 与已有理论的定向核对

日期：2026-10-16。使用deep-research简报流程中的问题界定、文献检索、核验和综合；按已明确的理论问题推进。本文件是定向阅读记录，不是系统综述或穷尽优先权检索。

检索词包括speculative decoding、expected acceptance length、total variation、sequence acceptance、continuation value、weighted overlap、maximal coupling、multiple distributions及指定论文标题。只将一手论文作为结论依据，搜索摘要仅用于定位。

## 1. 直接相关来源

|来源与固定版本|本次读取范围|已有内容与本笔记的关系|
|---|---|---|
|[Leviathan等，Fast Inference from Transformers via Speculative Decoding，ICML 2023](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)|Algorithm 1，§3.1–3.3，Theorem 3.5|标准拒绝规则和固定历史overlap/TV关系已有。几何长度公式采用i.i.d.简化，不能直接套到任意历史依赖。|
|[Zou等，Variational Speculative Decoding，arXiv:2602.05774v1](https://arxiv.org/html/2602.05774v1)|§4.1、§4.4 Eq.(9–11)，附录B相关段落|明确写出路径条件接受连乘，以及对q路径求期望的接受长度。因此本笔记式(3)属于已有框架。其下界提高不能自动推成真实长度单调提高。|
|[Xia等，Acceptance-Aware Draft Model Training，arXiv:2609.24150v1](https://arxiv.org/html/2609.24150v1)|§2.4、§3.1–3.4|WTV和EAL直接面向窗口接受；§3.1 Eq.(4)明确采用给定prefix后的条件独立处理。固定轨迹overlap连乘不能替代一般历史树上的路径求和。|
|[Chen等，D-Loop，arXiv:2610.06011v1](https://arxiv.org/html/2610.06011v1)|§2、§3.1、附录A.1–A.2、A.4|已把具体并行失配现象与依赖信息损失联系起来，并明确likelihood改善不保证接受长度改善。A.4使用argmax草稿及target匹配规则，不能与本笔记soft q的标准拒绝条件混用。|
|[Sun等，Block Verification Accelerates Speculative Decoding](https://arxiv.org/html/2403.10444)|§3–4、Definition 1、Theorem 1–2，附录B部分；该URL未固定版本|研究改变验证规则后能否提高接受长度。其最优性不能转用为固定标准verifier、改变q时的最优性。未采用其结果推导本笔记命题。|
|[García-García与Williamson，Divergences and Risks for Multiclass Experiments，COLT 2012](https://proceedings.mlr.press/v23/garcia12/garcia12.pdf)|R9已核§2–4；本次复核官方来源，继承该阅读范围|信息结构、多个分布与最优风险关系属于已有决策论工具。不是本项目首次发现信息细化的价值。|

D-Loop、VSD和Acceptance-Aware三项是这次理论核查必须加入的近邻，不能只与Domino/DSpark比较。

## 2. 与前次近邻阅读的连接

- [Domino v1](https://arxiv.org/html/2605.29707v1)：§3–4.2讨论并行计算与前缀依赖修正；§5.3检查训练数据、策略和修正头。该工作不能被描述为只会“补一个前驱”，其GRU汇总前缀。
- [DSpark v1](https://arxiv.org/html/2607.05147v1)：§3.2–3.3使用前缀存活、位置加权TV及调度。其附录A的选择偏差反例与本笔记局部TV反例回答不同问题。
- [SpecTr，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/6034a661584af6c28fd97a6f23e56c0a-Paper-Conference.pdf)：§4–5从最优传输处理多候选接受。不能把其更改候选/验证接口的结论直接当作固定单链修复定理。

以上版本/章节范围由本轮文献角色核验，root复核VSD、Acceptance-Aware、D-Loop的关键原文以及Leviathan/COLT官方来源。角色核验不意味着独立统计证据或穷尽来源覆盖。

## 3. 新颖性结论

|候选说法|结论|
|---|---|
|首次把接受长度写成路径生存和|不成立，VSD已有直接形式|
|首次用TV或前缀权重训练草稿|不成立，DSpark及Acceptance-Aware等已有|
|首次指出并行边缘预测遗漏依赖|不成立，已有并行解码文献及D-Loop|
|本笔记D判别、共同欠量非必要例子以及g−BN界是否有独立新颖性|未确立；都是标准概率/序关系的简短推导，不能凭有限检索未定位同式就称原创|
|本笔记是否能作为本地R9的解释工具|可以，须保留有限历史、强条件与一般TV安全的区别|

对VSD的进一步优化解释，本次仅使用其可直接核对的路径期望公式。正文的接受函数含q，不能不检查这种依赖就把固定接受函数的后验最优结论套给任意drafter优化；本轮没有据此评价整篇论文有效性。

读到的文献足以排除几个过宽的新颖性主张，尚不足以认证一个新的理论贡献。技术比较不代表对这些预印本的录用状态或所有实验作了核验。
