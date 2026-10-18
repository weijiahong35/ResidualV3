# Mismatch 如何服务投机解码的草稿主体

最新架构见[顶层设计v2](TOP_LEVEL_DESIGN_CN.txt)。[96提示](../mismatch_expert_pilot_20261012_v1/RESULTS_CN.txt)仅流程试跑。[大样本方案](../mismatch_expert_pilot_20261012_v1/LARGE_SAMPLE_PLAN_CN.txt)按用户要求每领域>=10000条头训练样本、冻结主体只训头，总量>=42288、3种子，尚未启动。body候选保留为文献问题，不混入这轮头训练；系统更新前保持GPU停止。

2026-10-12，北京时间。用户最新要求：主体改进应围绕投机解码本身，最好从 mismatch 出发。此次资料整理聚焦**拒绝边界改变之后，哪些计算可继续利用、哪些需要修复或重算**。通用层剪枝、量化不作为这次主线。

结论：最直接减少主体重复前向的参考是 **Lyanna**；最贴近块内拒绝边界的计算分配是 **D²SD**；DFlow、ReTrace、Carryover提供跨轮信息处理方法，但主要提高草稿质量，原主体仍完整计算。验证反馈指导稀疏KV的SparseSpec/SpecAttn则实际减少attention读取。将这些机制与现有repair组合是否有效、是否有独立创新，均未验证。

## 先分清 mismatch 的对象

| 对象 | 具体差异 | 不能混为一谈的情况 |
|---|---|---|
| 分布 mismatch | 同一实际prefix下target p与proposal q不同 | T1随机拒绝不是语义严重错误的证明；T0 argmax不同也不是TV本身 |
| 条件/prefix mismatch | 旧草稿路径的前词，被真实采样或target替换token改变 | 原parallel body表示缺少新条件，与旧verifier suffix包含错误前词，是两种不同失配 |
| 训练–部署 mismatch | teacher feature/teacher-prefix与实际draft自反馈输入不同 | feature自反馈、token rollout和多轮native teacher监督的成本/效果不同 |
| 计算分配 mismatch | 计算均匀分给整块，但接受只保留首次拒绝前的连续prefix | 少位置/少层并不保证少GPU时间，更不保证更高TPS |

前两种区别的例子：草稿`a b x d e`，target保留`a b`并把`x`替换成`c`。旧verifier在后部产生的表示含`x`条件，不能当`a b c`下的正确target状态；但它仍可能为后续proposal提供信息。另一方面，parallel draft body在块内采样之前就完成，未必读过具体`x`；它的问题可能是还没吸收新确认的prefix。不能把所有旧hidden都称“已被错误token污染”。[DFlash §4](https://arxiv.org/html/2602.06036v1)、[DFlow §3–4](https://arxiv.org/html/2609.06498v1)。

## 优先阅读的机制

| 论文与原文位置 | WHY：投机特有的问题 | HOW：怎样处理 | WHAT：究竟省什么、需要什么 |
|---|---|---|---|
| [Lyanna / Make Every Draft Count](https://arxiv.org/html/2602.21224v1)，§4–7.3 | first rejection后已付出的draft hidden计算被丢弃 | 重hidden轨迹与token选择解耦；正确bonus重新融合旧raw logits，重采样候选，再与正常候选合并verify | 重采样候选免新draft Transformer；需新body/late-fusion训练。主T0、Llama2/Vicuna、A800/H800，不能直接迁移本E2 |
| [D²SD](https://arxiv.org/html/2606.04446v1)，§3–4.2 | 首错位置限制连续接受，统一增加block无助于难点 | 估计首次拒绝边界，第二个variable-prefix drafter在top-K边界续写，cascade验证 | 增加而非省掉原body计算，改善计算投入位置；需第二drafter训练。H200、K4/block16，T0/T1表不替代本地树采样合同 |
| [DFlow](https://arxiv.org/html/2609.06498v1)，§4–6；[ReTrace v2](https://arxiv.org/pdf/2608.29748v2)，Eq4–9 | 下一轮重叠位置失去旧拒绝轨迹的信息，新prefix又不同 | 用真实拒绝/替换条件修正、移位旧verifier或draft表示，作为下轮输入 | 原完整body仍运行＋relay/gate；提高AC，不是每轮少body层。均需相应训练 |
| [Carryover Drafting](https://arxiv.org/html/2609.14717v1)，§2–5 | rejected target state有信息但不能进入committed历史 | 临时只读draftKV保留一轮拒绝表示，标记其身份 | 完整body＋额外KV读取；需适配训练和branch验证。信息复用不是精确新prefix KV缓存 |
| [FlexDraft](https://arxiv.org/html/2605.20022v1)，§3–4.3 | 提前生成下一轮时不知道接受长度与bonus | bonus条件校准；target末层attention适配draft，大小batch选择多分支并行或已确认prefix单分支执行 | 减少重复分支及共享prefix计算；新attention/MLP训练，非现五层E2即插；A100、主bs1与温度附录 |
| [SparseSpec](https://arxiv.org/html/2512.01278v1)，§4.1–5.1；[SpecAttn 2602.07223](https://arxiv.org/html/2602.07223v1)，§3–5.1 | 草稿KV工作集与变化的target上下文不匹配，独立检索又有开销 | 从正常验证提取attention信息，指导下轮稀疏draftKV；SpecAttn只收集两行以压成本 | 实际减少draft KV读取/attention；training-free self-spec，完整target保留。长推理/H100，当前2048窗与≤4096上下文需另验；不是同名有损target稀疏项目 |

其他有用的边界资料：[DistillSpec](https://proceedings.iclr.cc/paper_files/paper/2024/file/8766fbc68e1ed1cdef712ce273e0a363-Paper-Conference.pdf)针对p/q；[HASS](https://arxiv.org/html/2408.15766v3)、[EAGLE-3](https://arxiv.org/html/2503.01840v3)针对feature自反馈；[LongSpec，ACL2026](https://aclanthology.org/2026.acl-long.83/)针对位置曝光与targetKV接口。它们不能统一解释成“增加on-policy就好”。[DRelay](https://arxiv.org/html/2610.01439v1)主要改候选/实际prefix修复selector，不能把同类head改名当body创新。逐篇WHY/HOW/WHAT、阅读范围、正式venue与未确认项见[书目](phase2_investigation/mismatch_focus_20261012/BIBLIOGRAPHY_ALIGNMENT.md)与[拒绝前缀核验](phase2_investigation/mismatch_focus_20261012/VERIFICATION_PREFIX.md)。

## 必须保留的负结果

- **Lyanna：省body不等于净提速。** §7.3的B2消融只开重采样而没有verify fusion，Llama2/Vicuna吞吐分别下降16.1%/20.6%。额外验证可以吃掉全部草稿节省。[原文](https://arxiv.org/html/2602.21224v1)。
- **DFlow：条件训练组织会决定成败。** §6、Qwen3-1.7B T0中，Domino AC4.02；加DFlow但不使用实际draft rollout降到2.22，使用后4.50。这不是固定E2低成本添加上一轮hidden的成功证据。[原文](https://arxiv.org/html/2609.06498v1)。
- **HASS：feature alignment与token rollout不是同一件事。** Appendix A.2中保留训练数据token、只对齐feature最好，更多draft采样token反而下降。[原文](https://arxiv.org/html/2408.15766v3)。
- **SpecAttn：反馈并非免费。** §3.3收集全部12query logits时原attention开销显著增加；只收两query仍有代价。其长上下文收益不能换算成本地C32收益。[原文](https://arxiv.org/html/2602.07223v1)。
- **本项目已有负结果。** 10/04历史复审第七节记录验证边界p−q投影/PAST_Q/margin反馈未稳定胜出；此前on-policy约慢8倍且无一致收益。不能将“拼接上一轮mismatch”或“重跑真实前缀监督”当未做的新实验。[历史证据](../mismatch_design_reassessment_20261004_v2/REASSESSMENT_CN.txt)。这是历史版本的约束，不把该文的旧head结构描述自动套当前E2。

## 与当前 E2 的接口，以及候选研究问题

当前冻结[repair/model.py](../../../residualV3-nonthing/code/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/native_runtime/repair/model.py)在`prepare`中准备raw/hidden特征，`rollout`随后用真实采样的previous token逐位置修复。块内计算已经有“重body先算、轻head处理实际前驱”的分工；这只是已有接口事实，不证明跨轮缓存可以直接精确复用。源码SHA与本地历史身份存于[ROOT_EXTRACTS.json](phase2_investigation/mismatch_focus_20261012/ROOT_EXTRACTS.json)。现有contextKV也已增量缓存，候选要省的是query body/原始LM计算，不能再次包装缓存历史prefix。

**候选问题：在block-parallel草稿中，已观测的拒绝边界和新bonus能否决定重叠位置应复用、廉价修复还是重新计算，从而减少下一轮body工作，并避免新增整轮target验证？** 这是基于本次资料提出的未测假设，不是已确认原创。

可能的动作界面（仅设计）：

| 动作 | 条件与对象 | 要证明的成本变化 |
|---|---|---|
| 复用 | 仍有预测价值的旧draft raw/hidden作为新proposal的输入，重新条件化和采样 | 这些位置实际免body前向，不只是多加缓存 |
| 廉价修复 | 将旧表示与真实bonus/已确认前缀结合，再由轻head形成实际新q | 修复和存储成本低于省下的body/LM成本 |
| 重算 | 新位置或旧表示退化明显的区域，运行正常主体 | 改变执行行数/读取范围，避免全部稠密计算后mask |

“退化明显”不是已存在可靠的预测器。T1拒绝可来自概率耦合，不能把一次拒绝直接等同大语义变化；仅embedding距离也不足以证明可复用。现q45/latent标签来自teacher-prefix argmax/analytical overlap，不直接当真实prefix生存概率。

Lyanna已经做canonical hidden重采样；DFlow/ReTrace已做mismatch-conditioned输入；因此仅把这两者与本头拼接并不自动新。可能的差别必须由**parallel body中的实际部分工作避免、mismatch驱动的复用范围、C32混合请求执行与无新增verify轮次**来证明，并与固定范围复用、始终重算、仅信息反馈比较。论文优先级由实验决定，不在此次资料整理中承诺A会。

## 下一轮先测什么（尚未执行）

1. 用新冻结诊断收集少量实际round的首拒绝位置、bonus、跨轮重叠位置与body/LM成本；分别处理T0/T1、C1/C32。只诊断既有推理，不新增teacher监督训练。
2. 在相同新prefix上比较旧raw/hidden经现头重新条件化与完整重算的proposal变化和执行成本。先回答是否有复用价值，不训练复杂controller，不据hidden相似度宣布成功。
3. 候选先验实际token/前驱/真实q/bonus/KV与Graph合同，再16/80短测。必须计入缓存、映射、重采样及target额外工作；若没有净收益不追加截断长测。

缓存draft表示可作为近似proposal，不能当新prefix下精确target状态；T1的正确性取决实际proposal q、正确条件的target p与残差采样。旧target suffix KV不得commit，旧已抽样token不能在改变条件和选择规则后仍冒用旧q。发布论文的T1表格不能替代本地实现合同。[投机采样基础](https://proceedings.mlr.press/v202/leviathan23a.html)。

## 检索与证据边界

这是deep-research的WHY/HOW/WHAT定向扫描，使用AI检索、原文核验与整合，不是系统综述、不宣称用户已阅读全文。原始论文/正式论文集提供机制证据；arXiv存在性不等于同行评审或独立复现。已复用10/04原始PDF，未重复下载大批资料。Semantic Scholar访问降级，未虚构API匹配；逐条scope和metadata状态保留。

检索与提取：[SEARCH_ROOT.json](phase2_investigation/mismatch_focus_20261012/SEARCH_ROOT.json)、[SEARCH_ALIGNMENT.json](phase2_investigation/mismatch_focus_20261012/SEARCH_ALIGNMENT.json)、[SEARCH_PREFIX.json](phase2_investigation/mismatch_focus_20261012/SEARCH_PREFIX.json)、[ROOT_EXTRACTS.json](phase2_investigation/mismatch_focus_20261012/ROOT_EXTRACTS.json)、[VERIFICATION_PREFIX.json](phase2_investigation/mismatch_focus_20261012/VERIFICATION_PREFIX.json)。本次没有修改E1/E2、运行GPU测试、训练新模型或恢复旧队列；候选收益与创新性均未测。
