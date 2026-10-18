# ResidualV3：转向 target 验证与草稿主体的成本

**2026-10-12最新设计与用户约束：[可见mismatch线索 → 修复动作选择](TOP_LEVEL_DESIGN_CN.txt)。** [96提示](../mismatch_expert_pilot_20261012_v1/RESULTS_CN.txt)仅流程试跑。[大样本方案](../mismatch_expert_pilot_20261012_v1/LARGE_SAMPLE_PLAN_CN.txt)现为数学/代码/对话每领域>=10000条实际头训练样本、冻结主体只训修复模块；总量>=42288，3种子，3072独立留出，尚未启动。没有专家路由native AC/TPS；系统更新前不再启动GPU。

**2026-10-12最新研究约束：主体改进围绕投机解码的mismatch。** 已另整理[专用资料与候选路线](MISMATCH_BODY_CN.md)：优先研究拒绝边界改变后草稿表示的复用、廉价修复和必要重算，区分信息复用与实际body工作避免。Lyanna是最直接的body复用参考，D²SD是拒绝边界驱动计算分配的参考；二者也限制了可宣称的新颖性。下文通用预算/三层加两层建议保留为历史候选，不能自动成为当前唯一主线。尚未运行新候选实验或训练。

2026-10-12，北京时间。结论：**优先验证 target 的真实计算压缩；草稿侧先确定 attention/KV 与 FFN 的占比，再考虑更小的上下文预算。** 调小阈值、缩短一个逻辑 prefix、再做一次头融合，都不足以成为新贡献。以下是结合原始论文与现有负结果的候选研究方向，尚未测得新收益，也没有认定创新性已经成立。

本次采用 deep-research 的定向三向扫描：原文机制、适用条件、可验证问题；书目与核验分别保存在 [注释书目](phase2_investigation/BIBLIOGRAPHY.md)、[搜索记录](phase2_investigation/SEARCH_LOG.json)、[来源核验](phase2_investigation/SOURCE_VERIFICATION.md)。这是定向扫描，覆盖有边界；不是系统综述。使用 AI 检索、核验与起草，未声称作者已阅读全文。

## 现在实际卡在哪里

下表来自 [probe32真实服务报告](../../../residualV3-nonthing/results/epoch2_tail_cut_short_iter_20261011_v3/probes/full_head_32_native_cost_snapshot/RESULTS_CN.md)。单位是 **ms/整批/一个decode block**，不是每token。采用 E2 同一联合主体的 FULL15_NOFILL；CUDA事件诊断会改变图，输入是 GSM8K 早期短上下文，不是整段正式测速。C32 尚无相同边界的分段归因。

| 设置 | 草稿query主体 | 原始全词表LM | repair＋采样 | context KV准备 | 完整propose | target forward |
|---|---:|---:|---:|---:|---:|---:|
| C1 T0 | 4.4412 | 1.5448 | 1.2989 | 0.2398 | 8.0068 | 31.8400 |
| C1 T1 | 4.4395 | 1.5470 | 1.3422 | 0.2383 | 8.0503 | 31.8431 |
| 历史C40 T0 | 8.4693 | 4.2797 | 5.3710 | 0.4715 | 19.4653 | 60.6308 |
| 历史C40 T1 | 8.4844 | 4.2893 | 6.1434 | 0.4733 | 20.2557 | 60.5542 |

完整propose已经覆盖其内部阶段，不能把嵌套计时再相加。target forward排除prefill，target词表投影/拒绝采样另计，不能称完整验证周期。上表只支持研究优先级，不能给C32预计TPS。

当前两个容易遗漏的事实：E2草稿五层已经全部使用 **2048-token sliding attention**；vLLM0.26已经把各层context KV投影融合成一次GEMM，并把当前target特征写入已有prefix缓存。它不是每块重算所有历史KV。源码与配置证据在 [LOCAL_EVIDENCE.json](LOCAL_EVIDENCE.json)。因此“加滑窗”“融合五层KV投影”“开始缓存prefix”不能再当新方案。

## DSpark、Domino、DFlash-2各做了什么

| 方法 | target 验证 | 草稿主体 | 可借鉴的部分与边界 |
|---|---|---|---|
| [DSpark](https://arxiv.org/html/2607.05147v1) | confidence估计前缀生存概率，按实际硬件容量曲线分配验证token；生产系统支持变长query的物理执行 | 并行body后接串行修正头；body仍先生成整个块 | §5.2用历史预测提前定容量，§5.3把不同请求有效token压平并保留attention依赖。后者依赖DeepSeek-V4引擎，不能当Qwen/vLLM现成能力。原文也承认固定整块body成本无法由验证截断收回 |
| [Domino](https://arxiv.org/html/2605.29707v1) | 主结果依靠AC增加，让每输出token平均需要更少验证轮次；没有另给target物理压缩机制 | 五层并行body＋轻量因果编码/修正；融合头的实现开销 | 头优化有用，但不能写成省掉body层或target层；原文1.20ms的边界也不能直接套到我们的C32 |
| [DFlash-2](https://inco.ai/blog/dflash2/) | 选择一致路径，提高AC；标准target验证不因此减少每轮必需计算 | 在attention/FFN前后加短卷积，提高块内建模能力，反而新增少量工作 | selector-only是无卷积加头版本；完整卷积方案需要新训练。它优化“单位成本得到多少接受”，不是直接减少原五层body工作 |

**你的判断成立：DSpark截断的目的在于减少低价值的target验证工作。** 我们目前主要测的是“少修头部尾巴”，DEVICE前缀解决旧D2H同步，但target仍固定补齐图；少修的位置数不能换算成target计算节省。DSpark原始调度还要求每个token的准入只使用该token采样前的信息，不能看到完整块之后反向筛选早期token；否则可能引入选择偏差。参见原文§3.2/§5.2/Appendix A。

## 相关论文：优先读哪些

| 论文 | 压缩的位置 | 核心机制 | 对当前工程的适用程度 |
|---|---|---|---|
| [BASS，ACL Findings 2024](https://aclanthology.org/2024.findings-acl.489/) | batched attention / ragged负载 | 请求接受长度不同，拆分attention而批量执行FFN/QKV；由历史接受结果调整草稿长度 | 单A100、多序列证据较近；SPLIT少pad，但多kernel launch可能抵消。它没有证明全部target FFN都可免费变长 |
| [DeFT，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html) | 验证attention/KV IO | 按共享KV组织query、paged读取与负载均衡，避免重复搬前缀 | 原文有A10080GB/8B；主要树验证，t=32是树token数，不是C32。我们原生paged kernel已有基础复用，要先找具体重复读写 |
| [Sequoia，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html) | 验证预算 | 用期望接受长度和实际draft/verify成本选择树大小、深度 | 可借成本建模；当前线性DFlash不能直接移植树结构，增加节点也可能降低吞吐 |
| [OPT-Tree，TACL 2025](https://aclanthology.org/2025.tacl-1.8/) | 自适应验证候选 | prefix-closed累计概率选树；期望增益不足时停止构造 | 不是简单减少target层数；树化需要新的条件特征、mask、采样合同，当前优先级较低 |
| [AdaServe，EuroSys 2026](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf) | 服务级verify预算 | 按请求SLO与全batch token预算选子树，剩余预算提高吞吐 | 适合研究mixed requests分配；原文4×A100、大target/多SLO，不是我们单卡C32固定请求TPS |
| [TurboSpec，原SmartSpec](https://arxiv.org/html/2406.14066v3) | draft/verify预算与闭环调度 | 离线成本profile＋在线acceptance/load反馈，选择提议/验证长度 | 有Graph桶与真实负载反馈；v3题名/机制与v1区分。单纯整块关闭不符合当前尾部优化问题 |
| [MagicDec，ICLR 2025](https://arxiv.org/abs/2408.11049) | 草稿KV读取 | sparse/fixed-budget draft KV；target保留完整上下文校正 | 训练免费但改变proposal q；高batch、长上下文、多卡证据。已有2048窗后还能省多少要测，原文4K/C32也有0.90×负例 |
| [TriForce，COLM 2024](https://arxiv.org/abs/2404.11912) | 草稿上下文读取 | 检索一部分KV做中间草稿，小模型再加速；最终full target验证 | 主要120K+长上下文；我们≤4096可能被检索和新增层级开销吃掉，不宜直接照搬 |
| [H-Spec，2026预印本](https://arxiv.org/html/2609.24197v1) | 草稿KV投影/存储 | 原位读target KV＋末token hidden summary，Mamba-attention并行body，不另建draft KV | 很贴近“主体之外找创新”的问题，并有Qwen3-8B并发实验；但改body和训练，不可把当前E2直接换KV。该论文已明确head可以独立搭配，简单换成我们头不自动构成新贡献 |
| [VocabTrim，2025预印本](https://arxiv.org/html/2506.22694v1) | 原始草稿LM词表投影 | 固定保留校准数据中的高频token，仅计算对应LM矩阵行；target保留完整词表 | 不重训权重，但改变proposal q，代码跨域词表可能掉AC。原文主要greedy，MBSU由参数比计算，不能当作A100实测TPS；本地T1真实稀疏q/残差合同须另验 |
| [FR-Spec，2025](https://arxiv.org/html/2502.14856v2) | 原始草稿LM词表投影/softmax | 频率排序选子词表，物理裁剪LM矩阵；target保留完整词表 | 无需重训；有单A80080GB、EAGLE-2的T0实际TPS。作者单层draft的瓶颈占比不能套到本项目五层body/C32 |
| [DynaSpec，2025预印本](https://arxiv.org/html/2510.13847v1) | 动态草稿LM词表投影 | 训练轻量router选token簇，前两步给较大词表预算，后续衰减；融合索引与GEMM | EAGLE-2、单A6000；v1主要报告AC与词表大小，未给端到端TPS表。router需新训练，逐步AR设计不能直接移到DFlash并行15query |
| [QSpec，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.240/) | 草稿线性层/FFN低精度执行 | 共享W4权重，用W4A4快速draft、W4A16验证并覆写KV | 有A10040GB/vLLM真实并发1–32；原文主要greedy、3proposal，target本身也是量化模型。可借低精度draft思路，但完整配方不等于保留本项目BF16 target |
| [QuantSpec，ICML 2025](https://proceedings.mlr.press/v267/tiwari25b.html) | 草稿权重和KV IO | 草稿4bit权重/KV；target历史KV由上下两组4bit重构到INT8，最近KV保留高精度 | 主要长上下文、同架构self-spec。其INT8 target与FP16“精度接近”不是同概率分布；若借鉴本地草稿量化，必须保留原BF16 target和真实新q |
| [SuffixDecoding，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf) | 草稿主体调用 | 从prompt/历史后缀树生成候选，命中时避免神经draft | 无训练；已核验原文采用greedy验证，不能直接当T1无损实现依据。适合重复的agent/code负载，独立HumanEval/MT首轮未必命中；不是相同q/hidden的精确缓存 |
| [SSD / Saguaro，ICLR 2026](https://arxiv.org/html/2603.03251v1) | draft/target等待 | 提前猜接受结果，在另一个设备缓存下一块草稿 | 主要target4×H100＋draft1×H100；是重叠和额外算力。单A100 C32不能把两条流当免费加速 |

更完整的机制、版本、硬件与读到的范围见注释书目。PEARL、SwiftSpec、PipeInfer也可读，但当前单卡固定权重问题下排在后面。Carryover Drafting需要draft–verify–draft新训练；DFlare是容量/条件建模改善，也不等于固定body减工作。最新长上下文offload论文与当前模型全部驻留显存的设置不同。

## 建议的两个研究问题

**第一优先：在不改变proposal的情况下，能否让设备端有效prefix真正缩小target的执行规模，并保住CUDA Graph？** 两个比较臂修满15位置、使用同一E2权重和真实q，只变化提交验证的prefix及其执行方式。这能把“停修头的成本”与“缩小target的收益”拆开。预算应按实际token数、上下文长度和Graph/tile台阶建模，不能只最小化token数量。历史信息预定本轮总容量，再在设备端维护prefix和映射，是值得验证的工程假设；仍需独立证明采样前因果约束。

这里必须超越旧负结果：iter01/03已经试过ragged/二维Graph，C40仍慢约7%；旧路径有约8ms读回同步，DEVICE消同步后又保留全padding。target_logits_02仅uniform cut10有收益，mixed长度BM128/BM64反而慢4.4%/12.6%。新问题应是**同时去掉同步和大块padding**，并避免gather/细碎GEMM吞掉收益。仅重新运行旧ragged、固定block12/8或再次改阈值不能回答它。

最低可审查证据：实际执行Graph行数下降、attention与MLP/词表投影的实际计算边界下降、无新增GPU→CPU等待；完整decode关键路径和同窗TPS同时改善，并覆盖C1/C32及混合上下文。MASK短了但GEMM仍跑512行，不能算成功。所有实际token/前驱/q/prefix/bonus、请求结束、KV commit/rollback和Graph合同先通过；若改KV实现，新增独立KV字节/有效位置比较，不能继续把发射计数等价称字节等价。

**第二优先：我们的repair头能否降低维持同样吞吐所需的draft上下文预算？** 草稿已经2048滑窗，进一步只读512/1024等预算或少量检索页，会改变proposal分布；用真实q和完整target校正可以保持target分布，但AC可能掉。应检验repair是否能补偿这种低成本body的退化，并把检索、搬运和新的Graph调度成本全部纳入。如果能在多领域/C32保留AC而减少body attention时间，才有独立结果。

这仍只是研究假设：MagicDec/TriForce已经做KV预算，H-Spec已经做target KV复用。可能的新问题在于**repair能力与draft/verify两个计算预算的协同**；普通缩窗、复制DSpark scheduler、换一个head组合，都不能直接宣称原创。要用“仅repair、仅预算优化、两者组合”的消融证明组合的价值；若FFN/LM而非KV占主体主要时间，应转向相应算子，而不是继续做上下文裁剪。新结构/新训练只作为后续候选，本次没有开启epoch3或昂贵native监督。

**草稿侧另一个明确成本项：原始LM词表投影。** 当前草稿词表151,936，probe32中C1约1.54ms、C40约4.28ms；这个投影不是repair头，不能因为repair已优化就忽略它。VocabTrim给出无需重训的固定词表裁剪，但它改变q、牺牲一部分覆盖率，且原文报告的MBSU是基于参数比的理论指标。可检验repair是否能补偿更小词表的退化，使用既有训练/独立校准数据确定候选token，不拿HumanEval/MT-Bench测试生成反过来选词表；target仍执行完整词表验证。T1必须重建截断后真实q及全词表拒绝残差，不能沿用原始全词表q。它与旧target_logits_02按query行压缩是不同问题，也不是现成新贡献。[原文§3–4.1](https://arxiv.org/html/2506.22694v1)、[补充核验](phase2_investigation/VOCABTRIM_VERIFICATION.md)。

这里还需要覆盖FR-Spec与DynaSpec：前者已经做无需重训的频率子词表，并报告T0实际TPS；后者已经做按上下文、按位置分配词表预算，需要训练router。DynaSpec关于动态集合优于静态集合的定理依赖理想选择器，不能当学习router必然提速的保证；原文宣称两条CUDA流可隐藏router开销，也需要在本地C32关键路径重测。因此“动态词表”或“尾部逐渐减小词表”本身都已有相关工作。尚可研究的是repair对低成本proposal退化的补偿是否改变最优预算，需要同主体消融与真实TPS证明。[FR-Spec原文§3–4](https://arxiv.org/html/2502.14856v2)、[DynaSpec原文§3–5及A.2](https://arxiv.org/html/2510.13847v1)、[补充阅读边界](phase2_investigation/ADDITIONAL_PRIMARY_SOURCES.md)。

**若主体主要耗时在FFN/线性层，低精度draft比缩KV窗口更对症。** QSpec有真实A100 C32服务证据；QuantSpec同时展示权重与KV低精度的取舍。但两者发布设置都包含量化target，不能直接作为保留当前BF16 target的完整无损方案。研究候选是只转换E2草稿推理副本，检查repair对量化误差是否有补偿，再由完整BF16 target与实际q校正。量化会改变hidden、raw概率与辅助统计；不是同q的kernel替换，且A100 C32×16query的矩阵规模与AR逐token不同，需计入量化/反量化、Graph和AC损失。当前只整理文献，未转换权重或运行该候选。[QSpec原文§3与Appendix A.4](https://arxiv.org/html/2410.11305v3)、[QuantSpec原文§4.2](https://arxiv.org/html/2502.10424v1)、[补充核验](phase2_investigation/QUANTIZATION_VERIFICATION.md)。

服务归因的重要背景还有MLSys2026的[Speculative Decoding: Performance or Illusion?](https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html)：官方摘要描述了vLLM中跨方法、负载和batch的实测，发现target验证占主要时间，AC随位置、请求与任务变化。本次仅阅读官方摘要；它支持补C32实际服务归因的必要性，不能代替本地profile，也未作为可直接实现的加速方法。

## 明确的下一步：研究动态主体预算，而非增加模块数量

当前头有三领域收益，但“修正头”“头允许浅主体”“按confidence截断验证”都已存在。DSpark §4.3.2在Qwen3-4B、7proposal设置下展示两层DSpark的AC超过五层DFlash；H-Spec §3.3.3明确其主体可配不同修正头。因此，前文“repair能否补偿便宜草稿”的静态实验只是机制探针，不足以单独认定新贡献。[DSpark](https://arxiv.org/html/2607.05147v1)、[H-Spec](https://arxiv.org/html/2609.24197v1)。

建议主线是一个待检验问题：**是否可以利用请求与位置的差异，联合分配草稿主体和target验证的真实计算预算，并在同样吞吐下使用更少计算，或在同样成本下产生更多接受token？** 相比固定整块五层body，控制器要在主体工作尚未全部支付之前决定预算；头是已有的纠错组件。与DSpark的区别需要体现为实际削减body工作及联合预算优于仅verify调度，不能只把其scheduler移植到vLLM。

已有q45/latent辅助头的标签来自teacher-prefix argmax一致/analytical overlap，不能直接把其输出当真实rollout的prefix生存概率。初期先利用已有测量和轻量诊断确认决策信息是否有用，不以该路线为由新增昂贵native监督。

一种后续架构候选是：所有位置先做共享的浅层计算，只给有价值的前缀位置追加深层计算；随后按真实出口分布执行repair与target校正。这里的“浅层+追加层”是设计例子，不是已经训练可用的E2模型。controller决策必须满足因果约束；不能根据未来已经采样的候选反向筛掉早期token。深层query必须物理减少，稠密计算后仅加mask不成立。是否保留前缀依赖、出口hidden是否适合现有head、KV是否正确，都需独立验证。

普通早退也已有LayerSkip等工作；本次仅核验其官方摘要所述layer-dropout/early-exit训练，不声称新候选已与全部动态深度工作完成查重。未经出口训练直接跳E2层的失败，不足以否定专门训练的预算架构；成功也不能替代真实端到端结果。[LayerSkip，ACL2024](https://arxiv.org/abs/2404.16710)。

| 顺序 | 做什么 | 必须回答的问题 |
|---|---|---|
| 1 | 对现有E2补C1/C32混合长度服务归因：target attention/MLP/LM、draft attention/FFN/LM、repair、prepare/cache、stream等待 | 哪个成本实际可压缩？旧C40早期计时不能决定C32新实现 |
| 2 | 先把相同有效prefix的target执行由固定padding改成真实紧凑执行，保持proposal策略不变 | 无新增读回等待时，真实行数/关键路径/TPS是否减少？这是执行基础，单独移植不自动等于算法创新 |
| 3 | 根据第1步只选一个body预算轴做机制探针：attention预算或深层计算预算；FFN/LM占主导则转向对应算子 | repair是否改变成本—AC曲线，值得进一步设计动态预算？不同时混入量化、词表、深度多项改动 |
| 4 | 仅在有成本空间后设计动态主体/验证联合预算，并与最佳固定预算、仅verify调度、仅body压缩比较 | 联合方案是否有独立收益，而不是模块简单叠加或调参收益？ |

评估以独立单训DFlash作为正式基线，保留当前FULL15_NOFILL；同主体开/关头只作机制消融，不能替代正式基线。先新冻结代码、实际token/q/prefix/bonus/KV/Graph合同与16/80短测，再做稳定性复核；没有截断净收益不追加正式长测。新预算会改变proposal q，T1必须使用实际新q与完整target校正。需要出口训练时单列新结构实验，不默认开启epoch3，不新增昂贵native监督。上述是路线建议，本次未启动新GPU测试或训练。

## 当前执行状态

三领域C32补测于2026-10-12 02:12北京时间全部完成，最终审计通过：12正式窗、6252请求，182份源码及10份权重/依赖核验、全部隔离/Graph和1563题T0输出一致检查通过。本批GPU服务、控制器及报告器于02:13宿主核验均已退出；磁盘`du -sh /data/weijiahong`为496G。

仅比较E2独立DFLASH与联合FULL15_NOFILL，不包含本文候选新机制。数量GSM8K1319、HumanEval164、MT-Bench80（首轮），T0/T1、max2096/context4096、KV44GiB，与原C40原题及顺序相同。之后高并发默认C32，历史C40保留。

| 数据集 | C32 T0：AC / TPS增长 | C32 T1：AC / TPS增长 |
|---|---:|---:|
| GSM8K | +13.70% / +6.86% | +18.75% / +11.26% |
| HumanEval | +14.46% / +9.43% | +18.94% / +14.89% |
| MT-Bench | +8.48% / +3.17% | +13.47% / +7.23% |

[完整C32报告](../../../residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/RESULTS_CN.md)、[最终审计](../../../residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/FINAL_AUDIT.json)、[本批进程退出快照](../../../residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/parallel/FINAL_PROCESS_SNAPSHOT.json)。此前GPU1/7的smoke因两臂切换前出现外来进程占卡已归档，不计入正式成绩；各领域有效正式两臂均在GPU4同卡隔离完成。

下一步实施候选之前，先补C32实际混合长度服务的target、draft attention/FFN/LM、prepare/cache与stream等待归因。现有probe32边界只支持候选优先级，不能用内核时间相加预测新吞吐，也不能宣称repair已达到最优实现。
