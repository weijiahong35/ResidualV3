# 论证架构与待审实验设计

科学问题：跨历史可共同修正的偏差，是否跨两种已训练主体存在，并从 target 历史迁移到两种草稿自行产生的历史？

已知：联合主体内有共同修复，原头也利用了部分共同需求。未知：它是否局限于这对共同训练的组件及 target 历史测度。

设计：同一批新36题（GSM8K/HumanEval/Alpaca各12），同 C 与 i；2个被诊断主体 D/J × 3种历史来源 P/D/F；每来源 K64=fit16/eval48，anchor1/65、主位置3/8/15，1/2作控制。三来源固定同一 target 生成的块前 C；研究块内历史测度，未研究长期 C 漂移。

对每个 m、W，P-fit 至少2种完整 H 时，用 R5 的 u=max(0,min_H(p_H−q_m))、o=max(0,min_H(q_m−p_H))、M=min(sum u,sum o)，构造 a=(M/2)u/sum u−(M/2)o/sum o，M=0则a=0。只用 P-fit 构造两项主研究的动作，不看原头 d 或 eval target。D/F-fit 只构造预定次要同源探针，独立访问。

四个主端点：A_D=P-eval 上 D 探针收益；A_J=P-eval 上 J 探针收益；B_D=D-eval 上 D 的 P-fit 动作收益；B_J=F-eval 上 J 的 P-fit 动作收益。A两端点必须同状态配对。B两个分别是各自草稿历史，不把其差解释为同状态模型效应。

四主量集合是有合格 P-fit W 且完整 H 未出现于同 C/i 全部 P-fit 的 eval 自然频数，称 new_to_P_fit。D/F-fit 是否出现该 H 不影响主资格；不能叫相对所有 fit 严格未见。全部 matched 自然频数另作描述，并分 P-fit_seen / only_other_fit_seen / strict_allfit_novel 三桶精确还原贡献；P-fit_seen 的 M/2 改善是构造保证，不能靠它确认泛化。严格novel（同 C/i、三来源全部fit H的并集之外）另表。缺匹配为NA，不填0；M=0留在估计中。

统计：域内题面bootstrap5000，三域等权、题面等权、anchor/位置等权、W自然频率。四主量98.75%双侧区间；正效应lower>0，实质正效应lower>0.01；每端点每域至少6题有定义且6题有非零动作，否则覆盖不足。固定同source同W比较 m 间差异；跨source的总均值差不自动等于方向迁移变化，另做共同W标准化分解。

参考 reached：P/D/F 均从 BF16 logits 生成 FP64 完整概率缓冲，CPU PCG64 类别采样直接使用该缓冲，之后不再用另一次 softmax 代替 actual generation q。D/F eval 用独立验证随机流对已生成路径进行标准 first-reject 参考验证，保存全部1..15合法位置的 sampled-token p/q、buffer SHA、uniform、accept/reached/EOS/预算；仅主/控制位置保存完整分布。reached含拒绝位，局部量只是原策略到达状态上的探针诊断，不是部署探针后的接受收益。与旧FP32及生产vLLM采样口径不同，不宣称native等价或消除旧警报，不把路径overlap乘积当接受概率。

W标准化：每C/i只取P与对比来源S都具有有定义主状态的W，以P-eval主状态自然频数在共同支持内归一加权；报告双方支持率。分别报同W内收益差与W权重改变，不能把总自然均值差当方向改变。三类q必须分名：diagnostic raw q_D/q_J、generation q_S、probe q_m+a；F来源验证使用q_F，不用q_J或probe。

反解释：联合阳性、独立未支持≠独立不存在；两者阳性≠容量造成或全部方法普适；迁移减弱可能由W权重/共同支持/EOS选择改变，而非方向变坏；相同历史在两来源重复不是独立样本。

共同采集后独立回答两项研究；技术问题可使某项partial，不以另一项效果阴性停止。预算和代码身份限制优先，失败不启新轮。
