# 执行指南

## 0. 工作范围与读取顺序

仓库 `/data/weijiahong/speculators-main-lora`。先读仓库 `AGENTS.md`、本包README、研究计划、资料指南、设计JSON、两份审核。新run为本包 `runs/<UTC>_mechanism_discovery/`。

目标是实际完成旧数据诊断与候选取舍，不能只再写计划。所有数据已暴露，结果均为回顾性。新训练、加载模型、新题采集、原头改造、native/TPS任务不在范围内。

唯一主要概率来源：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

先核对该run的 `COMPLETION.json`、`protocol/SAMPLE_MANIFEST.json`、`code/collect_r8.py`、`code/common.py` 及每C的 `META.json`。只读旧代码；不要直接运行会写入旧ROOT的模块。需要的纯数值函数复制到新run并记录来源SHA。

## 1. 概念和面板

- C：本block开始前的固定token上下文。H：本block中当前位置之前已生成的token序列；不是hidden。
- p：target在同C、同H处的完整下一token分布。
- q_D/q_J：独立DFlash主体/联合主体的raw分布，取同C和位置i。
- q_F：联合主体加冻结原头后的完整分布。F来源为该完整策略生成的H，不是J raw自采样。
- P/D/F来源：target、独立草稿、联合草稿加原头分别生成的block内前缀。

固定面板为R8 `data/fresh_test` 全36题、69C、P/D/F的eval槽、位置1/2/3/8/15，D/J两种raw。`fresh_test`只是旧目录名。禁止P-fit匹配、novel、reached、accept、头成败过滤。自然EOS位置若合法且有完整p则保留，EOS后无效位置不补采。不同source的同slot不是同一条H，不跨来源配对。

**代码中的精确筛选是 `record['pool'] == 'fresh_eval'`，不是字符串`eval`；排除`fresh_fit`及dev。** 这里的eval是简称。保留每C输入身份、H与target_row映射检查；不能只按行号假设不同C共享target缓存。

作者阶段元数据快照：anchor1共36、anchor65共33；每source×position有3312条eval记录，合计49680条，展开D/J为99360行；选定eval涉及12157个唯一target行、2611个F head行。执行时重核合法性，不把快照当作强制有效分母。单位是36题，不能把这些行当独立样本。

每格保留expected、valid、invalid、missing/NA，以及题/C/自然槽/独特H数；列明每个排除原因。档案缺字段就NA或阻断受影响分析，不从其他来源拼接、不补题。

## 2. 数值与内存合同

每C有 `raw.npz`、`target.npz`、`F.npz`、`META.json`。前者D/J数组按i−1索引；p用记录的target_row；F用F_head_row并核对position/previous。raw存在J_hidden不等于存在D_hidden，本轮不新训hidden探针。

logits是uint16的BF16位模式，正确流程为：转uint32并左移16位→view FP32→转FP64→稳定的全词表softmax。不能把uint16数字当logits、不能用FP16解码。沿用R8分布定义，要求finite且sum(p)偏离1≤1e−12；TV在[0,1]。log概率使用稳定log-softmax；熵按0log0=0处理。tie统一选择最小token ID，TopK边界并列按token ID升序。

全词表约151936，以实际shape核实。完整p仅有五个位置；其他位置sampled-token标量不能补成完整分布，也不能拼造反事实连续接受率。

逐C读BF16数组，按唯一target_row/F_head_row归一化并复用标量；小批次处理FP64概率。禁止全库FP64矩阵。NPZ的mmap参数不等于能按成员行映射。大npz保持原位只读，约8.76GiB旧概率档案不整份复制。只保存必要标量、聚合、合同、少量案例与审计。

测试限于实质数值风险：BF16位模式解码、极端logits归一化、并列排序、重复H权重、自然EOS资格、不同来源不误配。用小型确定性fixture，不加载模型。

## 3. 固定指标与汇总

### A：raw错误图谱

每状态固定保存：

1. TV(p,q)=0.5×sum(abs(p−q))。
2. a=argmax p，b=argmax q；a=b；p(a)、q(a)、p(b)、q(b)。token ID只作可追溯索引。
3. p/q前二概率差；p/q熵，单位nats。
4. sum_{v∈Top16(q)}p(v)、sum_{v∈Top256(q)}p(v)；a是否在两候选集内。

预定表只按source×body×position及top1一致/不一致分组。每个分组同时报频率、均值及“分组频率×该组TV均值”对总体TV的贡献；两个top1组的贡献应加回总体均值。不要只展示少数高TV组。

汇总顺序固定：每个source/body/position内，C中自然槽等权→题中有效C等权→域内题等权→总体三个域等权。top1子组贡献使用这一层级下的联合权重，不能给子组重新归一后冒充总体贡献。另报告子组内条件均值，明确分母。

重复H保留自然频数。预定敏感性只改C内为独特H等权，其他层级相同；输出全量与等H结果，不能挑有利版本。跨位置或来源的差异仅是该面板的描述，不能直接叫误差累积、容量或分布迁移因果。

不新增确认性显著性检验。保留每题方向、每域汇总、覆盖与NA；不以大量槽数制造精确CI。

### B：监督差异的描述性诊断

固定每C/i/source，使用合法eval槽的自然频数，令：

```text
m(v) = mean_H p_H(v)
r(v) = mean_H 1[argmax p_H = v]
```

对D/J保存TV(m,r)、TV(q,m)、TV(q,r)、TV(q,r)−TV(q,m)，及对应有效槽/H数。再按题/域层级汇总，等H版仅为上述预定敏感性。

B需在每C内按i/source在线累加完整m/r向量及自然槽权重；不能从A的标量表重建。完成对应比较即释放向量，无需落盘全词表中间矩阵。

m不是平均TV最优解；TV(q,m)不等于mean TV(q,p_H)。r也不是本项目真实训练标签分布，更不是D-PACE加权训练的已知最优。i=1可能只有一个H，此时m/r之差是概率硬化的数学结果，不能当新发现。比较只为提出或削弱一个明确预测，不证明硬监督造成当前误差。

### C：最终选点封存后的原头连接表

预处理可在首遍读取时计算并单独封存F标量，以避免重复IO；保存代码/文件SHA与访问日志，在D1/D2完成并写入 `protocol/FINAL_SELECTION.json`（最多1个或0个）之前不查看其结果。若隔离做不到，推迟到最终选点封存后再计算。

最终选点封存后，只按同C的F-source eval记录，用target_row和F_head_row配对p、q_J、q_F。固定报告全F面板、每个冻结场景内及场景外的：覆盖、TV(raw)、TV(head)、配对TV改善、数值容差1e−7下的受损率。沿用A的权重和等H敏感性。不得为此改变场景阈值，D-only候选如无J对应则NA。

这是一张固定附表，已包括在预算内，不占候选D1/D2的6项名额，也不开放第三项候选诊断。它不作D2来源证据，不能依结果重选候选。结论仅限F生成历史上的局部概率变化，不外推J自身轨迹、native接受或TPS。

## 4. 文献与候选锁定

按[资料指南](sources/READING_GUIDE_CN.md)完成问题—近似—证据—方法矩阵，引用原文具体章节。旧资料提供索引，正式决定前重读最接近候选的原文；未获取全文则记录待核，不补写内容。

A/B后生成0–3候选。候选卡必须包含：条件、错误方向、近似环节、至少两种来源、必要预测、不同预测、可用/缺失字段、可否定条件及与近邻的具体差异。范围或指标还没定义清楚的卡不能进入分析。

将候选卡、D1/D2合同、代码/输入访问范围、结果判别规则写入 `protocol/CANDIDATE_LOCK.json` 并签名，随后才读取新增诊断结果；D1/D2完成后封存 `protocol/FINAL_SELECTION.json`，最后解封C表。这是探索过程的访问记录，不能称新鲜留出或完整预注册。D1可引用A/B既有结果，并明确已见。D1必须可能在满足场景定义的样本上失败，筛选条件或概率恒等式保证的结果仅作描述。

每候选最多两项诊断，每项是冻结的“预测—比较—主指标”组合。禁止追加阈值搜索、域筛选、额外预测器、重新求共同修复oracle或改历史长度。缺失的D2写unavailable，不能把缺数据写成反证。所有候选及失败结果同表保留。

跨域门槛和五类结论遵守研究计划。满足跨域门槛也不自动成为创新；最近邻已覆盖问题和边界时，淘汰创新表述但保留为竞争解释。

## 5. 性能门禁与资源账本

所有run累计上限：6 CPU进程树小时、8 GiB新增磁盘、16 GiB自有RSS，GPU=0。失败、技术重试、临时文件计入；换run不重置。单数值worker，BLAS/OMP/MKL/HiGHS等所有数值线程1，在import前设置。无GPU预约、无模型加载、无环境安装。

性能样本在读取科学结果前由元数据确定：①group_sha256排序首题的C1；②选定eval唯一target行数最多的C，并列按group_sha256、anchor排序。相同则仅一个。测试完整A/B流程及C的封存标量提取，记录CPU、RSS、输入字节和输出大小；这些题保留在回顾性全量面板。

用每唯一行和每输入字节的较保守实测吞吐，估算完整扫描及最多6项诊断，乘1.5余量；总估计不得超过剩余CPU预算的80%，留20%给审计与收尾。记录估计依据，不能承诺六小时必完成。候选锁定时按其实际计算重做一次预算核对。

不满足时可优化IO或复用核对过的标量，每阶段最多2次有定位的技术修复，不改科学面板。仍超限则 `partial_resource_limited`，交付已完成内容和具体缺口；未测候选标未测。不能选便宜/阳性的子集写complete。

每阶段记录进程树CPU、峰值RSS、新增磁盘、wall时间；操作只限核实UID/PID/PGID/startticks/CMD/run归属的自有进程，不碰其他控制器/作业。旧资源快照不用于新准入。当前不需要卡，“允许共卡”的历史偏好不构成GPU工作要求。

阶段内也须持续检查累计CPU/RSS/磁盘（监督轮询间隔≤5秒），每个C及每次大文件写入前检查额度；达到预算即停止自有计算并收尾，不能只在阶段结束后发现超限。

## 6. 交付与停止

至少交付：

```text
protocol/INPUT_MANIFEST.json、ANALYSIS_LOCK.json、CANDIDATE_LOCK.json、FINAL_SELECTION.json
results/RESULTS_CN.md
results/RAW_ERROR_MAP.*、SUPERVISION_DIAGNOSTIC.*、HEAD_CONNECTION.*
results/CANDIDATE_DECISIONS.*、NEXT_CONTROLLED_TEST_CN.md
results/LITERATURE_MATRIX_CN.md、CLAIM_EVIDENCE.json
qa/CORE_QUESTION_REVIEW_CN.md、NUMERICAL_AUDIT.json、RESOURCE_AUDIT.json
COMPLETION.json、NEXT_CHAT.md、FILE_MANIFEST.json
```

没有候选时NEXT_CONTROLLED_TEST明确“无合格候选，不建议自动实验”。选中时按模板给一个检验，资产或成本不明确写待定，不伪装成可直接启动协议。最多6个案例：按候选冻结条件中每域group SHA及C/位置/slot固定排序，保留反例；不挑头改善最大样本。

报告第一页≤600中文字，回答：找到哪个具体问题；依据是什么；与最接近论文差在哪；哪些仍是假说；下一步唯一检验或停止理由。正文给全体候选、负例、分母和文件链接，不只交均值表。

`complete`要求三项固定分析、适用的候选诊断、文献核查和收尾都完成。0候选可complete；字段缺失、资源不足或必要工作未做则partial并列明，不把结果阴性当partial。结束前按AGENTS核心问题条款再审，只有主张超出证据时改主张，不为阳性改实验。退出自有作业，不自动下一轮。
