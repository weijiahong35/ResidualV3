# 机制发现启动 prompt

复制分隔线以下全文给执行 chat。收到后完成固定分析，不再只交下一轮计划。

---

你接手ResidualV3的具体失配机制发现。仓库 `/data/weijiahong/speculators-main-lora`。

本轮问题是：**在什么明确条件下，低成本草稿稳定地犯哪一种错；哪个具体近似环节可能造成它；什么观测能区分其他解释？** 用户需要大背景下的一个小切口。不能把“大小模型不协同”、多看几个词、预测原头成败或探针平均有效当答案。此前“输出适配优先”推荐已撤回，不预设输出层、硬监督或其他来源为主线。

先读仓库AGENTS.md及本包：

```text
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/README_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/RESEARCH_PLAN_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/sources/READING_GUIDE_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/protocol/CANDIDATE_TEMPLATE_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/qa/CORE_QUESTION_REVIEW_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/qa/FEASIBILITY_REVIEW_CN.md
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/qa/INPUT_AUDIT.json
```

已有R1–R9保持归档。本轮主要概率来源是：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

执行四步：

1. 按阅读指南核查一手论文与本地负结果，提取“异常→具体近似→竞争解释→关键证据”。不要只列论文名，也不要把已有方法简化成只补前驱。所有因果判断标清作者主张、实验证据与自己的推断。
2. 实际完成固定raw错误图谱A和监督描述图B：全部36题/69C、P/D/F eval、位置1/2/3/8/15、D/J；不得按头成败、P-fit匹配、novel或reached筛样本。文件名fresh_test不代表新数据。自然重复H保留。
3. 从A/B与文献中生成最多3张候选卡，冻结后每张最多D1必要预测、D2竞争解释不同预测两项诊断。最多6项，不换阈值/筛域/换指标救结果。D2缺干预则明确unavailable，不能用相关性假装确认来源。
4. D1/D2完成后，先在protocol/FINAL_SELECTION.json封存最多1个小切口或零候选决定，再解封原头连接附表C。C不能参与选点或重选；无方法关联就报告缺口。为选中者写一个可区分解释的后续受控检验卡，本包不执行新训练。0个合格候选也允许完成，给出停止理由，不自动换题或追加轮次。

候选必须写清可计算条件、具体错误方向、源码/原文可定位的近似、至少两个来源、可否定预测、档案可测/不可测部分、近邻之外的具体增量认识。即使原头没有收益，候选问题也应独立成立。模型有效不保证一定藏着一个新机制。

D1必须可能在满足场景定义的样本上失败。由筛选条件或概率恒等式保证的结果只作描述，不能当支持来源候选的证据。

五类结论分开：必要预测未观察到、已有工作覆盖创新主张、数据不能区分、局部线索、可进入受控检验。既有论文覆盖不等于它已被证明是当前模型原因。跨域主候选须至少两域各≥4有定义题，且各≥2/3题满足冻结方向；这是探索选择门槛，不是确认性检验。保留所有负例及NA。

数值严格遵守指南：uint16 BF16位模式→FP32→FP64全词表softmax，不能直接数值转换；p=target(C,H)，q=D/J(C,i)。固定TV、top1及对应概率、前二概率差、熵nats、p在rawTop16/256的质量。并列按最小token ID。监督图比较m=自然槽均值p、r=argmax频数及q之间的TV；m不是TV最优，r不是实际训练标签分布，不能证明硬监督因果。

META中eval的精确字段值是`fresh_eval`；只选`record['pool']=='fresh_eval'`，排除`fresh_fit`及dev，不用`pool=='eval'`得到空面板。

每source/body/position：C内自然槽等权→题内有效C等权→域内题等权→总体域等权，另做C内等H敏感性。top1两组的加权TV贡献应加回总体。完整报告expected/valid/invalid/NA及题/C/槽/H分母。EOS当前合法位置保留、EOS后无效不补。不把99360结果行当独立样本。

原头连接C只能按同C、同F-source eval记录的target_row和F_head_row配p/q_J/q_F；不跨来源同slot配对，不叫J raw轨迹。只报全F及冻结场景内外的TV前后差、损伤率和覆盖；无新机制指标。它是三项固定分析之一，不占6项诊断名额，也不能充当来源D2或用于选点。可首遍计算封存F标量，最终选点封存前不读结果。没有可靠隔离就最终选点封存后再算。

先锁基础指标、样本身份和分析代码；按身份预选首个group SHA题C1及元数据唯一target行最多C（并列固定排序）做完整性能门禁，不看科学效果选题。逐C读取旧npz，唯一行归一化后复用，禁止整档复制或全库FP64矩阵。NPZ的mmap不提供成员逐行映射。根据实测唯一行和字节吞吐做1.5倍保守预测，总估计≤剩余CPU预算80%，留收尾余量；候选冻结后重核。仍超限就partial_resource_limited，不能改科学面板或把未测候选写成反证。

**收到本prompt后，授权你在新run自主写CPU分析代码、读旧数据、查原文、做规定诊断及收尾。** 全部run累计6 CPU进程树小时、8GiB新增磁盘、16GiB自有RSS，单数值worker、所有数值库线程1，import前设置；GPU预算0。现有数据用于本阶段，无需等卡。每阶段最多2次有定位技术修复，失败也计账，不换run重置。

不加载模型、不训练、不采新题、不改结构/损失/安装环境、不启动native/TPS，不重启任何旧stage、controller、parallel或收尾。只管理核实身份的自有进程。后续若需训练干预，在检验卡明确范围、资产缺口和成本，不能自行启动。

输出：

```text
residualV3_main_v1/research/mismatch_mechanism_discovery_20261016_v1/runs/<UTC>_mechanism_discovery/
```

先简报实际输入与可做/不可做的判断，然后实施。交付A/B/C实测表、全部候选决策、文献矩阵、最多一个后续受控检验卡、claim–evidence、数值/资源/文件审计、COMPLETION和NEXT_CHAT。第一页≤600中文字回答“找到的具体问题、证据、与近邻区别、尚未证明什么、唯一下一步或停止理由”。

每次改方案以及最终报告前，都按AGENTS检查是否直接回答所声称的问题；不通过就修改再审，保存记录。审核通过仅指能作有效判断，不保证阳性、创新或因果成立。固定工作做完且无候选可complete；必要分析未做则partial。退出自有作业，不自动下一轮。
