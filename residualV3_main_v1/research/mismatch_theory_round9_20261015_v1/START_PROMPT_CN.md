# 第九轮启动 prompt

复制分隔线以下全文给执行chat。本轮研究失配的可修边界，使用已有完整概率存档；不是训练或预测原头成败。

---

你接手ResidualV3第九轮失配研究。仓库：`/data/weijiahong/speculators-main-lora`。收到后在以下固定范围自主实施，交付实际分析，不再只写下一轮计划。

先读仓库`AGENTS.md`及：

```text
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/README_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/THEORY_PLAN_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/sources/READING_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/qa/CORE_QUESTION_REVIEW_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/qa/FEASIBILITY_REVIEW_CN.md
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/qa/INPUT_AUDIT.json
```

**核心问题是失配本身：什么样的失配能共同修好，为什么，何时必须区分历史才能避免取舍。不能替换成“同TV下预测原头成败”，也不能再以某个探针平均有效作为答案。** 每次写完或改动执行方案，审核是否直接回答核心问题；不通过就重写再审，通过后才执行。审核和修改记录随run保存，不反复要求用户重申方向。

R8已完成，唯一主要概率来源：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

R1已有KL分解；R4已有受限操作共享/安全修正；R5–R8已有共同探针与迁移。**本轮新增问题是：完全放开修复表达能力，仅因共用同一个预测，究竟有什么不可兼顾限制。** 不重新包装旧oracle，不宣称识别模型规模或训练因果。

固定同C/i、真实历史族{p_H}和raw q_D/q_J，修复允许任意完整词表合法分布。比较四种信息：I0全部历史共用；I1按前驱；I2按最近两token；IH完整历史只作数学参照。p/q和频数始终固定，不训练头。

**必须同时求三个量：**

1. A：最大平均TV改善，允许部分历史损伤。
2. M：能保证每条历史至少改善多少。
3. S：保证每条不恶化时的最大平均改善；数值主表使用S^τ，τ=1e−7，明确只是容差安全。

M≈0不等于任何平均改善必伤；只有收益超过S^τ的认证上界，才可推出至少一条历史必须损伤超过τ。平均最优动作的受损率不是必然受损率。两两可改善不保证整组可改善。

两历史M=(d1+d2−TV(p1,p2))/2可解析验证；多历史求完整simplex的可行见证和优化上界。平均TV最优按指南的共同分位阶闭式计算。允许active+tail LP作外层松弛，但候选必须完整词表评估；合桶安全不等于完整安全。不收敛保留区间，不能写不可修。raw始终为合法零改善fallback。

**固定数据：**全量风险图使用R8全部36题、69C、三来源P/D/F、主位置3/8/15全部合法eval槽，不加P-fit匹配或novel门槛，不看头/探针成败选样本。重复完整H保留自然频数。M/S证书固定每域完整group SHA首4题，共12题，只取已有anchor1、三来源、三个主位置、D/J、I0/I1/I2。所有数据都已暴露，不能称新fresh确认。

全36题另做不同H等权、16次固定同组大小假分组的R对照：I1全组内置乱，I2在真实I1内置乱。单调性和IH风险为0不是发现；i3的I2等于完整H，只作控制。历史内容额外价值的解释看i8/15真分组对假分组。该对照只检验风险，不能替M/S排除所有组数效应。

先完成数学合同、输入身份、固定dev工程计时，再锁源码和执行规则。每个一般信息组×raw的M/S两目标合计60秒solver wall（各30秒，含active阶梯），数值目标间隙1e−4；实质分类幅度ε=0.01。原始连续界、未判定分母、每域覆盖全部保留。主seed2026101591、bootstrap2026101592；5000次整题分层95%区间仅描述，不能包装成未见确认。具体算法、NA和聚合遵守指南。

本prompt授权在新run编写纯CPU代码、读取旧存档、执行全部规定分析与收尾。**本轮GPU预算0：现有数据足够，不加载模型、不采新题、不等待空卡。** 不训练、不增加预测器、不改结构/损失/安装环境，不重启旧stage、controller、parallel、CPU收尾或native/TPS/RNG任务。此前“有卡可共卡”不等于此轮需要GPU。

所有run合计CPU进程树24小时、新增磁盘16GiB、自有RSS32GiB；单数值worker、所有数值库与HiGHS线程1，import前设定。每阶段最多2次有定位的技术修复，失败计账；科学阴性不重试。只管理核实身份的自有CPU进程，旧资产只读。

写入：

```text
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/runs/<UTC>_mismatch_r9/
```

先简报实际读到的R8终态、三个边界各回答什么、可用概率文件与固定面板，然后执行。交付完整A/M/S边界、信息分组与假分组对照、数值见证/上界、全部覆盖、最多6个固定排序案例、理论卡片、claim–evidence、资源/文件审计、COMPLETION和NEXT_CHAT。

第一页必须用简单中文回答：**哪些失配还有共同安全余量；哪些改善幅度确实要求牺牲某些历史；区分历史解除了什么限制；失配的哪些来源仍未识别。** 安全余量仅排除相应幅度下的必要损伤，不能据此判断保留原头实际收益是否也能无损。没有不可兼顾证书就不得声称已识别信息强迫的取舍；若只因证书不够窄，则写未判定，不能说不存在。

所有固定工作完成允许阴性/不确定；必要工作未完成为partial。报告完成后再审核是否回答核心问题，不能只交一张平均收益表。结束退出自有作业，不自动下一轮。
