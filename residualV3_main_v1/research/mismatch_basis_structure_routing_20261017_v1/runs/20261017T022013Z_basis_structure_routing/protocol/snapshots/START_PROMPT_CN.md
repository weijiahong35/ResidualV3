# 基底内部结构与上下文选择：执行prompt

复制分隔线以下全文给新的执行chat。本次制文没有运行科学实验；这个prompt被用户明确作为执行指令时，才启动下述CPU工作。

---

你接手ResidualV3的基底研究，仓库`/data/weijiahong/speculators-main-lora`。用户要求只回答四个问题，不许偏题：

1. 不同错误究竟共用了基底中的哪些修正方向？
2. 上下文变化时，只要调整方向组合/幅度，还是现有空间确实不够？
3. 如果组/空间有互补性，可见上下文能否提供选择依据？
4. 先按训练残差聚类，再分别建基底组，不同组是否更适合一类任务？

先读AGENTS.md及本包全部合同：

```
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/README_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/EXPERIMENT_METHOD_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/THEORY_CONTRACT_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/protocol/GROUP_AND_ROUTER_CONTRACT_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/qa/INPUT_AUDIT.json
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/qa/CORE_QUESTION_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/qa/FEASIBILITY_REVIEW_CN.md
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/sources/READING_GUIDE_CN.md
```

先用普通话简报实际可读资产、四问对应实验、可做和未定，再实施，不只再写计划。此前basis_source来源特殊性阴性保持封存；不再用配对、成本比或原头平均收益替代四问。

**授权**：仅在新run写CPU分析代码、读静态W/embedding与旧档、构造规定基底、做规定凸优化、18次闭式小路由拟合、折外评价和收尾。不是训练修复头/LoRA/主体，不加载或前向运行LLM，不采新题，不安装环境，不用GPU/native/TPS，不重启旧stage/controller/收尾，只管理核实身份的自有CPU进程。

新包所有run累计6 CPU进程树小时、16GiB自有RSS、8GiB新增盘、GPU0。单数值worker；所有数值库线程1，import前设置；使用现有`/data/weijiahong/miniconda3/envs/dflash/bin/python`并禁CUDA。失败和门禁都计账，不能换run重置。

固定面板只读R8与basis_reuse缓存：P源、`pool=='fresh_eval'`、q=J raw、i3/8/15、36题/69C/207 C-i/9936自然槽/2571唯一(C,target_row)。三折各24训练题/12折外题，沿用域内SHA排序mod3。所有题已暴露，折外不是新确认。按自然槽→C→位置→题→域权重，重复H保留，EOS当前合法槽保留。不得按头成败、reached、novel、P-fit匹配或修复收益筛样本。

**执行顺序：**

1. 锁样本/源码/数学合同，重验旧档与tensor身份，做小词表核验。制文已核cache/E SHA和元数据，但没有新实验性能实测。
2. 按身份固定首group C1和P唯一target最多C做完整门禁，含fold0完整聚类/基底构造、14基本空间、8删除空间、E512两级方向稳定性、完整TV、ridge和收尾。基础求解约118266个状态/空间，另2571个E方向续算；不能只按2571条估时。按各维度每步吞吐与固定迭代上限保守预测，乘1.5后≤剩余CPU预算80%且留1200CPU秒收尾。超限partial_resource_limited，不改面板/rank/求解上限，不用GPU。
3. 每fold只用训练单位残差做K=4轴向聚类；固定同一簇分别建IN（E512内128方向）、OUT（允许4096维中取128方向）；RND在域/位置/精确唯一历史权重层内置乱簇标签。重复H整体走，保每组人数/权重边际；随机对照换组权重<10%标无信息。不足128秩不补方向、不降rank。完整GLOBAL512和GLOBAL128始终保留。
4. 同p/q/Top256/.8||h||下，求精确全词表forward KL动作项`log(q_tail+Σ_S q exp(Aa))−p_SᵀAa`的球约束最优；按合同保留可行解和FW最优值界。主间隙1e−6、最多2000步，E方向续算1e−8、累计4000步。未收敛不作不可修证据。BF16先位重解释FP32再FP64；tail不重归一、不用合桶TV。
5. Q1报告方向/长度/符号与六类概率错误；对4条训练首轴及4个IN方向组分别从完整E512删除后重解，区分单轴共用和组依赖。不稳定或零动作的方向NA。Q2用OUT对完整E512的最优值界判断当前空间限制；仅胜128不能这样声称。同C/i不同H单列。
6. Q3固定18次ridge：3fold×IN/OUT/RND×X0/X1；X0为现有h/q/i，X1仅增加实际前驱embedding；不输入p、H全文、域/题/slot/cluster标签。按合同λ=.001、训练权重/标准化、无调参。比较训练常量组、训练领域组、X0/X1和oracle；算相同可见输入类的最好统一选择上界。所有路由评价标selection_only_with_oracle_coefficients，不能冒充系数已学会。
7. Q4对每个折外任务把四组全部交叉评价，任务优选组只在训练题确定，三个固定域对交叉优势全报。每fold簇号分开，不把cluster0跨折平均成同一功能。簇里某领域多不等于擅长该领域。
8. 主判别KL .001 nats、完整TV/top1/损伤副报，5000域内整题bootstrap仅描述。保全部负例/NA和数值界；不通过就按预设结果分支写，不换K/rank/剂量/目标/seed或追加训练。四问分别回答，不只交平均收益。

输出仅写：

```
residualV3_main_v1/research/mismatch_basis_structure_routing_20261017_v1/runs/<UTC>_basis_structure_routing/
```

每阶段最多两次有定位技术修复，保存diff/失败/耗时；旧run只读。严格按执行指南交付Q1方向与删除表、Q2空间/历史表、Q3路由/信息上界表、Q4任务交叉表、四张核心图、最多6个身份预选案例、claim–evidence及数值/资源/文件审计。

**做完再按AGENTS审核**：逐问核查是否用了直接证据，是否混淆方向/空间、128/512、选择/修复、簇成分/任务适配。每次改设计先保存核心审核，最终不通过先改结论再审，不用更多实验救报告。第一页≤600中文字直接给四个答案、限制和停止理由。

必要工作未做partial；全部固定工作完成但阴性可complete；求解按合同完成仍有宽区间可complete_with_unresolved。交COMPLETION、NEXT_CHAT、封存清单，退出自有作业，不自动下一轮。
