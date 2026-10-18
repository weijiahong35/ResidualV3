# R2 理论卡片：失配大小、允许修正与实际收益

此卡片依据[锁定协议](../protocol/PROTOCOL.json)、[一手文献矩阵](../sources/ROUND2_LITERATURE_MATRIX.md)、dev及两份已经一次开启的固定留出摘要。最终更新仅读固定摘要并以[结果正文](RESULTS_CN.md)核验机械范围，没有读原始fresh记录、重新统计、拟合或选择子集。锁SHA为 `8f0e641692524fc698a9c3a3659dc6279f7342a903b14295a5103e0377044c57`。标准数学、已有模块或同名术语缺失都不构成首次发现。

## 第一层：标准工具及其条件

固定同一状态，p为target、q为联合raw、r为实际修复后的完整词表归一化分布。S是同一次raw预测中头实际使用的Top256，U为逐token补集。仅修改S的logits、使用相同温度和支持处理时，`r(v)=c·q(v), v∈U`。**尾部总质量可以变，尾部内部条件比例不变**：`r(·|U)=q(·|U)`。这不是“不能改变任何尾部概率”，也不同于裁剪draft词表令尾部零支持。

在允许S内任意修改的放宽集合中，q(U)>0时：

\[
 D_S^*=\inf_r TV(p,r)=p(U)TV(p(\cdot|U),q(\cdot|U)),
\]
\[
 \inf_r KL(p\|r)=p(U)KL(p(\cdot|U)\|q(\cdot|U)).
\]

一个oracle取 `r*_S=p_S, r*_U=p(U)·q(·|U)`。TV式由尾部L1与候选质量差的三角不等式得到；forward KL式由S/U分组链式分解得到。KL方向是**p||r**，不能互换为r||p，也不能直接移用相反方向的投影结论。[已核链式分解来源](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/184197ca5d5418da2415d37e929860b9_MIT6_441S16_chapter_2.pdf)

p(U)=0时两下限为0；真实q(U)=0且p(U)>0时TV下限为p(U)、forward KL无穷。U内单点q(v)=0而p(v)>0亦会令forward KL无穷。有限日志概率的数值下溢另记，不能伪装成真实零支持。TV/KL的误差形状及排序不同，不把KL改善自动换算成接受改善；受限模型中的这种目标差异已有认识。[LK Losses，§3–4](https://arxiv.org/html/2602.23881v1)

令 `D=TV(p,q), D_h=TV(p,r), A=D−D*_S, Delta=D−D_h`，实际r属于该集合时有 `D_h≥D*_S, Delta≤A`。但oracle放宽了真实rank、dose、有限logit、跨状态共享参数和可见信息约束，**不是当前头可学习、可部署的最优收益**。`gap=D_h−D*_S=A−Delta`混合实际表达限制、学习/优化和信息因素；它不是纯容量误差或独立新机制证据。不能用gap推出扩大模型一定能补齐。[保真与优化的竞争解释](https://proceedings.neurips.cc/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf)

坐标改善 `delta_S+delta_U=Delta` 精确闭合；尾部项仅表示U内坐标L1的变化，不能称为学会了尾部内部结构，也不能转为AC/TPS因果份额。全词表温度校准通常改变U内条件比例，是集合外参照；它低于D*_S时不构成下限失效。

**CPU说明例，不是E2新发现：** S={1}、p=(.60,.20,.20)，q_A=(.78,.11,.11)与q_B=(.60,.38,.02)的raw TV都为.18，D*_S分别为0和.18。相同误差大小可以有不同放宽修复空间。即使A>0，实际头仍可恶化；仅验证不等式不能证明H_R。[合成合同](../analysis/GEOMETRY_CPU_AUDIT.json)

## 第二层：本项目可被推翻的预测

| 假设 | 锁定的新留出检验 | 支持条件与反证出口 |
|---|---|---|
| H_R：尾部条件形状提供超出raw TV、位置、域及覆盖的离线解释。 | 冻结dev的同类ridge：M0已有D/pS/qS及固定少量非线性；M1仅加D*_S。比较i2–15每prompt MSE(M0)−MSE(M1)。 | 95%下界须超过冻结实质阈值0.0007908361。无实质增量、量近0或与覆盖冗余，都允许收缩主张；不改A版本、扩网格或换子集。 |
| H_N：现有头的局部TV改善迁移到native真实达到状态。 | B2达到且独立路径有效G成立的i2–15，同H下shadow raw−actual repair TV；每请求先等权，再prompt/域等权。 | 95%下界须超过冻结0.005。缺合格状态记缺失，不填0；无正效应则削弱迁移。首位置和all-proposal单列。 |
| 接受后果：局部迁移同时伴随实际接受收益。 | B2−B1每请求已发accepted proposal数／实际proposal验证轮数的配对差；全调度轮实际输出为辅助。 | 后果需同向且对应区间支持；局部为正、后果不确定时只写局部迁移支持。分母0缺失，终止、short、ordinary及零输出轮保留。 |

两项主要假设各自给95%区间；总体确认性表述要求H_R/H_N各自97.5%区间满足对应阈值。**HF损失**按branch→position→C→prompt汇总；**native局部量和迁移MSE**每请求内等权平均所有合格状态，再prompt／域等权，没有先把轮或位置等权。**native后果**先算完整请求的计数比，再按prompt配对比较。概率零均值诊断使用请求总计数差，不用随机验证轮数作分母。各项使用prompt bootstrap，不能混用权重；完成48个分片不等于48个prompt都有合格主状态。首位置另报，其Dstar或预测增量不数学必为0。

D*_S、D和Delta共用target p；新留出可以防本轮拟合泄漏，不能解除代数依赖或建立因果机制。新增高度共线的列也可能改变ridge惩罚。支持上限是**冻结低容量流程中的增量离线预测**，不是头感知了oracle或形成可部署门控。HF模型将原样用于native合格达到状态，单独报告迁移损失；不在native新test重新拟合。target-blind探针未实施，不作部署预测主张。

## 第三层：已测证据与待确认结果

**已支持的dev合同：** 36 prompt、72 C、8,640状态来源与保存C/H/token一致，集合／oracle／坐标闭合检查失败0；tail条件TV最大1.789e−15。HF几何是同BF16 logits的FP64稳定归一化；FP32采样概率另测误差，最大TV为6.780e−7。它不冒充native实际p/q。[完整dev契约](../analysis/GEOMETRY_DEV_CONTRACT.json)

**dev尚未支持H_R：** M0/M1已选CV MSE为0.07908361/0.07903783，增量4.58e−5，远低于冻结实质阈值。平均D*_S=.057124，接近已有覆盖量p(U)=.057298；这是冗余的开发警点，不能说覆盖已解释大部分实际收益。实际Delta=.267670，gap=.352699，约12.85%加权状态恶化。CV有选择乐观性，最终dev拟合后的区间不是留出证据。两模型alpha均.01，scaler/系数不再改。[冻结模型](../protocol/GEOMETRY_FIT.json)、[固定dev报告](../analysis/GEOMETRY_REPORT_dev.json)

**native机械准入范围：** 已审核工程6题×三臂18个完整请求、1,758验证轮的真实映射与整数输出/HTTP计数闭合。36条B2 dev请求的原／优化头同H parity最大TV为0，固定容差2e−5；fresh依冻结设计继承此准入，未重做原／优化头parity。fresh全部12,538验证轮的实际q与proposal生成q绑定，56,355个B2几何状态的数值检查通过；后者不是局部主量的13,728个reached且G有效状态。机械准入本身不是H_N科学成功，也不证明理想概率定律。[桥接审核](../analysis/NATIVE_BRIDGE_REVIEWED_AUDIT.json)、[最终结果的机械限定](RESULTS_CN.md)

| 最终主张 | 证据状态 | 固定证据 |
|---|---|---|
| H_R新48题离线增量 | **未支持／实质增量主张削弱**：MSE差1.528e−5，95%CI[−1.476e−5,4.058e−5]，远低于阈值7.908e−4。 | [HF固定报告](../analysis/GEOMETRY_REPORT_fresh_test.json) |
| H_N局部达到状态 | **已支持，限当前runtime／测度**：TV改善0.127825，[0.119611,0.137178]，超过0.005阈值。 | [native固定报告](../analysis/NATIVE_REPORT_fresh_test.json) |
| 接受／输出后果 | **已支持，限实测计数**：已发accepted proposal／验证轮增加0.892487，[0.703574,1.087621]；全调度轮输出增加0.847449，[0.671629,1.026772]。不是TPS结论。 | 同上；144完整请求，整数计数误差0 |
| HF→native冻结预测迁移 | **未支持pool增量**：MSE差−5.717e−6，区间含0；GSM小正，HumanEval／MT-Bench小负。 | 同上，独立native权重与测度，无再拟合 |
| G乘积在当前runtime的理想期望身份 | **未建立** | 原警点及真实实现边界继续保留 |

HF留出48题均有主状态，i2–15共10,640状态；实际Delta=.251713、Dstar=.063438、gap=.374797，14.267%的分层加权状态恶化。Dstar接近p(U)=.063694，说明该新增量接近已有覆盖量，不能说覆盖解释了全部Delta。数学不变量和大平均收益成立，不等于增量预测理论成立。两项主要检验的**联合确认条件没有满足**；H_N的正结果不补偿H_R的失效边界。

native局部主量每请求合格状态等权，共13,728状态／48 prompt；三域局部及接受后果方向均正。HF平均Delta与native平均Delta不使用同一测度或权重，不能除出“迁移保留百分比”，也不能把局部TV改善换成全部接受收益的因果份额。区间限于固定dev模型、本轮题面及单seed设置，不覆盖训练与RNG重复的不确定性。[最终解释审查](../analysis/FINAL_THEORY_INTERPRETATION_REVIEW_CN.md)

## 接受价值的边界

标准单状态定理在实际X~q和相应随机验证下给出接受期望`1−TV`。[Leviathan等，§2–3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf) 理想连续first-reject规则还需要完整proposal Q及正确验证随机性：`E_Q[G_k·prod_i b_i]`，其中`b_i=min(1,p(X_i|H_i)/q(X_i|H_i))`。G只含proposal路径EOS、有效长度和轮开始预算，不能包含已实现接受事件；真实reached可筛局部统计，却不能再次塞进G重复加权。普通draft或teacher路径上的overlap乘积不替代此期望。

本runtime的log-softmax重建与Triton实际缓冲并未建立bitwise概率一致，条件于完整proposal H的理想随机独立性也未证明。复现所有实际first-reject事件、整数计数闭合和某个辅助区间包含0，都不能独自证明理想G乘积定律；同seed亦不保证跨臂相同轨迹。保留实际计数与工作期望检查的区别，不把尚未建立的定律写成已验证实验结论。

新48题三臂的请求总量差区间均包含0，只说明在这份固定摘要下与0相容；不证明等价或随机独立，不抵消工程6题B2原警告（均值8.4388，95%CI[3.7056,13.1719]）。保留两批各自身份，不合并或替换原警告。

teacher-prefix、target-branch/alive-at-i、native/all-proposal及native/reached/G-valid是不同测度；HF EOS为[151645]，native使用其真实EOS集合。局部距离、实际proposal接受、bonus/correction/ordinary输出分别记账。全调度轮指标按定义保留零输出轮，但fresh自然零输出轮实际为0，该分支仅有CPU fixture证据；调试eager插桩不支持TPS结论。前驱修正、候选oracle、前缀接受目标与条件失配已有近邻认识；本轮价值取决于新增固定证据及其失效边界，不能靠结构清单或跨论文数字排名建立创新。
