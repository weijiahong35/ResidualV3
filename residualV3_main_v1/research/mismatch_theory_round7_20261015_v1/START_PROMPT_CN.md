# 第七轮启动prompt

复制分隔线以下全文给执行chat。收到后按限定范围自主执行，不只再写计划。

---

你接手ResidualV3第七轮失配研究，仓库 `/data/weijiahong/speculators-main-lora`。

先读：

```text
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/README_CN.md
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/THEORY_PLAN_CN.md
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/sources/READING_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/qa/INPUT_AUDIT.json
```

R6已完成并归档：

```text
residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6
```

R6支持匹配未见历史上的共同流探针改善，未确认新增历史的风险/方向预测优势；12对中11对存在松弛共同改善候选，实际basis/dose可达性未知。所有负结果及旧警报保留，不恢复R6任何已完成stage。

**本轮补一条明确联系：跨历史共同的失配，是否确实对应原头实际概率移动中具有正边际价值的部分？** 从失配的共同需求出发，再检查原头动作，不给模块换创新名称，不转成训练更好的AR头。

执行以下固定研究：

1. 只读复用R6全部48题做开发与描述，核验现有概率与输入。用fit目标分布的共同低估/高估容量，与原头实际移动d=qh−q0相交，按指南构成质量平衡的子动作c。它只是保守提取，不是全部共同修复空间。
2. 比较q0、qh、qc=q0+c、qr=qh−c，以及均匀移除同等质量的qu。缩减比例必须用本轮子动作质量m/TV(qh,q0)，不能用R5/R6的M_common。
3. 新采GSM8K/HumanEval/Alpaca各12题，共36；K64=fit16/eval48、anchor1/65、主位置3/8/15。主集合要求同一C/i/精确Z的fit子组内部至少2种不同完整H，eval匹配该组Z；novel仍相对同C/i全部fit历史（含其他Z组）判定，不能跨Z凑fit两种H。保留m=0，seen另作描述，不补题追覆盖。
4. 两主量分别检验：η=TV(p,qr)−TV(p,qh)，去掉共同部分是否损害完整效果；κ=TV(p,qr)−TV(p,qu)，是否比同质量均匀移除更有害。只有两项各自支持才使用完整的正边际价值及区别性表述。
5. 用Gc/Gr/Gh与完整token账本解释失败：共同部分本身迁移失败、单独有益但叠加有害、共同部分仍有帮助但其他损伤更大，以及零/边界。固定最多8个案例，不挑最大效果，不据账本识别训练原因。

这是**概率分布诊断**。qc/qr/qu不保证原basis/dose可达，不上线解码；fit Gc=m及I=Gh−Gc−Gr≤0是数学合同，不是新发现；m/t不是收益解释率；r不能叫历史特异失配。本轮不能归因独立DFlash→联合模型的全部正式AC/TPS。

先锁协议/代码/样本，再采新题。独立导出fit/eval bank和去除D_raw/D_head/gain等效果字段的元数据。动作构造只读fit target与实际q0/qh，保存并签名全部动作后才评估eval。旧read_capture全量入口会读eval，不能直接调用后再宣称隔离。全文件SHA由独立审计入口核验，准确记录访问范围。

本prompt授权在新run内写代码、只读复用旧数据、必要冻结重放、固定新题采集与分析、报告和清理。**不拟合预测器、不训练主体/头、不新增LP/优化见证、不改结构/输入/损失、不做native/RNG/TPS或剂量网格。** 不因效果负而换指标、子动作规则、域、阈值或对照。旧数据工程正确后按固定计划完成新确认。

**有显存足够的卡就用，允许共卡，不等空卡、不要求utilization=0，不固定GPU0/7。** 每阶段实时核预约与工作集峰值+4 GiB余量；同时最多1卡、自有峰值默认≤32 GiB，不终止他人进程。无卡容纳最低工作集时继续CPU工作并复查，无需重复询问共卡许可。

全轮上限8 GPU小时、24 CPU小时、新增磁盘700 GiB；CPU worker≤4、数值计算线程合计≤8，import前库线程设1。失败、编译、等待、重试与临时文件全部计账，不换run重置。同阶段最多2次有新定位的技术重试。只管理核实身份的自有作业与预约，不碰原controller/parallel/收尾或历史资产。

输出：

```text
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/<UTC>_mismatch_r7/
```

先简报读到的R6终态、核心问题与可复用数据，然后自主执行。最终交付RESULTS、方法/案例/图表、claim–evidence、封存动作与访问审计、预算/身份/静态SHA、COMPLETION和NEXT_CHAT。第一页用简单中文回答共同部分是否对原头实际收益有用、是否超过同质量对照、何时仍会失败。

负结果可以complete；必要工作未完成则partial。结束退出自有进程、释放预约，保留全部旧警报，不自动开启第八轮。
