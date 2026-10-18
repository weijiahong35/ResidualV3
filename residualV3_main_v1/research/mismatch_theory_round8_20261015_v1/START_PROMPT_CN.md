# 第八轮启动 prompt

复制分隔线以下全文给执行 chat。收到后按限定范围自主实施，不再只交一个计划。

---

你接手ResidualV3第八轮失配研究。仓库：`/data/weijiahong/speculators-main-lora`。

用户明确要求两项一起做：**共同可修正偏差的跨主体对照，以及从target历史到草稿自行生成历史的迁移。** 理论对象是失配；不转成训练更好的AR头。

先读：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/README_CN.md
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/THEORY_PLAN_CN.md
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/EXECUTION_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/sources/READING_GUIDE_CN.md
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/protocol/DESIGN_SPEC.json
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/qa/INPUT_AUDIT.json
```

R7已完成，原run：

```text
residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/20261015T075350Z_mismatch_r7
```

它支持共同部分与原头收益的有限联系，没证明跨主体、跨历史来源或部署收益；旧负结果、概率警报和资源披露保留。不要重启任何旧stage。

本轮同时实施两种被诊断raw主体D/J × 三种历史来源P/D/F：D=E2独立DFlash，J=E2联合主体raw；P=target自采样，D来源=独立草稿自采样，F=联合主体加完整原头自采样。固定同题同C，两个主体都对三个来源的H打分。F不能写成joint raw自采样，不把joint头移植给独立主体。

从P-fit的共同欠量/过量构造R5的M/2概率探针，不依赖原头方向，不训练参数，不把探针投入生成。四主量是A_D、A_J在P-eval上的收益，以及B_D在D-eval、B_J在F-eval上的收益；始终使用P-fit动作。两研究一并采集、独立判定，不等A阳性才做B。

先用R7旧fresh题每域SHA排序前2题，共6题做工程；每来源dev K16=fit4/eval12。旧数据已暴露，只能开发。随后锁定新36题，各域12；每来源K64=fit16/eval48，anchor1/65，主位置3/8/15、控制1/2。合并最新机制暴露记录再选题，不能补题或换域救覆盖。当前余量快照48/12/422必须重核。

四主集合要求同W=(C,i,前驱)的P-fit内至少2种完整H，并且eval完整H未见于同C/i全部P-fit，称new_to_P_fit。D/F-fit出现与否不改变主资格；严格三来源all-fit-novel另表。保留M=0；未匹配NA不填0。全部matched自然频数按P-fit_seen、only_other_fit_seen、strict_allfit_novel三桶精确拆解；seen的M/2是数学保证，不能拿它证明泛化。

六格全报。四主量用5000次整题分层bootstrap、98.75%双侧CI；每端点每域至少6题有定义且6题有非零动作才过覆盖，门槛不删零动作。下界>0为正改善支持，下界>0.01另标实质幅度。不确定不等于不存在，不能说联合训练是唯一原因或模型大小差造成失配。

同时完成预定次要分析：同来源D/J配对、P权重的共同W标准化、D/F-fit同源探针、完整token损伤与补偿账本、最多8个固定排序案例、原策略参考reached状态局部量。严格按指南，不能增加预测器、oracle或新统计口径救效果。

**概率身份特别重要：** 统一BF16 logits→FP64完整概率缓冲，CPU独立PCG64类别采样直接使用该缓冲；参考验证使用同一actual generation q。F验证用q_F，不是q_J或probe。P/D/F、fit/eval、context/proposal/reject随机流分开。全部D/F-eval 1..15合法位置保存sampled-token p/q、buffer SHA、uniform、accept/reached/EOS/预算，不能从五个保存位置拼造first-reject。

这是新锁定的HF参考采样口径，与旧FP32或生产vLLM不逐位等价。参考reached只说明原策略到达哪些状态；探针未部署，不能声称探针接受增益、TPS或native无偏。旧R2 passed=false警报继续保留。

本prompt授权在新run写代码、冻结重放、新题固定采集、全部规定分析和收尾，收到后自主执行。**不训练、不改结构/损失/安装环境、不启动native/TPS/RNG对照、不动旧权重/协议/报告/控制器/parallel/CPU收尾。** 数值和采样身份先过dev，再封存协议与代码。动作只读对应fit，先签名全部动作，后评eval，准确记录访问范围。

**有显存足够的卡就用，允许共卡，不等空卡、不要求utilization=0、不固定卡号。** 实时核UUID/预约/预计工作集+4GiB余量；同时最多1卡、自有峰值默认≤32GiB。不能终止他人进程；无合适卡时推进CPU工作并复查，不重复询问共卡许可。

全轮所有run累计上限8 GPU小时、24 CPU小时、700 GiB新增磁盘；失败/编译/等待/共享变慢与临时副本计账，不换run重置。CPU workers≤4、数值线程合计≤8，库线程import前设1。每阶段最多2次有新定位的技术重试，科学负结果不重试。只管理核实UID/PID/PGID/startticks/CMD/run身份的自有进程及预约。

输出：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/<UTC>_mismatch_r8/
```

先简报读到的R7终态、两个问题、题面余量和可复用资产，随后执行。最后交付四主端点/六格结果、覆盖/三桶/标准化/案例/参考到达记录、方法与理论卡片、claim–evidence、访问/数值/资源/文件审计、COMPLETION和NEXT_CHAT。

固定工作完成但阴性或覆盖不足可complete；必要采集/分析没完成则partial。第一页用简单中文分别回答“共同偏差是否跨主体存在”“换到草稿历史后是否仍有效”“哪些情况失败”“仍未证明什么”。结束退出自有进程、释放预约，不自动第九轮。
