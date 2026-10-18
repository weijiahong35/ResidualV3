# ResidualV3 第二轮自主执行任务书

版本：2026-10-14 v1。仓库：`/data/weijiahong/speculators-main-lora`。
先读：[归档索引](ARCHIVE_INDEX_CN.md)、[理论问题与实验契约](THEORY_PLAN_CN.md)。[启动 prompt](START_PROMPT_CN.md)供新 chat 直接使用。

**本轮目标：补齐 native 测量，并检验修复可达范围是否比失配大小本身更能解释实际收益。** 现有冻结 E2 是研究对象；不以再证明“前驱有用”结束，不预设新假设必须成立。

用户已要求给执行 chat 新一轮指引并执行。下述新目录内的实现、工程修复、限定采集和诊断拟合均在本轮授权内，满足资源条件就推进，不再请求“重开旧 C”的确认。本聊天只制作归档和任务书，尚未启动本轮模型实验。

## 1. 原始资料与不可改变的资产

定义仓库相对路径：

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
MAINT=residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1
```

按顺序读取：

1. R1 `results/RESULTS_CN.md`、`RUN_AUDIT.json`、`CLAIM_EVIDENCE.json`、`NEXT_CHAT.md`；R2归档清单。旧 A/B已完成，旧 C不确定。
2. R1 `protocol/PROTOCOL.json`、`LOCK.json`、`SAMPLE_MANIFEST.json`、`CALIBRATION.json`；`analysis/EXPERIMENT_REPORT_test.json`、`FINAL_INTERPRETATION_REVIEW_CN.md`、`SECONDARY_B_KL_TEST.json`。这些只作已暴露历史证据，不重新拟合、选旧test案例或更换旧主效应。
3. R1 `analysis/NATIVE_BRANCH_EXIT_CN.md`、`NATIVE_BRANCH_EXIT.json`、`NATIVE_C_IMPLEMENTATION_CN.md`、`EOS_RUNTIME_IDENTITY_CN.md`、`RESUME_METHOD_AUDIT_CN.md`、`DEV_TARGET_CACHE_PROBE.json`。
4. R1 `code/native_scheduler_capture.py`、`native_verifier_capture.py`、`native_accounting.py`、`run_native_stage.py`、`native_diagnostic_launch.py`及对应CPU测试；再查 `sources/ASSET_AUDIT.json` 中指向的实际安装版 vLLM源码，不根据最新版网页猜接口。
5. R1 `code/hf_frozen_adapter.py`、`repair_replay.py`、`ab_position.py`、`collect_ab.py`、`mismatch_math.py`，以及旧 `analysis/aggregate_ab.py`、`report_experiments.py`；复用身份和统计经验，不导入会将新输出写回R1的硬编码路径。
6. R1 `sources/LITERATURE_MATRIX.md` 和原文；本轮补充阅读问题见 THEORY_PLAN §7。
7. `GROUP_MEETING_20261013_CN.md`、`EXPERIMENTS_NONTHINKING.md`、R1 `analysis/HISTORICAL_EVIDENCE_CN.md`；保留on-policy、尾部停修、专家路由等负结果。

接手核对 FORMAL 的 status/progress/FINAL_AUDIT 和 MAINT 的 status/COMPLETION_AUDIT；本任务书创建时原批次已48/48、24/24、两审计passed。原任务完成不是等待条件，不再起原 controller、parallel.py 或收尾。进程/GPU空闲仍须实时核验，旧释放快照不能作为新准入证据。

## 2. 自主执行范围与成本

- 新建 `R2/runs/<UTC时间>_<标识>/`，原run及归档包只读。复制必要代码到新run，修改RUN路径和新协议；不得改旧 gate、旧源码/权重/报告、安装环境或原正式评测文件。
- 同时最多1张确认空闲且无预约冲突的GPU；本轮总额 **8 GPU小时**，包含启动、失败、dev、新test、兼容性检查与退出。桥接阶段最多 **2 GPU小时，计入8小时**；理论/HF/dev建议≤3，锁定留出/native建议≤3。不能按arm/stage重置账本。
- CPU worker≤4、总计算线程≤8；新增run磁盘≤50 GiB，记录实际文件系统剩余。CPU约4小时是首个工程进展检查点，不是保证工期或自动停工上限。
- 允许冻结模型采集、诊断数学/报告代码、限定ridge/logistic探针。此轮不训练头或主体，不改结构，不扩成多卡/全近邻复现，不发布插桩TPS。
- 父进程监督累计预算并预留保存/清理时间；到限前停止经身份核验的自有GPU进程，保留数据。记录UID、PID/PGID、启动ticks、完整CMD、GPU UUID与新run输出路径。禁止宽泛杀进程；外来占用只退出自己的工作。

**新工程重试规则：** 不继承旧run“总计三次失败”的计数，也不修改它。每次新启动前记录根因、源码定位、改动SHA、针对性CPU检查和下一验收点。不同已定位缺陷可在桥接2小时总额内持续修复。对同一缺陷无新证据/实现变化不得重启；连续两次修订仍停在同一验收点且没有定位进展时，停止该支路并写具体缺口。不能通过更名重置时间或停滞记录。

无需确认的事项：授权范围内修复、数据去重、限定采样、数值测试、负结果分析、运行恢复和报告。需要超预算、全量训练或改冻结资产时，先交具体变更；其余可做工作继续。桥接预算用尽只关闭本轮native支路，HF理论诊断仍可推进。

## 3. 样本：真正的新验证集

本次只读可用性核验：原清单GSM8K 200题剩136、HumanEval164题剩100、MT-Bench80题剩16，均已排除R1每域64题。使用与R1相同的 NFKC/空白规范化和GSM固定末尾提示去除规则；最终还需核查重复及其他已暴露机制分析。

默认方案：

| 用途 | 样本 | 使用规则 |
|---|---|---|
| native工程验收 | 每域2条R1 dev，共6题 | 在旧dev中按group SHA固定顺序选；先1条B0再三臂；不能动旧test |
| 本轮科学开发 | 每域12条R1 dev，共36题，包含上面6题 | 开发新度量/拟合规则。已有C/H可只读重放以测新量，标记用途和provenance；不重做旧主分析 |
| 新确认性test | 每域16条，共48题 | 排除R1全部192题及本轮调试题；MT-Bench用剩余16，另两域固定seed从剩余组抽16 |

新test仍来自历史评测面板，只称“未参与前轮机制研究和本轮开发的分析留出”，不称从未被项目/预训练使用。锁前审计题面重合和已知训练重合；若实际可用数不足，先记录并缩小，不按结果挑替代题或自动换数据集。

保存 `SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`，对每个group记旧dev/旧test/本轮dev/新test身份、来源/模板/seed。所有分支、native臂和HF状态随prompt同组。旧120题已经用于形成新假设，不能重新当本轮test；新48题采集到开启主分析前也不得用于调参。

HF新量默认沿用R1 cached参考程序、T1完整词表、原EOS身份，最多两个C(anchor1/65)、K8、15 proposal；已有dev的C/H在 `data/target_branch/dev/*/COVERAGE.json` 中查实际文件名和保存字段再重放。缺失的p/q全向量需要重算以获得**新诊断字段**时允许，记录为新测量而非旧实验重跑；不重生成已保存的H来挑结果。新test按固定seed生成，EOS/缺anchor不补采。

native默认C1、T1、15proposal、max512、context4096、eager、同步调度；SamplingParams显式指定temperature=1/top_p=1/top_k关闭和无额外惩罚。实际EOS集合从runtime读取；不得为与HF表面一致而改正式模型。不同测度单独标识，不合表。

## 4. 第一优先：把 native 数据链路做对

### 4.1 已定位的工程问题

上轮IPC长路径和warmup误识别已有修复记录；保留短临时目录和显式CAPTURE_ARMED。最后的错误不应仅靠删除 `−1`检查解决：scheduler hook还错误要求CPU发布的proposal token与GPU逐项相同。

正确权威分工：

- scheduler：内部/外部request映射、schedule/轮次、输出预算、context预算、实际EOS集合、proposal**长度**。
- GPU worker：实际槽位与request映射、anchor/proposal token、同轮hidden/raw/repair分布、真实进入verifier的处理后p/q。
- scheduler完成事件：停止处理后的实际输出新增片段、请求状态/原因，含显式零输出事件。
- HTTP：最终成功response及usage.completion_tokens，核对全请求输出。

CPU `[-1]`只可作为已核实接口中的长度占位；不得变成负token采样、推断真实token，或让未知负ID一概通过。用权威request ID＋schedule/轮次＋长度＋worker槽位绑定，再由实际GPU数据验证。

### 4.2 分层验收

1. CPU检查：占位符/真实token分离、request映射、普通/零输出prefill、EOS/预算、stale/duplicate事件。缺事件是缺失，不能由“start数−emission数”推成零输出。
2. 单条B0完整HTTP请求：真实proposal验证记录>0、已完成请求>0、p/q及token对齐；只有health或anchor不算通过。
3. 每域2旧dev题×B0/B1/B2：独立raw、联合raw、联合＋头身份分开，B1必须从自己的raw q采样；不能把B2 token配上B1 q。
4. B2同H原始/优化头parity、shadow raw与实际q核对。先固定容差与fail条件；旧代码仅存误差/标parity_done不等于验收通过。容差来自同runtime、同算术参考的dev数值检查，不用R1 HF cached/full的0.0608作native容差。
5. 验证S/U尾部比例不变量及理论数值下限；超容差先修采集/分布定义，不把违反数学约束当好结果。

计数合同要求整型精确闭合：真实输出=已发accepted proposal＋已发bonus＋已发correction＋ordinary输出；全请求与HTTP usage相等。明确记录每次调度完成事件、零输出原因和post-stop行为。未自然遇到的EOS/bonus等分支可用标注的工程fixture验证实现，同时报告自然数据覆盖；不强迫持续生成直到碰到罕见事件。

概率期望是另一层：实际随机接受和 G∏b 的估计不要求逐轮相等；按prompt聚合并报告抽样不确定性。G只含proposal路径EOS/有效长度/轮开始预算，不含实际拒绝随机数。不能用普通draft路径∏overlap替代。

通过后写 `NATIVE_BRIDGE_AUDIT.json`，包含每项观测次数、容差、差值和源SHA；禁止未检查字段初始化为0后当通过。未通过则留技术不确定，并继续不依赖它的工作。

## 5. 第二优先：新理论量与留出检验

执行 THEORY_PLAN §3–6，先CPU验证闭式解及反例，再在同次联合raw/repair/target上流式采集 D、D_h、D*_S、p(S)/q(S)/r(S)和候选内/尾部改善。全词表精确量不替换为Top256+聚合tail后仍称相同指标。尽量只存标量、token、上下文及少量验收向量，避免海量logits。

对额外重算的旧dev模型状态记录“新量采集”；复用旧C/H但不修改旧文件。若S/U集合假设在真实runtime不成立，先解释差异；不得偷偷改修复机制以满足理论。

dev36用于固定M0/M1特征、少量非线性项、正则、训练权重、数值容差、统计规则及是否执行可选target-blind探针。两模型预测同一Δ，训练时每prompt等权；test使用相同层级，不能token数量加权成另一目标。

主要H_R效应为新test上每prompt预测MSE(M0)−MSE(M1)；使用i=2…15，按branch→position→C→prompt汇总平方误差，首位置另报。给绝对差、基线MSE、相对变化及按prompt bootstrap的区间。数值/实质阈值先由dev确定并说明，不按test改。不能只报相关、代数上界或oracle能达到多少而宣布当前头已实现。

如果候选内/尾部分解显示变化主要来自尾部总质量，也只报告坐标质量变化，不等同“学习了尾部内部结构”。温度校准改变尾部条件比例，保留为集合外参照；沿用R1 dev冻结温度时注明来源，不重拟合旧test。

## 6. 第三优先：native机制与跨测度

桥接通过才采本轮dev/新test native B0/B1/B2，尽量复用工程验收中的已完成合格dev请求；升级采集字段/协议时解释哪些旧数据可复用，不能混用。

H_N主要局部量：B2真实reached且i=2…15状态上同H的 shadow raw−actual repair TV。每请求内等权平均合格状态，再prompt/域等权；无合格状态记缺失，不填0。分别另给所有proposal位置、首位置与真正reached的均值/覆盖。actual reached是实测筛选集合，不能再乘真实接受指示去重复加权概率乘积。

实际后果：B2−B1的每请求“已接受proposal数/实际proposal验证轮数”为预定关键后果指标；实际输出/全部调度轮数为辅助指标，各自按prompt配对比较并报告区间，分母0记缺失。记录EOS、长度与普通/零输出轮构成。同seed控制请求设置，不保证各臂生成相同轨迹或共同随机数。B0整体配方参照另列，不给训练/推理收益分摊百分比。Δ_N为正只支持局部迁移；要宣称接受收益迁移，后果也须同向且其不确定性支持。否则明确“局部迁移支持，接受后果不确定”。

若 dev 锁定了迁移测试，把HF拟合的M0/M1原样用于native reached，单独报告预测损失改善。不要在native test重新拟合或把两测度混合出更好的系数。native几何指标仍需使用实际native p/q，不能把HF重算分布冒充实际验证器概率。

H_R/H_N作为两项预定主要科学检验分别报告；若给总体“显著”结论，预先约定多重检验处理。默认给95%区间、连续效应和逐域结果，样本仅16/域时明确精度限制。不要把窗口、token或分支当独立重复。

## 7. 实验锁定、失败与恢复

本轮建立自己的 `PROTOCOL.json`、`LOCK.json`、数据清单、代码/依赖身份与预算；绝不重新锁R1。锁前可在dev修代码、改分析并留修订记录；锁后test只执行固定主分析。

采集中途若纯实现缺陷影响test正确性：先停止，隔离受影响分片、记录是否已经看到效应及修订范围。协议/算术改变可能使原数据不再可合并；不能改SHA后称原预注册未变。此前已暴露效果的样本不再用作同一修订假设的未见验证；在报告中区分技术修复、探索性重分析与新验证。

状态机：`archive_checked → CPU_contracts → bridge_pilot → dev_diagnostics → protocol_locked → fresh_test → report`。bridge失败可分支到`HF_only_locked_test`；不会将整项研究伪装为native完成。预算、严重身份/概率不一致和无进展重复缺陷是退出条件；科学负结果则正常归档，不追加调参搜索。

新run建议目录：

```text
manifest.json / DECISIONS_CN.md / NEXT_CHAT.md
protocol/{PROTOCOL,LOCK,SAMPLE_MANIFEST,EXPOSURE_LEDGER}.json
code/                         # 独立修改后的诊断实现
data/{engineering,dev,fresh_test}/
analysis/                     # 合成检查、分布不变量、固定统计、真实图
sources/                      # 新增来源及旧来源引用/版本
results/{RESULTS_CN.md,CLAIM_EVIDENCE.json,RUN_AUDIT.json}
logs/                         # 自有作业、缺陷证据、预算、恢复
```

## 8. 交付标准

必须回答：

1. native桥接是否真的完成；请求、p/q、计数和parity证据是什么？
2. 当前修复操作的S/U不变量是否成立；放宽下限/实际差距的分布如何？
3. 新几何量是否超出原始TV/位置/覆盖率提供留出解释；负结果是什么？
4. 当前头在哪些状态实现/未实现允许的改善；是否存在实际负收益？
5. HF解释是否迁移到native reached；局部效果与实际接受/输出分别怎样？
6. 哪些是标准数学、哪些是新增经验发现；与近邻是否仍高度重合？
7. 接下来是保留理论、收缩主张还是有证据驱动的最小方法改动？

交报告、claim–evidence、真实图表、模型/测度/样本身份、源码/数据审计、预算和NEXT_CHAT。每条主张标支持/削弱/不确定；没有数据不画模型结果图。新的方法完善建议至多两项，必须对应本轮发现、最小对照和额外成本，本轮不自行训练新结构。
