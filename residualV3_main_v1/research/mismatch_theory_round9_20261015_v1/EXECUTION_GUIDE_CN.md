# 第九轮执行指南

与[理论方案](THEORY_PLAN_CN.md)、[启动prompt](START_PROMPT_CN.md)、[核心审核](qa/CORE_QUESTION_REVIEW_CN.md)配套。编写本包没有启动科学实验。执行chat收到完整prompt后，在下列范围自主完成。

## 1. 范围与入口

仓库 `/data/weijiahong/speculators-main-lora`。新建且只写：

```text
residualV3_main_v1/research/mismatch_theory_round9_20261015_v1/runs/<UTC>_mismatch_r9/
```

旧R1–R8协议、结果、权重、安装环境与控制器保留；不得执行旧main。复制必要纯CPU数学函数后修改新ROOT；import前设置PYTHONDONTWRITEBYTECODE=1，缓存与日志进入新run。

本轮是完整概率存档再分析，**GPU预算0，不加载模型、不采新题、不等卡**。有卡就用的既有偏好仍适用于需要GPU的任务；本轮全部目标有存档可算，不为占卡增加推理。若概率资产缺失，不擅自补采/改题，明确缺口。

允许：只读存档、建立新CPU数学实现、数值合同、固定有限族优化、分组干预、图表/报告和收尾。禁止：训练模型/头/预测器，基于原头成败选样本，重启正式native/TPS/RNG评测，变更旧报告，把撤回的R/r方案重新启用。

## 2. 实际资料与身份

必读仓库AGENTS.md，以及本包THEORY_PLAN、CORE_QUESTION_REVIEW、sources/READING_GUIDE、protocol/DESIGN_SPEC和qa/INPUT_AUDIT。

源run：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8
```

读取其COMPLETION、NEXT_CHAT、results/RESULTS_CN、code/common.py与code/collect_r8.py确认概率身份。数据来自`data/fresh_test/<prompt>/C{1|65}/`的META.json、raw.npz、target.npz、COMPLETION.json。目录名fresh_test是旧身份，不代表本轮未见。

- raw.npz中的D/J是各主体原始BF16 logits，全15位置；只读对应i−1。
- target.npz logits按META.records的target_row映射完整H；不能把自然槽序号当target_row。
- 仅取source∈P/D/F、pool以eval结尾、position∈3/8/15、valid=true、target_row有效的原记录。不要应用旧P-fit匹配/novel筛选。
- 相同C/H/i的target身份应一致；重复H保留次数，意外重复记录按(source,pool,slot,position)检测并fail。
- 不需要原头F.npz，也不需要计算原头或R8探针收益。F仅说明这些H如何产生。
- 每个实际读取文件核SHA对照源per-context COMPLETION；只声明重验范围，不全盘拷贝/重验八轮权重。

采样测度继承R8：BF16 logits→稳定FP64完整softmax；不重新采样，不导入旧FP32概率，不截top-k，不clip或偷偷重归一化输入。允许对优化器产生的候选显式投影回simplex，但须记录投影、重新评估完整词表，不能投影模型输入来掩盖身份错误。

BF16存档实际是uint16位模式，不是整数值logit。按源`common.py`的`bits_float`恢复：uint16→uint32左移16位→view float32，再转FP64，随后使用源`distribution`的稳定完整softmax。不能将uint16数值直接转浮点后softmax。将解码与概率构造身份列入数值合同。

## 3. 先审核，再写执行锁

1. 读完后用一页`protocol/CORE_ALIGNMENT_CN.md`说明：本轮A/M/S分别回答什么核心问题；何种结果反驳信息强迫取舍；哪些工作只是辅助。若发现实现准备把目标变成原头预测/某探针均值，重写后再审，不启动计算。
2. 完成只读清单与固定样本身份，不读旧Gh/probe gain来选面板。
3. 完成下述数学合同和小词表独立LP核对。旧R4线程/数值审计警点不能因复用源码消失。
4. 在旧R8 dev资料按各域SHA首题、anchor1、i8、source P的3个固定面板做工程计时，D/J均检查。dev结果只用于实现/资源准入，不调科学规则、不混入正式统计。
5. 固定源码、参数、样本、种子、权重、输出表和资源规则到LOCK。由于数据已暴露，称“本次再分析执行锁”，不称事前未见预注册。
6. 完成全量风险和固定证书面板，最后一次汇总报告。局部完整分片可以续跑，不能重算成功分片追结果。

每阶段最多2次有新定位的技术修复；修复记录在qa，旧失败保留，预算不重置。科学阴性不重试。

## 4. 平均TV最优值：完整词表直接计算

对信息组内n个自然槽，P为n×V。可按不同H缓存概率，但自然频数风险应等价于展开全部槽。不同H等权的辅助分析使用不同H各一行。

令a_k(v)为逐token升序的第k个值，s_k=Σ_v a_k(v)。找到s_k≤1≤s_{k+1}，令：

```text
alpha = (1-s_k)/(s_(k+1)-s_k)
r = a_k + alpha * (a_(k+1)-a_k)
d = (2*k-n)/(2*n)
F(x) = mean_j TV(p_j,x)
lower = F(a_k) + d*(1-s_k)
upper = F(r)                  # r需合法、完整词表实算
```

公共次梯度d给出下界，候选r给出上界。n=1直接r=p；所有分布相同直接共同p；s_k=s_{k+1}=1时取a_k避免0/0。等值/零概率保留。非单位质量a_k的F只用于L1凸函数下界，不能把它当概率候选。

保存k、alpha、输入引用、实际质量误差与含浮点guard的上下界。目标间隙1e−8，若不满足则报告数值未决、定位实现问题，不能以解析式应当成立跳过验证。任何必要质量修正必须留记录后完整评估。

每个信息组独立求R，按该组在面板中的自然频数聚合。I0/I1/I2/IH用同一批槽。R只依赖p和分组，同一面板两个raw共用R；E_D/E_J与A_D/A_J分别计算。

## 5. M与安全平均S：完整simplex证书

固定每域完整group SHA首4题，共12题，只取已有anchor1、三来源、三主位置，分别D/J、I0/I1/I2。身份见qa/INPUT_AUDIT，不随结果改变。全目标面板108个、raw面板216个；一般组计数在执行时重核。

重复完整H先合并权重；完全相同p可在求解器内部再合并，需保留到原H的映射。M不受重复权重影响，S按自然频数权重计算。

### 单/双目标

- 单目标：R=0，M=S=d（S^τ同样不超过d）。
- 双目标M用THEORY_PLAN闭式，完整评估见证；不使用当前头动作。
- 双目标严格S可在p1到p2线段上求最优：t>0时λ∈[max(0,1−d2/t), min(1,d1/t)]，平均loss=t*[w2+(w1−w2)λ]，取使线性目标最小的端点；t=0直接共同p。S^τ把d1/d2分别替换d1+τ/d2+τ并截[0,1]，最终收益仍相对原始d计算。退化并列允许确定性选点。

### 三个及以上目标

优化变量是任意完整分布r；M为最大最差分支gain，S^τ为逐分支gain≥−τ下的最大自然频数平均gain。τ=1e−7。raw为显式可行fallback，M/S下界至少0。

为限制LP规模，可将active词集合T之外合为一个tail桶，但**这只产生外层松弛上界**。原始d_j必须保留完整词表TV，不可将d_j也合桶。

完整词表候选从active值与tail质量重构，tail形状固定为q在补集内的条件分布；若q补集质量在给定FP64数组中为0，用该组平均target补集形状；二者都为0时用最小token编号作为合法退路。候选空间只是用于提供可行下界，不能拿它替代完整simplex上界。

对任何这样的合法r：

```text
0 <= TV_full(p_j,r) - TV_bucket(p_j,r) <= min(p_j(U),r(U)) <= p_j(U)
```

因此压缩M的有效优化上界与完整候选下界之间，可用最大target尾质量+优化间隙+数值guard控制；必须实算完整loss验证。active按所有p_j的需求选择：以max_j p_j(v)降序、token编号破同分，使用64/256/1024/4096/16384固定阶梯，目标max_j p_j(U)≤1e−6。不能只取raw Top256。达到支持集/时间上限仍不闭合就输出区间。

**S^τ需额外核验逐分支可行性：** 合桶LP满足安全约束不代表完整r满足。只有完整gain减数值guard≥−τ的候选才可贡献S^τ下界；否则保留raw或已核可行候选。可用已认证M>0的候选与该候选按凸性所需最小混合量恢复安全，再完整检查，记录混合量与两个来源。若没有这样的见证，下界保留0，不能把合桶值或违反安全的候选当下界。合桶S^τ的有效上界仍可保留。

优化上界需导出dual/Lagrangian证书并独立重算，不能只信solver.fun/success。R4 `tv_oracles.py`的box-infimum思路可复用，但它的操作集不同、默认threads=2，必须在新run适配，改成threads=1；不能直接拿旧函数当full-simplex oracle。

在启动优化前即保存合法初始界：M∈[0,min_j d_j]，S^τ∈[0,A的认证上界]（若A尚未计算则用E作宽上界），附浮点guard。即使LP没有闭合，也必须留下有效区间；不能把缺优化值填0。若M被几乎已正确的分支卡住，单列这一情况，不据M小宣称历史冲突。

每个一般信息组×raw的M与S^τ**合计最多60秒求解器wall**，包含全部active阶梯/迭代，不是每次LP各60秒；两目标各预留30秒，不把一个做完后挤掉另一个。全轮硬预算另约束。M/S目标区间宽度≤1e−4；更宽不是不可修，只是未判定。保留求解器终止原因、对偶残差、浮点guard、候选原始/投影后质量与所有branch gains。

分组聚合：A=E−Σw_z R_z，M=min_z M_z，S^τ=Σw_z S_z^τ。上下界分别按单调规则组合，不把几个区间的端点当同时可达的点分解。

## 6. 必须通过的数学合同

至少覆盖：单目标、全部相同、自然重复权重、非均匀权重、零概率、ties、q已等于某target、两个target解析结果、三者逐对有解却整体M=0，以及M=0但S>0。

两个必须出现的反误判fixture：

1. p1=e1,p2=e2,p3=e3，q=(1/3,1/3,1/3)：逐对M>0，整体M=0。
2. p1=(.5,.5,0,0)、p2=(0,0,.5,.5)、p3=(.5,0,.5,0)、q=(.25,.25,.25,.25)：M=0，但存在部分改善、其余持平的r=(.30,.20,.25,.25)，所以S>0。

再用p1=(1,0)、p2=(0,1)、q=(.5,.5)、权重(.75,.25)检查A>0而严格S=0；这不是新模型证据。

小词表独立直接LP核对平均闭式、M、S^τ与tail松弛界。完整词表还要核概率身份、R嵌套、0≤M≤S^τ≤A（带guard）、信息细化边界单调、IH参照和i3的I2恒等控制。fixture不当成自然出现的实测结果。

## 7. 假分组、统计与案例

全部36题风险面板做不同H等权的辅助计算。固定16次假分组：I1在该面板不同H之间置乱真实I1标签；I2在每个真实I1组内部置乱真实I2标签。各自保留标签计数，不能拆开同一个完整H。I0不置乱，IH与i3/I2的零风险只作控制。

主seed=2026101591，bootstrap=2026101592。每个置乱seed由canonical JSON数组`[master,prompt_sha,anchor,source,position,info_level,replicate]`取SHA256前16字节big-endian，显式PCG64；不使用Python hash。H按token序列字典序固定。

假分组不优化M/S，不宣称其排除了M/S中的组数效应。16次仅作固定随机化参照，不能声称精确检验p值。保留自然与distinct两个测度，不跨口径相除。

按THEORY_PLAN给类别与ε=0.01；同时保留全部连续区间。汇总允许描述性95%整题分层bootstrap5000次，P/D/F单列，D/J同题配对；数值区间与抽样区间分开。未认证面板必须留在总分母，给可确认比例的下/上界，不仅对成功求解面板报占比。

最多6个固定排序案例；完整词表贡献用于解释见证，不拟合成败预测器、不按当前头坏状态挑样本。未观察到某类就写未观察到。

## 8. 资源与完成标准

全轮跨所有run累计：GPU=0；CPU进程树user+system≤24小时；新增磁盘≤16GiB；自有RSS峰值≤32GiB。CPU单数值worker，各数值库/HiGHS线程均1，import前设OMP/MKL/OPENBLAS/NUMEXPR线程1；同一时刻不叠加数值任务。只读解压与临时文件也计资源，不复制完整旧存档。

逐context流式处理；平均解与假分组不重复保存完整V维向量，保存输入SHA、分组映射、k/alpha等可重构信息。优化见证保存active值、tail质量、tail形状引用及完整评估结果；最多6个展示案例另保留完整词表账本。任何压缩存储都必须能重建并复核见证，不以磁盘上限为由省略证书。

只管理核实UID/PID/PGID/startticks/CMD/run身份的自有进程。不得终止旧controller/parallel/收尾或同UID其他任务。本轮无GPU预约可释放，不捏造GPU准入检查。

先做元数据成本预估及固定dev计时；若预计超预算，优先缓存/流式/避免重复排序等不改科学问题的实现优化。仍无法覆盖固定工作则partial，不改样本、阈值或只选容易求解组。

输出至少包含：

```text
protocol/{LOCK,INPUT_MANIFEST,POPULATIONS,RESOURCE_POLICY}.json
protocol/CORE_ALIGNMENT_CN.md
analysis/{MATH_CONTRACTS,IDENTITY_AUDIT,MEAN_RISK,SHAM_CONTROL,
          COMMON_BOUNDARIES,CERTIFICATE_AUDIT,COVERAGE,CLASSIFICATION}.json
results/{RESULTS_CN.md,THEORY_CARD_CN.md,MECHANISM_CASES_CN.md,
         CLAIM_EVIDENCE.json,RUN_AUDIT.json,RUN_FILE_MANIFEST.json}
COMPLETION.json
NEXT_CHAT.md
```

固定全部面板已按预算合同尝试并给合法区间/未判定说明，可标工作complete，科学结论另写支持/反证/不确定；必要面板连计算和合法初始界都未完成则partial。complete不等于核心理论成立。

报告写完再次执行核心问题审核：是否逐项回答形成机制、可修条件、失败边界？若只能给平均收益，重写分析解释；若证据无法判定，具体写出未判定范围，不补造答案、不追加支路。退出自有CPU作业，记录资源与静态文件SHA，保留旧概率/线程/证书披露，不自动第十轮。
