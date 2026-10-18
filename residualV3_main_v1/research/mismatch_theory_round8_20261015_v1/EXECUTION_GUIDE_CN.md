# 第八轮执行指南

配合 [研究设计](THEORY_PLAN_CN.md)、[启动prompt](START_PROMPT_CN.md) 和 [固定参数](protocol/DESIGN_SPEC.json)。编写chat没有运行实验；执行chat收到prompt后，在本指南范围内自主完成。

## 1. 运行范围

仓库 `/data/weijiahong/speculators-main-lora`。只写：

```text
residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/<UTC>_mismatch_r8/
```

允许：新run内复制/改造采集与分析代码；旧资产只读；6道旧开发题的工程重放；固定36道新机制题、两主体×三历史来源采集；固定概率探针、参考验证、统计/图表/报告与清理。

禁止：训练主体/头/预测器；更换结构、损失、输入；剂量/K/种子网格；新LP/oracle优化；安装/升级依赖；改冻结源/权重/旧协议/自动报告；启动生产native、RNG比较或TPS实验；启停原controller/parallel/CPU收尾；预算剩余时追加支路。E1/RAW不在本轮主或次要采集范围。

两项研究一并实施，不把A效果阳性作为B准入。不因任何科学阴性、低覆盖或不够新颖重采、换题、换规则。

## 2. 资料、资产与代码复用

按 [资料索引](sources/READING_GUIDE_CN.md) 读取R7终态、R1两主体身份、R5共同流和R2概率边界。先核R7 COMPLETION/签名以及将实际读取的源文件SHA；不必为开始本轮重算七轮全部大型数据SHA。记录验证范围，不声称全档已重验。

模型使用正式E2导出：`epoch1_epoch2_native_compare_gsm8k_20261010_v1/models/E2_DFLASH`、`E2_JOINT_NATIVE`、`E2_JOINT_HEAD`。精确路径、basis、target及配置从R7 LOCK和正式plan交叉读取，不从文件名猜。保留target原有stat身份限制：当前新算SHA不补造历史全SHA证明。

R1 `FrozenABReference` 已有两主体加载及target_layer_ids一致性核验；R7提供自然C、冻结原头、BF16位保存、fit/eval导出和资源包装。**复制到新run再改ROOT/入口**；import前设 `PYTHONDONTWRITEBYTECODE=1`，缓存也放新run，不直接运行旧main。

不能原样套旧collector：它只取joint，历史来自target，默认采样为FP32，而且只有五个保存位置。本轮必须增加D主体、D/F生成、统一FP64概率采样和全部15位置参考验证字段。

R5 M/2共同流可复用；R7与原头d相交的c不是本轮主探针。旧全量read_capture会接触eval效果，不能在新fit-only动作入口直接调用。

## 3. 顺序与工程准入

1. **只读审计。** 核对R7完成、资产、最新暴露、可用面板。建立新run和总预算账本。
2. **CPU合同。** 两主体/三来源映射、W/H键、索引、三桶权重、M/2动作、主端点/NA、共同支持标准化、first-reject/EOS/预算及缺失fixture。
3. **固定旧dev6题。** 从R7原fresh36每域按group SHA排序取前2题。每来源K16=fit4/eval12，anchor1/65；旧C可原样重放以测工程，明确是旧题旧C上的新FP64条件。不能把旧FP32分支冒充新自然分支。dev只判断身份、资源和实现，不据效果改科学设计。
4. **锁定。** 完成采样buffer、两主体和头身份检查；写实际代码SHA、模型身份、题面、全部seed、访问/统计/资源规则；新题采集前签名LOCK。
5. **固定新36题共同采集。** 每完成一个C流式落盘P/D/F及D/J资料，所有来源都完成后再转下一C；不先跑完某一问题再按效果决定另一问题。
6. **动作封存。** 按来源导出fit/eval bank，构造所有合格fit组动作并签名，之后评估eval。P主动作入口不能读D/F-fit或eval target来选择方向/幅度。
7. **一次主分析与报告。** 六格、四端点、三桶、严格novel、来源标准化、同源探针、失败账本和参考reached均按固定规则输出，结束后收尾。

工程失败可定位修复；每阶段最多2次有新诊断的技术重试，失败时间和产物计入原总账。只续缺失分片，不重新运行完成且身份通过的分片。新eval效果暴露后若发现影响科学结果的代码错误，保留原产物和更正说明；更正重分析标post-exposure，不重称原封存确认。必要工程不能成立或预算不足时partial，不伪造complete。

dev资源测量给出全轮耗时/磁盘上界估计。若预计超过剩余硬上限，先做不改变科学定义的流式/缓存/微批优化；仍不可行就保存已有工作和阻碍，不能擅减新题/K或超支。估计不构成完成时间承诺。

## 4. 题面、分支与种子

新36题：GSM8K/HumanEval/Alpaca各12，沿R7三个固定panel及DeepSpec提交。按最新全部机制manifest、专家/headonly等实际暴露记录排除；继承NFKC/空白/GSM后缀规范化、完整messages与token身份核验。机制未见不等于未参加正式评测或预训练未见。

编写时发现的机制记录排除后余量48/12/422，见INPUT_AUDIT；它不是未来准入。HumanEval仅够12；若执行时不足，保持旧数据工程可做部分，报告新确认partial，不用已暴露题、不替换域、不缩n。旧MT-Bench已全部机制暴露，不引入。

按合格group SHA排序，以 `Random(2026101508 + sum(ord(c) for c in domain))` 洗牌取前12。长度资格 `prompt_tokens+512<=4096`，禁止截prompt。保留选中题后自然缺anchor/EOS，不能补题追覆盖。

新题的块前C由target新FP64参考采样产生，最多65 token；anchor1/65；anchor本身若EOS或不存在记缺失。固定15 proposal，T1、完整词表、无top-k/top-p裁剪和额外penalty，EOS/config从原锁读取。每C留够15位置且不超过context4096。只生成到自然EOS/15/预算三者先到；当前位置抽到EOS时该位置仍有效，以后无效。

三来源P/D/F各fit16/eval48。主位置3/8/15，控制1/2；中间位置必须连续生成。原始q_D/q_J每C/i只算和存一份。两主体同一H的target概率只评一次，可引用去重缓存，但自然重复抽样槽仍分别保留频数。

固定种子：master `2026101538`，bootstrap `2026101539`。用SHA256将完整元组映射到128位整数：

```text
(master, namespace, prompt_group_SHA, anchor, source, pool, branch_slot, position)
namespace ∈ {context, proposal, reject}
pool ∈ {context, dev_fit, dev_eval, fresh_fit, fresh_eval}
```

序列化采用UTF-8 canonical JSON数组（无空格），取SHA256前16字节的big-endian整数。context使用固定anchor=0/source=P/pool=context/branch=0，位置为上下文生成步；一次最多65步的轨迹同时供anchor1/65使用，不分别生成两条C轨迹。各proposal/reject使用自己的source/pool/slot/i。每个元组初始化独立PCG64，先检查不同逻辑元组的seed唯一性；同一元组的缓存重用不算碰撞。不同元组冲突则fail，不加盐救跑。不要使用Python内置hash或torch全局RNG。锁Python/NumPy版本及显式SEED_TABLE；统计bootstrap用独立PCG64。

## 5. 三种q与采样身份

每条记录分名：

```text
raw_q_D / raw_q_J        # 被诊断对象；与来源H无关，但依赖C/i
generation_q_S          # 产生该历史实际token的分布；F时为完整头q_F
probe_q_m_from_P_fit    # q_m+a_m^P；仅诊断，绝不喂给生成器/验证器
target_p_on_H           # 在这条完整实际H下的target
```

所有sampling logits先保存/绑定原BF16位，按锁定稳定FP64 full softmax构造唯一完整概率缓冲；CPU `numpy.random.Generator(PCG64(seed)).choice(V,p=buffer)`直接使用该缓冲。P/D/F以及块前C均遵守同一规则。验证所用q是**该次传给choice的原缓冲**，不能再用FP32 softmax或另一次重算替代。记录dtype、算法/版本、buffer SHA与选中token概率。

F每一步用GPU原冻结头，读取当前J hidden、实际上一采样token、原始q特征；仅替换原Top256 logit，FP64完整词表采样，下一步反馈实际抽到的token。不递推完整soft，不启aux/truncation/expert。采样FP64不意味着把头内部原FP32/BF16计算擅自改为FP64。

dev检查：

- 固定同C/i重复调用D/J各自raw稳定；跨模型只对齐C、索引、target taps，不断言q相同。
- F同C/i/前驱的完整可见输入与原头更新一致，原/复用头全词表TV容差2e−5；仅声明实际检查的dev记录数量，不捏造fresh逐条parity。
- 对所有dev D/F eval合法位置保存full generation/target buffers，独立CPU复现采样输入身份与全部first-reject；fresh按锁定代码和下述记录继承准入。
- i1/2主要检查索引和历史身份；不足2种完整H是结构性NA，不放宽共同组资格。

## 6. 数据保存与参考验证

每状态保存C/i、source/pool/slot、完整H与前驱、选择token、q/p buffer身份、模型与概率口径。主/控制位置1/2/3/8/15保存全词表target BF16 logits；两主体raw全15位置、F实际更新后的candidate logits按C/i/前驱去重保存。

**全部D/F eval合法位置1..15**保存：sampled-token p、actual q、生成/target缓冲SHA、独立验证uniform、threshold、accept、reached、first_reject、当前及先前EOS、轮初预算、有效路径长度。P只作target分支诊断，不把其自我验证混入D/F。

参考规则：q(selected)>0必须成立；阈值min(1,p(selected)/q(selected))，uniform来自reject命名空间，接受比较固定`u<threshold`。reached只依赖此前有效位置的接受，含当前拒绝位；首拒后不再累计接受，采集的后续proposal历史仍可用于离线状态诊断。先前EOS/预算门控不包含本位接受事件。

fresh其余10个位置仅保留sampled-token标量和buffer SHA，不宣称可离线重构每个完整概率向量；能重放first-reject决策，并通过采集源码/运行断言绑定标量来自实际缓冲。需增强身份检查只能按预定dev完成，不在fresh效果后选例加证。

参考验证只分析原D/F策略，**探针不改变proposal、不重抽token、不产生bonus/residual输出**；这不是完整生产decoder。只报原参考块接受/到达计数及reached状态上的G，不宣称probe部署接受提升、native无偏或TPS。旧R2警报原样保留。

## 7. fit/eval访问与数值合同

独立导出P/D/F的fit/eval target bank。元数据白名单只留身份、H、索引、pool/source、采样/验证记录等固定字段，不含TV/gain或任何效果摘要。动作入口读取对应fit target、两主体raw和身份；P主动作入口禁止读其他来源fit/eval数值。构造**所有合格fit组**，不按是否有eval匹配或好效果挑组。

先签全部 `ACTION_BANK_SEALED.json`，后运行评估。完整性审计可以读取原始eval字节，但它不向动作构造回传数值/效果；准确记录入口、文件访问和能力边界，不宣称同UID实现了OS安全隔离。

FP64合同默认绝对容差1e−10：概率非负、和为1、sum a=0、TV(q,q+a)=M/2、每个fit H的G=M/2、三桶贡献还原总体、标准化两项还原共同支持差。M≤1e−12时a=0、effective M=0，不做0/0；保留raw M。原BF16/FP32内部特征用各自原容差，不拿FP64合同要求GPU运算逐位等价。

不得clip、额外归一化、截tail合桶或放宽容差掩盖失败。replay需要确定性数值执行/缓存：同一C/H的p身份复用一致，否则fit_seen的数学保证不能直接检查为同观测。CPU合同覆盖M=0但可补偿修复、fit中两H、seen保证、三个新旧桶、未匹配NA、各类损伤、空路径/零reached和EOS场景；fixture不是自然出现的证据。

## 8. 统计、图表和问题清单

主统计严格按THEORY_PLAN第4–5节，不重新选择主集合。全部matched、new_to_P_fit、strict_allfit_novel和参考reached分表；所有自然重复H保留抽样频数，同一记录的意外副本去重。各来源之间相同H不当独立题面；bootstrap单位为题。

最终第一页用简单中文回答：

1. 独立与联合主体是否都存在共同可改善的偏差？各主CI/覆盖是什么？
2. 两者共同方向是否类似？只是总体误差类似，还是方向也类似？
3. P-fit动作到D/F历史是否仍有效？两个主端点独立回答。
4. 来源间差异有多少伴随W权重/覆盖改变，固定W后是否仍可观察？
5. 受损状态实际发生反向、越界还是正负补偿不足？给完整token账本支持。
6. 在参考first-reject真正到达的主位置是否仍有同样现象？不能据此宣称探针部署收益。
7. 哪些理论表述保留、收缩或仍不确定？

图最多4类：2×3收益/损伤及四主CI；自然→匹配→new_to_P_fit→strict/reached覆盖；共同W标准化与同源重新识别；固定真实案例的target需求与概率移动。R7历史效果可作背景，不与新采样口径数值拼成重复试验。

## 9. 共卡、预算和中断

**显存足够就用，允许共卡；不等待空卡、不要求utilization=0、不固定GPU0/7。** 每阶段实时查UUID、空余显存、预约与自有预计峰值。已有独占预约只跳过那张卡，不把共享环境全停用。

同时最多1卡；自有峰值默认≤32 GiB，准入为预计工作集+4 GiB余量。R7峰值只供估算，双主体和新buffer由dev测。必要时流式加载第二主体/缓存raw、减微批，不改变样本和科学定义，不量化模型。无卡容纳最低工作集时推进CPU工作并短周期复查，不重复询问共卡许可；等待时保持用户可见更新。

本轮所有run合计上限8 GPU小时、24 CPU小时、700 GiB新增磁盘。GPU进程存活墙钟含编译/等待/失败/共享变慢；CPU进程树user+system累计；所有临时文件、副本、失败分片算磁盘，换run不重置。保留报告和退出所需余量，接近上限先checkpoint并停止新分片。

CPU workers≤4、数值线程合计≤8。import前OMP/MKL/OPENBLAS/NUMEXPR均设1，Torch intra/inter-op各1，监控进程树；驱动/IO线程另记，不能掩盖数值超限。旧线程/证书记录缺口保留，不用本轮通过覆盖。

记录UID/PID/PGID/startticks/CMD/run路径/GPU UUID/lease；只操作逐项核实的自有进程和预约，不杀同UID他人任务或按裸PID判断。正常/异常退出都释放自有资源；保留原正式控制器及收尾终态。

## 10. 交付与完成标准

```text
protocol/{PROTOCOL,LOCK,SAMPLE_MANIFEST,EXPOSURE_LEDGER,SEED_TABLE,
          MODEL_IDENTITIES,SAMPLING_IDENTITY,STATISTICS_RULES,RESOURCE_POLICY}.json
analysis/{IDENTITY_AUDIT,INPUT_ACCESS_AUDIT,ACTION_BANK_SEALED,
          CROSS_BODY_SOURCE_MATRIX,PRIMARY_ENDPOINTS,COVERAGE,
          THREE_BUCKET_ACCOUNTING,SUPPORT_STANDARDIZATION,
          SOURCE_REFIT_DIAGNOSTIC,FAILURE_ACCOUNTING,
          REFERENCE_REACHED_AUDIT,NUMERIC_AUDIT}.json
results/{RESULTS_CN.md,METHODS_CN.md,MECHANISM_CASES_CN.md,
         THEORY_CARDS_CN.md,CLAIM_EVIDENCE.json,RUN_AUDIT.json,RUN_FILE_MANIFEST.json}
COMPLETION.json
NEXT_CHAT.md
```

全部固定采集/主分析/必要审计完成，即使结果阴性或覆盖不足也可complete，科学项另标unsupported/inconclusive。固定采集或必要分析缺失为partial，具体列未完成项；不要把coverage-inconclusive当技术partial，也不要反过来。

结束复核静态文件SHA、冻结资产、暴露和预算账本、自有进程退出与预约释放。只声明实际完成的验证范围。不自动发起第九轮，不追加训练“解决”本轮负结果。
