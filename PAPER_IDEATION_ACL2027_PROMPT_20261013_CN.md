# ACL 2027论文构思专用启动prompt

复制分隔线以下全文作为新对话的第一条消息。本文件的动态结果快照为北京时间2026-10-13 19:45左右，接手必须重新读取实时文件。

---

你现在协助我构思一篇以现有ResidualV3工作为主体、目标为**ACL 2027主会长文**的论文。仓库是`/data/weijiahong/speculators-main-lora`。

## 1. 任务与用户已明确的偏好

**这次专门研究论文构思、科学问题、贡献表述和论证逻辑。先使用已有结构及实验，不发明新结构，不把任务自动转成追加训练、调参或大规模复现。**

我已经反复纠正过方向：希望**从mismatch出发论述我们的工作**，再解释现有结构和训练为什么是对这个问题的回应。不要先罗列hidden/q/三层/方向/剂量等模块，再给每个模块包装一个创新名称。

论文构思应当回答：

1. 我们研究的mismatch具体是什么？哪些是已有认识，哪些是本工作可支持的新贡献？
2. mismatch怎样导致投机解码中的连续前缀接受损失？
3. 现有方法修复什么、依靠什么信息、在哪些条件下可能有效？
4. 相对Domino、DSpark和DFlash-2，我们有什么准确、可检验的区别与价值？结构区别、待验证优势和已经测得的优势要分开。
5. 现有结果能支撑哪些主张，如何组成一篇逻辑自洽的ACL论文？

不要为了声称全面优于三个近邻而歪曲它们。优先构建有明确问题、方法、证据与边界的论文故事。补证据可以作为建议列出，不能当成已完成工作，也不构成启动新GPU任务的授权。

本对话以只读研究、论证稿和论文构思为主。既有评测由原控制器和CPU收尾程序继续运行；不要启停GPU进程、另启动`parallel.py`、重跑已完成实验、修改冻结源码/权重/协议，或直接改自动生成的结果报告。先交付论文构思，不要求我重复确认已有事实。

## 2. 首先读取的材料

进入仓库后，按需读取以下材料。遇到动态状态冲突，以实时结果为准。

### 基本事实与已有汇报

- `HANDOFF_20261013_CN.md`
- `HANDOFF_20261013_STATE.json`
- `HANDOFF_20261013_PROCESS_SNAPSHOT.json`：这是13:01旧进程快照，不是当前状态。
- `GROUP_MEETING_20261013_CN.md`：结构、训练、RAW/E1/E2、旧三领域主结果、相关工作与模块成本。
- `RESIDUALV3_START_HERE.txt`
- `EXPERIMENTS_NONTHINKING.md`、`EXPERIMENTS_THINKING.md`：完整历史。当前论文主体是nonthinking，thinking结果不能混入同一主表。
- `HANDOFF_20261011_CN.md`：完整沿革与负结果；旧动态状态不能覆盖新约束。

### 正在运行的六数据集批次

设：

```text
R=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
K=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1
```

先读R下：

- `status.json`、`progress.json`、`RESULTS_CN.md`、`COMPARISON.json`
- `plan.json`、`DATASET_AUDIT.json`、`PREFLIGHT.json`
- `parallel/controller.log`、`parallel/controller_identity.json`、`parallel/DISPATCH_MANIFEST.json`
- `FINAL_AUDIT.json`若存在；缺失表示不能宣称整批最终审计通过。
- 已完成数据集的`{dataset}/C{1|32}/formal/summary.json`；当前Arena另读`arena-hard-v2_C1_formal.log`及其formal/status。

CPU收尾程序在：

```text
residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1/
```

读其`README_CN.md`、`identity.json`、`status.json`、`watcher.log`；若存在则读`COMPLETION_AUDIT.json`。
它在原控制器成功结束后核对48窗/24配对/16000请求和最终审计，再锁定更新总账及周报4.5。不要重复启动收尾程序。

### 结构与训练源码

- `residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/repair/model.py`
- 同根`native_runtime/runtime/components.py`
- `residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/head_config.json`
- `residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/losses.py`
- E1协议：`residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/protocol.json`
- E2协议：`residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/protocol.json`

### 文献证据

目录：`residualV3_main_v1/research/group_meeting_sources_20261013/`。
包含Domino/DSpark原文HTML、DFlash-2作者报告及模型卡、机器抽取原表、`TABLE_ROWS.json`、`FETCH.json`、`REPORT_AUDIT.json`、`DSPARK_TRUNCATION.json`。
具体论文主张应核对一手来源；已保存快照与最新网页如有差异，注明版本，不能悄悄混用。

## 3. 现有方法：准确描述，不添加不存在的机制

### 推理结构

- target为Qwen3-8B，冻结；主体为五层并行DFlash。
- 主体与原始LM投影对整块只算一次。block16是anchor＋15 proposal。
- 修复头逐位置运行。三路输入为当前位置hidden、实际前驱token的冻结target embedding、当前位置原始q分布的basis特征。
- 首位置前驱为anchor，后续位置前驱为上一位置**修复后实际采样**的token。
- hidden维4096；三路各投影至256，融合后通过三个256维SwiGLU残差块，每块inner768。
- 输入/输出basis rank512。q特征由原始Top256候选概率加权的冻结输入basis构成。
- 头输出512维修正方向和一个剂量；方向归一化，剂量为sigmoid×0.8，并按hidden范数缩放。
- 在当前原始Top256候选上施加logit残差，**最终仍从修复后完整词表分布采样**。
- 输入basis和输出basis用途不同；不能混称一张表。
- 新增可训练头参数为**4,353,537**。冻结basis、embedding及缓存仍占存储/显存，不能把新增可训练参数少等同于总显存少。
- 三层指三个SwiGLU残差块，不是三层Transformer。
- `no-soft`表示不递推上一位置完整soft分布为第四路输入；不是取消softmax或概率采样。
- 当前正式FULL15_NOFILL不启用尾部截断、辅助预测器或专家路由。NOFILL是取消局部调试empty预填充的实现版本，不是单独科学贡献。

文字流程：

```text
已验证上下文与anchor
  → 五层并行主体一次计算整块hidden/raw logits
  → 当前hidden＋实际前驱＋当前q特征
  → 三层修复头预测方向与剂量
  → 在Top256上施加残差，完整词表采样
  → 当前样本反馈给下一位置
  → target按原生规则验证完整草稿
  → 接受连续前缀与bonus
```

### 训练

- 独立DFlash与联合主体＋三层头从同一下载的nonthinking DFlash起点出发，各追加两个完整epoch，不是从随机初始化开始。
- 每轮833,176有效会话组，约1.019B输入token；两轮是1,666,352次呈现，不是这么多不同独立样本。
- max4096，每包≤65,536输入token、≤1,536完整anchors，每包更新一次。
- E1更新15,847次，E2更新15,854次，合计31,701次/模型。
- 一个共享冻结teacher供应两个独立learner，梯度、权重和优化器分别维护。
- 独立分支训练raw CE/D-PACE；联合分支训练raw CE/D-PACE＋repair CE/D-PACE＋repair OUTPUT，三个主项按1:1:1相加。
- CE使用teacher argmax。修复训练使用teacher-prefix真实前驱，推理使用实际采样前驱；不是额外native rollout标签训练。
- OUTPUT为teacher-prefix下的前缀存活代理：令`o_i=sum_v min(p_target_i(v),q_repair_i(v))`，则`L_OUTPUT=1-(1/15)*sum_{k=1}^{15}prod_{i=1}^{k}o_i`。
- E2的detach辅助预测器已训练，但正式FULL关闭。E1/E2权重、完整master/Muon/Adam/RNG/cursor断点必须保留；epoch12就是epoch1＋epoch2，没有epoch3。

## 4. 从mismatch出发的候选论证，尚不能当作已证实机制结论

此前讨论形成的候选主线是：

```text
并行预测未根据块内实际采样前缀更新
  → 与target在该实际前缀下要求的条件分布可能失配
  → 概率重叠降低
  → 连续前缀接受规则放大输出损失
  → 利用原始预测状态与实际前驱学习有限修正
  → 检验接受长度收益能否超过修复成本
```

可以形式化`q_raw_i(v|C,i)`与`p_target_i(v|C,x_<i)`的条件信息差异。不要把实际训练出的q_raw直接等同于target的精确边缘分布，也不要把所有p/q偏差都归因于缺少前驱；小模型容量、训练与估计误差也可能参与。

固定状态下，T1标准拒绝采样的期望接受概率等于`sum_v min(p(v),q(v))=1-TV(p,q)`。T0要用贪心一致性，不能直接套T1概率重叠解释全部结果。

务必守住以下边界：

- 头推理时看不到当前位置target分布，学习的是有用修正，不是在线测得真实mismatch或显式错误标签。
- q是原始预测的显式表示，不是独立于hidden的新信息源；其价值需要证据支持。
- 当前头显式读取紧邻前驱，没有GRU式完整块内历史；不能宣称恢复完整target条件分布。
- 方向/剂量限制修正，不自动构成KL信任域、置信度校准或理论最优修复。
- OUTPUT是teacher-prefix代理，不直接等于真实native rollout的期望接受长度。
- 主表两臂主体权重不同。整体模型AC/TPS正收益不能全部归因于推理加头，也不能单凭主表证明mismatch是唯一原因。
- 条件mismatch及轻量顺序修正已有近邻讨论，不能宣称首次发现问题、首次补充因果依赖或首次前缀加权训练。

请把这些候选表述作为待推敲的论文构思，而不是我已经确认的唯一结论。

## 5. 之前已经完成的关键实验

以下是关键摘要。完整历史在总账和原始报告；不要重跑，也不要把不同协议的绝对TPS直接相除。

### 5.1 旧E2三领域正式主结果：12个C1/C32配对

独立E2 DFlash→联合E2完整三层FULL15_NOFILL。T0/T1、15 proposal、max2096、seed42、BF16/batch-invariant/Graph。
GSM8K C1=200/C32=1319；HumanEval164；MT-Bench80首轮。正式KV C1=52GiB/C32=44GiB、util0.92。

| 数据集 | 设置 | AC增长 | TPS增长 |
|---|---|---:|---:|
| GSM8K | C1 T0 | +13.86% | +9.94% |
| GSM8K | C1 T1 | +18.19% | +14.09% |
| GSM8K | C32 T0 | +13.70% | +6.86% |
| GSM8K | C32 T1 | +18.75% | +11.26% |
| HumanEval | C1 T0 | +14.48% | +10.85% |
| HumanEval | C1 T1 | +21.50% | +17.48% |
| HumanEval | C32 T0 | +14.46% | +9.43% |
| HumanEval | C32 T1 | +18.94% | +14.89% |
| MT-Bench | C1 T0 | +8.52% | +6.78% |
| MT-Bench | C1 T1 | +10.66% | +8.78% |
| MT-Bench | C32 T0 | +8.48% | +3.17% |
| MT-Bench | C32 T1 | +13.47% | +7.23% |

原始COMPARISON/summary在：

- `residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/`
- `residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/`
- `residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/`

以上隔离/Graph审计通过，旧HumanEval/MT-Bench的T0逐题输出一致。历史C40另外保留：GSM8K TPS+6.98%/+12.04%、HumanEval+7.47%/+14.58%、MT-Bench+3.97%/+8.43%，均按T0/T1；不能把C40写成C32。

### 5.2 RAW与E1/E2历史

RAW是下载的完整nonthinking DFlash，无本项目追加训练。RAW/E1历史采用KV36GiB、util0.70、acceptance_only=true、speed_evidence=false，只能作为AC历史：

- GSM8K C1 T0：RAW6.738→E1单训7.421/联合8.414；T1：5.741→6.266/7.356。
- C40 T0：6.858→7.591/8.545；T1：5.961→6.374/7.651。
- 不计算RAW→E2吞吐，不把RAW填进当前C32或其他领域。

E1/E2统一runtime比较为16正式窗，结果根：
`residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/`。
GSM8K C1=200/C40=1319、max2096、KV52/46GiB、util0.92，无辅助/截断；每轮联合模型相对同轮独立DFlash：

| epoch | 设置 | AC增长 | TPS增长 |
|---|---|---:|---:|
| E1 | C1 T0 | +13.50% | +6.03% |
| E1 | C1 T1 | +17.34% | +9.34% |
| E1 | C40 T0 | +12.79% | +4.67% |
| E1 | C40 T1 | +20.13% | +11.54% |
| E2 | C1 T0 | +13.86% | +6.20% |
| E2 | C1 T1 | +18.49% | +10.27% |
| E2 | C40 T0 | +13.70% | +5.68% |
| E2 | C40 T1 | +18.92% | +10.21% |

联合模型E2相对E1的AC增量0.56%–1.60%，TPS增量0.51%–1.60%。只支持这个GSM8K面板上的增量，不证明完全收敛。
后来NOFILL、Graph模式、KV和批次发生变化；不能把历史E2→新正式版的TPS差全部归于多训一轮或某个kernel优化。

### 5.3 两层/三层、模块成本与负结果

- 两层/三层直接对照根：`residualV3-nonthing/results/epoch2_head2_vs_head3_three_domains_C1_C32_20261012_v1/`。
  24正式窗完成；同一联合主体与同一三层权重，两层只是推理跳第三块，未重训。三层→两层AC12配对全降，TPS11/12配对下降，唯一正例HumanEval C32 T1约+5.03%。该对照C1 KV16GiB/C32 44GiB，不与正式C1 KV52GiB绝对TPS混算。不能据此证明三层全局最优。
- 三层模块计时根：`residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1/`。
  只有C1/C40，没有C32模块时间。C1 T0/T1的ms/批/block：主体4.460/4.461、raw LM1.544/1.545、头＋采样1.297/1.340、完整propose8.058/8.095、target forward31.936/31.955、target起点至验证终点33.511/33.545。跨度嵌套，不能相加或改写成每token延迟。
  满40直接组件T0仅3份、T1缺失；见`TIMING_COVERAGE_CN.md`。细事件重放约有2.2ms插桩扰动，不能当真实服务成本。
- 旧尾部停修iter28/29已结束，CUT/FULL四窗均无净TPS收益；不把它包装成第二创新。
- 30k专家三种子静态训练已完成，learned mismatch router未胜通用头，没有native收益。用户要求先不放进周报；论文构思阶段也不以它作为正贡献。原始负结果与资产保留。
- thinking的层数/方向剂量/输入等历史消融，以及早期nonthinking诊断，可以从两总账追溯。若引用，必须检查模型、训练数据、实现和指标身份，不能冒充当前E2配方的完整贡献消融。

## 6. 正在跑的新六数据集实验：19:45快照

两臂仅E2独立完整单训DFLASH与E2联合主体＋三层FULL15_NOFILL，主体权重不同。
C1/C32、T0/T1、BF16、batch-invariant、CUDA Graph、nonthinking，固定15 proposal、max2048，无截断/辅助/专家。
固定KV C1=52GiB/C32=44GiB、util0.92，smoke与formal相同。五套context4096，Arena context12288；无prompt截断。

数据固定DeepSpec提交`005e03b81cec38b7da6399833d609ee89a2587f2`，按作者抽样顺序与首轮，请求seed980406＋index：

| 数据集 | 每窗来源行 | 不同题面 |
|---|---:|---:|
| HumanEval | 164 | 164 |
| MBPP | 256 | 255 |
| LiveCodeBench | 500 | 500 |
| MT-Bench | 80 | 80 |
| Alpaca | 500 | 500 |
| Arena-Hard-v2 | 500 | 500 |

MBPP重复是作者上游来源重复，原样保留。每模型×温度×并发共2000来源行；全批48正式窗、24配对、16000请求，不是16000不同题。
本批对齐DSpark数据及部分口径，仍用我们的vLLM与15 proposal，DSpark离线主表是7 proposal，不是完整DSpark复现。
旧max2096/seed42批次必须保留；不跨批TPS相除或把旧新结果拼成一组重复试验。

**北京时间2026-10-13 19:45左右：42/48有效正式窗、20/24配对。**

- HumanEval、MBPP、LiveCodeBench、MT-Bench、Alpaca的C1/C32全部完成。
- Arena C1在GPU6 formal，独立DFLASH T0/T1已完成，联合FULL15_NOFILL正在测T0；C32尚未开始。
- 尚有6个正式窗未计入完成汇总、4个配对未形成；正在执行的请求包含在内。
- 控制器PID4138310、UID1024；CPU收尾PID664385、UID1024。二者19:45在宿主重新核验存活。
- Arena evaluator快照PID1000514、UID1024。PID/卡号是快照；若核验进程，使用宿主uid/cmd与GPU查询，不能把沙箱不可见当作退出。
- 此时`FINAL_AUDIT.json`和收尾`COMPLETION_AUDIT.json`尚不存在；整批最终审计没有完成。

### 已完成20配对的新批次增幅

以下来自R/COMPARISON.json未舍入原值的两位小数展示，**不能与第5节旧批次混用**：

| 数据集 | 设置 | AC增长 | TPS增长 |
|---|---|---:|---:|
| HumanEval | C1 T0 | +14.48% | +10.57% |
| HumanEval | C1 T1 | +18.43% | +14.38% |
| HumanEval | C32 T0 | +14.46% | +8.81% |
| HumanEval | C32 T1 | +18.03% | +10.80% |
| MBPP | C1 T0 | +14.46% | +10.70% |
| MBPP | C1 T1 | +16.28% | +12.36% |
| MBPP | C32 T0 | +14.32% | +8.28% |
| MBPP | C32 T1 | +16.80% | +24.54% |
| LiveCodeBench | C1 T0 | +9.67% | +5.78% |
| LiveCodeBench | C1 T1 | +15.81% | +11.71% |
| LiveCodeBench | C32 T0 | +9.72% | +2.10% |
| LiveCodeBench | C32 T1 | +13.82% | +8.03% |
| MT-Bench | C1 T0 | +8.52% | +5.12% |
| MT-Bench | C1 T1 | +15.35% | +11.53% |
| MT-Bench | C32 T0 | +8.48% | +2.42% |
| MT-Bench | C32 T1 | +14.61% | +9.23% |
| Alpaca | C1 T0 | +8.93% | +5.43% |
| Alpaca | C1 T1 | +11.86% | +8.14% |
| Alpaca | C32 T0 | +8.95% | +3.63% |
| Alpaca | C32 T1 | +12.27% | +8.92% |

这20个条件的TPS增幅中位数约8.86%、AC增幅中位数约14.39%。它们不是20个独立重复试验，不支持据此虚构统计显著性。

Arena C1独立DFLASH日志已出：T0 AC3.3409528897708407、TPS80.58077252532357；T1 AC2.680992150076493、TPS64.54447975764681。
联合头尚未完成，所以没有Arena有效配对增幅，不能估算、填0或复制其他数据集结果。

AC/AL包含bonus，不是答题准确率。本批没有代码pass@1或对话judge评分；不能把接受长度增长写成任务质量增长。

## 7. 近邻工作与已经核验的口径

### Domino

- 一手来源：`https://arxiv.org/html/2605.29707v1`。
- 五层并行主体＋GRU累计采样前缀＋低秩全词表logit残差；teacher forcing与base-anchored curriculum。
- 主表1/2作者OPB训练Domino，baseline为公开DFlash权重；不能声称双方训练预算已对齐。
- 表3双方ShareGPT重训，更适合讨论匹配训练对照；与主实验是同一头架构，不是“头版/完整版”两种结构。
- 表3 AC未分并发，TPS由AR TPS×已舍入速度比推算时必须写≈；表2温度未注明，不补成T0。
- 同联合主体关/开头的表6不能替代独立单训DFlash基线。

### DSpark

- 一手来源：`https://arxiv.org/html/2607.05147v1`；作者代码固定提交如上。
- 默认Markov头用实际前驱对应的低秩转移偏置；另有RNN变体。confidence head与硬件感知前缀调度属于完整方法。
- 默认Markov**修正项**不直接依赖当前hidden/q，但最终分布通过原始logits仍依赖上下文。RNN变体另可读hidden，不能把默认头的差异推广到所有DSpark。
- 离线主表：Qwen3-4B/8B/14B和Gemma4-12B，Eagle3/DFlash/DSpark统一框架、同OPB数据重训10轮；T1、7 proposal，九套数学/代码/对话数据；**验证调度关闭**，报告AC，不是C1/C32服务TPS主表。
- 完整生产系统第5节另测DeepSeek-V4、最多5 proposal、调度开启，baseline为MTP-1。不能把该完整系统吞吐增益填成相对DFlash或单个头的收益。
- 静态阈值裁剪、置信度校准与动态生产调度证据分别对待。DSpark已有TV与位置权重等训练设计，不能宣称此前工作完全忽略概率重叠或前缀价值。

### DFlash-2

- 一手来源是作者技术报告`https://inco.ai/blog/dflash2/`和官方模型卡`https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2`；不是用原DFlash论文替代。
- 选择头保留Top16候选，按hidden门控的相邻候选对分数做路径选择；候选对打分并行，路径遍历顺序执行。完整版本另有主体局部卷积。
- 无卷积、仅选择头的表2只公开Qwen3-4B GSM8K：T0 AC4.27→4.61(+7.96%)，T1 3.78→4.25(+12.43%)。
- HumanEval公开5.43→6.28(+15.65%)是Qwen3.5-4B卷积＋选择头，不能全算成选择头收益。
- Muse完整版有C1/C32吞吐，但target、H200、reasoning与我们的设置不同。
- Top16草稿候选限制影响草稿表达范围，不能据此声称target输出分布失真；正确拒绝采样可保留target分布。
- 它已有很轻的选择头和并行打分优势。我们全词表采样、Top256修正是设计取舍，不是已证明的全面速度/AC优势。

### 本地历史近邻实现

仓库还有`Baseline_dspark_full_vocab_v1/`和`Baseline_dflash2_full_vocab_v1/`。如果讨论已有本地对照，先读各自最新README、SOURCE_AUDIT与实际结果文件。
它们是移植头/历史配方，DSpark未实现完整调度，DFlash-2未实现主体卷积；旧缓存/固定轨迹代理和较新的BF16/native入口要区分。
本地自由token codebook参数计数不能等同作者轻量头计数。64条、eager、C1、7 proposal、max128等历史短测不能冒充当前E2/Graph/C1-C32正式公平对照。
这里不预先指定其结果有效性，必须逐文件核验后才引用。

跨论文绝对AC/TPS、不同baseline下的相对增幅都不能直接排名。先说明我们与它们处理的共同问题，再界定各自近似和成本取舍。

## 8. ACL 2027目标与交付要求

目前已核验官方征稿：`https://2027.aclweb.org/calls/main/`。
包含Efficient Methods for NLP；长文正文8页。公布的最晚ARR投稿截止为北京时间2027-01-05 19:59（官网2027-01-04 AoE），commitment等动态信息需再查官网。
论文质量讨论可参考`https://aclrollingreview.org/reviewform`，但不要把构思任务变成行政材料清单。

请先完成以下交付，主要用中文，必要时给英文标题/贡献句/摘要草稿：

1. **一页论文主张**：核心科学问题、准确mismatch定义、一句话中心主张、现有方法如何回应、主要证据、适用边界。
2. **2–3种叙事方案与推荐方案**：每种必须围绕现有方法，说明为何值得ACL读者阅读、与近邻的关系，以及最可能遭质疑的论断。不能只是给同一结构换几个名称。
3. **引言论证图与可读初稿**：按问题→mismatch→接受损失→已有工作→当前方法→贡献推进；先写逻辑，再写句子。明确已有知识与本工作主张。
4. **8页长文章节与图表安排**：每节回答一个问题；给图表功能、数据来源、主文/附录分配。新旧批次分开，旧历史和系统细节不挤占中心证据链。
5. **近邻比较与优势表述**：Domino/DSpark/DFlash-2的准确机制，我们的差异，已证实优势、待验证设计假设、无法支持的说法。
6. **claim–evidence对照表**：每条贡献对应现有证据路径，标记“已支持/待完成/待验证/不应声称”。主表整体收益与修复机制归因要分开。
7. **有限证据缺口**：只指出哪些缺口影响中心主张，以及最小验证思路；不发起新实验、不扩展结构。不以继续加模块代替想清楚论文。

请先快速报告你读到的最新批次状态，再进入论文构思。已有结果足够开始写作，不要停在笼统地说“需要更多实验”，也不要通过措辞把未验证机制写成事实。
