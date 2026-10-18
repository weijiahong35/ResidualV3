# ResidualV3 三层 mismatch repair：组会汇报

**汇报日期：2026-10-13；本版已重新核对三项相关工作的一手资料、原表及模型卡。**

历史三领域结果：五层并行DFlash＋三层修复头追加训练两个完整epoch，在数学、代码、对话的C1/C32共12组正式对照中，接受长度和吞吐均提高。C1吞吐增加6.78%–17.48%，C32增加3.17%–14.89%。正文按“我们的结构 → 训练与E1/E2 → 相关方法结构 → 按任务比较 → 模块耗时”展开；外部实验逐行写明训练/基线来源与推理条件。

本次六数据集扩展重测另完成48正式窗、24配对、16000请求，固定max2048/seed980406，最终审计通过；完整新成绩见4.5，旧批次成绩继续保留。

## 1. 我们的方法：五层并行主体＋三层修复头

并行DFlash一次生成整个草稿块，各位置尚不知道前面实际采样了哪个token。后续预测可能与实际前驱不一致，而投机解码只接受连续正确前缀：前面被拒绝，后面的草稿就无法贡献本轮输出。我们的方案在一次并行主体计算之后，用**当前hidden＋实际前驱＋当前草稿分布**逐位置修正logits，再交给target按原生规则验证。

本报告的**AC/AL是平均接受长度，含bonus，不是答题准确率**；吞吐为整窗输出token/s。C1/C32/C40表示请求并发，T0/T1表示温度0/1。本项目“单训→联合头”指独立训练DFlash→联合训练三层模型；外部对照统一写“DFlash→增强模型”，左侧为各实验自己的完整已训练DFlash。RAW与跨epoch比较另行标明。“—”表示没有符合该口径的有效数据。跨论文的target、训练、引擎和块长不同，相对增长用于说明各自收益，不能直接据此排名绝对性能。

**完整推理流程**

```text
已验证前缀
   ↓
target提供上下文特征，上一轮bonus作为anchor
   ↓
anchor＋15个MASK → 五层DFlash并行主体 → 15个hidden和原始logits
                                                    ↓
                                      三层修复头逐位置修正并采样
                                                    ↓
                                      15个草稿token＋实际草稿概率q
                                                    ↓
                                      target一次并行验证整个草稿块
                                                    ↓
                                      接受连续前缀＋生成bonus → 下一轮
```

主体只执行一次整块forward和原始LM投影。块内15个位置的修复与采样顺序执行；第一个位置使用anchor作为前驱，之后使用上一位置**修复后实际采样的token**。正式版不启用尾部停修、置信度调度或专家路由。

正式实现为NOFILL冻结版，取消头部确定性empty调试预填充；没有把后来的概率prepare融合实验加入这批正式成绩。

**单个位置的三路输入与修复**

```text
当前主体hidden h_i，4096维
   → LayerNorm＋Linear → 256维 ───────────────────────────────┐
                                                            │
实际前驱token x_(i-1)                                        │
   → 冻结target embedding查表 → 向量归一化                  │
   → LayerNorm＋Linear → 256维 ───────────────────────────────┤
                                                            │
当前原始分布 q_i                                            │
   → 原始logits Top256候选                                   │
   → 候选概率加权的冻结输入basis特征，512维                   │
   → LayerNorm＋Linear → 256维 ───────────────────────────────┘
                                                            ↓
                                            拼接：256×3＝768维
                                                            ↓
                                            LayerNorm＋Linear
                                                            ↓
                                                          256维
                                                            ↓
                                       SwiGLU残差块 × 3 → LayerNorm
                                                            ↓
                                     ┌──────────────────────┴────────────────┐
                                     ↓                                       ↓
                         Linear → 512维修正方向                   Linear → 1维剂量
                              → L2归一化                          → sigmoid×0.8
                                     └──────────────────────┬────────────────┘
                                                            ↓
                                  冻结输出basis映射＋hidden范数缩放
                                                            ↓
                                  只修正当前Top256候选的原始logits
                                                            ↓
                                  从修复后完整词表分布采样当前token
                                                            ↓
                                  当前token成为下一位置的实际前驱
```

三路输入分别回答：**当前上下文允许什么、前面实际选了什么、主体当前本来倾向什么**。当前分布特征不是单一最大概率，而是候选概率在冻结输入basis中的加权表示。

每个SwiGLU残差块为：

```text
输入x，256维
  → LayerNorm
  → Linear，分成两个768维分支 a、g
  → a ⊙ SiLU(g)
  → Linear，768→256
  → 与输入x相加
```

这里的“三层”是三个SwiGLU残差块，不是三层Transformer。`no-soft`表示不递推上一位置完整soft分布作为第四路输入；概率归一化和采样仍然存在。头没有GRU式的完整块内历史状态，显式因果输入是紧邻的实际前驱token。

**修正形式与参数**

令当前Top256候选集合为 $\mathcal C_i$，预测的单位方向为 $u_i\in\mathbb R^{512}$，剂量为 $d_i\in(0,0.8)$，冻结的输出basis词表行为 $B_{\mathrm{out}}[v]$：

$$
\Delta z_i(v)=d_i\|h_i\|_2\langle u_i,B_{\mathrm{out}}[v]\rangle,
\qquad v\in\mathcal C_i.
$$

候选之外 $\Delta z_i(v)=0$，最终 $z_i'=z_i^{\mathrm{raw}}+\Delta z_i$。输入embedding构建的特征basis与LM-head构建的输出basis用途不同；不能把两者混成同一张词表特征表。

这个设计把“怎么改”和“改多少”分开：方向负责调整候选之间的相对偏好，剂量控制修改强度；原始分布作为残差底座继续保留。Top256限制的是**被修正的集合**，不是把最终采样词表限制为256个token。

| 配置 | 数值 |
|---|---:|
| DFlash主体层数 | 5 |
| 主体hidden维度 | 4096 |
| 修复头宽度 | 256 |
| SwiGLU残差块数 | 3 |
| 每块FFN中间维度 | 768 |
| 输入/输出basis rank | 512 |
| 修正候选数 | 256 |
| 剂量上限 | 0.8 |
| 修复头新增可训练参数 | **4,353,537** |

源码与配置：[头结构](residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/repair/model.py)、[SwiGLU实现](residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/runtime/components.py)、[头配置](residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/head_config.json)。

## 2. 训练与RAW、Epoch1、Epoch2对照

**训练数据与轮数**

训练prompt来自Open-PerfectBlend再生成流程，回答由Qwen3-8B在nonthinking模式下生成。最终使用**833,176条去重有效训练会话组/样本**，每轮约**1,019,443,945输入token**。输入长度上限4096；数据清单包含符合完整监督块条件的有效截断前缀，不能简单表述成所有超长原始会话都被丢弃。

独立DFlash和联合模型从同一个下载的nonthinking DFlash初始化，各追加训练**两个完整epoch**。这不是从随机初始化开始训练；当前主结果测试第二轮结束的E2权重，下面使用E1与E2各自结束的权重比较训练效果。

| 项目 | 设置 |
|---|---|
| target / teacher | Qwen3-8B，冻结 |
| 独立对照 | 完整DFlash主体训练，不带修复头 |
| 我们的模型 | DFlash主体＋三层修复头联合训练 |
| 每轮数据量 | 833,176条有效样本，约1.019B输入token |
| 追加训练轮数 | E1＋E2，共2个完整epoch |
| 两轮样本呈现量 | 每个模型1,666,352次样本呈现，不是新增了同等数量的独立样本 |
| 两轮优化器更新 | E1 15,847次＋E2 15,854次＝31,701次/模型 |
| 每包输入token上限 | 65,536 |
| 每包anchor上限 | 1,536；仅选择完整监督16-token块 |
| 草稿块 | anchor＋15个proposal |
| 更新方式 | 每包一次更新；累计包数1 |
| teacher供应 | 一个共享teacher供应两个独立learner，梯度与优化器分别维护 |
| 冻结资产 | target、embedding、LM-head、输入/输出basis |
| 主体优化器 | Muon＋AdamW；头使用AdamW |
| E2续训 | 继承完整master、优化器状态、RNG和数据游标；新epoch学习率周期 |

训练时修复头使用teacher-prefix里的真实前驱；推理时使用实际采样前驱。该训练方式围绕“前面被接受后，当前位置才有输出价值”的投机解码机制构造监督。

E2后段另训练了两个detach辅助预测器，但本报告的FULL15_NOFILL正式推理没有启用它们；它们不计入上述三层修复头参数量。

数据依据：[训练数据清单](residualV3-nonthing/data/all_nonthinking_ctx4096_20261006_v1/manifest.json)。训练依据：[E1协议](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/protocol.json)、[E2协议](residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4/protocol.json)。

**训练目标**

```text
共享冻结teacher提供监督
          ↓
  ┌───────┴────────────────────────────────────────┐
  ↓                                                ↓
独立DFlash                                    DFlash＋三层头
  ↓                                                ↓
raw CE/D-PACE                               raw CE/D-PACE
                                                 ＋ repaired CE/D-PACE
                                                 ＋ repair OUTPUT
```

联合模型保留raw分支目标，避免仅让头补偿一个退化的主体。三个主损失项按1:1:1相加；两条训练路线不合并梯度。

代码中的CE使用**teacher argmax**作为监督标签，D-PACE根据前缀的预测质量对位置CE加权，强调前面位置的价值。OUTPUT进一步针对连续接受前缀：令 $o_i=\sum_v\min(p_i^{\mathrm{target}}(v),q_i^{\mathrm{repair}}(v))$，则每个15-token块的目标形式为

$$
\mathcal L_{\mathrm{OUTPUT}}=1-\frac1{15}\sum_{k=1}^{15}\prod_{i=1}^{k}o_i.
$$

因此，监督同时约束原始预测、修复后预测和前缀存活价值。它是teacher-prefix下的训练目标，不是额外采集昂贵native rollout标签。[损失实现](residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/losses.py)。

**RAW起点：追加训练前的历史接受长度**

RAW是下载的完整预训练nonthinking DFlash，尚未做本项目E1/E2追加训练，没有修复头。早期GSM8K批次同时测了RAW与E1：C1=200/C40=1319、T0/T1、max2096/context4096、15 proposal，KV36GiB、memory utilization=0.70。下面各箭头均来自这一同批次；其中E1数值与后来的统一实现评测略有差别。协议明确`acceptance_only=true`、`speed_evidence=false`，因此只报告AC增长。

| 数据集 | 设置与比较路线 | AC：RAW→E1 | AC增长 | 吞吐：RAW→E1 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| GSM8K | RAW→E1单训，C1 T0 | 6.738→7.421 | +10.15% | — | — |
| GSM8K | RAW→E1单训，C1 T1 | 5.741→6.266 | +9.14% | — | — |
| GSM8K | RAW→E1单训，C40 T0 | 6.858→7.591 | +10.69% | — | — |
| GSM8K | RAW→E1单训，C40 T1 | 5.961→6.374 | +6.94% | — | — |
| GSM8K | RAW→E1联合三层，C1 T0 | 6.738→8.414 | +24.89% | — | — |
| GSM8K | RAW→E1联合三层，C1 T1 | 5.741→7.356 | +28.12% | — | — |
| GSM8K | RAW→E1联合三层，C40 T0 | 6.858→8.545 | +24.61% | — | — |
| GSM8K | RAW→E1联合三层，C40 T1 | 5.961→7.651 | +28.37% | — | — |

RAW日志TPS依次为C1 T0 **429.0**、C1 T1 **370.5**、C40 T0 **3794.5**、C40 T1 **3289.5**，仅保留原始读数，不作为性能证据。RAW→E1联合模型包含主体训练与头的共同作用。后续E1/E2统一评测及当前E2正式对照没有同批RAW，不能据此计算RAW→E2吞吐，也不能把旧RAW填入当前C32或另外两个领域。来源：[RAW/E1历史报告](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/evaluation_dspark_gsm8k_recovery_20261009_v2/RESULTS_CN.txt)、[模型身份与协议](residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1/evaluation_dspark_gsm8k_recovery_20261009_v2/native_protocol.json)。

**E1/E2：在同一实现下比较追加训练效果**

下面16个正式窗口统一使用同一个冻结runtime；E1/E2各自与同轮独立单训DFlash对照。GSM8K C1=200/C40=1319，T0/T1，15 proposal，max2096/context4096，BF16、batch-invariant、CUDA Graph；C1 KV52GiB/C40 KV46GiB、memory utilization=0.92，辅助预测器与截断关闭，隔离审计通过。

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| GSM8K | E1 C1 T0 | 7.384→8.381 | +13.50% | 179.7→190.5 | +6.03% |
| GSM8K | E1 C1 T1 | 6.331→7.429 | +17.34% | 154.8→169.2 | +9.34% |
| GSM8K | E1 C40 T0 | 7.561→8.527 | +12.79% | 3197.9→3347.1 | +4.67% |
| GSM8K | E1 C40 T1 | 6.382→7.667 | +20.13% | 2705.0→3017.2 | +11.54% |
| GSM8K | E2 C1 T0 | 7.479→8.515 | +13.86% | 181.9→193.2 | +6.20% |
| GSM8K | E2 C1 T1 | 6.326→7.496 | +18.49% | 154.8→170.7 | +10.27% |
| GSM8K | E2 C40 T0 | 7.609→8.652 | +13.70% | 3217.7→3400.6 | +5.68% |
| GSM8K | E2 C40 T1 | 6.483→7.710 | +18.92% | 2751.5→3032.5 | +10.21% |

同一训练路线从E1到E2的增量如下；百分比按原始值计算：

| 设置 | 独立单训AC增量 | 独立单训TPS增量 | 联合三层AC增量 | 联合三层TPS增量 |
|---|---:|---:|---:|---:|
| C1 T0 | +1.28% | +1.24% | +1.60% | +1.40% |
| C1 T1 | −0.08% | +0.01% | +0.90% | +0.85% |
| C40 T0 | +0.64% | +0.62% | +1.46% | +1.60% |
| C40 T1 | +1.59% | +1.72% | +0.56% | +0.51% |

在这个GSM8K面板上，第二轮联合模型的额外AC增长0.56%–1.60%、TPS增长0.51%–1.60%。增量较小，但只有一个训练seed和一次正式测量，不能断言完全收敛，也不能推广到其他领域。来源：[E1/E2统一实现报告](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/RESULTS_CN.md)、[C1窗口](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C1/formal/summary.json)、[C40窗口](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C40/formal/summary.json)。

**实现差异：历史吞吐与当前主表不能混算**

| 项目 | 历史E1/E2统一实现 | 后续E2 NOFILL正式版 |
|---|---|---|
| 权重 | E1/E2各自结束的权重 | 已完成的同一套E2权重，没有新一轮训练 |
| 头内empty调试预填充 | 开启 | 局部取消 |
| Graph模式 | FULL_DECODE_ONLY | FULL_AND_PIECEWISE |
| KV预算 | C1 52GiB，C40 46GiB | C1 52GiB，C40/C32 44GiB |
| 测试批次 | C1 GPU5、C40 GPU6 | 后续独立批次；C32另行补测 |

同一套E2权重的GSM8K C1 T0相对单训TPS增长，历史为+6.20%，后续NOFILL为+9.94%。差别包含实现、Graph及测试批次，不能全部归因于去掉empty，更不能算作多训练一轮的收益；T1跨批次AC的细小变化也不当作新训练收益。历史C40始终标为C40。协议：[历史C1](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C1/formal/protocol.json)、[历史C40](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/C40/formal/protocol.json)、[NOFILL C1](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/C1/formal/protocol.json)、[NOFILL C40](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/C40/formal/protocol.json)。

## 3. 相关方法：头结构、额外模块与训练贡献

本次核验的是[Domino论文v1（2026-05-28）](https://arxiv.org/abs/2605.29707v1)、[DSpark论文v1（2026-07-06）](https://arxiv.org/abs/2607.05147v1)，以及[DFlash-2作者技术报告（2026-08-18）](https://inco.ai/blog/dflash2/)和官方模型卡。DFlash-2作者给出的引用对象是技术报告；原始DFlash的论文不能代替DFlash-2的结构与实验来源。

| 方法 | 生成头 | 头之外的结构／调度变化 | 训练与实现贡献 |
|---|---|---|---|
| 我们的E2三层 | 当前hidden＋实际前驱＋q特征；方向／剂量残差 | 当前正式版保留五层DFlash架构，固定15 proposal | 主体与头联合训练；raw＋repair＋OUTPUT；NOFILL实现 |
| Domino | GRU累计前缀＋低秩logits残差 | 所引实验保留五层并行主体，普通target验证 | teacher forcing、base-anchored curriculum、kernel融合与Graph |
| DSpark | 默认Markov转移偏置；另有RNN变体 | confidence head＋验证前缀调度；调整anchor输入位置 | CE＋TV＋confidence训练，校准和服务调度实现 |
| DFlash-2 | Top16候选对打分与路径选择 | 主体attention／MLP前后的两tap动态卷积 | 保持候选打分并行，最后顺序选择 |

**Domino的主实验与ShareGPT同数据实验采用同一种头结构。** 两组的区别写在第4节的训练与评测条件中。DSpark“调度开／关”、DFlash-2“卷积开／关”才是这里需要明确区分的额外模块开关。

**Domino：GRU因果状态＋低秩logits修正**

**完整流程：并行主体算一次，GRU与修正头逐位置补充已采样前缀。**

```text
target上下文特征＋anchor＋MASK草稿块
  ↓
五层DFlash主体【整块并行】
  ↓
各位置hidden H_i → 冻结target LM投影 → 原始logits L_base_i
  ↓
GRU因果编码＋低秩修正头【逐位置执行，见下图】
  ↓
草稿token＋实际proposal分布 → target验证 → 接受前缀＋bonus
```

**单个位置的头：两路输入汇合，再预测全词表残差。**

```text
块内此前实际token → embedding → GRU递推 → 1024维状态 S_(i-1)
当前主体表示 H_i ───────────────────────→ 与 S_(i-1) 拼接
                                                ↓
                                       Linear → 256维
                                                ↓
                                              SiLU
                                                ↓
                                       Linear → 词表维度
                                                ↓
                                          logits残差 ΔL_i
                                                ↓
                           原始logits L_base_i ＋ ΔL_i
                                                ↓
                                    归一化／采样当前token x_i
                                                ↓
                  embedding(x_i)＋旧GRU状态 → 新状态 S_i → 下一位置
```

**创新是用小GRU累计块内前缀，再用256维瓶颈修正全词表logits。** 重的主体与原始LM投影保持并行，轻的因果分支顺序执行。训练用真实前缀进行teacher forcing，并逐步调整主体／修复后损失权重，防止修正分支掩盖较弱主体；实现另有kernel融合与CUDA Graph。[论文第4节](https://arxiv.org/html/2605.29707v1#S4)。

**DSpark：Markov修正＋target前缀调度**

**完整流程：修正头负责“生成什么”，置信度与调度器负责“验证多长”。** 下图以默认Markov头为例。

```text
target上下文特征＋anchor＋MASK草稿块
  ↓
五层DFlash并行主体 → hidden h_i＋原始logits U_i【整块并行】
  ↓
Markov头修正与采样 ＋ confidence head预测【逐位置执行】
  ↓
候选token／实际分布q ＋ 校准后的前缀存活概率
  ↓
结合硬件吞吐曲线，为每个请求确定验证前缀长度
  ↓
target验证所选前缀 → 接受前缀＋bonus
```

**修正路径：**

```text
实际前驱token x_(i-1)
  → 在可训练W1中查表 → 256维前驱表示
  → 乘W2，映射到词表维度 → Markov转移偏置
  → 加到当前原始logits U_i
  → 归一化／采样 x_i → 成为下一位置的实际前驱
```

**置信度与验证调度路径：**

```text
当前hidden h_i ＋ 同一前驱的W1表示
  → 拼接 → Linear → sigmoid
  → 预测当前位置条件接受概率 c_i
  → 经校准并沿前缀累乘 → 预测“前i个都接受”的概率
  → 与当前负载下实测的SPS(B)共同评估
  → 为每个请求分配验证长度 → target只验证保留前缀
```

**创新同时覆盖因果修正和验证计算分配。** 默认Markov修正只直接读取紧邻前驱，hidden另供置信度头使用；论文还提供累积前缀状态的RNN修正变体。调度须保持因果约束，不能根据未来候选反向决定是否保留早期token。作者也把anchor位置用于第一个预测，减少一个草稿输入位置。[论文第3节及生产调度实现](https://arxiv.org/html/2607.05147v1#S3)。

**DFlash-2：局部卷积＋候选路径选择**

**完整流程：主体先改善候选，再由选择头把相邻候选接起来。**

```text
target上下文特征＋anchor／MASK草稿块
  ↓
DFlash并行主体＋局部卷积【整块并行】
  ↓
各位置hidden h_i → LM投影 → 原始logits U_i
  ↓
每个位置保留Top16候选
  ↓
选择头：计算所有相邻位置的候选对分数【并行打分】
  ↓
从anchor出发，按实际前驱选择下一候选【顺序遍历】
  ↓
草稿token＋实际proposal分布 → target验证 → 接受前缀＋bonus
```

**主体里的卷积做什么：** 每层attention和MLP子层的前后都插入两tap动态深度卷积；下面只画局部混合，省略原有norm与残差连接。

```text
当前位置表示 x_i ＋ 前一位置表示 x_(i-1)
  → 分别乘按内容调整的逐通道系数 → 相加 → 新表示
  （块首从最后一个已验证token读取前驱表示）

每层示意：卷积 → attention → 卷积；卷积 → MLP → 卷积
```

**选择头怎样给“前驱a→候选b”打分：**

```text
前驱候选a → 256维表示 A(a)
当前hidden h_i → 256维门控 H(h_i)
  → 两者逐元素相乘
  → 与当前候选b的256维表示 B(b) 做点积
  → 得到“a后面接b”的相容分数
  → 加上b的原始logit U_i(b) → 候选对分数

实际前驱确定 → 读取对应的一行已算分数
  → T0取最高分／T1按分布采样 → 当前token成为下一位置的前驱
```

**创新对应两处：卷积改善尾部候选质量，选择头改善候选组合的连贯性。** 候选对提前并行打分，最后逐位置选择；无卷积选择头与完整“卷积＋选择头”的成绩分别列出。[作者结构说明](https://inco.ai/blog/dflash2/)。

## 4. 测试结果：三领域DFlash对照与DSpark截断实验

**先固定本项目的评测口径。** 主表使用已完成的E2三层NOFILL成绩，target为Qwen3-8B，A100 80GB／vLLM，BF16、batch-invariant、CUDA Graph；nonthinking，T0/T1，15 proposal，max2096/context4096。C1 KV52GiB、C32 KV44GiB，memory utilization=0.92。左右两臂分别是E2独立单训DFlash和E2联合主体＋完整三层头，主体权重不同。

| 任务 | 数据集 | C1每个温度、每个模型的请求数 | C32每个温度、每个模型的请求数 | 测试内容 |
|---|---|---:|---:|---|
| 数学 | GSM8K | 200 | 1319 | nonthinking生成 |
| 代码 | HumanEval | 164 | 164 | 全部题目；未执行pass@1评分 |
| 对话 | MT-Bench | 80 | 80 | 每组只取首轮；未计算judge分数 |

正式窗口通过隔离与Graph审计，T0两臂最终输出逐题一致。HumanEval/MT-Bench未重复原题补足数量。按DSpark扩展的六数据集E2评测另有[独立报告](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/RESULTS_CN.md)，本批已完成并单列于4.5，下面既有成绩保留原批次口径。

**外部结果的训练、基线和推理设置。** 下表是数字的适用条件；“未注明”表示本次核对的一手表格／相邻说明没有足够信息，不用其他实验补齐。

| 实验来源 | 增强模型训练 | 左侧DFlash来源 | 推理设置与可引用指标 |
|---|---|---|---|
| Domino主实验，[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 约1.42M OPB prompt经target再生成；默认3轮，训练长度≤3072 | 公开z-lab/Qwen3-8B-DFlash-b16；论文未证明双方训练预算一致 | Qwen3-8B，A100 80GB，Transformers低并发，T0/T1，15 proposal，max2048；有AC与AR加速比，没有绝对TPS，表中未单列C值 |
| Domino主实验，[表2](https://arxiv.org/html/2605.29707v1#S5.T2) | 主实验模型 | 公开完整DFlash | Qwen3-8B，A100 80GB，SGLang；列出C2/4/8/16/32 TPS；表注未明确温度，不能直接标T0；本报告取GSM8K C32 |
| Domino同数据对照，[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | ShareGPT；头结构与主实验相同；该表未单列样本量／轮数 | 同样在ShareGPT上独立训练的完整DFlash | Qwen3-8B，A100，T0，16-token预算；C1–32速度比与单独AC列；引擎未在该表单列，绝对TPS由AR TPS×速度比估算 |
| DSpark离线，[表1](https://arxiv.org/html/2607.05147v1#S4.T1) | 约1.3M OPB prompt经target再生成；10轮 | 同框架、同数据重训的完整DFlash | Qwen3-8B，nonthinking T1，7 proposal，Markov头，**验证调度关闭**；有AC，无C1/C32绝对TPS；表中未分并发 |
| DSpark完整生产系统，[第5节](https://arxiv.org/html/2607.05147v1#S5) | 生产系统配置 | **对照是MTP-1** | DeepSeek-V4真实流量，3层MoE草稿主体、最多5 proposal，验证截断开启；完整结果单列于4.4，不填成DFlash增益 |
| DFlash-2仅选择头，[作者表2](https://inco.ai/blog/dflash2/#table-2) | 训练样本量／轮数未披露 | 同表五层Qwen3-4B DFlash | 卷积关闭，GSM8K T0/T1；有AC与周期开销，无绝对TPS；并发、硬件及该表块长未独立注明 |
| DFlash-2卷积＋选择头，[作者表3](https://inco.ai/blog/dflash2/#table-3) | 作者称匹配设置训练；样本量／轮数未披露 | 匹配设置训练的Qwen3.5-4B DFlash | thinking T1，top-p=.95、top-k=20、presence penalty=1.5；有AC；并发、硬件及该表块长未独立注明 |
| DFlash-2卷积＋选择头，[Muse模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#evaluation) | 从官方DFlash微调；样本量／轮数未披露 | 官方Muse-Glimmer-30B-assistant | Muse-30B，单H200，SGLang/FA3，high reasoning，T1/p=.95/k=64，15 proposal，max4096；有独立AC表和C1/8/32 TPS表 |

OPB指Open-PerfectBlend。不同论文的1.3M／1.42M是作者各自描述的原始数据规模，不能直接与我们过滤后的833,176条等同。Domino的[表6](https://arxiv.org/html/2605.29707v1#A1.T6)使用同一联合主体关／开头，左端不满足“独立单训DFlash”，因此不填入主比较表。

**如何读下面三张表：** 每行的箭头都在该行自己的实验内计算。AC是平均接受长度，吞吐单位token/s；“—”保留未报告项。“AC未分并发”不会移填到C1或C32。我们的增长率由未舍入的本地JSON计算；外部增长率由作者公布值计算。Domino表3的“≈TPS”用AR TPS乘已舍入的速度比，因此也只是近似值。表头“增强模型”涵盖三层联合头、Domino、DSpark或DFlash-2，不将主体卷积误称为纯头。**其中DSpark三行AC均未启用验证截断；开启截断的证据与完整系统收益见4.4。**

### 4.1 数学：GSM8K

| 数据集 | 设置：方法／训练与基线／target／温度并发／来源 | AC：DFlash→增强模型 | AC增长 | 吞吐：DFlash→增强模型 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| GSM8K | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T0 | 7.479→8.515 | +13.86% | 178.4→196.2 | +9.94% |
| GSM8K | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T1 | 6.336→7.489 | +18.19% | 151.9→173.3 | +14.09% |
| GSM8K | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T0 | 7.609→8.652 | +13.70% | 2978.9→3183.1 | +6.86% |
| GSM8K | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T1 | 6.483→7.699 | +18.75% | 2549.9→2837.1 | +11.26% |
| GSM8K | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T0；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 6.590→10.030 | +52.20% | — | — |
| GSM8K | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T1；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 6.000→8.340 | +39.00% | — | — |
| GSM8K | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；C32，**温度未注明**；[表2](https://arxiv.org/html/2605.29707v1#S5.T2) | — | — | 2801.0→3650.0 | +30.31% |
| GSM8K | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；AC未分并发；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | 3.900→4.650 | +19.23% | — | — |
| GSM8K | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；C1；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | — | — | ≈246.6→276.9 | ≈+12.31% |
| GSM8K | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；C32；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | — | — | ≈1867.2→2124.1 | ≈+13.76% |
| GSM8K | DSpark Markov头，固定长度验证（未截断）；双方OPB训练；Qwen3-8B；T1、7 proposal；C未注明；[表1](https://arxiv.org/html/2607.05147v1#S4.T1) | 5.330→6.170 | +15.76% | — | — |
| GSM8K | DFlash-2选择头，卷积关；Qwen3-4B；T0；C未注明；训练规模未披露；[作者表2](https://inco.ai/blog/dflash2/#table-2) | 4.270→4.610 | +7.96% | — | — |
| GSM8K | DFlash-2选择头，卷积关；Qwen3-4B；T1；C未注明；训练规模未披露；[作者表2](https://inco.ai/blog/dflash2/#table-2) | 3.780→4.250 | +12.43% | — | — |
| GSM8K | DFlash-2卷积＋选择头；Qwen3.5-4B、匹配设置DFlash；thinking T1；C未注明；[作者表3](https://inco.ai/blog/dflash2/#table-3) | 4.990→6.200 | +24.25% | — | — |
| GSM8K | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；AC未分并发；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#acceptance-length) | 5.430→6.570 | +20.99% | — | — |
| GSM8K | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C1；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 247.8→293.7 | +18.52% |
| GSM8K | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C32；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 2330.3→2818.3 | +20.94% |

### 4.2 代码：HumanEval

| 数据集 | 设置：方法／训练与基线／target／温度并发／来源 | AC：DFlash→增强模型 | AC增长 | 吞吐：DFlash→增强模型 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| HumanEval | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T0 | 5.542→6.344 | +14.48% | 134.1→148.7 | +10.85% |
| HumanEval | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T1 | 4.681→5.688 | +21.50% | 113.5→133.4 | +17.48% |
| HumanEval | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T0 | 5.542→6.343 | +14.46% | 1981.0→2167.8 | +9.43% |
| HumanEval | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T1 | 4.682→5.568 | +18.94% | 1635.4→1878.9 | +14.89% |
| HumanEval | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T0；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 6.520→7.390 | +13.34% | — | — |
| HumanEval | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T1；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 5.440→5.980 | +9.93% | — | — |
| HumanEval | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；AC未分并发；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | 3.780→4.350 | +15.08% | — | — |
| HumanEval | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；C1；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | — | — | ≈237.4→259.4 | ≈+9.30% |
| HumanEval | Domino同数据对照；双方ShareGPT训练；Qwen3-8B；T0；C32；[表3](https://arxiv.org/html/2605.29707v1#S5.T3) | — | — | ≈2036.0→2280.3 | ≈+12.00% |
| HumanEval | DSpark Markov头，固定长度验证（未截断）；双方OPB训练；Qwen3-8B；T1、7 proposal；C未注明；[表1](https://arxiv.org/html/2607.05147v1#S4.T1) | 4.640→5.520 | +18.97% | — | — |
| HumanEval | DFlash-2卷积＋选择头；Qwen3.5-4B、匹配设置DFlash；thinking T1；C未注明；[作者表3](https://inco.ai/blog/dflash2/#table-3) | 5.430→6.280 | +15.65% | — | — |
| HumanEval | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；AC未分并发；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#acceptance-length) | 4.110→5.660 | +37.71% | — | — |
| HumanEval | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C1；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 210.5→266.2 | +26.46% |
| HumanEval | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C32；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 2170.0→2780.2 | +28.12% |

### 4.3 对话：MT-Bench

| 数据集 | 设置：方法／训练与基线／target／温度并发／来源 | AC：DFlash→增强模型 | AC增长 | 吞吐：DFlash→增强模型 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| MT-Bench | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T0 | 3.145→3.413 | +8.52% | 77.6→82.8 | +6.78% |
| MT-Bench | 我们E2三层；同数据独立单训对照；Qwen3-8B；C1 T1 | 2.821→3.121 | +10.66% | 69.6→75.7 | +8.78% |
| MT-Bench | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T0 | 3.148→3.415 | +8.48% | 952.5→982.7 | +3.17% |
| MT-Bench | 我们E2三层；同数据独立单训对照；Qwen3-8B；C32 T1 | 2.777→3.151 | +13.47% | 889.2→953.5 | +7.23% |
| MT-Bench | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T0；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 4.180→5.180 | +23.92% | — | — |
| MT-Bench | Domino主实验；OPB训练/公开DFlash对照；Qwen3-8B；低并发 T1；[表1](https://arxiv.org/html/2605.29707v1#S4.T1) | 3.870→4.610 | +19.12% | — | — |
| MT-Bench | DSpark Markov头，固定长度验证（未截断）；双方OPB训练；Qwen3-8B；T1、7 proposal；C未注明；[表1](https://arxiv.org/html/2607.05147v1#S4.T1) | 3.110→3.720 | +19.61% | — | — |
| MT-Bench | DFlash-2卷积＋选择头；Qwen3.5-4B、匹配设置DFlash；thinking T1；C未注明；[作者表3](https://inco.ai/blog/dflash2/#table-3) | 4.260→5.200 | +22.07% | — | — |
| MT-Bench | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；AC未分并发；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#acceptance-length) | 3.520→4.420 | +25.57% | — | — |
| MT-Bench | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C1；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 164.6→197.4 | +19.93% |
| MT-Bench | DFlash-2卷积＋选择头；Muse-30B/官方DFlash；high reasoning T1；C32；[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2#throughput) | — | — | 1668.0→1975.5 | +18.44% |

DSpark完整系统的生产收益以MTP为基线；本表没有把它换算成DFlash增益。DFlash-2作者表2只提供GSM8K的无卷积选择头结果，Domino表3没有MT-Bench，均保留这一覆盖边界。Muse模型卡的社区DSpark权重也不替代DSpark论文的自训版本。

本地数字来源：[GSM8K C1](residualV3-nonthing/results/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/COMPARISON.json)、[HumanEval/MT-Bench C1](residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/COMPARISON.json)、[三领域C32](residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1/COMPARISON.json)。所有新表行的原值、来源与推算记录见[本次核对记录](residualV3_main_v1/research/group_meeting_sources_20261013/TABLE_ROWS.json)。

### 4.4 DSpark验证尾部截断：诊断与完整系统结果

**截断属于完整DSpark。** 论文表1为单独评估草稿质量而关闭调度；截断另有两组证据，不能把表1写成完整系统结果。[第4.2节](https://arxiv.org/html/2607.05147v1#S4.SS2)

**固定阈值诊断：** Qwen3-4B随阈值提高，数学／代码／对话的接受率分别从76.9%→92.5%、67.6%→92.0%、45.7%→95.7%。这是裁掉低置信尾部后的接受比例，**不是AC增长或吞吐增长**；也不是完整硬件感知调度。[第4.3.3节、图5](https://arxiv.org/html/2607.05147v1#S4.SS3.SSS3)

**完整动态截断：** 下表采用DeepSeek-V4 preview真实流量，基线为MTP-1；包含草稿、修正头、置信度调度及服务实现的共同收益。[第5.4节、图7](https://arxiv.org/html/2607.05147v1#S5.SS4)

| 目标模型 | 验证截断 | 同等单用户速度要求下，总吞吐增长 | 同等总吞吐下，单用户生成速度增长 |
|---|---|---|---|
| DeepSeek-V4-Flash preview | 开启 | **+51%**（要求≥80 token/s/user） | **+60%～85%** |
| DeepSeek-V4-Pro preview | 开启 | **+52%**（要求≥35 token/s/user） | **+57%～78%** |

这些是**完整DSpark相对MTP-1**的收益，不能归为截断单模块收益，也不能补进上面“DFlash→增强模型”的吞吐列。论文未提供所需的三数据集Qwen3-8B、C1/C32、DFlash→开启截断DSpark配对AC/TPS。

<!-- DSPARK_E2_20261013_BEGIN -->
### 4.5 E2六数据集扩展评测：2026-10-13独立批次

整批48正式窗、24配对、16000请求已完成，最终审计通过。
两臂为E2独立完整单训DFlash和E2联合主体＋完整三层FULL15_NOFILL，主体权重不同。
C1/C32、T0/T1，BF16、batch-invariant、CUDA Graph，nonthinking，固定15 proposal、max2048；无截断、辅助预测或专家路由。
固定KV C1=52GiB、C32=44GiB，utilization=0.92，smoke与formal一致；五套context4096，Arena-Hard-v2为12288，无prompt截断。

数据取自DeepSpec固定提交`005e03b81cec38b7da6399833d609ee89a2587f2`：HumanEval164、MBPP256、LiveCodeBench500、MT-Bench80首轮、Alpaca500、Arena-Hard-v2 500；按作者抽样顺序，请求种子980406＋index。MBPP上游256来源行包含255个不同题面，按原样保留。
本批对齐数据和部分评测口径，仍使用本项目vLLM与15 proposal；DSpark离线表使用7 proposal，因此不能称完整复现。AC含bonus，不是pass@1；本批未执行代码正确率或对话judge评分。

| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |
|---|---|---:|---:|---:|---:|
| HumanEval | C1 T0 | 5.542→6.344 | +14.48% | 134.4→148.6 | +10.57% |
| HumanEval | C1 T1 | 4.840→5.731 | +18.43% | 117.5→134.4 | +14.38% |
| HumanEval | C32 T0 | 5.542→6.343 | +14.46% | 1999.1→2175.2 | +8.81% |
| HumanEval | C32 T1 | 4.810→5.678 | +18.03% | 1793.4→1987.0 | +10.80% |
| MBPP | C1 T0 | 4.975→5.694 | +14.46% | 121.2→134.2 | +10.70% |
| MBPP | C1 T1 | 4.325→5.029 | +16.28% | 105.8→118.8 | +12.36% |
| MBPP | C32 T0 | 4.980→5.694 | +14.32% | 1968.3→2131.3 | +8.28% |
| MBPP | C32 T1 | 4.269→4.986 | +16.80% | 1458.8→1816.8 | +24.54% |
| LiveCodeBench | C1 T0 | 6.140→6.733 | +9.67% | 143.8→152.1 | +5.78% |
| LiveCodeBench | C1 T1 | 4.494→5.204 | +15.81% | 105.7→118.1 | +11.71% |
| LiveCodeBench | C32 T0 | 6.140→6.737 | +9.72% | 1732.4→1768.8 | +2.10% |
| LiveCodeBench | C32 T1 | 4.495→5.117 | +13.82% | 1181.7→1276.7 | +8.03% |
| MT-Bench | C1 T0 | 3.145→3.413 | +8.52% | 77.3→81.2 | +5.12% |
| MT-Bench | C1 T1 | 2.765→3.190 | +15.35% | 68.0→75.8 | +11.53% |
| MT-Bench | C32 T0 | 3.148→3.415 | +8.48% | 963.7→987.1 | +2.42% |
| MT-Bench | C32 T1 | 2.779→3.185 | +14.61% | 824.3→900.3 | +9.23% |
| Alpaca | C1 T0 | 2.930→3.192 | +8.93% | 72.0→75.9 | +5.43% |
| Alpaca | C1 T1 | 2.633→2.945 | +11.86% | 64.7→70.0 | +8.14% |
| Alpaca | C32 T0 | 2.930→3.192 | +8.95% | 1149.9→1191.6 | +3.63% |
| Alpaca | C32 T1 | 2.634→2.958 | +12.27% | 1005.5→1095.2 | +8.92% |
| Arena-Hard-v2 | C1 T0 | 3.341→3.542 | +6.01% | 80.5→82.8 | +2.86% |
| Arena-Hard-v2 | C1 T1 | 2.681→2.919 | +8.86% | 64.6→68.1 | +5.47% |
| Arena-Hard-v2 | C32 T0 | 3.340→3.542 | +6.05% | 1213.7→1220.9 | +0.59% |
| Arena-Hard-v2 | C32 T1 | 2.674→2.912 | +8.91% | 957.8→977.7 | +2.08% |

增长率均按本批未舍入JSON计算；4.1–4.3中的旧max2096/seed42批次保留为历史，不跨批绝对TPS相除，也不据跨论文增幅排名。
48窗均通过隔离、吞吐证据与target/draft Graph检查，无eager或fallback；12组T0两臂最终输出逐题一致。183份冻结源码及3份调度/报告/审计脚本身份一致，权重与依赖由控制器最终verify重新核验。
[本批原始结果](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/COMPARISON.json)、[完整报告](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/RESULTS_CN.md)、[最终审计](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/FINAL_AUDIT.json)、[收尾审计](residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1/COMPLETION_AUDIT.json)。

<!-- DSPARK_E2_20261013_END -->

## 5. 模块耗时与当前结论

下表是已完成的GSM8K三层服务诊断，固定C1 KV52GiB；单位为ms/整批decode block均值，不是正式吞吐测试的每token延迟。

| 模块/跨度 | C1 T0 | C1 T1 |
|---|---:|---:|
| 草稿主体 | 4.460 | 4.461 |
| 草稿原始LM投影 | 1.544 | 1.545 |
| 完整修复头＋草稿采样 | 1.297 | 1.340 |
| 完整propose，含准备与等待 | 8.058 | 8.095 |
| target forward | 31.936 | 31.955 |
| target forward起点至验证终点 | 33.511 | 33.545 |

跨度有嵌套，不能逐行相加。该轮高并发诊断按此前要求测的是C40，**没有C32模块细分时长**。头内细分重放存在约2.2ms事件扰动，未将其当作真实服务头成本；也不把Domino论文的1.20ms直接当成同测量边界的横向对照。[计时报告与覆盖说明](residualV3-nonthing/results/epoch2_head3_module_latency_C1_C40_20261012_v1/RESULTS_CN.md)。

当前证据支持：三层完整模型相对本项目独立训练DFlash，在三个领域C1/C32的12组AC/TPS均为正收益；代码T1收益较大，对话较小。历史统一实现下，E2相对E1额外收益有限，后续实现优化需单独解释。

我们的明确结构特点是**当前分布q感知＋实际前驱条件＋方向/剂量分离的受控残差**。目前尚未单独证明每路输入、方向/剂量或OUTPUT目标各自的贡献；也没有统一训练与引擎的跨方法复测。从架构范围看，Domino和我们都在并行主体后增加因果修正头；DFlash-2另加主体卷积，DSpark另做验证调度。Domino两组训练实验不能写成两种结构版本。现有跨论文结果用于描述各自增益，不能据此直接排列方法优劣。

本报告集中整理已完成的三层结果。独立与联合模型主体权重不同，收益属于完整训练路线，不能全部解释为固定主体推理加头；HumanEval未报告pass@1、MT-Bench未报告质量评分。源码、训练协议、历史窗口和正式结果均在对应段落链接，可直接核对。

本版资料、网页快照SHA与修改范围见[来源及文档审计](residualV3_main_v1/research/group_meeting_sources_20261013/REPORT_AUDIT.json)。
