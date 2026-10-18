# 尾部截断的性能实现核查

核查时间：2026-10-10T15:28:39.122728+00:00。C1 四组正式窗口已完成，C40 仍在运行。

结论：C1 的接收长度基本保留，但新增特征提取、掩码路径和流同步没有被充分抵消；target 前缀确实变短，图执行行数仍补齐到16。存在明确的性能实现问题，目前证据不能把吞吐下降解释成预测头没有训练好。

## 同轮正式对照

| 方法 | T | 接收长度 | token/s | AL对FULL15 | TPS对FULL15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL15 | 0 | 8.515070 | 192.100 | +0.00% | +0.00% |
| FULL15 | 1 | 7.488577 | 169.527 | +0.00% | +0.00% |
| Q45_CUT | 0 | 8.483994 | 172.924 | -0.36% | -9.98% |
| Q45_CUT | 1 | 7.445799 | 152.228 | -0.57% | -10.20% |
| Q45_LATENT_CUT | 0 | 8.490007 | 172.090 | -0.29% | -10.42% |
| Q45_LATENT_CUT | 1 | 7.519466 | 153.359 | +0.41% | -9.54% |

C1=200、GSM8K nonthinking，T0/T1、block16=anchor+15 proposal，max_new2096/context4096，BF16，同一GPU、同一FULL_AND_PIECEWISE配置。四组均有隔离证据，正式窗口没有eager/fallback。接收长度指标为1+接受proposal数/草稿轮数。

## 实际执行工作

| 方法 | T | 平均修复位置 | target逻辑行/轮 | target图行/轮 | piecewise占比 | 墙钟摊销ms/轮 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL15 | 0 | 15.0000 | 16.0000 | 16.0000 | 0.00% | 44.4054 |
| FULL15 | 1 | 15.0000 | 16.0000 | 16.0000 | 0.00% | 44.2563 |
| Q45_CUT | 0 | 13.1310 | 14.1861 | 16.0000 | 45.72% | 49.1481 |
| Q45_CUT | 1 | 12.6650 | 13.7123 | 16.0000 | 54.83% | 49.0056 |
| Q45_LATENT_CUT | 0 | 13.1650 | 14.2217 | 16.0000 | 45.22% | 49.4213 |
| Q45_LATENT_CUT | 1 | 12.4920 | 13.5417 | 16.0000 | 58.60% | 49.1287 |

墙钟摊销=正式窗口wall_seconds/decode_calls，包含prefill和请求服务开销，不能当作纯target或纯头延迟。Q45_CUT每轮约多4.74–4.75ms；没有逐模块正式计时，尚不能把这部分全部归于预测头或piecewise。

## 已确认的代码开销

1. [q45特征提取](../../code/epoch2_trained_aux_tail_cut_20261010_v1/native_runtime/head_latency/q45_features.py#L39)每个位置重新归约全词表normalizer/熵并topk256。归约为每行一个CTA分段扫描词表；在C1只有一个CTA。小MLP只有1538/9730参数，特征提取和同步值得优先查，不能只缩小MLP。
2. [signal](../../code/epoch2_trained_aux_tail_cut_20261010_v1/native_runtime/head_latency/trained_tail.py#L73)在读取alive前无条件提取特征和预测。[rollout](../../code/epoch2_trained_aux_tail_cut_20261010_v1/native_runtime/head_latency/tail_pipeline.py#L60)仍运行15个位置，停止后尾部继续q45特征提取和原采样。掩码省了部分repair矩阵计算，但prepare、部分归一化、采样及图内launch位置仍保留。C40的16行tile只要有一行活跃，整tile矩阵计算继续，因此多并发收益需单独评估。
3. [target图尺寸](../../code/epoch2_trained_aux_tail_cut_20261010_v1/native_evaluate.py#L117)为16的倍数，C1只捕获16。缩短的11–15行前缀仍被补到16行；attention逻辑长度可能减少，但没有减少这些捕获图的执行行预算。大量变长轮次切换piecewise，仍是CUDA Graph，没有退到eager。
4. 预测使用第二条CUDA stream，与下一位置repair重叠；主流每步等待j−2的预测事件。重叠并不消除词表读取、计算资源占用及依赖等待。

## 微计时对照与正确性

已有[MICRO_GATE.json](MICRO_GATE.json)在C1/T0合成状态下：FULL15约2.0514ms，启用预测但禁止截断约3.4675ms；强制第10位置停止，流水线仍约3.4103ms。此处包含头和实际采样，不能替代服务TPS，但支持“现有流水线开销大于省下的头计算”。
当前C1 T0，两个CUT相对FULL15的最终文本均为0/200差异。40个合成数学/流/原生q一致性案例通过；冻结源码全部167项哈希仍一致。没有发现导致这次下降的token逻辑错误；这些证据不等于证明所有C40/T1路径完全等价。

## 下一版优先改动与验证

先在独立副本给target建立实际前缀长度对应的图尺寸，验证decode_graph_rows确实下降；再对停止后的q45计算和采样作设备端跳过，保持有效前缀的原生q和采样正确；随后融合post-repair q归约和topk，复用可共享统计量。保持当前修复后q45定义、冻结预测权重、阈值和修复权重，先做特征值/决定/有效前缀logits与token校验，再测正式TPS。
至少加入“计算预测但不截断”的对照以分离新增预测开销，以及细化target图尺寸的对照。逐模块计时与无插桩TPS分开，避免插桩改变正式吞吐。

本次仅写入诊断和更新总账，未修改正在运行的冻结推理源码、未中断C40、未删除或改写训练权重。原始证据：[即时C1日志](single_card_C1_formal.log)、[窗口FULL15](C1/formal/round_1/FULL15/windows.json)、[窗口Q45](C1/formal/round_1/Q45_CUT/windows.json)、[窗口Q45+latent](C1/formal/round_1/Q45_LATENT_CUT/windows.json)、[核查数值](PERFORMANCE_DIAGNOSIS.json)。
