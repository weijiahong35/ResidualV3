# dflash2 对比头：BF16 与原生在线测试

这是移植到本地 DFlash body 的 **dflash2 对比头**，不是完整外部模型。代码、采样和 native 入口都在本目录；`body_only_draft` 是本地 body 资产（大权重只读硬链接），不导入 ResidualV3 实验代码。

## 当前数值和采样契约

- head 权重、token codebook、线性投影、修正 logits 加法及返回边界为 BF16；概率归一化/损失为 FP32。
- native T0：target/draft 均贪心；T1：均随机。宿主使用 stock vLLM Gumbel、位置/请求种子及拒绝采样。
- 每个 proposal 反馈实际选中 token，首位置使用已验证 anchor。验证器使用本次实际采样写入的 processed logits/q。
- DFlash2 保留 unary Top16 候选限制、仅在候选内采样；所有位置的 raw TopK/hidden projection 预先批量计算。不能把算法候选限制改成全词表采样并声称同一头。
- DSpark 使用 vanilla Markov 全词表 bias 和 confidence 线性头；不引入完整 DSpark 调度器。
- eager、TP=1、并发1、7 proposals；固定既有64条开放诊断面板，T0/T1，seed=42+index，max_new_tokens=128。

## 入口区分

| 文件 | 用途 |
|---|---|
| `Baseline_train.py` | 缓存 hidden 上只训练头，BF16 |
| `Baseline_train_online.py` | 现场获取 target hidden 并重算 DFlash body；默认冻结 body，`--train-body` 才更新 body |
| `Baseline_test.py` | 固定轨迹的离线诊断，不能称为 native 接受长度 |
| `Baseline_test_trajectory.py` | 原来的 test_online，已改名；重算 hidden 但依旧是固定轨迹离线诊断 |
| `Baseline_test_online.py` | 新的原生生成测试：按实际 proposal 前缀验证，接受长度和吞吐同窗口采集 |
| `Baseline_native.py` / `Baseline_sampling.py` | 仅进程内安装的头采样集成和 BF16 递归；不改已安装 vLLM |
| `Baseline_sampling_test.py` | CPU 精度、强制非 Top1 反馈、支持集、T0/T1 和梯度检查 |

在线生成测试必须显式指定 checkpoint。新训练默认输出到 `eval_results/Baseline_dflash2_full_vocab_bf16_v2_seed42`，不会覆盖历史 FP32 权重。读取旧 FP32 checkpoint 必须传 `--convert-legacy-fp32`，只在内存转换；不宣称等同于 BF16 重新训练。

```bash
PY=/data/weijiahong/miniconda3/envs/dflash/bin/python
# CHECKPOINT 指向所选固定权重，OUT 指向新的结果目录。
"$PY" Baseline_dflash2_full_vocab_v1/Baseline_test_online.py \
  --gpu 1 --checkpoint "$CHECKPOINT" --output-dir "$OUT" --smoke --allow-shared-gpu
# 同一个 OUT、checkpoint 和协议，在 smoke 成功后执行两轮反向顺序测量。
"$PY" Baseline_dflash2_full_vocab_v1/Baseline_test_online.py \
  --gpu 1 --checkpoint "$CHECKPOINT" --output-dir "$OUT" --allow-shared-gpu
```

如果选择历史 FP32 checkpoint，上述两条命令都追加 `--convert-legacy-fp32`。smoke 训练权重仅允许 smoke 测试，不能误作正式模型结果。`--allow-shared-gpu` 表示接受共驻并记录限制，不得宣称隔离性能。空闲显存启动门槛45,000 MiB、运行保护6,000 MiB；只清理测试自身进程组。

## 在线输出与验收

`protocol.json` 冻结 checkpoint/body/source/panel 哈希、温度、精度和设备。历史 body+head checkpoint 会把相同 body overlay 用于 raw/head 两臂；绝不把不同 body 当作公共 raw。

Smoke 每臂 T0/T1 各2请求，32 tokens；实际 GPU 检查逐位置采样与批量 stock Gumbel一致、processed q一致、采样反馈一致，以及 T1 非 Top1 样本。smoke 有诊断同步，计时不能作为模型性能证据。

正式测量 raw/head 正反两轮，每温度4次 warmup，随后完整冻结面板。窗口内不写文件、不读 metrics、不做逐 block 诊断。窗口前后读取 native counters；窗口结束后保存每请求起止时间、elapsed、response/usage、总输出 token 数、wall time、tokens/s、延迟均值/p50/p95、含 bonus 的 A、各位置 S_j/D。TTFT/TPOT 没有流式时间戳则保持 null。

`runtime_summary.json`、`RUNTIME_RESULTS.md` 保存窗口结果；`smoke_summary.json`、`SMOKE_RESULTS.md` 分开保存集成验证结果。失败 trial 不自动覆盖；归档失败尝试后才能续跑。

## 未在本次悄悄改动的训练差异

本次修复限于 T、BF16、在线评测。DSpark 仍保留本地 `CE(target Top1)+TV+BCE` 配方，与官方响应 token 标签和 L1 权重存在差异；DFlash2 candidate CE 仍是实验性训练方案，不声称复现未公开的原作者配方。旧结果为历史诊断，不能当作新 BF16/native 结果。详见 `docs/BASELINE_HEAD_AUDIT_20260924.md`（修复前审计）。
