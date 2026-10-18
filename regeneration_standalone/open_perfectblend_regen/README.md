# Open-PerfectBlend：843,028 条 Qwen3-8B non-thinking 训练池

2026-10-05 当前配方改为 non-thinking，prompt + answer 总长度最多 4096。
使用 RedHatAI/Qwen3-8B-speculator.dflash 的新版 non-thinking 权重，固定 revision
`1a11b170eb65c8a62c80ecd01dfe22a5907298e6`，block size 16 / 15 个草稿 token。
权重保存在 Hugging Face 默认缓存，旧版 block size 8 的 thinking 权重不覆盖。
`profiles.py` 将新旧配方的数据、worker 日志和调度器目录分开。
旧 thinking 数据保留在 `pool_500k_seed42/regenerated_ctx8192/thinking/`；
其历史进度可用 `progress.sh --profile thinking_ctx8192` 查看。
切换脚本 `rolling_nonthinking_handoff.py` 逐张卡操作；停止一张旧 worker 前在同卡
保留小 CUDA 上下文，新 non-thinking worker 保存有效会话后释放，再切换下一张。
切换记录在 `nonthinking_ctx4096_handoff_20261005/status.json`。

当前重生成全部 843,028 个训练侧会话，分为 85 片：84 片各 10,000 条，
最后一片 3,028 条。保留原有 50 个分片，追加 343,028 条，不重做已生成数据。
评测采用在线测试，
不重生成验证集。原始 `manifest.json` 中保留此前准备的 2,000 条留出提示词作为
抽样记录；它们不进入当前队列、生成请求或完成计数。当前范围见
`regenerated_ctx4096/nonthinking/regeneration_scope.json`。
每个分片均检查 MetaMathQA、Orca Math、Evol-CodeAlpaca、UltraInteract、UltraChat、
UltraFeedback、AutoIF、LMSYS Arena 八个来源是否全部出现。
计数单位是会话；多轮会话逐轮重生成，因此请求数和最终 JSONL 行数可能超过会话数。

## 数据配方

- 原始数据：[mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)。
- 从下载清单读取不可变 commit SHA；原始 Parquet 和 Arrow 留在 Hugging Face 默认缓存。
- 将角色统一为 `system/user/assistant`，抽样文件仅保留 system 和所有 user 轮次。
  原始 assistant 回答不进入生成历史；后续轮次使用本次目标模型生成的回答。
- 检查完整会话提示词重复项；同一个首轮问题的不同后续会话划到同一侧。
- 先按首轮提示词组用 seed=42 留出约 5% 组，过滤后训练侧共有 843,028 条，全部使用。
  原有 500,000 条保持原分片、ID 和顺序；从同一训练侧补齐其余 343,028 条。
  留出侧共 44,362 条，原准备阶段选出的 2,000 条提示词保留为历史记录，当前不重生成。
- 以过滤后真实 `source` 条数作为权重，用最大余数法确定抽样配额和分片配额。
  `source` 中 UltraChat / Orca Math 的仓库名可能与数据卡展示的上游仓库名不同；
  每条数据保留原始 source，并另按八个来源族统计。
- 使用固定版本 DSpark 官方九个评测数据文件的完整提示词池，以及已缓存的
  GSM8K test / HumanEval / MT-Bench 原始问题，检查重合。
  匹配范围是 NFKC 与空白规范化后的精确问题，未宣称解决改写题或语义重合。
- 每条记录保留源数据版本、原始行号、稳定 ID、提示词 SHA256、分组 SHA256、split 和 shard。

DSpark 的论文领域比例是数学 39.4%、代码 38.9%、对话 17.6%、指令遵循 4.1%。
本方案保留公开数据真实来源占比，不把跨领域来源硬归为某一个领域。
DSpark 原文使用 non-thinking、约 1.3M 条及 10 epochs；这里使用去重、评测重复过滤
和分组留出后的全部 843,028 条训练侧数据。
我们的模型、DSpark 和 DFlash 比较时应共享这份重生成结果。
参考：[DSpark §4.1](https://arxiv.org/html/2607.05147v1#S4.SS1)、
[官方数据准备脚本](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/download_and_split.py)。
官方脚本按行划分 95/5，本实现加入了评测重复检查、会话去重和首轮问题分组。

## 下载、准备、核验（只使用 CPU）

使用现有 `llm_dflash_eval_v026` 环境和父目录 `.client_deps`，不安装新包：

```bash
cd /data/weijiahong/speculators-main-lora/regeneration_standalone
HF_ENDPOINT=https://huggingface.co bash open_perfectblend_regen/download.sh
HF_ENDPOINT=https://huggingface.co bash open_perfectblend_regen/prepare.sh
bash open_perfectblend_regen/verify.sh
bash open_perfectblend_regen/status.sh
```

默认 Hub 缓存为 `/data/weijiahong/.cache/huggingface/hub/`，
Arrow 缓存为 `/data/weijiahong/.cache/huggingface/datasets/`。
不设置 `cache_dir/local_dir`；HF 缓存环境变量如果已设置，按 Hugging Face 规则生效。
下载入口默认关闭 Xet 并使用标准 HTTP；可通过环境变量覆盖。
官方评测提示词保存到本目录 `eval_sources/<commit>/`，用于重复过滤。
首次准备会下载这些小文件；缓存齐备后可用 `prepare.sh --offline`。
已存在相同配方的训练池会核验各分片 SHA256 后复用，不覆盖已有清单。
改变版本、seed、规模或配方时，用新的 `--pool-dir`。
本次扩容通过 `extend_pool.py` 先在 CPU 上准备并核验追加分片，再原子更新清单。
原清单归档在 `manifests/base_500000_<SHA256>.json`；原生成契约、模型和采样设置保持一致。
扩容的准备、核验、进程接管记录在 `extension_843028_seed42_20261005/`。
实际路径继续使用 `pool_500k_seed42/`，兼容当前进程、检查点和 IDE 文件路径；
`pool_843028_seed42/` 是指向它的快捷路径。实际数量以清单和总进度为准。

```text
pool_500k_seed42/
  manifest.json                 # 固定配方、完整来源统计、分片 SHA256
  verification.json             # 全量 CPU 核验结果
  prompts/train/shard-000.jsonl  # ... shard-084.jsonl；最后一片 3,028 条
  manifests/base_500000_<SHA256>.json  # 扩容前的清单，原 50 片保持不变
  prompts/validation/shard-000.jsonl  # 历史留出提示词，不参与当前重生成
  regenerated_ctx4096/nonthinking/train/    # 当前训练使用的目标模型输出
  regenerated_ctx4096/nonthinking/regeneration_scope.json  # 仅 train，在线评测
  regenerated_ctx4096/nonthinking/generation_contract.json
  regenerated_ctx4096/nonthinking/training_data_paths.json
  regenerated_ctx8192/thinking/             # 保留旧 thinking 数据和轮次检查点
  regenerated/                  # 保留旧 16K 试跑输出，单独存放
  logs/<run_id>/
```

## 重生成

默认只打印计划。thinking 默认关闭，目标采样为 temperature=0.7、top_p=0.8、
top_k=20、min_p=0、seed=42，保存真实 completion token IDs。
上下文上限 4,096，并发从 40 开始。每次生成前调用服务端 `/tokenize`，
使用服务端实际 chat template 计算 prompt token 数（包含 system、历史、
本轮问题、特殊 token 和 non-thinking 模板前缀），请求的输出预算严格为
`max_completion_tokens = 4096 - prompt_tokens`。回答提前遇到 EOS 时正常结束。
多轮会话每一轮重新计算预算；保存的每个训练样本均为 prompt + answer ≤ 4,096。
与 `residualV3_main_v1/runtime/data.py` 对齐，每行至少有 8 个监督 token。
不足 8 个剩余位置的提示词不发生成请求；多轮会话保留此前有效轮次。
整段会话没有有效样本时，写入独立的 `shard-NNN.skipped.jsonl`，续跑时跳过。
提前结束且监督 token 不足 8 的样本不进入训练 JSONL。
采用原始 stock DFlash 草稿加速，BF16，单卡 TP=1 / DP=1；不导入实验修复补丁。

先看首批 1,000 条的命令：

```bash
bash open_perfectblend_regen/run.sh --shards 0 --max-samples 1000 --dry-run
```

实际运行需要设置已经分配的单卡编号，脚本不会自动选择显卡：

```bash
: "${GPU_ID:?先设置已经分配的单卡 GPU_ID}"
bash open_perfectblend_regen/run.sh --shards 0 --max-samples 1000 --run --gpu "$GPU_ID"
```

首批结果写入正式 shard-000，后续继续生成时跳过已完成 ID。日志里的
`pilot_report.json` 给出吞吐、截断数、输出字节数及粗略时间/空间估算；
扩容前的历史报告使用当时的 500K 目标，当前目标以总进度为准。
再完成前 10 个分片，即 100K：

```bash
bash open_perfectblend_regen/run.sh --shards 0-9 --run --gpu "$GPU_ID"
```

补齐其余 743,028 条（手动运行示例；自动调度会自行领取全部分片）：

```bash
bash open_perfectblend_regen/run.sh --shards 10-84 --run --gpu "$GPU_ID"
```

每次任务只启动一个模型服务，依次处理所选分片，完成后关闭自身服务。
同一输出分片使用独占锁。多卡任务需指定互不重叠的分片以及不同 `--port`。
thinking 与 non-thinking 分开存放；生成契约防止用不同模型版本、采样参数或长度续写同一池。
调整长度等参数时，用 `--output-dir /新的生成输出目录` 保留原数据的生成来源；
提示词清单继续复用，不必再次下载或抽样。查看该目录的进度可用
`status.sh --generation-dir /新的生成输出目录`。
每轮 HTTP 生成完成后，将真实 token IDs、已生成历史及下一轮位置写入
`shard-NNN.jsonl.checkpoints/`，通过 `fsync` 和原子替换持久保存。
整段会话完成后提交训练 JSONL，并 `fsync`，随后清理该会话的临时检查点。
意外中断后从最近已保存轮次继续，正在生成但尚未返回的请求会重试。
续跑会将末尾未完成的写入保存为 `.interrupted-tail-*.bin` 并恢复完整会话边界；
文件内部损坏会报错，不静默跳过。错误记录在 `shard-NNN.errors.jsonl`。

## 自动选择空闲卡：共享分片队列，按北京时间调整上限

2026-10-04 已按用户明确授权改为整机空闲卡调度，取消卡 1 / 卡 3 固定分片归属。
每天按北京时间（`Asia/Shanghai`）调整本任务的显卡上限：

- 00:00–08:00：最多六张空闲卡。
- 08:00–24:00：最多四张卡；有可用空卡就补齐，不预留空卡。

2026-10-05 按用户要求取消预留空卡限制：白天有空卡就补到三张，不要求保留一张空卡。
整机占满时保持已运行的三张，不再根据最后一张卡由谁占用而退回两张。
2026-10-06 按用户要求把白天上限从三张提高到四张，新增卡 1，原卡 2、3、6 的
监管进程、模型服务和客户端继续运行；接管记录见 `daytime_four_20261006/`。
已有两张继续保留并优先补齐；白天只在超过四张时按时段上限缩卡。
正常每 20 秒检查一次空卡状态。
2026-10-05 修复客户端失败导致健康模型服务被关闭的问题：客户端只补未完成 ID，
同一服务最多连续尝试三次；仍失败的分片暂缓 300 秒，继续领取其他分片。
未成功的会话继续计入剩余数量，错误和轮次检查点保留，不自动当作成功或跳过。
当其他分片被占用或正在等待重试时，worker 保留模型服务，继续等待共享队列。
监管进程意外失败时也保留仍存活的模型服务，调度器优先重新接管，避免释放显卡。
只有正常全池完成、按占卡规则主动退出，或模型服务本身已经死亡，才执行服务清理。
少于两张且仍有至少两个待处理分片时，每 2 秒检查一次可用卡；总状态显示
`state=waiting_for_second_gpu`、`minimum_worker_shortfall`，不再只显示 `running`。
领取新空卡后，在 CPU 规划和模型加载前启动小型 CUDA 占用，服务就绪后释放。
这能缩短显卡被空卡扫描程序领取的窗口；共享机器上的操作系统级独占仍需要管理员
或集群资源调度器支持。其他用户已占用的卡不会被终止或强行共享。
07:59 开始停止超过白天上限的 worker，给 08:00 缩回最多四张留出清理时间。
停止的分片保留已生成数据和轮次检查点，后续领取时继续生成。
`auto_scheduler_nonthinking_ctx4096_20261005/gpu_policy.json` 保存每日规则；总状态的 `capacity_schedule`
显示当前北京时间、时段上限和下次切换时间；`allocation` 显示可用空卡和当前调度目标，
`max_workers_limit` 为当前时段上限（白天四张、夜间六张）；
`allocation.reserved_idle_gpus_above_minimum=0` 表示不预留空卡。
各卡 `gpu_inventory.compute_processes` 列出 PID、UID 和用户名。
`scheduler.py --gpus all` 每 20 秒检查所有显卡；没有计算进程且可用显存至少为
总显存的 97% 时可领取。本配置先使用已经验证的 80GB 及以上显卡。
利用率为零但仍有计算进程的卡不算空卡；已有训练和评测进程继续运行。

每张领取到的卡各运行一个 Qwen3-8B non-thinking + 新版 Red Hat DFlash 服务和一个客户端，
单卡 TP=1，实际并发先用 40，服务容量 128。继续复用原有 CUDA 13 兼容库。
草稿长度可通过 `regenerated_ctx4096/nonthinking/speculation_policy.json` 的
`num_speculative_tokens` 设置全局默认；各卡 `speculation.json` 的试验设置优先。
`draft_length_tuning_20261006/` 保存卡 1 上 15 / 7 / 5 / 3 个草稿 token 的同提示对比、
重复确认和推广记录。诊断输出单独保存，不计入训练池完成数量。
2026-10-06 的 128 条同提示试验覆盖八个来源，40 并发测得 15 / 7 / 3 / 5 个草稿
约为 1,487 / 1,964 / 2,108 / 2,190 输出 token/s，5 个重复确认约为 2,248。
卡 6 的同并发生产确认约从 1,489 提高到 2,215 输出 token/s，增加约 49%。
因此全局默认改为 5；结果与当前部署状态见 `draft_length_tuning_20261006/final_status.json`，
原始对比结果在 `retry_v3/*-result.json`，生产确认在 `production_rollout_v4/status.json`。
切换时曾因暂停后的显卡查询超时触发错误清理，后续已修复并恢复任务。
显卡查询超时或失败现在重试，不从旧的空卡记录领取新卡；调度器意外异常也保留 worker。
只在显式停止调度器时关闭它的 worker；生成的数据和轮次检查点继续复用。
调整草稿长度只改变服务的预测和验证开销；目标模型、采样参数、4096 总长度和
重生成检查点配方保持不变。原 `generation_contract.json` 保留为不可变配方锚点，
运行清单的 `serving_execution.actual_speculative_config` 记录实际草稿长度；
`serving_execution_segments` 按输出文件字节位置记录同一分片历次服务配置。
模型服务切换前先在该卡建立 CUDA 占用，单卡切换完成并保存新会话后才处理下一卡。
每个 worker 尝试领取尚未处理完的分片，独占锁保护整个分片；锁已占用就领取下一片。
完成当前分片后继续领取下一片，直到当前可领取的训练分片都已处理。
较晚空闲的卡会自动加入，失败任务冷却后重试，并从已保存轮次续跑。
worker 保留服务等待未完成分片，直到全池处理结束；全池是否完成看总状态。

输出统一在 `regenerated_ctx4096/nonthinking/`，共有 85 个训练分片。
每个分片（包括最后 3,028 条）均包含八个来源。端口默认从 `18080 + GPU index` 开始，
已有服务沿用原端口；端口被占用时使用其他空闲端口，不停止占用端口的进程。
本次新配方从固定提示词池重新生成，旧 thinking 输出和检查点保持独立。

查看总任务状态及各卡归属：

```bash
cat open_perfectblend_regen/auto_scheduler_nonthinking_ctx4096_20261005/status.json
bash open_perfectblend_regen/progress.sh
```

`auto_scheduler_nonthinking_ctx4096_20261005/scheduler.log` 为调度日志；各卡服务、客户端、监管日志为
`gpu<N>_nonthinking_ctx4096_20261005/vllm.log`、`<split>-<shard>-client.log`、`supervisor.log`。
具体某个分片在哪张卡运行，以总状态中的 `workers` 为准。

意外中断后用相同入口续跑（先确认原调度器已退出；全局锁防止重复启动）：

```bash
mkdir -p open_perfectblend_regen/auto_scheduler_nonthinking_ctx4096_20261005
nohup bash open_perfectblend_regen/schedule.sh --gpus all --max-workers 4 --night-max-workers 6 \
  --profile nonthinking_ctx4096 \
  >> open_perfectblend_regen/auto_scheduler_nonthinking_ctx4096_20261005/scheduler.log 2>&1 &
```

`--gpus 1,3` 可将空闲卡调度限定到指定范围；`--max-workers` 设置白天上限，允许 2、3、4，默认 3。
当前任务显式设置为 4，有可用空卡就扩展到四张，不预留空卡；续跑请保留 `--max-workers 4`。
`--night-max-workers` 设置北京时间 00:00–08:00 上限，允许 1–6，默认 6。
整机已有其他训练任务的卡仍不领取；本任务已有 worker 计入当前时段的总上限。
调度器崩溃时已有 worker 和客户端继续运行；重新启动后识别并接管本任务的已有进程。
修复与接管记录见 `gpu_retention_fix_20261005/`；各卡失败重试记录在
`client_recovery.jsonl`，新服务的启动占用记录在 `startup_reservation.json`。
取消白天空卡预留的配置与接管记录见 `daytime_no_reserve_20261005/`。
每张卡的 `supervisor.lock` 和每个分片的 `.lock` 共同防止重复写入。
新配方首次从并发 40 测试 64 / 80 / 96 / 128，选择接近最高吞吐且无大量 KV 抢占的并发。
2026-10-05 实测 40 / 64 / 80 / 96 约为 1,766 / 1,722 / 1,576 / 1,785 输出 token/s，
40 并发利用率为 100%，吞吐在最佳值的 3% 以内。128 并发触发服务端 CUDA illegal memory access；
当前固定已测的 40 并发，保留失败记录。服务异常后会持久化较低并发并停止自动加压，
重启不会重复尝试失败档位。已完成的调优结果可在其他同配置卡上复用。
本次恢复记录在 `cuda_crash_recovery_20261005/`；原输出和轮次检查点继续使用。
现场并发可通过对应卡的 `concurrency.json` 原子更新，范围 1–128。
磁盘不足 8 GiB 时停止发新任务，保留检查点，并在总状态中标记磁盘不足。

此前 16K 配置在卡 1 测得并发 40 / 64 / 80 / 96 / 128 分别约
2,080 / 1,895 / 1,713 / 1,850 / 2,012 输出 token/s，均无 KV 抢占。
当前先沿用 40；这些旧短窗口数值不代表当前多卡配置的新性能测量。
`telemetry.jsonl` 持续记录各卡的实际利用率和投机解码计数。
切换到 8K 时复用的 407 条完整会话和 16 个部分会话检查点仍保留；
迁移明细和旧响应 token 哈希清单见卡 1 目录的 `ctx8192_migration.json`、
`ctx8192_reused_rows.json`，旧 16K 输出单独保存在 `regenerated/`。

## 如何确认所有数据重生成完成

```bash
bash open_perfectblend_regen/progress.sh
```

训练期望 843,028 会话，评测在线进行；计数单位是会话而非 JSONL 行数。
关注下列字段：

- `remaining_conversations = 0`：所有源会话都有有效生成结果或明确的跳过记录。
- `complete_shards = total_shards = 85`：85 个训练分片全部处理结束。
- `all_generated_without_skips = true`：843,028 训练会话均有有效生成结果。
- `all_processed = true` 但 `all_generated_without_skips = false`：已处理完，但存在跳过项；
  `splits.train.skipped_conversations` 单独显示数量，不计入成功生成数。

上述进度来自持续更新的分片状态。调度器检测全部处理结束后自动执行一次完整核验：
校验固定提示词 SHA256、源 ID 覆盖、重复提交、跳过记录及实际训练验证器
（prompt + answer ≤ 4,096，监督 token 至少 8），成功后写入
`regenerated_ctx4096/nonthinking/completion.json`。
其 `state = complete` 表示全部成功生成，`state = complete_with_skips` 表示处理结束但有跳过项。
核验失败不会写完成标记；训练文件清单随后同步更新。

全部处理完后也可手动重新核验：

```bash
bash open_perfectblend_regen/progress.sh --verify
```

逐分片进度可加 `--details`。仍有 worker 持有分片锁时，全量核验会提示等待，
避免读取写入中的 JSONL。旧错误日志可能保留已成功重试的历史错误，不能据此判断未完成。

## 训练输入与状态

生成后将 `training_data_paths.json` 中 `train` 列出的 JSONL 作为混合训练输入；
`validation` 列表留空，评测使用现有在线测试流程。输入已经包含 `input_ids/loss_mask`，
可供现有训练预处理流程读取。`finish_reason=length` 的记录保留截断标记。
全量会话是否均已处理，要看 `progress.sh` 的 remaining_conversations、
complete_shards 与 all_processed；有效训练会话数单独统计，不把跳过项计入。
各卡的 supervisor 每 5 秒更新运行分片状态；客户端每完成会话更新 `.progress.json`。
正在生成的轮次已保存在 `.checkpoints/`，未完成会话不计入完成数量。

已有 OpenMath/OpenCode 生成结果继续保存；本训练池不自动拼入历史生成数据。
CPU 测试：父目录 `test_perfectblend.py`，覆盖来源抽样、重复过滤、分组隔离、
真实 tokenizer 的模拟 HTTP、thinking、多轮、源 metadata、断点续跑和文件篡改拒绝。
