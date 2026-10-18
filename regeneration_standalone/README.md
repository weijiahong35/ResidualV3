# 独立重生成入口

此目录可以作为一个整体使用，不再导入仓库 `speculators` 包。
原 `examples/train/regenerate_data_math.sh`、`regenerate_data_simple.sh`、
`scripts/response_regeneration/script.py`、预处理源码均保留未改动。
2026-10-04 按用户要求扩展主仓库 `configs.py`，并同步到独立配置副本。
`source_origin.json` 保留原文件历史 SHA256，并记录配置文件当前 SHA256 和修改原因。

- `openmathinstruct2_regen/run.sh`：OpenMathInstruct-2，原 `train_1M` split。
- `opencodeinstruct_regen/run.sh`：OpenCodeInstruct，原 `train` split。
- `gsm8k_regen/run.sh`：默认 GSM8K `main/train`，7,473 条；`--split test` 为评测集 1,319 条。
- `humaneval_regen/run.sh`：HumanEval `test`，164 条。
- `mt_bench_regen/run.sh`：MT-Bench `train`，80 组双轮问题。
- `open_perfectblend_regen/`：Open-PerfectBlend 全部 843,028 条训练侧提示词，
  85 个训练分片，保留八个来源并检查评测问题重合；当前采用 non-thinking、
  prompt + answer 最多 4096、新版 Red Hat DFlash。原 thinking 输出单独保留，评测在线进行。
  [下载、准备与运行说明](open_perfectblend_regen/README.md)。
- `regenerate.py`、`regenlib/`：独立客户端、数据配置及辅助代码。
- `prepare_data.py`：独立预处理副本，非自动运行阶段；重生成 JSONL 已含 token IDs 和 loss mask。
- 每个数据集的 `regenerated/` 保存数据，`logs/<run_id>/` 保存后续运行日志。

## GSM8K、HumanEval、MT-Bench 下载与重生成

**用途纠正（2026-10-04）：用户需要重生成训练语料。** GSM8K 入口默认训练集；
HumanEval 与 MT-Bench 入口保留供评测/诊断，两者没有官方训练 split。
它们不能作为 EAGLE-3、DFlash 或 DSpark 的论文训练语料。

论文的训练数据规模：

| 方法 | 训练数据 | 规模与训练轮数 |
| --- | --- | --- |
| [EAGLE-3 §4](https://arxiv.org/html/2503.01840v3#S4) | ShareGPT + UltraChat-200K，目标模型重生成 | 约 68K + 464K = 532K 条；DeepSeek-R1 模型额外用 OpenThoughts-114k-math |
| [DFlash §5 / A.1](https://arxiv.org/html/2602.06036v2#S5) | Nemotron Post-Training Dataset V2 + CodeAlpaca，目标模型重生成 | 约 800K 条，6 epochs；多数消融取 100K 条 |
| [DSpark §4.1](https://arxiv.org/html/2607.05147v1#S4.SS1) | Open-PerfectBlend prompts，目标模型重生成 | 约 1.3M 条，10 epochs；DSpark / DFlash / EAGLE-3 对照模型都用此数据重训，非 thinking |
| [DFlash-2 官方资料](https://inco.ai/blog/dflash2/) | 未明确披露 | 未确认；不能套用 DFlash 原版的 800K |

以上是独立数据条目规模，不能与 token 数或乘 epochs 后的重复访问量混为一谈。
当前公开 Open-PerfectBlend 元信息为 1,420,909 条，DSpark 官方准备脚本默认
`train_test_split(test_size=0.05, seed=42)`，按此计算训练部分 1,349,863 条，保留部分 71,046 条。
这是根据当前版本和默认划分计算的数量，最终训练数量还受生成成功率和预处理影响。
大规模训练应使用已有 OpenMath / OpenCode 训练语料，或上述论文语料重生成；
GSM8K train 仅适合小规模数学训练实验。下面 HumanEval / MT-Bench 命令属于评测数据生成。

评测对齐的详细记录见 [论文数据对齐清单](../reports/paper_dataset_alignment_20261004.json)。
EAGLE 官方配套的 GSM8K / HumanEval / Alpaca / CNN-DailyMail 各 80 题，MT-Bench 为 80 组双轮；
DSpark 官方配置为 GSM8K 500、MATH500 500、AIME25 30、HumanEval 164、MBPP 256、
LiveCodeBench 500、MT-Bench 80、Alpaca 500、Arena-Hard-v2 500，seed=980406。
DSpark 评测代码只取 MT-Bench 首轮。DFlash / DFlash-2 原始资料没有明确确认各项实际评测数量，
其官方脚本的可用数据池大小、默认请求数和 README 示例数量不能直接认定为论文实测数量。

下载原始数据，使用现有 Python 环境，不安装包：

```bash
cd /data/weijiahong/speculators-main-lora/regeneration_standalone
HF_ENDPOINT=https://huggingface.co bash download_datasets.sh
```

默认下载 GSM8K 的 `main/train` 和 `main/test`、HumanEval 的 `test`、MT-Bench 的 `train`。
不传 `cache_dir`，不使用 `--local-dir` 或 `save_to_disk`。当前机器默认位置为：

- Hub 原始文件：`/data/weijiahong/.cache/huggingface/hub/`。
- Datasets Arrow 缓存：`/data/weijiahong/.cache/huggingface/datasets/`。

若其他终端设置了 `HF_HOME`、`HF_HUB_CACHE` 或 `HF_DATASETS_CACHE`，Hugging Face 会遵循这些覆盖项；
要使用默认位置，可在该命令前用 `env -u HF_HOME -u HF_HUB_CACHE -u HF_DATASETS_CACHE`。
`download_manifest.json` 记录每个数据集的 commit SHA、split、条数和缓存文件。
各数据集目录的 `download.sh` 可单独下载；记录会保留其他已下载的数据集。
入口会从下载清单自动读取数据版本，也可用 `--dataset-revision SHA` 明确指定。

先检查计划：

```bash
bash gsm8k_regen/run.sh --dry-run
bash humaneval_regen/run.sh --dry-run
bash mt_bench_regen/run.sh --dry-run
```

实际生成时把 `0` 替换为分配的 GPU 编号，三个命令顺序运行：

```bash
bash gsm8k_regen/run.sh --run --gpu 0
bash humaneval_regen/run.sh --run --gpu 0
bash mt_bench_regen/run.sh --run --gpu 0
```

三个新入口默认 `--thinking on`，明确发送 `chat_template_kwargs.enable_thinking=true`，
并使用 Qwen3 推荐的 `temperature=0.6, top_p=0.95, top_k=20, min_p=0`。
`--thinking off` 会另存文件，并采用非 thinking 默认采样；`--sampling-params` 可覆盖采样值。
总上下文与生成上限默认 8192，实际生成预算会扣除 prompt 长度；截断记录在 summary 中。
如需更长的 thinking，可根据显存调整 `--max-model-len`、`--max-tokens` 和 `--concurrency`。
不改变已有 OpenMath / OpenCode 入口的默认文件名与采样来源。

每个新数据集的默认输出为 `数据集名_regen/regenerated/数据集名_qwen3_8b_SPLIT_thinking_onpolicy.jsonl`。
MT-Bench 每组两次生成，一轮一条 boundary sample，正常全量为 80 组 / 160 行；
第二轮使用本次生成的第一轮回答。`--max-samples` 统计会话数，而不是 JSONL 行数。
输出保存真实 `input_ids`、`loss_mask` 和解码文本，thinking 生成 token 也受监督；
每行 metadata 记录原始数据集、split、版本和 source ID。

主仓库配置名为 `gsm8k`（训练集）、`gsm8k_test`、`humaneval`、`mt_bench`。
GSM8K 测试入口追加逐步解答和 boxed 答案提示；HumanEval 只发送代码题目的 prompt，
不发送 `canonical_solution` 或测试代码。HumanEval 和 MT-Bench 配置只有问题，需重生成后再预处理。
GSM8K 默认用于小规模训练；测试集需显式 `--split test`。HumanEval / MT-Bench 用于评测/诊断。
GSM8K 训练命令：

```bash
bash gsm8k_regen/run.sh --split train --run --gpu 0
```

训练与测试、thinking 与非 thinking 分开保存。用于训练的题目不应再用于报告独立评测成绩。

仅使用已下载缓存运行客户端/服务端时，可在运行命令前设置
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`；模型仍使用本机原权重目录。
新增 CPU 验证见 `test_benchmarks.py`、`benchmark_cpu_validation.log`；
旧入口回归验证见 `cpu_validation_20261004.log`，原 `cpu_validation.log` 保留。

## 现成 Qwen3-8B thinking 数据

- [Qwen3-8B-Regenerated-Collection](https://huggingface.co/datasets/inference-optimization/Qwen3-8B-Regenerated-Collection)：
  文件包括 `ultrachat_output.jsonl`、`evol_codealpaca_train_Qwen3-8B.jsonl`、
  `metamathqa_train_Qwen3-8B.jsonl`、Nemotron 等。抽查上述前三个文件的助手回答均包含
  `<think>...</think>`；仓库没有完整生成参数说明，不能据抽样断言所有行都含完整 thinking。
  核对的文件列表没有 GSM8K、HumanEval 或 MT-Bench 文件。
  下载单个文件到默认 Hub 缓存：

  ```bash
  HF_ENDPOINT=https://huggingface.co /data/weijiahong/miniconda3/envs/llm_dflash_eval_v026/bin/hf download \
    inference-optimization/Qwen3-8B-Regenerated-Collection ultrachat_output.jsonl --repo-type dataset
  ```

- [samuki-hf/thinking-rollouts](https://huggingface.co/datasets/samuki-hf/thinking-rollouts)：
  DS-1000 / LiveCodeBench 上的 Qwen3-8B thinking 与非 thinking rollout，
  提供 `raw_text`、`reasoning_text`、`answer_text` 和 token 数等列；每题 100 个样本。
  对应 thinking 文件在 `rollouts/domain=.../model=qwen3-8b-think/temp=.../data.parquet`。
  这属于其他代码任务，格式也不同于本地 `input_ids/loss_mask`，不能直接代替本轮三套数据。

现成集合只做来源核对，没有批量下载；本轮原始数据的下载与它们相互独立。

## 数据迁移与续跑

两个旧 `regenerated/` 目录已迁移到对应的 `数据集名_regen/regenerated/`。
旧路径的兼容软链接仅供历史代码使用。当前 `residualV3_static_vocab_online_joint_v1`
及 `standalone_static_vocab_online_answers_v3_20260927/preregistration.json` 已直接指向本目录中的
OpenMath JSONL，不经过旧软链接；后续代码也应使用新路径。
每个文件的 SHA256、字节数、行数及迁移前 inode 记录在 `data_migration_manifest.json`，
迁移前后字节一致，没有重写旧行、合并或删除 `.current.jsonl`。

继续使用原文件名：

- `openmathinstruct2_regen/regenerated/openmath_qwen3_8b_onpolicy.jsonl`：10,000 条。
- `opencodeinstruct_regen/regenerated/opencode_qwen3_8b_onpolicy.jsonl`：10,000 条。

按 `primary_id` 跳过已完成会话并追加同一个 JSONL。标准投机解码的拒绝采样
以保持目标采样分布为目标，不要求同种子逐条输出相同，因此不因启用投机而拆文件。
后续行的 `metadata.run_manifest` 指向续跑的配置记录；历史行保持原样。
主文件达到 `--max-samples` 时不启动服务，不重新生成已有数据。
`.current.jsonl` 是原有快照（数学 10,000 条、代码 1,963 条），不会自动更新。

## 现有环境与 CU13

服务端和客户端均使用现有：
`/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026/bin/python`。
没有新建或修改 conda 环境，没有安装/替换 torch、vLLM、CUDA。

CU13 设置对应原 `examples/evaluate/evaluate_dflash_lora_math_cu13.sh`
和 `residualV2/_common.sh` 的启动部分：

```text
删除 LD_PRELOAD
LD_LIBRARY_PATH=<v026>/cuda-compat:<v026>/lib
VLLM_ENABLE_CUDA_COMPATIBILITY=1
VLLM_CUDA_COMPATIBILITY_PATH=<v026>/cuda-compat
CUDA_VISIBLE_DEVICES=<明确指定的 GPU>
```

不继承实验代码的 `PYTHONPATH` 或 `VLLM_*` 覆盖项，不加载 Residual/LoRA/自定义 native hook。
不在代码准备阶段调用原脚本中会初始化 CUDA 的 GPU preflight。

v026 原来缺少 `datasets`。本次在用户补充“不再安装”前已把少量客户端依赖补到
本目录 `.client_deps/`；仅客户端/预处理加入此路径，服务端不加载它。
`client_deps_install.log` 和 `requirements-client.txt` 保留记录。
**现在无需再运行安装脚本，启动器不会自动安装任何包。**
客户端只使用 CPU tokenizer/HTTP，禁用 transformers 的 torch/TF/Flax 后端。

## 模型与默认参数

- Target：`/data/weijiahong/huggingface/models/Qwen3-8B` 原权重。
- Draft：`/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash` 原权重。
- vLLM 原生 `method=dflash`，7 个 speculative tokens，`draft_sample_method=greedy`。
- 保留 draft 原始 32,000 行 LM head 及 d2t 映射，不转换成我们实验里的全词表 draft。
  Target 仍按自己的完整词表验证/生成，未限制目标模型的分布。
  vLLM 内部为映射生成目标词表宽度 logits 张量是原生实现的一部分。
- BF16、TP=1、DP=1；显存比例 0.93。
- 原脚本默认并发：数学 50、代码 40；客户端并发与服务端 `max-num-seqs` 一致。
- 不加 `--enforce-eager`，保留 vLLM 默认编译/Graph 行为；实际捕获与性能待 GPU 验证。
- 总上下文 8192；生成上限默认 8192，并按每条实际 prompt 长度扣除预算。
  使用服务端 `/tokenize` 渲染同一 prompt，再请求最多 `min(max_tokens, 8192-prompt_tokens)`；
  不截断原 prompt，不落盘 hidden。
- Target 采样默认使用 checkpoint 的 generation_config（本地为 T=0.6、top_p=0.95、top_k=20），
  与原重生成脚本的默认来源一致。`--sampling-params` 可显式覆盖。
  Draft 的 greedy 提案不等于强制 Target T=0。

## 使用

Open-PerfectBlend 已使用共享分片队列自动领取空闲卡，按北京时间白天最多三张、夜间最多六张，见
[自动调度说明](open_perfectblend_regen/README.md#自动选择空闲卡共享分片队列按北京时间调整上限)。
普通运行入口默认只打印计划，追加 `--run --gpu N` 才启动服务器：

```bash
bash /data/weijiahong/speculators-main-lora/regeneration_standalone/openmathinstruct2_regen/run.sh --dry-run
bash /data/weijiahong/speculators-main-lora/regeneration_standalone/opencodeinstruct_regen/run.sh --dry-run
```

后续明确分配 GPU 后，追加数据的命令形式（此处未执行；将 `N` 换成指定卡）：

```bash
bash /data/weijiahong/speculators-main-lora/regeneration_standalone/openmathinstruct2_regen/run.sh \
  --run --gpu N --max-samples 20000
```

`--max-samples` 是包括旧数据的总数，默认 10,000。`--concurrency`、`--port`、
`--gpu-memory-utilization`、`--max-tokens`、`--max-model-len` 均可覆盖。
`--dataset-path` 可指定原始 HF 数据集目录，不能指向重生成后的答案文件。
入口不依赖当前工作目录，也不调用 `conda activate`。

每个数据集只允许一个新入口写入者（flock）；旧脚本仍保留但不要同时向同文件运行。
错误写入原 `.errors.jsonl`，任何本轮失败都会返回非零退出码。
检测到 JSONL 尾部损坏时停止追加，不自动删改已有数据。
终止时仅清理本次启动的客户端和服务端进程组。

## CPU 检查与尚未验证

`test_cpu.py` 使用本机模拟 HTTP endpoint，不加载模型：验证旧数据保留与续跑去重、
异步请求实际重叠、按 prompt 预算限制输出、token IDs/loss mask、错误行隔离和非零退出码、
旧源码哈希/迁移软链接、默认不启动及显式 GPU 参数要求。
结果见 `cpu_validation.log`。

尚未在真实 GPU 上验证原生 DFlash 启动、32k 权重加载、实际 Graph、接受率和速度。
没有启动 GPU 实验、训练或后台等待进程。
