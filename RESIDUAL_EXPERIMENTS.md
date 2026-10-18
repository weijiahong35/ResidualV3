# 当前实验入口（2026-10-09）

请统一查阅 [Thinking 实验总账](EXPERIMENTS_THINKING.md) 和 [Non-thinking 实验总账](EXPERIMENTS_NONTHINKING.md)。两份总账按实验记录目的、对照、设置、正式 AL/TPS、诊断和原始证据，并自动更新。下面保留的是早期历史导航，状态以新总账及原始运行文件为准。

---

> 最新入口（2026-09-24）：[新窗口交接与高质量实验导航](HANDOFF_RESIDUALV3_NEXT.md)。新实验进度以 [pipeline_status.json](eval_results/standalone_k256_next_v1_20260924/pipeline_status.json) 为准；下方包含整理时期的历史状态。

# ResidualV2/V3 实验导航（2026-09-24 整理）

## 首要保留：ResidualV2 主实验

**ResidualV2 是重要主实验，必须独立保留，不属于可清理的旧依赖。**

- 源码：[residualV2/](residualV2/README_residualV2.md)，包含原 Stage1/2/3 训练、repair、basis 和 native 评测代码。
- 主训练数据、basis、checkpoint 和日志：[outputresidualV2](output/openmath/onpolicy/outputresidualV2/)。Stage1、Stage2 的 latest/final 权重和配置已确认存在；本次清理从未作用于此目录。
- 主源码此前被错误归档，现已恢复原路径，42 个文件逐项核验与归档前一致；未修改模型源码。
- 恢复记录：[residualV2_restoration.json](maintenance/residual_source_refactor_20260924/residualV2_restoration.json)。
- 必须区分：此前确实删除了 preexp5、preexp6（含不同输出版本）、preexp30、preexp31 的部分中间权重/缓存；这些不是主训练目录，且本次没有恢复这些已删除文件。报告与预实验源码仍在归档。完整影响清单：[residualV2_impact_audit.json](maintenance/residual_source_refactor_20260924/residualV2_impact_audit.json)。

后续清理不得移动或删除 `residualV2/`、`output/openmath/onpolicy/outputresidualV2/` 及其主实验结果。只有用户明确指定具体产物后才改变这项保留约定。此约定不改变两个 ResidualV3 目录各自独立的实现。

主线为 **full-vocab LM-head repair 的接受收益与实际 runtime，以及面向 DFlash2/DSpark 的公平比较**。32k raw/no-adapter 是保留的历史基线与迁移诊断，不要求 full 超过 32k。

统一优化主线见 [研究方向与约束](RESIDUAL_RESEARCH_DIRECTION.md)：全词表背景＋关键候选局部 mismatch 修复；小词表启示与已确认的 greedy Top-K 前提合并推进。

## 当前 32k：只保留 raw / no-adapter

按用户明确指令，已删除 adapter 模型/训练分支、对应权重和 4 个旧 adapter 源码目录。no-adapter checkpoint 剥离未使用 adapter 张量，有效 repair 权重逐位保留；ResidualV2、full LM-head 及数据/面板不动。详细清单见 [32k adapter 移除记录](maintenance/32k_adapter_removal_20260924/README.md)。旧报告中的 adapter 分数仅是历史记录。

当前两套 ResidualV3 已采用新的 BF16 数值设置，GPU acceptance/runtime 尚未重测；下述已完成复现结果对应修改前代码。

## 当前运行：独立目录复现 + 吞吐

2026-09-24 已在 GPU 3/5 完成独立目录的接受收益复现：32k 三臂、全词表两臂的 T0/T1 结果及 native 计数与历史完全一致。当前按用户要求在 GPU 2 只补测 32k raw/no-adapter 的吞吐，在 GPU 4 补测 full raw/LM-head C 的吞吐；本轮为共卡 runtime，输出在 runtime_retry_v2/，旧错误日志保留。只评测，不训练。当前状态、日志与最终报告入口统一放在 [本轮运行目录](eval_results/standalone_reproduction_runtime_20260924/README.md)。是否完成以其中 status.json 为准。

下面关于“未启动 GPU / 未补测吞吐”的描述是此前清理和源码整理阶段的记录，不代表本轮复现状态。

## 优先查看

| 保留实验 | 内容与地位 | 结果入口 |
|---|---|---|
| 最新 full-vocab LM-head C | 完整保留训练、epoch16、初始化、64 条面板的采样对齐三臂、requests/blocks、契约和日志 | [最新 RESULTS](eval_results/residualV3_full_vocab_lmhead_retrain_v1_seed42/native_sampling_aligned_v1/RESULTS.md) |
| 最新 32k matched | 保留 no-adapter epoch24、raw、256 条面板与历史记录；adapter 专用权重/代码已退役，no-adapter 冗余张量已剥离 | [最新 RESULTS](eval_results/residualV3_p1_adapter_matched_t01_v1_seed42/native_sampling_aligned_v1/RESULTS.md) |
| 最新汇总与执行记录 | 完整保留采样对齐、GPU 续跑记录、源文件哈希和转发文档 | [给 Chat 的完整 prompt](eval_results/residualV3_sampling_alignment_v1_seed42/LATEST_MODEL_EXPERIMENT_PROMPT_FOR_CHAT_20260924.md) · [实验汇总](eval_results/residualV3_sampling_alignment_v1_seed42/EXPERIMENT_SUMMARY_FOR_CHAT_20260924.md) |

这些是已打开的诊断面板。最新有效评测看 **native_sampling_aligned_v1**，不要把父目录旧 status 或旧 native_t01 当成最新采样协议。此轮未补测吞吐、未启动四臂、未训练模型。

## 保留的辅助证据

- [小词表 baseline validation](eval_results/residualV3_p1_baseline_validation_v1_seed42/)：保留验证数据与报告，是 matched 流程的输入之一。
- [小词表 native gate](eval_results/residualV3_p1_native_gate_v1_seed42/)：保留历史 native 验证。其历史采样设置需按原记录解释，不能替代最新对齐评测。
- [ResidualV2 来源审计](eval_results/residualV2_stage2_source_audit_20260908/)：保留来源审计证据。

baseline validation 的 stage2_snapshots 原本是指向 `/tmp/residualV3_p1_baseline_validation_v1_seed42_snapshots` 的软链接；本轮没有迁移它，1000 个目标分片仍存在。最新实验的数据依赖记录见下方。

## 为什么还有旧名字的目录

下面是**依赖保留区**，不再视为活跃预实验。为了保留固定路径、初始化来源和冻结哈希，没有改名或重写共享实现。

| 目录 | 保留理由 |
|---|---|
| [residualV3_acceptance_diagnostic_v1_seed42](eval_results/residualV3_acceptance_diagnostic_v1_seed42/) | 只保留被最新实验继承的同输入 body/head 数值契约；旧 input C 四臂不是当前结果。 |
| [residualV3_full_vocab_joint_v4_seed42](eval_results/residualV3_full_vocab_joint_v4_seed42/) | 仅保留初始化、C epoch16 历史对照权重及原协议，供 LM-head 重训来源核验和最新三臂中的 input C 对照使用；J 和旧运行产物已退出。 |
| [residualV3_full_vocab_unified_lag1_v1_seed42](eval_results/residualV3_full_vocab_unified_lag1_v1_seed42/) | 仅保留 repair basis、协议引用的投影文件及源码软链接所需历史元数据。 |
| [residualV3_full_vocab_unified_lag1_v2_seed42](eval_results/residualV3_full_vocab_unified_lag1_v2_seed42/) | 仅保留被后续协议引用的原协议；废弃 compact cache 已删除。 |
| [residualV3_full_vocab_unified_lag1_v3_seed42](eval_results/residualV3_full_vocab_unified_lag1_v3_seed42/) | 保留最新 LM-head 使用的 2432 个 compact 分片（2368 train + 64 final）、manifest、协议及固定 prompts；旧 input-embedding 模型和 resume 已删除。 |
| [residualV3_lmhead_feedback_diagnostic_v1_seed42](eval_results/residualV3_lmhead_feedback_diagnostic_v1_seed42/) | 仅保留 body_contract_reference.json；推理换表旧实验已退役，不是 LM-head 完整重训。 |
| [residualV3_pareto_safe_rank_v1_seed42](eval_results/residualV3_pareto_safe_rank_v1_seed42/) | 仅保留 capped_rank_projected/step_000528 父 checkpoint 及配置。 |
| [residualV3_predecessor_adapter_v1_seed42](eval_results/residualV3_predecessor_adapter_v1_seed42/) | 保留 32k matched 所需 3168 个训练/来源快照及 manifest、protocol、basis 及历史配置；adapter epoch24 权重已按新指令删除，数据依赖保留。 |
| [residualV3_predecessor_external_openmath_0_2000_v1](eval_results/residualV3_predecessor_external_openmath_0_2000_v1/) | 仅保留训练来源协议。 |

## ResidualV3 当前源码：两个独立目录

| 实验 | 唯一源码入口 | 主要文件 |
|---|---|---|
| 最新全词表 LM-head | [residualV3_lmhead/README.md](residualV3_lmhead/README.md) | model.py、repair_head.py、train.py、evaluate.py、analyze.py、paths.py |
| 最新小词表 matched | [residualV3_32k/README.md](residualV3_32k/README.md) | model.py、repair_head.py、train.py、evaluate.py、analyze.py、paths.py |

两个目录各自包含模型、训练、loss、native 采样及统计，不互相 import，不再运行时加载旧预实验目录。允许少量代码重复，让每个实验都能独立阅读和运行。数据/权重路径集中在各自 `paths.py`；历史结果位置不变。

此前将 **37 个**源码目录移入 [历史源码归档](_archive/residual_preexperiments_20260924/source/)，其中 `residualV2/` 被误归档，现已恢复。连同更早的 9 个，本轮又删除 4 个 adapter 专用源码目录，归档现保留 **41 个**目录。根目录保留 `residualV2/`、`residualV3_lmhead/`、`residualV3_32k/` 三套源码。原 `residual/` 是更早的 Residual/MoLE 实现，本次公共包尝试已撤回，未将它作为新方案依赖。

除明确删除的 adapter 专用源码外，其他旧源码字节与冻结哈希保留，原记录中的旧源码路径可按 `archive_source_plan.json` 映射到归档位置；历史文档不重写成新源码的实验结果。归档不保证直接运行旧入口，因为旧结果大文件已在上一轮清理。

验证：16 个模型/采样模块的函数 AST 对照、12 组非零修正/反馈/梯度 CPU 对照通过；3 个实际最终 checkpoint 严格加载通过；新入口重算两套 `summary.json`、`mechanisms.json` 与原报告一致。native 模块在 vLLM 环境导入成功，未初始化 CUDA。没有新增训练或 GPU 实验。

详见 [源码迁移记录](maintenance/residual_source_refactor_20260924/archive_source_completed.json)、[数值/权重检查](maintenance/residual_source_refactor_20260924/equivalence.json)、[报告重算对照](maintenance/residual_source_refactor_20260924/report_equivalence.json)。GPU smoke/runtime 未在此轮复测。

## 清理结果与追溯

- `eval_results` 中 ResidualV2/V3 顶层目录 **48 → 15**：3 个主要实验/汇总目录、3 个辅助证据目录、9 个依赖目录。
- 删除 **14,940 个**废弃中间权重、缓存及旧全词表诊断 traces；删除列表的逻辑大小约 231.16 GiB，不能当作释放空间。
- 将旧报告、协议、日志等移入 [历史结果归档](_archive/residual_preexperiments_20260924/results/)。归档仅用于追溯，已删权重/缓存的旧实验不保证可直接重跑。
- 将归档结果也计入后，目录实际占用减少约 **116.30 GiB**（`du` 分配块统计，硬链接去重；不是整个共享磁盘空闲量的测量）。
- 完整核验 **7,417 个保护项**；其中 **1,806 个文件 SHA-256 与清理前一致**。训练/验证 manifest 指向的 **6,600 个分片仍可访问**。大数据分片本轮检查 inode/size/mtime 与路径可达性，没有宣称重新计算全部分片哈希。
- 保留源码的 10 个状态/报告软链接目标在归档后核验中补充恢复，详情有独立记录。
- 未改变最新结果、模型数值或采样逻辑；未启动 GPU 工作，也未停止任何实验。外部目标模型、原始 `output/openmath` 数据未清理。

详细记录：[完成核验](maintenance/residual_cleanup_20260924/completed.json) · [逐文件清单](maintenance/residual_cleanup_20260924/plan.json) · [实际执行日志](maintenance/residual_cleanup_20260924/actions.jsonl) · [软链接目标恢复](maintenance/residual_cleanup_20260924/restored_symlink_targets.json) · [空间统计](maintenance/residual_cleanup_20260924/disk_after.json)。`plan.json` 是执行前清单；归档后恢复的 10 个文件以恢复清单为准。
