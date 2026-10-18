# 资料索引与阅读问题

路径均相对仓库 `/data/weijiahong/speculators-main-lora`。先读取已保存版本；网页若更新，另记版本和日期，不悄悄替换历史主张。

## 1. 研究轮次

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3
R4=residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4
R5=residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
```

|资料|必须回答的问题|
|---|---|
|R5 `results/RESULTS_CN.md`、`THEORY_CARDS_CN.md`、`METHODS_CN.md`、`CLAIM_EVIDENCE.json`|共同偏差探针与真实原头分别支持什么，条件子集如何形成？|
|R5 `results/MECHANISM_CASES_CN.md`、`MECHANISM_CASES_fresh_test_CN.md`、`FIXED_FLOW_CASES_fresh_test_CN.md`|真实一好一坏是否需求反向，是否也有过量；不能把两者混称|
|R5 `NEXT_CHAT.md`、`COMPLETION.json`、`results/RUN_AUDIT.json`、`RESOURCE_HISTORY_CN.md`|哪些已结束、哪些警报保留，不恢复旧作业|
|R4/R5 `protocol/SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`、`LOCK.json`|旧train/dev是否完整题面隔离，新题从哪里排除暴露|
|R4 `results/RESULTS_CN.md`、`METHODS_CN.md`、`RUN_AUDIT.json`|实际合法见证、概率松弛、同C留出及证书重构边界|
|R1–R3 `results/RESULTS_CN.md`，R2 `results/NATIVE_PROBABILITY_BOUNDARY_CN.md`|哪些假说已被削弱，旧native无偏性警报为何不能撤销|

R5完整数据在 `data/fresh_test/<domain_group>/anchor{1,65}/`，具体以manifest和coverage为准：

```text
TARGET_BITS.npy
STATE_ARRAYS.npz
HEAD_ARRAYS.npz
HEAD_BRANCH_ARRAYS.npz
INPUT_ARRAYS.npz
BRANCH_STATES.json
GROUPS.json
ALL_BRANCH_HISTORY.json
```

R4同类资产保留全部15位置，训练须裁到3/8/15。准备时抽查两轮npz头：hidden4096、qp256、previous embedding4096、fusion_concat768均存在；这不是全数据完整性或数值parity认证。

## 2. 代码：读懂后复用，不原样运行旧入口

- R5 `code/collect_mechanism.py`、`hf_frozen_adapter.py`：C/H生成、实际头输入hook、自然EOS、完整target概率。
- R5 `code/read_capture.py`、`repair_replay.py`、`numeric_gate.py`：BF16位、实际GPU head logits与full-vocab q；检查身份与重构误差。
- R5 `code/mismatch_math.py`、`analyze_mismatch.py`、`summarize_r5.py`：共同流M/2、分组资格、自然权重、零值和覆盖。R6不重开R5全部分析。
- R5 `code/prepare_inputs.py`：面板路径、规范化、曝光排除、采样；新R6另建seed和清单。
- R4 `code/tv_oracles.py`：S/U完整尾部、R_ind/R_shared/R_safe、数值界。只复用有限诊断；`basis_witness.py`不在本轮运行范围。
- R5 `code/gpu_budget.py`、`cpu_runner.py`、`resource_identity.py`、`disk_guard.py`：共卡、身份、资源；核对硬编码RUN和线程启动顺序。
- FROZEN `repair/model.py`、`runtime/components.py`：三路真实输入、basis区分、完整词表输出和冻结头身份。

训练配方只为确认模型身份按需读原E1/E2 protocol，不重新训练。正式六数据集48窗已完成，不重复评测。

## 3. 定向文献问题

本轮新增阅读的核心是预测实验如何避免过度解释。编写时核对了以下ACL Anthology书目与摘要页面；执行时如引用正文细节，须再读相应原文段落并记录页码。

|原文|为什么读|不能借它声称什么|
|---|---|---|
|[Hewitt & Liang, 2019, Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)|探针表现受到模型容量和记忆影响，需要合适控制|本轮同Z历史置乱不是对其word-type控制任务的直接复现，也不是通用信息充分性证明|
|[Voita & Titov, 2020, Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/)|探针单一准确率不能完整表达可提取信息|本轮不实施MDL，不用其理论替我们的Brier结论背书|
|[Banerjee等, 2005, Clustering with Bregman Divergences](https://jmlr.org/papers/v6/banerjee05b.html)|R5沿用的均值分组工具与标准分解|标准KL/均值恒等式不是新贡献，不是TV可加分解|
|[Leviathan等, 2023, Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)|区分理想拒绝采样局部重叠与实际runtime结论|不能据此消除本地旧随机源警报，也不能把本轮HF预测直接写成接受/TPS收益|

近邻沿R1 `sources/LITERATURE_MATRIX.md`、`ANNOTATIONS.json`、`SOURCE_BOUNDARIES_CN.md`定向核对：

- [Domino v1](https://arxiv.org/html/2605.29707v1)：历史条件缺失、GRU累计信息、训练控制；是否已讨论本轮拟用的解释。
- [DSpark v1](https://arxiv.org/html/2607.05147v1)：默认Markov与RNN变体、confidence和调度分别做什么；本轮离线损伤预测不等于已有或新部署调度头。
- [DFlash-2作者报告](https://inco.ai/blog/dflash2/)与[模型卡](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2)：候选路径选择与主体卷积的分工；不能把本轮诊断与其整套吞吐直接排名。
- R1其他已保存来源仅在确有对应主张时追溯；不要宣称全面检索或“没有同名方法所以创新”。

新增文献记录至少写版本、读到的段落、对应研究问题及不能支持的推论。文献用于约束解释，不用于给尚未出现的R6效果补证。
