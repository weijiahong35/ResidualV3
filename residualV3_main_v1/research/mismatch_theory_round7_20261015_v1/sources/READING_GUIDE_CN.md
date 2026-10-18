# 第七轮资料索引

所有路径相对仓库 `/data/weijiahong/speculators-main-lora`。先读保存版本；网页若变更另记版本，不混写。

## 1. 最新证据与待回答的问题

```text
R6=residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6
R5=residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5
R4=residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
```

|必读资料|要弄清楚什么|
|---|---|
|R6 `results/RESULTS_CN.md`、`METHODS_CN.md`、`THEORY_CARDS_CN.md`、`CLAIM_EVIDENCE.json`|共同方向、预测能力、松弛空间分别支持什么；为什么不能替原头机制归因|
|R6 `results/FIXED_CASE_INTERPRETATION_CN.md`、`MECHANISM_CASES_CN.md`|同Z下需求反向、幅度差异和跨token补偿怎样共同影响净TV|
|R6 `results/MODEL_CARD_CN.md`|固定预测器泛化不足与未证收敛边界；本轮不救跑它们|
|R6 `COMPLETION.json`、`NEXT_CHAT.md`、`results/RUN_AUDIT.json`|旧任务已结束、必要工作无剩余、旧警报不撤销|
|R6 `protocol/LOCK.json`、`SAMPLE_MANIFEST.json`、`EXPOSURE_LEDGER.json`|输入与模型身份、旧题全部已暴露、面板及排除规则|
|R5 `THEORY_PLAN_CN.md`（其研究目录）、R5 run `results/RESULTS_CN.md`|u/o容量、M_common/2构造保证与有限迁移，不把旧探针等同本轮c|
|R4/R1–R3 `results/RESULTS_CN.md`、R2 `results/NATIVE_PROBABILITY_BOUNDARY_CN.md`|合法见证与松弛边界、旧未支持假说和runtime概率警报|

## 2. 数据与入口

R6 `data/fresh_test/<domain_prompt>/anchor{1,65}/` 的实际完整性按每context COMPLETION核验。核心资产：

```text
STATE_ARRAYS.npz       # raw/hidden；prepared字段另有位置索引
HEAD_ARRAYS.npz        # 每精确Z组实际更新后的Top256 logits
HEAD_BRANCH_ARRAYS.npz # 逐分支输出与组输出对齐
INPUT_ARRAYS.npz       # 完整原头输入
TARGET_BITS.npy       # row_id对应BF16 uint16 logits位
BRANCH_STATES.json    # 位置、pool、H、row_id；同时含效果字段！
GROUPS.json            # group_id/row_ids/完整输入签名和bitwise身份
COMPLETION.json
```

R6 `code/read_capture.py` 是概率与身份的参考，不可直接作新题fit-only入口。其 `load_context` 校验完整target行，`position_data`读取全部fit/eval并比较D_raw/D_head；必须拆开新入口，保留身份核验而限制数值访问。

`code/mismatch_math.py` 的common_direction与TV可复用；原action为M_common/2，R7需单独实现c交集公式。`code/novel_stability.py` 的完整H与Z匹配可参考，移除其预测器封存依赖，不为复用代码重跑预测。

采集沿 `code/collect_mechanism.py`、`hf_frozen_adapter.py`、`repair_replay.py`、`inference_metadata.py`；资源沿 `gpu_budget.py`、`cpu_runner.py`、`resource_identity.py`、`disk_guard.py`。先检查硬编码ROOT、身份和线程启动顺序，复制到新run修改入口，旧源码不动。

面板、规范化、全机制暴露排除沿 `code/prepare_inputs.py`；R7固定12/域，不继承旧自动min(16,available)。实际余量核验见本目录qa/INPUT_AUDIT；执行前重新核，不能把旧空闲GPU快照或余量当实时事实。

FROZEN `repair/model.py`、`runtime/components.py`仅用于确认三路输入、实际头动作与basis身份。本轮不改它们；FORMAL六数据集已完成，不重开评测。

## 3. 原文与解释边界

本次准备核对以下一手书目/摘要页面；不声称重新阅读全文，也不开展新颖性全面检索：

- [Banerjee等，2005](https://jmlr.org/papers/v6/banerjee05b.html)：包含relative entropy的Bregman分组工具。R5/R6条件均值账本是标准工具；本轮TV动作诊断不能借用KL可加性。
- [Leviathan等，2023](https://proceedings.mlr.press/v202/leviathan23a.html)：标准投机解码保持目标输出分布的理论与算法。这里的HF固定状态TV诊断不建立本地native无偏性或连续接受收益。
- [Hewitt与Liang，2019](https://aclanthology.org/D19-1275/)：控制任务用于约束探针容量/记忆的解释。它提醒区分观测与解释，不为本轮同质量概率移除提供因果识别定理。

Domino、DSpark、DFlash-2等近邻沿R1 `sources/LITERATURE_MATRIX.md`、`SOURCE_BOUNDARIES_CN.md` 和R6 `sources/R6_READING_CN.md` 查已存版本。仅在论文主张确实涉及时再核原文段落，不把本轮描述性账本宣称成近邻从未讨论的理论。

本轮c合法性、fit Gc=m、同质量移除和I≤0的推导来自给定定义的代数与绝对值几何，见理论计划及CPU合同。它们是测量基础，不是发现新定理。文献不能替尚未测得的η/κ补证。
