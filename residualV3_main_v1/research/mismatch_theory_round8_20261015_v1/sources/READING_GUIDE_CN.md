# 第八轮资料与问题索引

所有本地路径相对 `/data/weijiahong/speculators-main-lora`。先读已保存版本；网页有更新时另记版本，不替换旧轮来源。

## 1. 研究证据

```text
R1=residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2
R2=residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2
R3=residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/20261014T073622Z_mismatch_r3
R4=residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4
R5=residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5
R6=residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6
R7=residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/20261015T075350Z_mismatch_r7
FORMAL=residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1
FROZEN=residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime
```

|先读|必须回答的问题|
|---|---|
|R7 `results/RESULTS_CN.md`、`METRIC_TABLES_CN.md`、`THEORY_CARDS_CN.md`、`CLAIM_EVIDENCE.json`|为什么支持共同部分有用，但没回答跨主体/来源？6.35%为何是选择覆盖而非解释率？|
|R7 `COMPLETION.json`、`NEXT_CHAT.md`、`results/RUN_AUDIT.json`、`protocol/LOCK.json`|哪些资产和任务已结束，哪些旧警报仍存在？|
|R1 `results/RESULTS_CN.md`|两主体同prefix的总体KL/TV比较已做了什么；为什么不能把它当共同结构比较？|
|R5 `results/RESULTS_CN.md`、`THEORY_CARDS_CN.md`；R6 `results/METHODS_CN.md`|R5 M/2动作怎样构造、什么是fit保证、什么需要留出检验？|
|R6 `results/RESULTS_CN.md`、`MODEL_CARD_CN.md`|为什么不再扩大历史预测器、也不能说历史不可预测？|
|R4 `results/RESULTS_CN.md`、`METHODS_CN.md`|为什么不重做共同/逐历史oracle；合法动作与全词表松弛探针有何区别？|
|R2 `results/NATIVE_PROBABILITY_BOUNDARY_CN.md`；R3 `results/RESULTS_CN.md`|参考q、actual generation q与native警报的区别；统一减半为何不是本轮方向？|
|FORMAL `RESULTS_CN.md`、`COMPARISON.json`、`FINAL_AUDIT.json`|正式整体性能已完成，不能从本轮机制结果重写其归因或重开评测|

## 2. 实现入口

|来源|用途/禁止照搬之处|
|---|---|
|R1 `code/hf_frozen_adapter.py`|`FrozenABReference`两主体、mask embedding、target taps；检查同架构而非同权重|
|R7 `code/hf_frozen_adapter.py`|target/context、body原始计算；旧`target_trajectory`为FP32采样，本轮必须新建FP64入口|
|R7 `code/collect_mechanism.py`|anchor1/65、BF16位、原头实际更新、组身份；旧版只取joint且是target分支|
|R7 `code/repair_replay.py`；R6 `code/read_capture.py`|概率重构与原头读取参考；全量reader不能当fit-only入口|
|R6 `code/mismatch_math.py`|共同u/o及M/2动作；不调用预测器/LP分析|
|R7 `code/prepare_inputs.py`|规范化、面板、暴露排除、tokenizer核验；新轮参数/seed和最新排除须显式替换|
|R7 `code/gpu_budget.py`、`cpu_runner.py`、`resource_identity.py`、`disk_guard.py`|预算/身份/磁盘实现参考；核硬编码ROOT、线程环境和共卡准入|
|FROZEN `repair/model.py`、`runtime/components.py`|F来源原完整头的输入和BF16更新身份；只读、不改安装环境|

模型导出精确资产以R7 `protocol/LOCK.json`及FORMAL `plan.json`核对，重点核：

```text
residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/models/E2_DFLASH/
residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/models/E2_JOINT_NATIVE/
residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/models/E2_JOINT_HEAD/
residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/head_config.json
residualV3-nonthing/results/full_nonthinking_joint_then_dflash_1epoch_2gpu_20261006_v1/basis/initial_pretrained_basis_512.safetensors
/data/weijiahong/huggingface/models/Qwen3-8B/
```

开发题只取R7原fresh36的每域前2个SHA，不重跑R7主量。旧数据是开发材料，旧目标分支不是新FP64来源测度的实测样本。

## 3. 一手资料与本轮作用

以下是前一轮整合已核正文范围的索引，不是全新系统综述。执行时只需核与当前论断相关的原文，保存URL/版本/访问时间及读到的节；网页不可得可用已归档快照并如实说明。

|原文|已核范围/要回答的问题|不能替本项目证明什么|
|---|---|---|
|[Stanton等，Does Knowledge Distillation Really Work?，2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/376c6b9ff3bedbbea56751a84fffc10c-Paper.pdf)|§4.4、§6.1–6.2：可表达与实际拟合为何不同？|经验偏差不等于本地容量下限|
|[Domino v1](https://arxiv.org/html/2605.29707v1)|§4.1–4.2：主体/修正分支训练分工为什么是竞争解释？|不能认定本地联合主体故意留下偏差；不能由两checkpoint识别训练因果|
|[GKD v3，On-Policy Distillation of Language Models](https://arxiv.org/pdf/2306.13649v3)|§3.1、§4：教师历史与学生历史如何不同？|它的训练收益不是当前冻结探针迁移证据，不授权on-policy训练|
|[DistillSpec v2](https://arxiv.org/html/2310.08461v2)|§3–4、A.2：训练测度与接受相关测度为何分开？|不移植有警点的界常数，不把平均TV变成block15 TPS|
|[Understanding Knowledge Distillation in NAT v3](https://arxiv.org/html/1911.02727v3)|§2.1、§3.1–3.2、§4.4：多模态目标和容量/数据复杂度已有哪些认识？|历史改变目标需求不是首次发现；翻译质量不等于接受长度|
|[Leviathan等，2023](https://proceedings.mlr.press/v202/leviathan23a.html)|标准投机拒绝采样、固定状态overlap|不能替本地native概率身份审计背书|
|[DCOCO v1，2026-10-08](https://arxiv.org/html/2610.12570v1)|§3–4及Appendix G：上下文修正已有相近路线；T1为确定性proposal|不能跨论文排名，不把其候选覆盖当共同失配理论|

Domino/DSpark/DFlash-2完整邻近定位继续读R1 `sources/LITERATURE_MATRIX.md`、`SOURCE_BOUNDARIES_CN.md`。本轮不进行完整近邻训练/服务复现。

本轮的新价值若获支持，应来自同一协议下的跨主体/历史来源鉴别；TV、共同流、KL账本、参考拒绝算法及“未找到同名理论”均不单独成为新贡献。
