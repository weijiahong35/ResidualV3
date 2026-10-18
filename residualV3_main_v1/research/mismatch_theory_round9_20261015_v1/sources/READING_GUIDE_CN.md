# 资料阅读与主张边界

按下面顺序读，与核心问题无关的模块计时、TPS及RNG细节本轮不扩展。旧警报保留在最终限制，不重开工程支路。

## 1. 用户方向与已撤回方案

- 仓库根`AGENTS.md`：核心问题与每份方案必须审核的约定。
- `../../mismatch_core_question_20261015_v1/ANALYSIS_SPEC_CN.md`：已撤回的“相近TV下预测原头成败”方案。没有执行；不可作为本轮协议。

## 2. 本地直接证据

下列路径以仓库根为前缀：

|材料|必须核对什么|
|---|---|
|`residualV3_main_v1/research/mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/results/RESULTS_CN.md`|条件KL/拟合KL的定义与负结果；不能把均值拟合差叫纯容量|
|`residualV3_main_v1/research/mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4/results/THEORY_CARDS_CN.md`及`RESULTS_CN.md`|已存在R_ind/R_shared/R_safe与153/163见证；当前头失败不等于共同不可修|
|同R4 `code/tv_oracles.py`|完整尾部评估和Lagrangian证书方法；其动作空间保留raw尾形状，不能直接代替full-simplex；旧默认线程2必须改|
|`residualV3_main_v1/research/mismatch_theory_round6_20261015_v1/runs/20261015T050613Z_mismatch_r6/results/RESULTS_CN.md`|风险/方向预测未支持，不重新加特征救它；11/12松弛共同候选不等于可达|
|`residualV3_main_v1/research/mismatch_theory_round7_20261015_v1/runs/20261015T075350Z_mismatch_r7/results/RESULTS_CN.md`|原头共同部分的有限联系，不能替本轮信息边界|
|`residualV3_main_v1/research/mismatch_theory_round8_20261015_v1/runs/20261015T110214Z_mismatch_r8/{COMPLETION.json,NEXT_CHAT.md,results/RESULTS_CN.md}`|终态、全部数据已暴露、来源测度、证书/采样边界|
|同R8 `code/common.py`、`code/collect_r8.py`、`data/fresh_test/*/C*/META.json`|BF16位→FP64概率、H/target_row/slot映射、P/D/F身份及EOS；完整概率仅保存指定位置|

全读八轮所有文件不是准入要求。核对需要使用的输入、历史反证与数学身份即可；不要用文献/审计无限扩展代替核心工作。

## 3. 一手文献：哪些认识已有

此前定向选读记录保存在`../../mismatch_core_question_20261015_v1/sources/LITERATURE_CHECK_CN.md`。其原任务表述已撤回，来源定位可复用，但不能继承“原头预测”为研究目标。

|来源|核对位置与用途|不可据此宣称|
|---|---|---|
|[Domino v1](https://arxiv.org/html/2605.29707v1)|§4.2主体与修正分工；条件依赖与课程|本项目共同偏差必由容量不足造成|
|[DSpark v1](https://arxiv.org/html/2607.05147v1)|§3.1/3.3条件依赖、Markov/RNN和TV训练|首次发现前驱或历史修复价值|
|[DFlash-2作者报告](https://inco.ai/blog/dflash2/)|候选Recall与路径选择|full-simplex边界等于该方法实际可达|
|[DistillSpec v2](https://arxiv.org/html/2310.08461v2)|模型生成历史与散度选择|换历史测度能自动解决本地冻结修正问题|
|[GKD v3](https://arxiv.org/pdf/2306.13649v3)|§3 on-policy历史与受限模型类|固定D/J对照识别模型规模因果|
|[García-García & Williamson, COLT 2012](https://proceedings.mlr.press/v23/garcia12.html)及[原文](https://proceedings.mlr.press/v23/garcia12/garcia12.pdf)|§2实验比较、§3–4风险与多分布差异的关系；信息结构和最优风险是已有决策论工具|本方案的单调性/分解是新理论，或该论文已证明本项目的修复边界实证|

2026-10-15已实时核对COLT官方条目并读取PDF相关定义；其他五项有本会话之前的一手定向核查。没有完成穷尽新颖性/引用审计，不能写“首次”。Blackwell原始馆藏页面本次403，不假称已经读到原文；使用可核的COLT原文讲已有工具。

## 4. 理论推导必须自行核验

本方案中的两历史M闭式、平均TV最优共同分位阶、A/M/S关系和tail松弛界由标准三角不等式与凸优化直接推导，属于待实现的数学合同。执行chat须独立给证明和小词表LP校验，不将本文件的公式当实验真值。

不能把数学必然、数值证书、有限模型数据结果、未见泛化和可部署性能合成一条“已证实”主张。最终claim–evidence按这五种身份分开。
