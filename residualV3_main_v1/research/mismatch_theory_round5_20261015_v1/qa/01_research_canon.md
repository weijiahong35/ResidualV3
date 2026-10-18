# 事实与约束

R4=`../mismatch_theory_round4_20261014_v1/runs/20261014T131428Z_mismatch_r4`（从本轮根相对定位）。

- R4 RESULTS_CN.md：153/163异号组存在实际操作内逐分支改善 raw 的共同动作；并非保住当前头全部收益。
- R4 fit→eval 主量相对当前头 TV 改善0.319211，限匹配子集、同C、target标签求动作；损伤约1.918%。没有训练新模型或证明跨C可学习。
- R4 THEORY_CARDS_CN.md：候选外条件比例不变；共享代价有限组定义；当前头未实现部分不等于纯训练/容量误差。
- R4 H2控制后覆盖不足，增量不确定。R1条件KL主导未证，R2几何增量未支持，R3统一减半不优。
- R4 RUN_AUDIT.json：科学完成，但 global_clean_resources_passed=false；线程历史、一次证书事后重构、旧概率警报均保留。
- 理论工具：KL均值分解是标准Bregman/条件期望工具，不是本项目创新。Banerjee et al. (2005), https://jmlr.org/papers/v6/banerjee05b.html ，Section 2–3。
- 标准投机验证/输出分布保证及单状态关系见 Leviathan et al. (2023), https://proceedings.mlr.press/v202/leviathan23a.html 。不将理论算法条件自动赋予本地旧警报runtime。
- 并行草稿缺少块内条件信息已由 Domino v1 讨论：https://arxiv.org/html/2605.29707v1 ，Sections 3–4；不能宣称首次。
- 用户要求：理论先于方法；有容纳阶段工作集的卡就共卡使用；700 GiB上限不回退为旧60 GiB。
- 未知：共同误差与历史变化谁在实际修复中发挥何种作用；有限fit里的共同方向是否能在独立分支维持；本轮不声称容量/训练原因识别。
