# 文档最终复审

状态：可交付执行文档；没有进行模型采集或统计检验。研究假说仍待验证。

机制/统计审查：9.13/10，无P0/P1；可行性审查：8.69/10，无P0，两项文字/路径P1已修。评分为角色审查意见，不是独立统计证据或论文录用概率。

最终修订：

- new_to_P_fit主检验与全部matched三桶/strict_allfit_novel副表一致，避免seen构造保证代替泛化。
- context随机元组固定anchor=0，pool枚举包含context，单条65步轨迹共用两anchor；不同逻辑元组才查碰撞，缓存重用不重复采样。
- 资料表full reader改为实际存在的R6 `code/read_capture.py`。
- 1e−5符号带只分整状态G，逐token损伤和方向判断不丢小量。
- A/B分开判定、六格全报；W标准化、P-fit-only访问、actual-q/FP64及reference-reached限制已复核。

未消除的执行不确定性：dev需测真实双主体显存、共卡速度、CPU采样往返耗时、W/新历史覆盖与采样buffer绑定。固定工作全部完成但覆盖不足仍是complete且科学未确认；预算/工程使必要工作缺失则partial。

researchwrite用于组织事实、论证和交接检查；本计划由AI辅助形成，不将建议写成已完成实验或新理论证明。
