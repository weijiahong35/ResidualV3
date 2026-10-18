# 后续短测入口

当前实现迭代集中记录在 [短测总结果](../epoch2_tail_cut_short_iter_20261011_v3/RESULTS_CN.md)，并同步到仓库根目录的 `EXPERIMENTS_NONTHINKING.md`。

- [C1 即时短测日志](../epoch2_tail_cut_short_iter_20261011_v3/single_card_C1_short.log)
- [C40 即时短测日志](../epoch2_tail_cut_short_iter_20261011_v3/single_card_C40_short.log)
- [实时状态](../epoch2_tail_cut_short_iter_20261011_v3/status.json)
- [数值核验与算子计时](../epoch2_tail_cut_short_iter_20261011_v3/probes/)

短测设置：GSM8K nonthinking、block16、BF16、C1 16题／C40 80题、每题最多256个输出token。对照为同一epoch2联合权重的完整15位置修复；独立训练DFlash的正式比较保留在epoch1／epoch2测评总账。权重、训练损失和截断阈值保持相同，只优化推理实现。

磁盘按整个用户目录600GB限制检查。历史代码存档去重已腾出60.58GB，epoch1／epoch2导出权重、完整断点和消融权重保留。
