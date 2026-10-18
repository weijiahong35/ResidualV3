# 查看本次两个版本的进度与结果

运行只读检查：

```bash
cd /data/weijiahong/speculators-main-lora
python eval_results/standalone_d1_dual_feedback_ab_gpu45_v1_20260930_attempt3/check_run.py
```

可在前面加 `watch -n 15`。退出码0全部完成，2运行中，1失败，3主动停止。

- 总阶段：pipeline_status.json，log字段给出当前日志。GPU4在线target，GPU5训练两套独立头；后续所有C1都在GPU5。
- 正式训练：training.log，training/LMHEAD/status.json 与 training/EMBEDDING/status.json；各384步为训练完成。同一批状态各自更新，无hidden缓存积累。
- 输入验收：smoke/LMHEAD/real_input_effects.json、real_input_gradients.json，EMBEDDING同名文件。四路输入必须有效；干预固定原logits、hard前驱、候选ID与输出基底。
- 真随机初始化：两版training/*/initialization.json参数hash相同；training/*/fresh_reset_verified.json证明独立seed重建，与smoke初始和训练后权重不同。
- 来源一致：feature_table_parity.json，比较native与训练的原LM-head/原embedding及投影行hash，防止误用DFlash mask修改表。
- 最终完成：pipeline_status.json为complete，native/C1/summary.json有12个窗口，results.json和RESULTS.md出现，cleanup.json清理通过。

先分别看T0/T1的接受长度与吞吐。相对RAW的吞吐比例>1才表示观察到提速；embedding与LM-head的直接比较见results.json。不能仅靠loss下降或smoke通过判效果。

旧三输入、原D1、objective audit、dense/K256等均保留。本次修复头从零训练，使用原训练128条与原测试20条。C1已有开发使用历史，两轮重复不代表独立统计样本。只跑C1，不跑C30。
