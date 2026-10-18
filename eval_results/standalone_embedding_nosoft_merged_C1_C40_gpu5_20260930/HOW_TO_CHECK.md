# 查看训练和测试进度

- `status.json`：总状态与当前阶段。`running`/`waiting_gpu`表示进行中，`failed`查error和对应log，`complete`表示全部完成。
- `M_no_soft_2layer_merged/training.log`：唯一新训练日志；训练总384步。
- `M_no_soft_2layer_merged/training/status.json`：训练进度；正式初始化证据在同目录`EMBEDDING/initialization.json`和`fresh_reset_verified.json`。
- `C1/C1.log`：RAW → N_no_soft_2layer → M_no_soft_2layer_merged，20prompt，T0/T1。
- `C40/C40.log`：同三模型，300prompt，40并发，T0/T1。
- `C1/formal/summary.json`与`C40/formal/summary.json`：两份均应complete，各6个正式窗口。smoke不算效果结果。
- `RESULTS.md` / `results.json`：全部完成后自动生成；包含acceptance length、token/s及相对RAW的变化。

旧两层epoch3仅复用于对照，其哈希见`N_no_soft_2layer_reference/reuse_proof.json`。不测试一层模型。GPU5完成训练、释放目标服务后依次测试；GPU4不受本实验调度影响。

在仓库根目录运行：

```bash
/data/weijiahong/miniconda3/envs/dflash/bin/python eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930/check_run.py eval_results/standalone_embedding_nosoft_merged_C1_C40_gpu5_20260930
```

返回码0=全部完成，1=失败，2=仍在运行。完成只代表流程成功，效果是否更好要看同套T0/T1的接受长度与吞吐，不以loss或smoke判断。
