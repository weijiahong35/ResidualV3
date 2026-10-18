> 最新指令：用户已取消等待和后续自动续跑。恢复进程846461已停止，状态stopped_by_user。只分析第一轮，见 [ROUND1_ANALYSIS.md](eval_results/standalone_k256_next_v1_20260924/ROUND1_ANALYSIS.md)。未经新指令不要重新启动。

> 2026-09-26更新：两组训练和第一轮五臂已完成；第二轮因GPU0出现其他进程被隔离guard中止。已添加[同卡续跑任务](eval_results/standalone_k256_next_v1_20260924/recovery_20260926/README.md)，最多等待GPU0空闲6小时，不重训。最新状态以pipeline_status.json和recovery_20260926/status.json为准。

# 新窗口接手：ResidualV3 K256 + greedy（2026-09-24）

仓库：`/data/weijiahong/speculators-main-lora`。请先读取本文件，再读最新 `pipeline_status.json` 和日志，**不要沿用本文件撰写时的进度，不要重复启动进程**。

## 1. 这次已经编写并启动什么

- 独立代码：[residualV3_k256_next_v1/README.md](residualV3_k256_next_v1/README.md)。源码、父权重、basis、body、验证状态均在该目录内；没有依赖兄弟实验代码的导入或链接。
- 输出：[standalone_k256_next_v1_20260924](eval_results/standalone_k256_next_v1_20260924/)。
- 实时状态：[pipeline_status.json](eval_results/standalone_k256_next_v1_20260924/pipeline_status.json)。
- 总日志：[pipeline.log](eval_results/standalone_k256_next_v1_20260924/pipeline.log)。
- 启动与GPU：[launch.json](eval_results/standalone_k256_next_v1_20260924/launch.json)。首次GPU0，主PID4191815；先验证PID命令和状态再操作，PID可能已经退出。
- 冻结协议：[preregistration.json](eval_results/standalone_k256_next_v1_20260924/preregistration.json)。运行中不得修改冻结源码或覆盖完成产物。

用户已授权实现并执行两条线：greedy接受收益与K256执行效率。不是只做规划，也不是新一轮大规模架构搜索。

### A. greedy固定前缀小规模训练

1. 从原2368组训练协议按既定顺序取32条不同prompt，排除本次64条评测prompt文本。已复制到新目录 `assets/train_prompts.jsonl`，来源记录 `training_selection.json`。
2. 用父K256模型T0真实生成，每请求保存前16个block；不根据成功失败选块。记录实际proposal、anchor、raw/hidden、父返回logits、原生验证器在对应真实前缀上的完整target processed logits。
3. `control` 与 `greedy` 同一初始权重、同数据、同顺序、同更新预算：4 epochs，4 blocks/update，seed20042；Stage1 LR1e-4、Confidence/Dose LR5e-4。BF16参数和AdamW矩，无FP32 master。冻结target/body/basis。
4. 共同损失是T1 overlap累计前缀surrogate加0.05 KL(parent||new)。greedy额外加0.1 Top16竞争hinge及0.2正确token保护，margin0.5。只在实际可到达位置加入greedy监督；target在raw Top16外时不强塞候选、不做该rank项。正确位置保护检查完整词表竞争者。
5. 训练反馈强制使用采集时实际token历史，保证teacher与prefix对应；新模型替代token不传播。**这是固定前缀pilot，不是free-recursive on-policy训练；只有T0训练轨迹，不能宣称消除T1历史差异。** 共同损失不再含旧direction/dose oracle辅助项，两个同预算新臂用于隔离新增greedy损失，父模型为零更新参考。
6. 保存固定最终epoch，不选最佳checkpoint、不扫LR/K/rank/seed。所有训练使用K256局部logit修正，保持全词表生成。

代码：`collect.py`、`collect_native.py`采集；`train.py`训练/损失；`test_objective.py`；`test_graph.py`。

完成后日志/结果：`collect.log`、`training/{control,greedy}/{status.json,training_summary.json,final.safetensors}`、`train_control.log`、`train_greedy.log`。

原生评测五臂：**raw_graph、dense父模型、K256父模型、K256 control、K256 greedy**。同一64条开放诊断面板，T0/T1各64请求，max128，并发1，两轮正反顺序。输出 `training_eval/{RESULTS.md,summary.json,RUNTIME_RESULTS.md}`；`eval_smoke.log`、`eval_formal.log`。

### B. 同权重执行融合

- `optimized.py:learned_step`保持原Stage1/Confidence/Dose运算与权重，尝试torch.compile融合小算子；实际lag-1、七步采样与反馈保持。dense和K256同时处理。
- 外层repair CUDA Graph不变；target/body/raw LM projection依然eager。不要把它说成整个vLLM都Graph化。
- `validate_fusion.py`先用6份真实状态、每臂12次动态重放，检验编译相对原算术的BF16 logits、q、采样ID逐位一致及编译Graph/eager一致。设置emulate_precision_casts与force_same_precision仍不等于自动通过检查。
- **数值门禁失败则保存证据、跳过该分支native计时**；不静默回退，不把新数值模型误报成同模型提速。训练分支不因此失效。
- 通过才运行五臂：raw_graph、dense_combined、local_combined、dense_compiled、local_compiled。同权重同面板同卡比较，`runtime_fusion/`保存结果。

流水线顺序：CPU测试 → 父Graph验证 → 采集 → 两组训练及权重Graph验证 → 训练五臂smoke/formal → 汇总 → 融合门禁 → 通过才跑融合五臂。

## 2. 最值得参考的已完成实验

### 第一优先：同卡Graph公平比较（当前速度参考）

- 代码：[residualV3_k256_followup_v1](residualV3_k256_followup_v1/README.md)。GPU0当轮实际三臂驱动在输出 `dispatch_graph_only.py`，协议最终是graph_only；目录旧名字带factorial并不意味着最终跑了全部eager组合。
- 报告：[ANALYSIS.md](eval_results/k256_followup_dense_factorial_gpu0_20260924/ANALYSIS.md)、[RESULTS.md](eval_results/k256_followup_dense_factorial_gpu0_20260924/RESULTS.md)、summary.json及runtime逐轮请求。
- 64条固定开放面板、7 proposals、T0/T1、BF16、并发1、max128、两轮正反顺序；监测无GPU共驻，CPU/频率并非独占控制。

| 模型 | T0 A | T0 tokens/s | T1 A | T1 tokens/s |
|---|---:|---:|---:|---:|
| raw_graph | 3.721645 | 140.973 | 3.163524 | 122.306 |
| dense_combined | 3.866792 | 145.905 | 3.656209 | 137.538 |
| local_combined K256 | 3.868606 | 143.311 | 3.656209 | 136.293 |

K256没有损失这里的接受均值；吞吐低于dense约1.78%/0.91%，两轮排序有波动，不足以解释稳定原因。dense相对raw吞吐+3.50%/+12.45%，K256+1.66%/+11.44%。不是相对纯AR或完整DFlash2/DSpark的速度。

原报告自动生成 `Graph/eager native parity passed: True` 来自空列表all；**不能引用为这三臂做过完整eager配对验证**。ANALYSIS已说明。新目录分析器已改为空列表时null。

### 第二优先：刚完成的真实前缀greedy与native时间线诊断

- 代码：[residualV3_k256_greedy_diagnostic_v1/README.md](residualV3_k256_greedy_diagnostic_v1/README.md)。
- 报告：[FINDINGS.md](eval_results/standalone_k256_greedy_diagnostic_v1_20260924/FINDINGS.md)、[DECISIONS.md](eval_results/standalone_k256_greedy_diagnostic_v1_20260924/DECISIONS.md)、summary.json。
- 原始greedy记录：输出 `greedy/{dense_combined,local_combined}/blocks.jsonl`。
- 时间线：`profile/*/profile_T*_p*.json`；分段events在event_spans.jsonl；kernel归属在trace_attribution.json。

16条固定诊断prompt，每臂508 blocks；1985个实际可到达位置，raw错557，修对176；raw对1428，修错51。dense和K256相同。

每臂432个首拒绝：35个原11档dose可修、预测未修；51个raw原本正确被改错；346个raw错且预测及原grid都不能修。86个局部grid∪pred改善空间不是native序列上界。346不能全部归咎于direction，有限grid与模型表示均有可能。

诊断Graph段dense约3.43/3.45 ms，K256约2.78/2.80 ms；K256候选准备约0.103 ms、投影0.056 ms、Stage1+Confidence/Dose1.49 ms。独立trace里的修复kernel总和dense3.11/K2562.44 ms/block，说明真实target/body交替下仍省投影。事件节点和同步有开销，不能把这些数当未插桩端到端时间。

### 第三优先：K256执行组织与局部范围证据

- 初始K256：[standalone_local_k256_v1_20260924/RESULTS.md](eval_results/standalone_local_k256_v1_20260924/RESULTS.md)。局部修256保留大部分接受收益；共卡eager吞吐只作辅助。
- 执行七臂：[standalone_k256_execution_v2_20260924/RESULTS.md](eval_results/standalone_k256_execution_v2_20260924/RESULTS.md)，代码 `residualV3_k256_execution_v2/`。有eager/Graph对照；GPU1共驻，不能用其跨臂百分比当干净净加速。
- 同状态：[k256_followup_same_state_gpu0_20260924/FINDINGS.md](eval_results/k256_followup_same_state_gpu0_20260924/FINDINGS.md)。热缓存微基准，不替代native。
- 早期成本：[standalone_k256_cost_diagnostic_v1_20260924/FINDINGS.md](eval_results/standalone_k256_cost_diagnostic_v1_20260924/FINDINGS.md)。准备写回问题的早期证据；后续Graph组织已改变，不能永久套用旧归因。

### 必须保留：全词表LM-head、小词表no-adapter、ResidualV2

| 项目 | 源码 | 权重/结果 |
|---|---|---|
| 完整重训full LM-head C | `residualV3_lmhead/` | `eval_results/residualV3_full_vocab_lmhead_retrain_v1_seed42/arms/C/epoch_16.safetensors`；该输出的`native_sampling_aligned_v1/RESULTS.md` |
| 32k no-adapter | `residualV3_32k/` | `eval_results/residualV3_p1_adapter_matched_t01_v1_seed42/arms/trained_no_adapter/epoch_24/repair_model.safetensors`；同父目录`native_sampling_aligned_v1/RESULTS.md` |
| 最重要ResidualV2 | `residualV2/` | `output/openmath/onpolicy/outputresidualV2/`，含Stage1/Stage2主权重、日志和basis；禁止擅自清理 |

历史采样对齐接受收益：32k raw T0/T1 A=3.933516/3.342239，no-adapter=4.134534/3.896397（+5.11%/+16.58%），256prompts。Full raw=3.721645/3.163524，LM-head重训=3.872240/3.644337（+4.05%/+15.20%），64prompts。面板不同，不能横比full和32k；这是历史数值路径，不冒充最新全BF16 Graph结果。

独立源码复现及共卡runtime：`eval_results/standalone_reproduction_runtime_20260924/{32k,lmhead}/runtime_retry_v2/RESULTS.md`。全BF16重新评测：`eval_results/standalone_bf16_full_vocab_gpu1_20260924/{analysis,runtime}/RESULTS.md`。这些共卡结果不优于GPU0 Graph公平对照的速度证据质量。

两个外部对比头的验证：`eval_results/baseline_bf16_native_validation_20260924/VALIDATION.md`；`Baseline_dspark_full_vocab_v1/`等是head-only基线，**不是完整DSpark/DFlash2复现**。

## 3. 新窗口必须遵守的口径和约束

- 主目标：完整词表支持下提高接受收益和实际速度，便于与DFlash2/DSpark比较；不是证明full大于32k。
- K=256是修正候选坐标数，不是basis rank。R=512不改，basis不用重新拟合。
- 保留真实lag-1；T0双方greedy，T1双方随机，反馈来自实际采样token，q来自实际processed返回logits。不能将argmax偷偷当T1反馈。
- Top16与K256属于同一局部修复方案：Top16用于特征/竞争训练，K256用于logit修正；全词表仍可生成。局部logit修改后的softmax会影响所有概率。
- 用户已给定历史raw Top16含target Top1约94%的事实；不再要求重验该覆盖率，也不能擅自改称特定条件覆盖率。
- A=1+accepted/D=1+sum Sj/D，按含bonus约定；不同臂block差和生存差不是共同block因果修复数。A增益不等于tokens/s增益。
- 当前评测是开放诊断面板，重复种子不是独立样本；无逐请求native计数则不伪造配对acceptance CI。
- 不修改旧C/J、历史实验/权重或ResidualV2；不恢复已删除adapter；不启动训练搜索；不因看到空闲卡自动复制同一流水线。
- GPU3此前按用户要求已释放；这次只在GPU0运行，不停止其他人的GPU任务。
- 运行环境 `/data/weijiahong/miniconda3/envs/llm_dflash_eval_v026`，CUDA兼容库和PYTHONPATH看launch.json；不要拿系统Python直接跑GPU入口。

新窗口建议第一条操作：读取本轮 `pipeline_status.json`、`pipeline.log`、当前阶段日志及nvidia-smi。根据实际状态继续排错/核对结果；如果阶段failed，先查错误并保留冻结协议与失败证据，不能覆盖旧结果假装一次成功。最终分别回答greedy损失相对同预算control是否改善，以及同权重融合是否通过数值门禁并带来native净提速。
