# 精确恢复记录：不能把本轮等待当成实验结束

Run：`20261013T154124Z_mismatch_e2`；目录不重建、不覆盖。先读 `manifest.json`、`results/RESULTS_CN.md`、`results/RUN_AUDIT.json`、`logs/RESOURCE_LATEST.json`。

## 已做，不重跑

原资料和当前有效summary审计；183冻结小源码SHA匹配；192个唯一prompt已固定（dev72/test120/pilot24）；训练group精确匹配检查；22条定向一手来源的章节核查及版本/范围审计（20全文原件，L05/L06完整下载缺失但browser原章节已保存）；理论卡片和反证审查；math30、AB28、native计数13项CPU检查通过。当前报告/图表只含历史真实数据和明确标示的CPU合成验证。

所有新GPU任务、pilot、A/B/C、温度拟合、探针和正式test主分析均未启动。GPU小时0；没有研究自有GPU PID，没有自动启动watcher；未创建 `protocol/LOCK.json`、`CALIBRATION.json` 或 `logs/TEST_ANALYSIS_OPEN.json`。

## 具体阻塞

原六域控制器 running；Arena C1 attempt2 正在运行；CPU watcher waiting_existing_controller。原GPU预约锁仍持有，FINAL/COMPLETION缺失。记录中的PID不是未来事实；接手重新宿主核验，不能只在sandbox执行ps或将旧PID消失等同成功。

原路径：

- `residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/{status,progress,COMPARISON,FINAL_AUDIT}.json`
- 同根 `parallel/controller_identity.json`、controller.log、各有效formal状态、重试记录；排除prior_attempts。
- `residualV3_main_v1/maintenance/complete_dspark_eval_20261013_v1/{identity,status,COMPLETION_AUDIT}.json`

只读宿主刷新使用本run `code/resource_preflight.py --host-confirmed`。该参数只能在宿主可见执行中使用，不是解除gate的覆盖开关。仍未终态则继续CPU，不能重启/终止/接管原parallel.py、收尾或占预约卡。失败终态需原因和预约释放，needs_attention单独一词不够。成功终态需核对FINAL/COMPLETION与原进程树退出。

## 尚需接通的独立接口

1. `hf_frozen_adapter.FrozenABReference` 已实现只读模型加载、target独立轨迹、fixed-C body和target分支。它有意要求有效lease对象 `assert_live_owned_and_budgeted()`；尚未实现/启动budgeted executor，也未实际加载E2。资源释放后在新run内实现租约/累计wall-clock supervisor，记录宿主PID/PGID/UID/fullCMD/start/GPU UUID；单卡累计24h、预留保存、所有分支和重试共额。需使用原共享GPU lease防竞态，不能改原资产或抢持有的锁。源码角色与模型文件都在ASSET_AUDIT。
2. A/B position采集用 `measure_position(..., replay_callback=replay_predecessors)`；这个接口已通过CPU随机小头集成。使用强制H，无target p输入head。A只给raw固定q做分解，B在同C/i live分支置换实际前驱；n1与i1规则已经实现。`branch_labels`保留同一C最多K×15的BF16概率标签在内存，每C释放；只存标量/token/H/coverage。
3. `native_verifier_capture.install(output, round_metadata)` 只可在新自有server进程安装，不在原服务安装。它读取实际sampling变换后的target logits、实际draft_logits、proposal token、b/a/G、sampleraccepted/bonus/correction。**尚缺权威轮开始request预算/身份的provider，以及scheduler EOS/预算后真实emit companion与join。** 不能把sampler输出当实际emit；`native_accounting.reconcile_emission`已有CPU验证。新launch/协议应独立于强制两臂的原native_evaluate.py。
4. B1 native可用同 `E2_JOINT_NATIVE` raw_graph，从raw实际采样、同raw processed q验证；不同于用B2 token配raw q。B0独立body、B2联合+头各走自己的轨迹。compute_logits与mm、HF/SDPA与native、anchor/EOS实际设置均需dev parity；当前target大shard只stat，小配置已SHA，是否限速补大SHA应在I/O空闲时决定并记录。

以上实现缺口是技术事项，不是科学反证。GPU等待期间可以继续接通新native接口；不因接口存在而宣称C已验收。

## 接下来沿用已授权本轮

只在gate通过、空卡exclusive并确认UUID后运行dev pilot（每域8，K4包含于dev）；测概率和sampling、偏移、replay/native parity、EOS/bonus、成本/显存/磁盘。技术重试最多2；保存失败原因，不重复已完成分片。

成本裁剪/主K只能pilot后改变一次并登记，随后拟合dev单温度、写真实 `analysis/PILOT_AUDIT.json`，用 `code/lock_protocol.py` 冻结阈值/样本/采样/源码hash。CPU audit不能替代该pilot。锁后A/B共用labels，test主分析一次；C需要同prompt三臂native实际q及发出计数。缺失覆盖报告，不补题、不扩搜到显著。预算耗尽前保存并停止已核实身份的自有作业。

分析接口：`analysis/fit_temperature.py <dev-grid.jsonl>` 拒test；`analysis/aggregate_ab.py <AB.jsonl> --split {dev|test}` 按prompt分组、报告anchor/n1/缺失域/B4，在锁定hash不匹配或已有test打开receipt时拒继续。不可删receipt重试获取更多test选择。

## 命令与身份

参考 `logs/COMMANDS_CN.md`、三份CPU audit、资源snapshot、来源恢复审计。CPU PID若属于sandbox命名空间不能用于宿主管理；原宿主job身份另存。下一chat先查重再恢复。原正式模型、完整断点、basis、协议、自动结果全部保护。本轮未提出新训练候选；只有核心证据完成后才可给≤2项下一轮方法建议。
