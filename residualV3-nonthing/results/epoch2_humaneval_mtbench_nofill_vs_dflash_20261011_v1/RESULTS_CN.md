# E2 HumanEval / MT-Bench：独立DFlash对联合完整头

2026-10-11 E2跨域对照：HumanEval164题、MT-Bench80题，均为DSpark第一轮全量原题。状态complete，正式窗16/16。humaneval C1 T0 AL+14.48% TPS+10.85% 隔离True；humaneval C1 T1 AL+21.50% TPS+17.48% 隔离True；humaneval C40 T0 AL+14.46% TPS+7.47% 隔离True；humaneval C40 T1 AL+18.95% TPS+14.58% 隔离True；mt-bench C1 T0 AL+8.52% TPS+6.78% 隔离True；mt-bench C1 T1 AL+10.66% TPS+8.78% 隔离True；mt-bench C40 T0 AL+8.48% TPS+3.97% 隔离True；mt-bench C40 T1 AL+13.72% TPS+8.43% 隔离True。

只测DFLASH（第二轮独立单训主体）和FULL15_NOFILL（第二轮联合主体＋三层修复头）。不同主体权重的整体模型对比；固定15个proposal，不启用尾部截断或辅助预测器。推理runtime与已完成GSM8K正式版逐文件相同，仅新建评测入口、数据协议及控制器。

HumanEval：C1/C40各164道唯一题。MT-Bench：C1/C40各80道唯一题，按DSpark的AL评测代码只取第一轮；不是完整两轮MT-Bench质量评分。原题不循环补足200/1319。每个数据集均T0/T1、两臂、单轮；共16个正式窗/1952次正式请求。C40有有限面板的启动和尾部排空，不能解释为1319请求持续负载。

保留此前max_new=2096/context4096、seed42+请求序号、BF16、batch-invariant、CUDA Graph、block16=anchor+15 proposal。DSpark默认T1/max2048/seed980406，本次对齐题量、官方prompt、第一轮与nonthinking口径，保留项目原推理设置；没有复现其权重、训练或离线推理引擎。

先在实际数据prompt上做smoke合同核验，再正式计时；两臂同卡同窗设置。C1 KV52GiB；C40 smoke38GiB、正式44GiB。所有GPU0–7均允许；只用宿主无进程且空闲至少78000MiB的空卡，不按GPU编号排除。AL是接受长度，包含bonus，不是答题准确率；TPS是实际生成token数除以HTTP整窗墙钟时间。

| 数据集 | 设置 | 单训 AL | 联合头 AL | AL 增长 | 单训 TPS | 联合头 TPS | TPS 增长 | 隔离 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| humaneval | C1 T0 | 5.541694 | 6.344182 | +14.48% | 134.114 | 148.662 | +10.85% | True |
| humaneval | C1 T1 | 4.681263 | 5.687675 | +21.50% | 113.522 | 133.365 | +17.48% | True |
| humaneval | C40 T0 | 5.541621 | 6.343126 | +14.46% | 2130.519 | 2289.712 | +7.47% | True |
| humaneval | C40 T1 | 4.681261 | 5.568287 | +18.95% | 1708.830 | 1957.990 | +14.58% | True |
| mt-bench | C1 T0 | 3.144794 | 3.412641 | +8.52% | 77.578 | 82.838 | +6.78% | True |
| mt-bench | C1 T1 | 2.820589 | 3.121363 | +10.66% | 69.598 | 75.711 | +8.78% | True |
| mt-bench | C40 T0 | 3.147619 | 3.414442 | +8.48% | 973.845 | 1012.534 | +3.97% | True |
| mt-bench | C40 T1 | 2.774017 | 3.154719 | +13.72% | 920.795 | 998.453 | +8.43% | True |

当前状态：{"status": "complete", "controller_pid": 3688127, "parallel_dispatch": true, "dataset_states": {"humaneval": {"status": "complete", "dataset": "humaneval", "utc": "2026-10-11T16:12:27.245643+00:00"}, "mt-bench": {"status": "complete", "dataset": "mt-bench", "utc": "2026-10-11T15:52:06.324240+00:00"}}, "utc": "2026-10-11T16:12:38.629121+00:00", "dispatch_manifest": "/data/weijiahong/speculators-main-lora/residualV3-nonthing/results/epoch2_humaneval_mtbench_nofill_vs_dflash_20261011_v1/parallel/DISPATCH_MANIFEST.json"}

[状态](status.json)、[宿主GPU检查](HOST_GPU_GATE.json)、[准备审计](PREFLIGHT.json)、[DSpark口径](DSPARK_PROTOCOL_AUDIT.json)、[原始比较](COMPARISON.json)、[控制器即时log](controller.log)。

humaneval C1：[smoke log](humaneval_C1_smoke.log)、[正式log](humaneval_C1_formal.log)。

humaneval C40：[smoke log](humaneval_C40_smoke.log)、[正式log](humaneval_C40_formal.log)。

mt-bench C1：[smoke log](mt-bench_C1_smoke.log)、[正式log](mt-bench_C1_formal.log)。

mt-bench C40：[smoke log](mt-bench_C40_smoke.log)、[正式log](mt-bench_C40_formal.log)。

[最终审计](FINAL_AUDIT.json)。

并行调度：HumanEval C1正式结果保留；未完成C40从GPU7迁卡，原部分smoke保存在humaneval/C40/prior_attempts。两个数据集各有独立待卡任务；同一C/T对照的两臂仍同卡同设置，另一卡可能同时运行另一个数据集。native runtime/evaluator及权重保持原冻结身份；新增调度源码单独冻结审计。外来进程进入后舍弃该尝试的速度数据，归档并重排未完成对照。

[并行控制器log](parallel/controller_v4.log)、[两路当前状态](status.json)、[调度身份](parallel/DISPATCH_MANIFEST.json)。
