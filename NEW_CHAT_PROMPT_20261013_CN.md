# 新对话启动 prompt（2026-10-13）

复制分隔线下全部内容作为新对话第一条消息。下面的进度是北京时间13:01左右快照，接手必须重新查实时状态。

---

继续 `/data/weijiahong/speculators-main-lora` 的 ResidualV3 mismatch repair。先读：

1. `HANDOFF_20261013_CN.md`、`HANDOFF_20261013_STATE.json`、`HANDOFF_20261013_PROCESS_SNAPSHOT.json`。
2. 设R=`residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1`，读R下 `status.json`、`progress.json`、`RESULTS_CN.md`、`COMPARISON.json`、`plan.json`、`DATASET_AUDIT.json`、`parallel/controller.log`。
3. `GROUP_MEETING_20261013_CN.md`、`RESIDUALV3_START_HERE.txt`。查实验历史再读两总账和20261011旧交接，不用重做已完成实验。

**当前六数据集E2重测仍在运行，不要另启动parallel.py。** 本批控制器快照PID4138310、UID1024，已在宿主核对命令。13:01左右10/48正式窗、4/24配对。HumanEval C1/C32全部8窗已完成，MBPP C1在GPU5，LiveCodeBench C1在GPU6，MT-Bench/Alpaca/Arena排队。PID／GPU／进度都可能变化，先重查uid/cmd和宿主GPU，不依赖沙箱缺失进程的结果。

本批只测E2独立完整单训DFlash vs E2联合主体＋三层FULL15_NOFILL，不用联合主体关头或RAW替代baseline；两臂主体权重不同。C1/C32、T0/T1，BF16、batch-invariant、CUDA Graph，nonthinking，固定15 proposal、max2048。**KV固定C1=52GiB/C32=44GiB、util=.92，smoke与formal相同，不得改显存预算。** 无截断／辅助／专家。本批两空卡上限，GPU0–7任意空卡可用，旧避开1/2规定已撤销。

数据按DeepSpec固定提交005e03b81cec38b7da6399833d609ee89a2587f2：HumanEval164、MBPP256、LiveCodeBench500、MT-Bench80（首轮）、Alpaca500、Arena-Hard-v2 500；种子980406＋index。MBPP只有255不同题面是上游重复，保留。五套context4096，Arena12288，无prompt截断。整批48正式窗、24配对、16000请求。我们仍是vLLM、15proposal，DSpark离线表为7，不能称完整复现。旧max2096/seed42批次保留，不跨批TPS相除。

新HumanEval已完成：C1 T0/T1 AC增长+14.48%/+18.43%、TPS+10.57%/+14.38%；C32 AC+14.46%/+18.03%、TPS+8.81%/+10.80%。以R/COMPARISON.json原值为准。8窗隔离与Graph通过，但整批最终审计未完成。正式日志在R/{dataset}_C{1|32}_formal.log，总log在R/parallel/controller.log；控制器自动更新报告及最终verify，别手改其生成MD。183份冻结源码SHA已重新核验；调度身份见DISPATCH_MANIFEST.json。

E1/E2全数据训练、完整master/Muon/Adam/RNG/cursor断点都完成且绝不能删；epoch12就是epoch1＋epoch2，不开epoch3。主体5层，三层no-soft SwiGLU修复头4353537参数，三路hidden/实际前驱/q特征，方向＋剂量Top256残差，仍全词表采样。每轮833176有效样本，每包65536输入token／1536完整anchors，block16。旧三领域C32、C40、2vs3层、模块计时、iter28/29均已完成，不重跑。两层只跳第三层、未重训；计时是C1/C40，没有C32模块时间。

周报已用文字箭头说明结构并修好公式。文献口径：Domino主实验用公开DFlash权重，ShareGPT双方重训表3更匹配我们口径，两组不是不同架构；DSpark表1关闭验证截断，完整调度结果另列4.4且baseline为MTP-1；DFlash-2仅选择头／无卷积消融只公开GSM8K，HumanEval的卷积＋选择头不能当纯头收益。DFlash-2来源是作者技术报告＋模型卡。原表、快照和审计在 `residualV3_main_v1/research/group_meeting_sources_20261013/`。不能凭跨论文绝对AC/TPS或增长率排名。

专家30k三种子训练完成，learned router未胜通用头，没有native收益；用户要求先不写进周报。尾部停修iter28/29也没有净TPS收益，不包装成第二创新。A会讨论提出公平近邻对照、贡献消融和泛化证据，只是后续建议，不是授权立刻额外长训。

磁盘用du -sh口径600G（二进制），接手快照523G；不是600GB十进制。保留模型／断点／basis／plan/order／共享cache，归档v4已核验。不要停他人或恢复旧控制器。CUDA环境复用runtime.utils.environment，当前Python为miniconda3/envs/dflash/bin/python。所有时刻向用户报告北京时间。

沿已有队列自主完成剩余授权评测，结束后核对48窗／24配对、隔离／Graph／代码身份／最终审计，在ledger.lock下更新registry.json后ledger.py refresh，再更新周报并保留旧成绩。先报告实时状态和log，不重复请求已授权事项。
