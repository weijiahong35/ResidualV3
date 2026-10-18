# 第三轮执行 chat 启动 prompt

复制下面分隔线后的全文作为新执行 chat 的首条消息。

---

你现在执行 ResidualV3 mismatch 研究第三轮。仓库：`/data/weijiahong/speculators-main-lora`。

请先完整阅读：

`residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/EXECUTION_GUIDE_CN.md`

然后按其 §1 读取第二轮真实报告、概率边界、样本清单、源码和一手文献记录。第二轮 run 是：

`residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2`

**研究目标：** 理论问题是大小模型分布不匹配及修正何时有效、何时失败；AR 修复头只是已有研究工具。当前平均修复与 native 每轮接受收益已有支持，但新增几何量解释成败的实质增量未获支持，约14.27% HF状态变差尚不能归因于过量修正。不要再证明“前驱有用”就结束，也不要预设下一假设必须成功。

**本轮仅执行两项：**

1. 隔离 overlay 中的接受随机源诊断：R2 原6条engineering dev × 8 seed × 原/分流两设置，仅完整B2，共96条请求；先核源码、实际buffer/计数和CPU小词表合同。共享Philox来源本身不证明有偏，区间含0也不证明无偏。
2. 冻结同一联合主体与头，固定强度乘数η∈{0,0.5,1}。乘数作用于已经预测的dose，之后走原BF16更新；做同状态HF对照和真实native三臂。检验半强度是否改善平均TV、救回坏状态且保住好状态收益，以及每验证轮实际接受是否相应改善。不得搜索更多剂量、训练新模块或把负状态率下降单独当成功。

在 `residualV3_main_v1/research/mismatch_theory_round3_20261014_v1/runs/<UTC时间>_mismatch_r3/` 建新run、新协议、新锁和新账本。**我授权你在任务书范围内自主实现、工程修复、运行、分析和收尾，不必再次询问是否开始。** 原R1/R2、归档、冻结权重/basis、安装环境、正式评测及其报告保持只读；不重启旧controller/parallel/收尾。只管理经实时身份核验的自有进程。

预算按任务书：同时最多1张确认空闲无预约冲突的GPU，**总上限3 GPU小时，其中随机源≤1、半强度≤2**，启动/失败/重试均计费；随机源CPU≤1 CPU小时，worker≤4、总线程≤8、新增磁盘≤20 GiB。记录并监督预算，到限前保存退出，不借新名字重置计数。预算是上限，不是必须用满。

半强度dev固定18题；新机制留出固定GSM8K/HumanEval/Alpaca各16题共48。按任务书去重与锁定规则抽取，排除R1/R2全部暴露题、本轮dev及其他已知机制题。MT-Bench80已全部暴露，不再作为新留出。Alpaca取正式500题面板，不从52002行原来源重抽。它是新的机制分析留出，不是从未参加过项目评测的数据。

按顺序推进：旧证据核验→CPU/源码合同→限定随机源诊断→半强度dev→按机械准入冻结统一RNG与统计协议→fresh一次采集/主分析→结果与资源收尾。RNG模式选择遵守任务书，不按效果挑实现。原模式机械身份合格而概率解释仍未建立时，可以报告可靠经验计数；token/q错配必须阻断native。native受阻时完成HF可做部分，不虚构全部完成。

两项主要科学量：同状态 `TV(full)−TV(half)`；native半强度减完整强度的每请求`已发accepted proposal/实际proposal验证轮数`。同时完整呈现raw/half/full取舍，按prompt配对、域等权报告区间；不要把状态/token当独立样本。执行任务书预设判据，不在fresh后修改。native同seed不等于同轨迹，不用teacher-prefix或普通draft overlap乘积替代真实连续存活。

原 `NATIVE_BRIDGE_AUDIT.json` 的 `passed=false` 必须保留；reviewed audit只通过机械准入，理想概率规律未建立。新证据另写新结论，不覆盖旧警报。半强度即使成功，也不能自动写成一般mismatch理论、最佳剂量或相对全部AR修复头的新优势。

第一条回复用通俗中文说明你将检验的两个问题、实时读到的旧轮终态和新run路径，随后推进。过程更新说明学到了什么与下一验收点，不停在计划。完成后交 `RESULTS_CN.md`、claim–evidence、审计、真实图表、`COMPLETION.json` 和 `NEXT_CHAT.md`；结果第一页回答“新增支持了什么、没支持什么、理论还缺什么、下一步是否值得做”。负结果正常收束，不追加第四轮。
