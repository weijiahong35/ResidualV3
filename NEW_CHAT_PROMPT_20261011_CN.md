# 新对话启动 prompt — 2026-10-12 整理后

我们继续 `/data/weijiahong/speculators-main-lora` 的 ResidualV3 mismatch repair 研究。
先读 `RESIDUALV3_START_HERE.txt`，再按问题打开其中链接；该页是当前唯一日常入口。
原始训练/结果路径保留；淘汰代码与诊断已校验存档后删工作目录，旧prompt全文在
`residualV3_main_v1/maintenance/research_cleanup_20261012_v4/before/NEW_CHAT_PROMPT_20261011_CN.md`。

当前已完成：E1/E2两轮训练；三层E2独立DFlash vs 联合NOFILL三领域C1/C40正式24窗；
C32三领域补测12窗；同主体两层/三层C1/C32直接对照24窗；三层C1/C40模块计时采集；
30k mismatch专家头研究的12模型训练/静态评估。不要重启完成队列或退役控制器。
宿主进程要重新核uid/cmd，历史PID不能当活跃证据；本轮整理未启停GPU服务。

核心结果与限制：
- 三层NOFILL同窗相对独立单训DFlash，三领域C1/C40及C32整体吞吐为正，详当前7项报告。
  AC/AL是接受长度。联合与独立主体权重不同，不能将完整收益解释成固定主体纯加头。
- 两层只在推理运行前两层，未重训；同主体对照AC12窗全下降，TPS11/12下降。
  对照C1 KV16GiB，旧正式C1 KV52GiB；不跨报告相除绝对TPS，禁止随意改预算重复实验。
- 最新模块计时三层，KV C1=52GiB/C40=44GiB、util0.92，无预算扫描。
  C40 T0满40直接组件仅3份；C40 T1缺满40主体/raw LM/头直接服务分段。
  已有不同活跃批大小分段与真实状态头重放，不能补成满40直接时长；详覆盖审计。
- 30k头训练：math/code/chat各10000条，另有router6144、selection3072、holdout3072，
  冻结E2主体/target/embedding/LM/basis/旧aux，仅训练新头/router，3种子×4对照。
  mismatch专家相对通用头T0一致率−0.212pp、T1 overlap−0.098pp，未胜随机分工。
  这是teacher-prefix静态指标，不是native AC/TPS；没有启动native候选。
- 尾部停修iter28/29无净TPS收益，所有负结果保留；选用代码仍iter20/21，最新测试/正式代码独立。
- 原两层vs独立DFlash仅16有效正式窗，HumanEval C1/C32隔离失败未完成，停止自动重试。

先读当前入口与 `HANDOFF_20261011_STATE.json`；训练细节/历史看 `HANDOFF_20261011_CN.md`，
完整总账看 `EXPERIMENTS_NONTHINKING.md`、`EXPERIMENTS_THINKING.md`。
今后多并发默认C32，历史C40保留；本次计时C40是用户单独指定。
E1根=`residualV3-nonthing/results/full_nonthinking_joint_body_1epoch_teacher1_3gpu_20261008_v1`；
E2根=`residualV3-nonthing/results/full_nonthinking_epoch2_aux_q45_latent_20261010_v4`。
两轮原始权重、完整共同断点、消融/新专家权重全部不能删；旧E2 plan/order、basis、
E1导出模型/panels、共享eval cache是依赖。主体5层，原头3层no-soft SwiGLU，4,353,537参数。
不擅自开epoch3、改训练目标或新增昂贵native/on-policy监督。

硬约束：磁盘上限600G，按 `du -sh` 口径；整理前约523G，接手重查。
GPU0–7任何编号空卡都可用，禁止碰其他项目/进程，不重启暂停VERIFY_STOP或退役控制器。
代码删前校验存档，硬链接二进制修改先独立copy，CUDA环境复用runtime.utils.environment。
归档索引：`archives/residualV3_research_cleanup_20261012_v4/INDEX.csv`，恢复先到新空目录审核。
总账必须在 `ledger.lock` 下改 registry.json 后 ledger.py refresh；保留负结果与代码/权重身份。
继续研究可先分析target验证/草稿主体的mismatch相关瓶颈；文献在当前入口04_history。
没有新稳定净收益不追加大规模截断长测。已授权范围自主推进，受限给出具体证据。
