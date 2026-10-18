# NEXT_CHAT：本轮已收束，勿重复已完成实验

Run `20261013T154124Z_mismatch_e2`；更新 UTC 2026-10-14T02:27:24.282782+00:00。

先读 [results/RESULTS_CN.md](results/RESULTS_CN.md)、[results/RUN_AUDIT.json](results/RUN_AUDIT.json)、[protocol/LOCK.json](protocol/LOCK.json)、[analysis/FINAL_RUNTIME_AUDIT.json](analysis/FINAL_RUNTIME_AUDIT.json)。本轮状态为**A/B完成，native C技术退出不确定**，不是资源等待，也不是C科学反证。

## 已完成与不可重跑

- 原批次48/48有效窗、24/24配对，FINAL与COMPLETION通过；12 summaries和最终链接SHA独立核验。勿补回任何archived attempt、勿启动原parallel/controller/CPU收尾。
- A/B：192唯一prompt（每域dev24/test40），pilot24包含dev72；标签复用、K8主分析，dev12题K4/8/16共同360C/i。formal120完成，一次固定主分析已写 `logs/TEST_ANALYSIS_OPEN.json`。不要重跑完成shard、refit温度、重新锁协议、改代码后冒充原结果、重挑案例。
- 校准jointT0.4965853/independentT0.4065697仅由devgrid拟合；test主效应.179090、raw改善.250882、head胜校准.236586，95%CI见固定report。约12.69%状态恶化，低raw-TV校准更好，低Top256 target质量收益较小但非0。
- A经验条件KL.723250/平均拟合2.472762 nats（joint/K8/i1..15）；条件项K16未饱和。不能说条件主导、容量比例或整体增益的因果百分比。
- B KL/overlap请求的描述性补充写 `analysis/SECONDARY_B_KL_TEST.json`；细节锁后但formal效果未开前声明，不是锁前预注册新假设，不改primary。TV与KL的置换/raw排序不同，保留该限制。
- 理论卡片、一手22项矩阵与阅读范围、历史负结果、原结果最终复核、四幅真实A/B图已归档。可选部署预测探针未做，不作可部署泛化预测主张。

## C为何不能继续

`analysis/NATIVE_BRANCH_EXIT.json`绑定三次尝试与日志，成功请求/verifier均0；第三次只有普通anchor[10061]，不是proposal数据。技术恢复初始+两次上限已耗尽，不得通过换arm/stage重置。本轮native全局gate已关闭。仍有GPU预算也不能绕过技术退出条件。

stock V2 `DraftTokensHandler`在非structured分支仅在CPU发布[-1]长度占位，真实token来自GPU `req_states.draft_tokens`。新hook误拒正常占位。下一轮测量桥接需scheduler权威request/预算/长度 + GPUworker真实token/p/q，再和scheduler实际发出及HTTP usage核对；EOS、bonus、correction、普通/零输出prefill均须保留。

HF参考是SDPA/BF16/cached，EOS151645；native普通首轮观测EOS151643/151645。48dev fresh cached重复bitwise一致，cached/full最大观察TV.060815；不是native漂移上界。不能把本轮head效应大于它写成native鲁棒。

## 资源和后续

全部7个自有GPUstage累计0.7519894215 GPU小时、始终只GPU6 UUID `GPU-16d48373-c956-39a9-3989-4676fb6aa22c`；所有已结束。最新宿主释放证据在FINAL_RUNTIME_AUDIT；未来占用必须重新读宿主状态/预约/UID/PID启动身份，不能沿用空闲快照。本轮无未来自动启动watcher，不控制他人进程。

本轮冻结源、权重、完整断点、原协议与自动结果保留；target大weight shard只具stat身份，未找到可信旧SHA，不伪造散列。新run累计写入远低于50GiB，精确预算和观察峰值见RUN_AUDIT。

若用户明确授权**新的限定轮次**，仅建议native桥接验收：CPU约2–4小时、GPU硬上限2小时、单卡，先每域2 dev题×三臂、max512、C1T1。合同不闭合即退出；闭合后才能研究native reached与连续存活，局部或真实发出无增益可削弱H_C。不要自动恢复本轮C、训练新头/主体或扩成结构搜索。低raw-TV是oracle分层，不直接成为部署gate。
