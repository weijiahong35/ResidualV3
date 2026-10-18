# 第二轮交接：固定实验已完成

本run：`residualV3_main_v1/research/mismatch_theory_round2_20261014_v1/runs/20261014T040004Z_mismatch_r2`。原R1、归档包、冻结资产、安装环境及原正式评测保留原样。本轮没有未完成的采集/拟合/主分析，不重启已完成stage，不因预算尚有余额添加支路。

先读 [RESULTS_CN.md](results/RESULTS_CN.md)、[THEORY_CARD_CN.md](results/THEORY_CARD_CN.md)、[CLAIM_EVIDENCE.json](results/CLAIM_EVIDENCE.json)、[RUN_AUDIT.json](results/RUN_AUDIT.json) 和本run `COMPLETION.json`。输入/源码、样本及模型身份见 `protocol/LOCK.json`、`SAMPLE_MANIFEST.json`、`GEOMETRY_FIT.json`，最终文件完整性见 `results/RUN_FILE_MANIFEST.json`。

## 已完成、不可作为未见数据重跑的事实

- dev每域12题共36题：HF旧C/H重放只采新量；native108条请求（工程18包含在内）。模型/scaler/正则/阈值/统计在fresh之前固定。
- 新分析留出每域16题共48题：HF11,400状态，主状态10,640；native144条完整请求、12,538验证轮。一个HF缺anchor未补采，全部48题都有主状态，无native零分母或缺合格请求。所有fresh效果已暴露。
- H_R预测损失差1.5276486e−5，95%CI[−1.4763676e−5,4.0583544e−5]，实质阈值.000790836；相对MSE改善仅.02015%。实质增量主张被削弱，不扩特征、换A版本或筛域救结果。
- HF平均实际改善.251713，gap .374797，约14.27%加权状态恶化。Dstar .063438接近pU .063694；不能解释为覆盖已解释全部收益，不能将gap纯归因容量或可学习空间。
- H_N native局部改善.127825，95%CI[.119611,.137178]；B2−B1每请求已发accepted proposal/验证轮差.892487[.703574,1.087621]，三域均正。实际输出/全调度轮辅助差.847449。B2总输出较B1少，不能写总token更多或TPS。
- HF预测器原样用于native的增量−5.71748e−6，CI含0，域方向不同。H_N不能补证H_R；锁定两主检验整体确认条件未满足。

## 工程与概率边界

桥接先有B0完整真实验证，后工程6题×三臂。scheduler提供request/轮次/预算/长度，CPU[-1]只作已核接口的长度占位；GPU真实token与actual generation q、同轮处理后p/q绑定。warmup、零/普通prefill、proposal验证、停止后输出和HTTP usage显式区分。

36条B2 **dev**请求检查原/优化头parity，最大TV=0，固定容差2e−5，越界fail。锁定capture在fresh不重做该parity，fresh继承同权重/源码准入；不能捏造48次fresh parity。raw shadow等于actual q的检查只适用于B0/B1，B2二者差异是修复本身；B2 actual q与实际生成buffer绑定另核验。root曾在辅助inline最终审计把这两项范围断言写错，已定位/修正并保留 `logs/FINAL_TECHNICAL_AUDIT_CORRECTION.json`；主源码、协议、数据、科学报告未改或重算。

原 `analysis/NATIVE_BRIDGE_AUDIT.json` 仍 **passed=false**：工程6题B2概率残差CI不含0。`NATIVE_BRIDGE_REVIEWED_AUDIT.json` 的passed仅指机械测量准入。CPU复现1758/1758 first-reject支持实际身份，不证明proposal与接受uniform条件独立；它们共享第一层Philox来源。新48题概率残差CI含0仅表示相容，不能撤销原警点。torch FP32重构并非Triton概率缓冲bitwise审计。ideal `G∏b` runtime law **未建立**。[边界说明](results/NATIVE_PROBABILITY_BOUNDARY_CN.md)

G仅含proposal路径长度、先前EOS、轮初预算，不含实现接受事件。局部主量是实际first-reject reached且独立G有效，含拒绝位置，不再次加权。HF target-branch与native自身轨迹分开；同seed不等于同轨迹，普通路径overlap乘积不能代替连续存活。自然零输出轮未出现，只有明确CPU fixture覆盖。原vLLM接口以锁定安装源码为准，不猜最新文档。

## 审计与资源

全轮累计1.397083 GPU小时，桥接.112450小时；八次监督启动包含一次已定位TIME_WAIT端口预启动失败，失败时间也计费。其他stage完成且清理正常。只用了GPU7 UUID `GPU-d74003c2-c31c-a723-8de9-4b1e1a41cf3f`，UID1024、PID/PGID/startticks/CMD均在GPU账本和各command记录。最终宿主检查GPU7无GPU进程/显存、预约锁释放；旧空闲快照不能供未来准入。精确新增磁盘/观测峰值、原资产身份和锁复核见运行审计。

原六数据集48/48窗、24/24配对，FINAL/COMPLETION均passed，16000请求收尾完成；R1全部1929文件与归档SHA一致。冻结导出/头/basis重新核SHA；大target分片缺历史可信全SHA，保留stat身份限制。禁止重新启动/终止原controller、parallel或收尾，未来只管理核实身份的自有进程。

## 下一轮建议，尚未执行

主报告仅保留两项：冻结头固定0.5剂量对照（预估≤2 GPU小时，另建题面后确认），以及隔离overlay中的独立拒绝随机源诊断（预估≤1 GPU小时、≤1 CPU小时，固定6 dev×8seed×两设置）。详见Q7的反证条件。成本是待验证估计，不是本轮追加任务；不改安装版，不训练新头/主体，不扩大剂量网格。

当前MT-Bench80题已全部参与两轮机制分析，未来确认题必须另建来源面板，排除所有已暴露题，不能重开本48题充当新留出。若用户明确授权新的限定轮次，使用新run、新锁和新预算；本轮终态与文献/实测结论保持归档。标准TV/KL工具、Top256、前驱有用、工程运行及未找到同名方法都不能单独成为新贡献。
