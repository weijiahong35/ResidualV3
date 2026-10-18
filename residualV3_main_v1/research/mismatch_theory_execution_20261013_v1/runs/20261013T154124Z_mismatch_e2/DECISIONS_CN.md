# 执行决定

- 用户已授权任务书范围内研究，沿用默认资源上限。
- 使用 deep-research 的来源验证、研究架构与反证角色；用户已给出明确问题、协议和授权，不重复研究选型或启动确认。
- 仅在新 run 写入；保留冻结源码、权重、原结果与原维护目录。
- 当前只开展 CPU；沙箱仅见本次命令进程、NVML 不可用，不能据此判断宿主资源释放。
- 文献检索为定向原文核验，不是系统综述；不写论文、不训练新头。

- 完成71项CPU验收与22条来源核查；未读新test标签。
- A/B单位置重放显式标anchor；28项CPU联检已验修正后的revision。
- 原formal计数先sampler后EOS/预算，C需实际emit companion，不沿用count−1充当终止后输出。
- source完整下载L05/L06缺失，保存browser一手章节选读并降级取得范围。
- 原controller/children/watcher与GPU预约仍活跃；本轮状态blocked，GPU小时0。
- native/HF后端仅静态实现；没有调用未验证后端、没有温度拟合、没有新训练建议。

- 2026-10-14恢复：原48窗24配对，FINAL/COMPLETION linkage与12 summary SHA独立核验；宿主原进程/预约释放，GPU6/7空。仅选GPU6。
- 单prompt的pilotK4与maindevK8共用同一组在线target标签；预选每域dev4题以K16嵌套子集做4/8/16敏感性。完成的dev shard不重跑。
- target轨迹只需生成到第65 token（在max256上限内）；anchor1/65规则不变，无EOS之后补样，不以效果选择C。
- 启动预算监督器累计各阶段/重试wall-clock GPU占用；单卡，compile worker1，GPU worker2计算线程；所有模型只读。

- native C attempt0在任何实际request之前因ZMQ Unixsocket路径超过107字节启动失败；已核实并清理自有进程。技术恢复1使用短/tmp符号链接指向run/data/tmp（实际文件仍归run计磁盘），保留attempt0日志。
- 48个pilot C/H缓存fresh重复bitwise相同；cached-vs-full最高TV0.060815，top1均相同。此为不同计算模式的概率差，不是head replay误差；HF cached与native runtime身份分别报告。

- native C第一次恢复（attempt1）服务warmup阶段因未armed的dummy request被当作真实请求失败；尚无实际request/采样。保留日志并清理自有进程；下一次attempt2是最后允许技术恢复，不按stage或arm重置此启动故障预算。

- 原批次最终独立CPU复核为48窗/24配对，AC范围6.0137%–18.4273%、TPS范围0.5930%–24.5424%；写入独立 `_FINAL` 产物，保留旧40窗快照，未修改原结果。
- native正式dev复用pilot24，只追加其余48dev；所有test collection均须核验协议、样本、校准、pilot及锁定代码哈希。
- HF缓存/整段计算差不进入head重放数值容差；它们是不同运行模式的概率标签，A/B结论限声明的HF-cached参考实现。native另在实际verifier分布上验收与分析，不据0.0608差异推断历史代理整体失效。

- dev72均完成（144 C，2160位置）；温度拟合首次CPU校验因32/4320 grid行TV略高于1退出，最大1.0000002384，所有finite/shape正确。未写拟合结果、未GPU重跑。锁前采用固定1e-6 FP32 TV界容差，保留原始值，不裁剪或添加epsilon；其规模小于既有overlap闭合误差8.35e-7。正式test仍未打开。

- native最后attempt2已失败：健康检查后首真实request在scheduler发布阶段出现-1 speculative占位token，当前capture不能将其当作已确定实际proposal。原stage/重试计数不重置，本轮C退出为inconclusive，成功request=0；保留attempt0/1/2和完整自有进程cleanup，不再启动C。技术缺口不是H_C被削弱。
- 正式test之前将锁定AB-only scope：只验收可证明的HF参考A/B概率/对齐/重放/EOS分支合同；native actualq与EOS/bonus合同未建立，不能标full pilot passed。Q5 native迁移不确定；仍完成A/B dev温度参照和120题test，完成全部可执行交付。

- 正式test预定范围为完整A/B120题、K8，温度joint0.4965853/independent0.4065697冻结。dev完整分析验收通过：72题/144C/2160C-i，未读取test。效应阈值按先定规则将为0.005；不因dev效应更改K、split、分层、主指标或阈值规则。本轮省略可选预测探针，不作部署特征可预测收益的结论。
- native权威EOS集合含151643/151645，HF参考明示151645；完整前缀/anchor/source层序约定已核对，但HF/native数值与终止身份不互换。

- 固定报告入口主统计仅汇总B TV/sign，任务书§7另要求B KL/overlap。在test采集中、尚未打开任何formal效应前，记录SECONDARY_KL_REPORT_SPEC：用已锁generic estimator对既存四臂KL与correct overlap做一次描述性补充，位置2..15/原prompt hierarchy/相同bootstrap。不改py、不更改主效应、不拟合、不挑样本；明确此报告细节不是锁前单独登记，B4 KL未采不伪造。

- 正式test120完整收束，一次主分析支持B对应信息；保留12.69%加权状态恶化、低raw-TV校准优胜、低Top256 target质量小而非零收益及描述性TV/KL排序分歧。没有新case挑选/温度refit/K扩搜/部署probe。
- 最终宿主26项运行审计通过：7job/24登记PID均退出，GPU6和所有预约锁释放；原183源码/installed14源码/锁25py/失败33artifact均完整。总GPU0.7519894215h，新增磁盘观察峰值约0.141GiB（非连续精确峰值）。本轮AB完成、C不确定；不自动恢复C或启新训练。
