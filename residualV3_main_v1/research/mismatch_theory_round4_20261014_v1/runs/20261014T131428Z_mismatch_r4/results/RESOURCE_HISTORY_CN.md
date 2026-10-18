# R4 资源变更与恢复记录

本记录解释资源层的偏差和恢复，不用于改变样本、动作、oracle、数值带或统计规则。最终累计用量及终态以 RUN_AUDIT.json 和 COMPLETION.json 为准。

用户后来明确要求“磁盘最大占用700g”。本轮按700 GiB执行；新增上限原为60 GiB。原协议和LOCK保留原字节，独立授权记录见 logs/DISK_AUTHORIZATION_700G.json，独立运行入口见 evidence_tools/resource_budget_entry.py。该入口只修改运行期磁盘守卫；GPU总额8小时、CPU总额24 CPU小时、单卡、固定K32和48题均未增加。700 GiB是上限，不是存储目标。

首次fresh采集曾实测工作进程共有10条OS线程，超过8条的字面上限。短时间CPU样本中主要是主线程消耗CPU，不能据此声称全程满足总线程上限。analysis/COLLECT_THREAD_SCOPE_AUDIT.json明确记录 literal_OS_thread_limit8_passed=false。核验UID/PID/PGID/startticks及命令后，仅停止本轮自有监督进程；原监督器清理自有子进程，没有管理他人作业。

资源恢复通过独立启动环境将BLAS线程设为1、禁用tokenizer并行和allocator后台线程；冻结采集源码仍执行原算术。30个已完成prompt及其完成收据原样保留，只隔离未完成anchor。第一次恢复实测worker OS线程8；后一次恢复worker为7。worker为8本身不能证明包括监督器的总线程≤8；历史超限及这一口径限制保留，不能把最终资源审计写成全程无偏差。对应记录见 logs/THREAD_CAP_CORRECTION_STOP.json、THREAD_CAP_RECOVERY.json。

随后用户主动中断执行turn。监督进程与GPU子进程在实时宿主核验时已退出；恢复前核对了完整命令身份和GPU进程列表。悬挂GPU账本按启动至确认终态的保守墙钟补记，不重置余额。CPU账本为不可见尾段补记保守上界，属于保守计费而非全段实测。只隔离未完成分片，已完成收据继续保留；同一固定新题清单、分支seed及fresh_open_count=1不变。见 logs/TURN_INTERRUPTION_RECONCILIATION.json 和 TURN_INTERRUPTION_PARTIAL_RECOVERY.json。

共卡准入按阶段工作集、剩余显存、UUID及预约核验。其他进程存在不阻断准入，不使用旧空卡gate，也未要求utilization=0。目标模型采集和纯见证优化分别按各自峰值加4 GiB余量准入；只对自有作业进行保存、退出和资源恢复。原正式评测与R1–R3只读，不重启旧controller/parallel/收尾。本轮不报告共卡TPS。

锁后为原求解器增加了独立证据观察器，保存实际使用的LP乘子、完整尾部切线和输入绑定。启用前后小词表结果合同通过；原科学源码不修改，观察器及其合同另有SHA与收据。它补充可重构的FP64数值证据，不代表有理数精确证明，也不把原未认证区间改称精确最优。

证书保存阶段另定位到原磁盘守卫的扫描开销：一次全树扫描约0.354秒墙钟、0.346 CPU秒；短窗口的71,070次系统调用均为文件树访问。这只说明该窗口的瓶颈，不是完整CPU消耗分解。独立新资源入口通过22项CPU合同后，仍每次检查物理余量，并用共享文件锁保留初始实测分配量、8 GiB未插桩小文件包络和累计写入预约上界。预约按块上取整，不因周期扫描、提交、失败或新stage清零。约10秒实测对账，超过上界或700 GiB即阻断；GPU父进程原20秒实测扫描保留。原分析器的大文件写入也逐次预约，不依赖8 GiB包络兜底。

新入口只路由资源回调，原LP/切平面、100次上限、初始化、动作、容差、样本、顺序及数学函数保持原样；小词表完整oracle输出合同通过。新资源代码与收据独立绑定，旧观察器9文件及科学LOCK仍保持原SHA。此次恢复只归档缺少完成收据的上下文及其证据，完成上下文原样保留；fresh打开次数仍为1。对应源码、合同、停止和恢复记录见 logs/DISK_CACHE_RESOURCE_RECEIPT.json 及本轮CPU账本。

恢复前核查已完成上下文的5,804个预期证据槽时，发现1个观察器保存缺口：`alpaca:02c58ec4a7b3247c6862 / anchor1 / g4 / fit_shared`。原科学记录完整；这是零迭代闭式情况，原下界为0，缺口目录为空。确切的保存异常原因未由历史日志建立，不能只凭空目录确定原因。其余5,803个槽已有完整PROOF/VERIFIED。

该槽允许从已签名科学记录和原完整输入独立重构闭式数值证据：不调用LP、不重求原完成上下文、不修改原数值，另标 `analytic_independent_posthoc_reconstruction`，并注明 `actual_return_capture=false`。历史空目录与缺口审计保留。原证书索引的实际捕获数量字段须结合新增来源审计阅读；事后重构项不能计成当时实际捕获的返回。此补证与原采样概率警报是不同事项，后者仍保留。
