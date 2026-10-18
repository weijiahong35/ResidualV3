卡1–5：先共同分片采集原生到达状态的XTX/XG/GG，后五组可预测基底拟合与短训。卡0旧实验继续，普通GG三组权重复用，不重训。
五组：RR_KL / RR_OVERLAP / RR_MARGIN（lambda1）；RR_OVERLAP_LOW/HIGH（lambda0.1/10）。rank512、no-soft两层、原始q、实际硬lag1、128×3=384步OUTPUT。
启动预留在native模型分配前释放；extra process记录不自动中止；hidden/logits仅在线使用，无持久缓存。
进度status.json，workers/gpu1..5的collect_status.json、collection/fit/progress.json、train_status.json、training/status.json。
完成判断：status.json=complete；completion_validation.json=complete且formal_windows=40、training_steps_each=384；controller_cleanup.json.finished=true。
结果FORMAL_RESULTS.csv、REPORT.txt；可预测性及几何DIAGNOSTIC_RESULTS.json。C1卡4=20/C40卡5=300，各10组×T0/T1；smoke不是测速。
如failed读error与相应日志；waiting_for_existing_controls表示等待卡0三组新GG训练结束，不自动重复训练。
固定随机输入投影和线性预测限制：不能从此断言全部输入信息是否足够，也不保证native改善。
