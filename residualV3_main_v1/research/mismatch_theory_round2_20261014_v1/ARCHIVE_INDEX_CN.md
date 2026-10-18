# 第一轮 mismatch 研究归档与第二轮入口

归档时间：北京时间 2026-10-14 11:23。原 run：`20261013T154124Z_mismatch_e2`。

## 归档对象

- [完整归档包](archive/ROUND1_20261013T154124Z_mismatch_e2.tar.gz)：原 run 的 1,929 个常规文件，包含报告、协议、代码、原始诊断记录、文献、图表、失败 attempt 和审计。包内文件内容逐一核对 SHA256。
- [归档清单与哈希](archive/ARCHIVE_MANIFEST.json)：原文件 147,375,321 bytes；压缩包 25,334,047 bytes，SHA256 `ec41b68c925a3dfb7378ffe288b383cffdcf601917424538ec62ec0e78b13a72`。
- [原结果报告](../mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/results/RESULTS_CN.md)、[原运行审计](../mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/results/RUN_AUDIT.json)、[原恢复记录](../mismatch_theory_execution_20261013_v1/runs/20261013T154124Z_mismatch_e2/NEXT_CHAT.md)。

原目录原位保留，未修改内容或权限。归档包覆盖该 run；外部模型权重、原正式评测目录与安装环境仍通过其中的路径和身份记录引用，不能把压缩包称为包含全部模型的独立复现环境。损坏/未完成的下载和失败 attempt 也按原样保留，不当作有效实验。

## 冻结的科学状态

| 项目 | 归档结论 |
|---|---|
| A/B | 完成；HF-SDPA-BF16、target-branch/alive-at-i；本次分析 test 120题 |
| 前驱对应信息 | 置换−正确 TV=0.179090，95%CI [0.164850,0.193114] |
| 同主体局部改善 | 联合 raw−正确 TV=0.250882，[0.234390,0.267329]；不等于 native AC/TPS 归因 |
| 条件/拟合 KL | K8 经验值 0.723250/2.472762 nats；K16 条件项未饱和，不能推总体占比或容量因果 |
| 失效边界 | 约12.69%状态恶化；低 raw-TV 子集中温度校准较好；分层仍属探索性 |
| native C | 三次工程尝试，成功请求/验证记录均0；不确定，不是模型反证 |
| 资源 | 原审计记录累计0.751989 GPU小时；原自有作业/预约已释放，未来需重查实时资源 |
| 原六数据集批次 | 已完成48有效窗、24配对、16000请求；FINAL和COMPLETION passed |

本归档不更新或覆盖原报告，不将原 C 状态改为“已完成”，不重新拟合温度，不重开原120题主分析。

## 第二轮怎么进入

1. [新执行任务书](EXECUTION_GUIDE_CN.md)：工程修复、资源上限、样本、实验顺序和交付。
2. [理论问题与实验契约](THEORY_PLAN_CN.md)：从“误差大小”进一步研究“在限定修复操作下能消除哪部分误差”。
3. [新启动 prompt](START_PROMPT_CN.md)：复制分隔线后的全文给执行 chat。

用户本次要求继续给执行 chat 新指引，构成新一轮限定执行的授权。新执行者使用新的 run 和账本；原任务书/旧 NEXT_CHAT 的“本轮 C 不再启动”继续适用于旧 run，不阻止本次新授权。新预算也不回填旧账。
