# ResidualV3

五层 DFlash 草稿主体与三层 NOFILL 自回归修复头的训练、推理和实验项目。**代码与重要实验文档已直接放在本仓库，可在线浏览或通过 Git 克隆。** E1/E2 大权重放在 [Release](https://github.com/weijiahong35/ResidualV3/releases/tag/v2.0-archive)，项目压缩包也保留供整包下载。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 方法、训练与正式 AC/TPS 汇总 | [组会报告](GROUP_MEETING_20261013_CN.md) |
| E1 训练 | [训练代码](residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/) · [主体与头](residualV3-nonthing/code/full_joint_body_shared_teacher_1epoch_20261008_v1/joint_model.py) |
| E2 训练 | [训练代码](residualV3-nonthing/code/full_joint_body_epoch2_aux_compare_20261010_v4/) · [主体与头](residualV3-nonthing/code/full_joint_body_epoch2_aux_compare_20261010_v4/joint_model.py) |
| 推理与正式评测 | [冻结实现](residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/) · [原修复头](residualV3-nonthing/code/epoch2_dspark_code_chat_C1_C32_20261013_v1/native_runtime/repair/model.py) |
| 六数据集正式结果 | [48 窗 / 24 配对报告](residualV3-nonthing/results/epoch2_dspark_code_chat_C1_C32_20261013_v1/RESULTS_CN.md) |
| E1/E2 同实现对照 | [结果报告](residualV3-nonthing/results/epoch1_epoch2_native_compare_gsm8k_20261010_v1/RESULTS_CN.md) |
| 机制与基底研究 | [研究目录](residualV3_main_v1/research/) · [项目导航](PROJECT_MAP_CN.md) |

## 获取代码和权重

```bash
git clone --depth 1 https://github.com/weijiahong35/ResidualV3.git
cd ResidualV3
# 只恢复 E2；需要两轮全部权重时去掉 --epoch E2
python scripts/restore_release_weights.py --repo weijiahong35/ResidualV3 --epoch E2
```

[Release 下载页](https://github.com/weijiahong35/ResidualV3/releases/tag/v2.0-archive)提供 E1/E2 各自的 body 与 joint 四份最终权重，附压缩和解压后 SHA256。body 是独立训练的 DFlash 主体，joint 是联合训练主体与原修复头。基础 target 模型、预训练初始化权重及运行环境需另行配置；现有实现中的服务器绝对路径也需按部署环境调整。

## 发布范围与阅读说明

当前训练/推理代码、必要依赖与配置/basis、重要报告和表格已展开。完整训练数据、概率银行、优化器恢复断点、退役研究源码及临时模型不在仓库。大权重继续从 Release 下载。

原始正结果、负结果、不确定性和概率警报均保留。部分历史题面中的凭据已脱敏；原协议的 SHA/token ID 指向原始输入。旧启动提示、NEXT_CHAT 和进程快照只是历史记录，不代表当前授权。AC/AL 指接受长度，不是答题准确率。

`PACKAGE_MANIFEST.json` 记录既有压缩包的文件身份；Git 首页等导航文字已单独更新，见 [发布说明](PACKAGE_README_CN.md)。本次仅调整展示方式，没有新增训练、推理或科学结论。GitHub Actions 保持关闭。

本项目基于 Speculators/DFlash，许可证见 [LICENSE](LICENSE)。
