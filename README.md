# ResidualV3

ResidualV3 当前训练、推理与重要实验文档的压缩发布入口，附E1/E2最终权重。
源码以压缩包发布，仓库不展开服务器全部历史文件，也不包含自动CI或Dependabot配置。

## 下载

[打开 Release](https://github.com/weijiahong35/ResidualV3/releases/tag/v2.0-archive)。

| 文件 | 内容 |
|---|---|
| ResidualV3_project.tar.gz | 当前代码及实际依赖、配置/basis、正式实验与机制研究的重要报告/表格 |
| body_epoch_01.safetensors.gz | E1独立训练DFlash主体 |
| joint_epoch_01.safetensors.gz | E1联合训练主体＋原修复头 |
| body_epoch_02.safetensors.gz | E2独立训练DFlash主体 |
| joint_epoch_02.safetensors.gz | E2联合训练主体＋原修复头 |

四份权重共用原始身份；body不是从joint中拆出的同一主体。压缩与解压后的SHA256分别记录于ASSETS.json和项目包内WEIGHTS_MANIFEST.json。
下载上述文件到同一个目录后：

```bash
sha256sum -c SHA256SUMS
tar -xzf ResidualV3_project.tar.gz
python ResidualV3/scripts/restore_release_weights.py --archives .
```

只需E2权重时，下载项目包与两份epoch_02文件，恢复命令追加`--epoch E2`。
也可在解压后的项目目录用`python scripts/restore_release_weights.py --repo weijiahong35/ResidualV3 --epoch E2`直接下载恢复。

## 阅读与运行边界

解压后先读PACKAGE_README_CN.md、PROJECT_MAP_CN.md和GROUP_MEETING_20261013_CN.md。
保留重要正式结果、负结果、不确定性及既有概率警报；发布没有新增训练或推理成绩。
现有代码保留原服务器路径，需要按运行环境配置。target基础模型、预训练初始化权重、环境、完整训练恢复断点和概率大数据不附带。
个别历史题面已脱敏；原协议中的token ID/SHA仍指向原始输入。
旧START_PROMPT、NEXT_CHAT、PID和预约记录不代表当前任务授权。

本项目基于Speculators/DFlash，许可证见LICENSE。Release编号仅是发布格式，不是新增模型训练轮次。
