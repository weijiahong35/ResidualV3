# 发布内容与压缩归档说明

当前 Git 主分支已展开 V3 训练/推理源码、必要依赖、配置/basis 和重要报告。GitHub 可直接浏览，克隆后无需先解压项目包。

既有 v2.0-archive Release 仍保留约65MiB项目压缩包和E1/E2四份gzip最终权重。压缩包内容及SHA256不变；PACKAGE_MANIFEST.json描述该压缩归档，Git中的README、.gitignore和导航说明已单独适配。源码来源与变化范围见SOURCE_PUBLICATION.json。

权重解压后按WEIGHTS_MANIFEST.json恢复至restored_path。仓库内scripts/restore_release_weights.py可从公开Release下载、校验并恢复。target基础模型、预训练初始化权重、运行环境、完整训练/概率大数据及恢复断点不附带。现有绝对路径需要按部署环境调整，本次没有执行训练或推理。

退役代码只在原服务器保留校验归档。PROJECT_MAP_CN.md中的服务器历史记录不表示全部大数据已包含。

少数历史题面中的凭据已脱敏；原协议内部SHA/token ID仍属于当时原始输入，不能据脱敏文本声明原始输入身份一致。旧概率警报和负结果保留。旧启动提示、NEXT_CHAT、PID和预约记录只供理解历史，不是新实验授权。
