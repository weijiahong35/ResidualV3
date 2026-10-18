# R5决策记录

- 明确授权新轮；旧R1–R4、安装环境和冻结模型只读。
- 科学范围严格执行E1–E4，禁止训练/新损失/LP/新见证/native/RNG/TPS。
- 优先流式复用R4完整概率；旧题只作开发/描述。
- 本轮新增磁盘上限700 GiB，GPU8h、CPU24h，最多1卡。
- 不做正式文献综述；来源核验限既有原文所列段落。

- 锁前token核验发现当前tokenizer默认返回BatchEncoding；原正式prepare显式return_dict=False。按原接口修复返回类型，未改tokenizer/环境/题目或模型；保留失败日志。

- 旧dev读取合同定位scalar shape问题：np.ascontiguousarray把0D hidden_norm扩为[1]，与原torch shape()签名不同。保留原数组形状后再取连续bytes；原数据/数值/协议不变，失败空目录保存在logs/partial_dev_scalar_reader_revision0。

- 全18旧dev可定义E2/E3，Alpaca5/6有定义；旧dev不作确认。约87%置乱配对步长为0，保留零值，不换K/位置/阈值/探针。新36题按原门槛每域≥6评价。
- 全18 dev重构误差≤4.45e-16，KL身份≤1.34e-14，fit身份≤5e-16；固定1e-10数值合同、1e-5损伤符号带和M≤1e-12零带，不上调阈值。
- 新fresh summary锁前添加严格入口与完成分片/来源SHA核验；额外E3双方raw参照损伤只读原冻结fit产物，不选新方向。
