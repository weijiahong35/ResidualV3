# R8测度与访问合同

D/J分别为E2独立DFlash raw与联合主体raw；P/D/F分别为target、独立草稿、联合主体加完整原头自采样。不能将F写为joint raw，也没有将joint头移植给D。所有来源共用同题同C；两主体对相同历史使用同一target标签，raw每C/i只算一份。自然重复槽保留频数。

全部采样输出从原BF16 logit位构造完整FP64概率缓冲，CPU PCG64 choice直接使用同一缓冲；选中q和SHA从实际缓冲采下。P/D/F、fit/eval、context/proposal/reject种子命名空间独立，SHA256元组前16字节big-endian uint128。显式seed表与NumPy/Python版本封存；context单轨迹最多65步供两个anchor。完整C/H的target缓存身份固定，KV只裁到公共前缀后逐token重放，重算前缀位不等即失败。

F保持原head.prepare/predict及Top256 BF16更新，内部FP32/BF16不擅改。开发full15原始rollout对照容差2e−5；首个单位置对照形状不一致造成的失败及重试保留。开发全部D/F eval合法15位置保存实际gen/target完整缓冲并独立CPU重放；fresh只在五位置存target全量位，其他位置保存实际标量/SHA，不补造全缓冲证明。

W=(C,i,前驱)，fit内至少两种完整H才构造u/o与M/2概率探针。只用对应来源fit，不用head移动方向。逐token u=max(0,min_fit(p−q))、o=max(0,min_fit(q−p))，U=sum u、O=sum o、M=min(U,O)；非零时a=(M/2)u/U−(M/2)o/O。两侧支持不相交且不越过任何fit需求，因此每条fit的TV减少M/2；这不是留出发现。完整词表保留，不clip、不额外归一化。M≤1e−12归零且保存raw M。fit改善M/2、概率守恒和TV步长为已知构造合同；M=0不等于其他补偿式共同修正不存在。两主体各自构造a_D/a_J，不将一个动作直接移到另一个raw上。动作不保证basis/dose可达、也不参与采样或验证。

动作构造与eval入口分开。独立exporter/auditor读过全量混合捕获；P构造器只接触P-fit、raw及白名单身份，D/F构造器只接触各自fit。全部动作封存后才运行eval入口。访问日志陈述实际代码路径，不声称同UID可防恶意OS越权。全SHA由独立审计入口完成，不将全文件hash称为未读eval字节。

四主集合用完整H new_to_P_fit，不同seed不是novel，D/F-fit是否见过不改变主资格；strict novel另列。三桶在原matched权重下精确分解，seen保证不能冒称泛化。主位置3/8/15，负控1/2；EOS当前位置有效，之前EOS/预算缺失不补齐。未匹配NA，零动作留在估计中。

自然槽→C/i→位置→anchor→prompt→域等权。5000次域内整题bootstrap，主98.75%、次95%；每端点每域至少6定义题和6非零动作题才准许正支持。统计条件于固定历史库与checkpoint。共同W标准化、同源fit重新识别均使用规定共同支持，不能替换四主量。

原D/F参考验证阈值min(1,p/q_actual)，独立reject uniform；reached只依赖此前接受且包含首拒位，G式有效路径指示不含本位接受事件。本轮没有计算普通路径overlap乘积、没有生成bonus/residual，也没有部署probe。原策略accepted数和reached局部G仅作描述，不报告TPS或native无偏。

