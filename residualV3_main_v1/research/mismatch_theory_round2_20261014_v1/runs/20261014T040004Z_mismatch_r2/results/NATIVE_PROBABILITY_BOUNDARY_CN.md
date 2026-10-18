# native 概率解释的保留边界

真实 token、同轮处理后 p/q、轮次与请求身份、原/优化头 parity 及整数输出对账已完成机械核验。**理想独立 uniform 模型的连续存活期望尚未被当前 runtime 建立。** 这两项结论必须同时保留。

原工程6题×3臂的 [NATIVE_BRIDGE_AUDIT.json](../analysis/NATIVE_BRIDGE_AUDIT.json) 保持 `passed=false`。唯一未通过项是 B2 的每请求“实测已发 accepted proposal 总数−工作 G∏b 总和”：均值+8.438779，95% prompt-bootstrap 区间[3.705637,13.171921]。没有重采这6题、更换seed或删除此警报；后续36题/新48题的区间不撤销它。

独立CPU源码审计在1758/1758轮重构 first-reject 接受数，与真实记录完全相同。每轮 seed、proposal位置、处理后 token p/q、阈值和输出身份有可复核证据。proposal Gumbel 的第一层 key 与 verifier uniform 使用同一 Philox word；本次未发现独立 stream salt。词表噪声随后再次按 token ID 哈希。这个源码事实不独自证明错误、不独自解释 B2 偏差，也不保证给定完整 proposal 后的接受 uniform 条件独立。

工作 prefix 方差与小样本区间敏感性分析表明原警报仍有统计不确定性；它们不能替换原检验或证明概率定律。原始 GPU position tensor 未持久化；审计位置由已核源码与轮初前缀推导，1758次计数吻合支持该映射，但不是 bitwise buffer 证明。实际 p/q 保存的是 sampler 处理后 logits 的 torch FP32 数值重构；Triton LSE/log 算术的逐位身份未建立。

经 [NATIVE_BRIDGE_REVIEWED_AUDIT.json](../analysis/NATIVE_BRIDGE_REVIEWED_AUDIT.json) 明确限缩批准的 gate 只覆盖机械测量：完整 HTTP/验证记录、真实 token/slot/轮次绑定、同 H 原/优化头 parity、raw q 身份、S/U 不变量及 post-stop 整数计数。它允许检验真实 reached 的局部 TV 与实际接受/输出，不把上述理想期望规律设为已通过事实，也不把桥接成功当理论创新。

G 只由 proposal 路径长度、先前 EOS 和轮初输出预算构造，不含已实现接受事件。`b_i=min(1,p(token_i)/q(token_i))` 对实际 proposal token 计算。只有在适用随机性假设下，`G_k∏b_i` 才是对应的条件存活工作期望；普通 draft 路径的 overlap 乘积和 teacher-prefix 乘积均不能代替它。每请求概率残差用总数诊断，以避免随机验证轮数分母引入额外偏差。每prompt一次seed的 bootstrap也不是同题多seed的Monte Carlo误差估计。

因此，本轮可支持的 native 经验结论限于记录完整的局部距离与实际已发后果。不能由结果区间覆盖0宣称采样无偏、理想连续存活已证实、通用期望长度恒等式成立，或报告插桩服务 TPS。详细证据和限制见 [NATIVE_RNG_SOURCE_REVIEW_CN.md](../analysis/NATIVE_RNG_SOURCE_REVIEW_CN.md) 与 [MEASURES_AND_IDENTITIES_CN.md](MEASURES_AND_IDENTITIES_CN.md)。
