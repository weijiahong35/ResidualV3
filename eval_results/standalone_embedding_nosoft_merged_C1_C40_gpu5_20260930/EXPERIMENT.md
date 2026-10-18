# 去软 lag-1 两层：direction + dose 合并向量头

本实验仅训练 `M_no_soft_2layer_merged`。`N_no_soft_2layer` 使用既有 epoch3 checkpoint；不训练、不测试 no-soft 一层模型。全部在 GPU5；GPU4 的实验保持独立。

输入：当前 hidden → LN/Linear → 256；当前 u 的 q=softmax(u)，I=Top256(u)，Σ q[v]·(Embedding[v] Bᵀ) → LN/Linear → 256；实际前驱 token 的冻结原 embedding 行 → 行归一化/LN/Linear → 256。三路拼接768 → LN/Linear(768,256) → SwiGLU残差块×2（expansion3）→ LN → latent256。没有软lag-1。

合并头：x=Linear(256,512)(latent)，r=0.8x/√(1+Σx²)。v∈I 时 δ(v)=‖hidden‖·〈C[v],r〉，C[v]=原 LM-head[v] Bᵀ；候选外增量为零。z=u+δ。保留原幅度上限，但仅有一个可训练向量头，没有单位direction头或独立标量dose头。公共运行接口返回 `(r, ones)`，ones 是固定兼容乘数，不是可训练或预测的dose。范数计算FP32，r回到BF16；保留原BF16残差更新顺序。

这是有界联合向量参数化实验，并非移除幅度约束的实验，也没有突破固定输出基底。除输出头外，输入、两层主干、Top256、OUTPUT目标、训练数据、优化器参数不变。参数量由3,761,409变为3,761,152，只少257；不预设明显提速。向量权重随机小初始化，使其系数范数期望约0.01；初始函数不宣称与分离头相同。共享主干随机初始化逐张量与原方案fresh seed一致，旧checkpoint仅用于测试。

128条既有重生成prompt+answer ×3epoch=384步，每条≤1536锚点；目标vLLM和DFlash在线状态同在GPU5（0.55/0.40），hidden只经/dev/shm读取后unlink。BF16 forward，FP32 master/梯度累积。正式seed300931、smoke300932。CPU/CUDA输入干预、梯度和因果检查、初始真实native T0/T1 smoke、在线hidden对齐与因果检查、2步训练smoke通过后，正式重新随机初始化，保存并核对hash，禁止加载旧repair/optimizer/smoke权重。

固定I、原始u、前驱、hidden与输出基底，单独改变q必须改变delta；检查q/fusion/vector_head梯度。新runtime对旧split模型的初始化、logits与梯度精确一致，旧checkpoint严格加载。Graph/eager、实际采样前驱、processed verifier q继续验证。训练结束与旧两层384步的答案/锚点hash逐步比较。

评估顺序 RAW → N_no_soft_2layer → M_no_soft_2layer_merged；各T0/T1，正向一轮，不逆向。C1固定20prompt，C40固定300prompt，生成4096、总长8192、显存0.93、CUDA Graph；每套6个正式窗口。各套先同三模型smoke后正式评估。结果报告acceptance length与token/s及相对同套RAW的变化。

`run.log`为总调度；`M_no_soft_2layer_merged/training.log`为新头训练；`C1/C1.log`和`C40/C40.log`分别汇总三模型。根status.json=complete及RESULTS.md/results.json生成表示训练和两套测试均完成。failed时查其stage/log，不自动恢复或从失败权重续训。
