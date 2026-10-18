# CPU代理头的唯一合同

本文件只固定受控代理网络，不改变已训练原头，不声称与native路径逐位一致。

## 源码身份

仓库相对路径：

- `residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/repair/model.py`，SHA256 `a350a84243dda1fd7b1e298fd5943c0ca417150feed41673ef33b040c91f7bf8`。
- `residualV3-nonthing/code/epoch2_nofill_vs_dflash_gsm8k_20261011_v1/native_runtime/runtime/components.py`，SHA256 `8483bb55ca0e03561acdb53d99c0bb8864d7a6d8fb2d30500b1349d7fe175cfd`。

执行时核验后，只将DualFeedbackRepair的网络构造/predict、SwiGLUBlock和normalized_rows复制到新run `code/reference_head.py`，归档新SHA。不导入旧rollout/controller；prepare改为下列CPU FP32固定输入计算，不调用hidden_norm_bf16或bf16_logit_update。

## 网络与输入

初始化必须保留原构造顺序：hidden_proj、previous_proj、q_proj、fusion、三个blocks、output_norm、direction_head、dose_head。

- hidden_proj：LayerNorm(4096)→Linear(4096,256)，输入存档J_hidden解码后FP32。
- previous_proj：先将对应前驱的静态embedding行x变为x/max(||x||₂,1e−6)，再LayerNorm(4096)→Linear(4096,256)。前驱来自该条存档记录；不进行自回归rollout。
- q_proj：LayerNorm(512)→Linear(512,256)，输入固定`Σ_{v∈rawTop256} q_v(E_emb,v B_Gᵀ)`，q为完整分布概率，不在候选内重归一化。input G跨三臂相同。
- fusion：依hidden、previous、q顺序拼接，LayerNorm(768)→Linear(768,256)。无soft_previous。
- 三个SwiGLUBlock：LayerNorm(256)→Linear(256,1536)，按原chunk顺序拆a,g各768，计算silu(g)*a→Linear(768,256)，加回输入。dropout=0；down.bias置0。
- output_norm：LayerNorm(256)；方向Linear(256,512)输出u，剂量Linear(256,1)输出z。

所有LayerNorm使用eps=1e−5、elementwise_affine=True、bias=True；所有Linear含bias。网络参数与前向FP32；KL归约FP64。不安装新torch，记录现有版本并使用deterministic algorithms与单线程。

## 初始化与系数

每个fold和训练seed构造前`torch.manual_seed(seed)`；fold不改变seed。其余Linear/LayerNorm依锁定的现有torch默认初始化，按上述顺序构建。随后direction_head.weight和dose_head.weight各normal_(std=.01)，direction_head.bias=0，dose_head.bias=log(.01/(.8−.01))，block down.bias=0。保存一份state_dict，E/G/R同fold、seed加载同一份；不得让基底构造消费torch初始化RNG。

`hn=max(||h||₂,1e−6)`、`direction=u/max(||u||₂,1e−6)`、`dose=.8*sigmoid(z)`、`a=dose*hn*direction`，范数按FP32计算。u=0时a=0。a为用于候选映射A的完整系数，不得再次乘dose或hn。无dropout随机性。

真实原代码的hidden_norm/BF16日志更新被有意替换为这个FP32代理。FP32可饱和/舍入，必须报告实际范数超出理论预算的量；超过相对1e−6且非可定位舍入问题则技术失败，不默默clip改变动作。主评价将a转FP64并使用canonical A，不把FP32训练乘法当精确容量合同。

## 随机流与初始化审计

训练seed为2026101692/1693/1694，R基底seed依次2026101696/1697/1698；R由NumPy Generator(PCG64(seed))生成标准高斯，不消耗torch RNG。

shuffle输入字节唯一为：`json.dumps([2026101691,fold,seed,epoch],separators=(',',':'),ensure_ascii=True).encode('utf-8')`。取SHA256 digest前16字节，用`int.from_bytes(...,'big',signed=False)`初始化PCG64，再对冻结训练行序做permutation；fold=0/1/2，epoch=0…11。训练行初序按完整group SHA、anchor数值、position数值、slot数值排序（仅P-source）。保存每epoch置乱SHA。

epoch0需报告实际初始分布扰动。参数初始化相同不代表三个输出映射的初始函数相同，不能据配对seed宣称排除了初始化解释。
