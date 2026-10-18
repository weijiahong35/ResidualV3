# R7方法与测度

理论对象是大小模型失配；AR头是固定工具。仅分析同一联合主体q0→qh，不分摊独立DFlash到完整模型的正式AC/TPS。

## 固定概率子动作

同C/i/完整Z组内，u=max(0,min_fit(p−q0))，o=max(0,min_fit(q0−p))；d=qh−q0。a=min(d+,u)，b=min((−d)+,o)，m_raw=min(sum a,sum b)。m_raw≤1e−12时m=0,c=0；否则c=m a/sum a−m b/sum b。r=d−c，t=TV(qh,q0)，rho=m/t；t=0内部rho0、显示NA。rho不使用旧M_common。qc=q0+c，qr=qh−c，qu=q0+(1−rho)d。

完整尾部保留。五分布合法/同质量及fit Gc=m都是测量合同，不是新发现。I=Gh−Gc−Gr≤0，η=Gc+I为几何恒等式；没有第三项主检验或“解释百分比”。这些概率动作不保证实际Top256-logit/basis/dose可达，r不等于历史特异部分。

## 样本与自然频数

R6旧48开发，新36三域各12按固定SHA排序、种子洗牌选定，排除R1–R6、其他已知机制panel；完整messages/token规范化核验。机制新留出不代表从未正式评测/训练；未知预训练与语义重合不作排除保证。K64自然采样fit16/eval48，保留重复H频数，复制观测拒收。缺anchor/EOS/不碰撞不替换。

资格只在同C/i/精确Z内计算两种fit H。所有合格fit组都构造，不按eval匹配或效果删组。novel查同C/i所有fit H（包括其他Z）；不匹配与单H组无定义，不能填零。seen/全部合格匹配另外描述。i1/2为身份负控，同Z的i2不能有不同更早H，不强行断言不同Z的i2修正相同。

## 锁与访问

协议、代码、样本、seed、数值/统计在新采集前LOCK。独立身份worker/导出器允许读全部原始字节，分出fit/eval bank与严格白名单。构造worker拒读eval bank、原始混合效果记录及summary，只用fit target和q0/qh；全部动作数组与五分布SHA签名后，evalworker读取eval target。独立事后审计读取全部raw与旧效果作概率身份检查，不能把它描述为构造阶段盲测。Python访问hook+白名单+源码+SHA是过程审计，不是同UID安全隔离。

## 统计与固定失败类

η=Lr−Lh，κ=Lr−Lu共用核心novel集合。每C/i自然频数平均，再位置、anchor、题面、域等权；三域缺一主量NA。分域整题bootstrap2000，97.5%主CI（0.0125/0.9875），95%描述。每域至少6有定义题，η另需6含m>0题，κ另需6含TV(qr,qu)>1e−12题；只用于判定，不改人群。两项分别满足覆盖且CI下界>0才支持，均通过才联合表述。

损伤/正收益带1e−5；连续均值不阈值化。原头Gh<−1e−5时四类按预设Gc与η符号分配，频率按加权事件质量/全部或受损质量计算，不删无坏题后平均比例。每类固定排序前2不同H、总≤8，无例不补；背景同C/i/Z字典序第一不同eval H。token b=½(|p−q0|−|p−q|)，全词表正负及净值闭合；展示原头正负各≤8token，完整数组保留。

## 资源与解释边界

8 GPU小时、24 CPU小时、700 GiB新增；同时1卡，允许共卡，自有32GiB，阶段峰值外4GiB余量。4数值库import前1，Torch intra/inter1，worker≤4/合计计算线程≤8；GPU线程采样包含驱动/IO且不构成全时刻生命周期证书。所有启动失败及共卡wall计账，预算是上限。

没有拟合预测器、头/主体训练、LP/见证、native/RNG/TPS或剂量网格。本轮既不证明一般理论，也不撤销任何旧警报。报告工具位于evidence_tools，不参与科学LOCK；全run静态SHA绑定其实际版本。

访问日志的时间范围：动作worker先核验科学LOCK的静态SHA，再安装Python open hook；静态核验会读取旧dev摘要的原始字节作SHA。LOCK不含fresh采集/银行文件，因此这一步不会读取fresh eval target或效果；hook记录的是随后数据构造范围。独立审计另核全文件SHA。以上不是声称进程从启动起从未读过任何旧效果字节。

采样精度沿冻结R6参考实现：target原始BF16 logits转FP32 softmax后由torch.multinomial自然采样；TV标签/五分布比较用相同保存BF16 logits的稳定FP64重构。没有把这两个有限精度步骤写成逐位相同的随机分布，也不从它们推出本地native理想概率规律。

旧48题开发时另保存6个同规则工程账本，以核验案例导出；它们不展示为新题证据，不与本轮正式最多8例或新题失败频率混合。运行版本记录中的旧R6置乱说明已单列更正于logs/RUNTIME_SCOPE_CORRECTION.json；R7没有执行历史置乱实验。
