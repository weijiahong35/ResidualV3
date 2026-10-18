# 阅读边界：为四问服务

先读本地basis_reuse和basis_source的RESULTS_CN及本包researchwrite/01_research_canon。前者成本线索和学习未确认、后者来源阴性都保留。旧双专家头使用同一固定基底，不能当不同空间路由已经失败。无需为本包再展开一轮失配来源综述。

|一手来源|已核对内容|本轮借用|不能推出|
|---|---|---|---|
|[Towards Modular LLMs by Building and Reusing a Library of LoRAs](https://arxiv.org/html/2405.11157v1)，§3–4|作者先按适配器相似性建模块组，再用hidden等信息选模块；Arrow用LoRA算子的首右奇异向量|把“库是否各有所长”和“输入能否选择”分开|本残差基底已适合路由；聚类加路由本身新颖|
|[Sparse Subspace Clustering: Algorithm, Theory, and Applications](https://arxiv.org/abs/1203.1005)，摘要与方法定位|作者研究多个低维子空间之并的聚类，识别保证需要条件|区分一个空间内坐标变化和多个局部空间；本包采用固定轴向分组作检验|本残差满足SSC假设；本包是SSC复现或四个簇天然真实|
|[Revisiting Frank-Wolfe](https://proceedings.mlr.press/v28/jaggi13.html)，对偶间隙框架|作者用primal-dual间隙刻画凸优化误差|本包给容量解保留最优性误差界|标准间隙公式是新理论；FP64保护区间为严格数值证明|

数学推导见THEORY_CONTRACT_CN.md，是本包对既有工具的明确应用；科学假说是共享方向、现有空间限制、选择信息和任务偏好，均待运行。网页于2026-10-17核查。没有引用未核实的paper ID或用二手总结替代原文。
