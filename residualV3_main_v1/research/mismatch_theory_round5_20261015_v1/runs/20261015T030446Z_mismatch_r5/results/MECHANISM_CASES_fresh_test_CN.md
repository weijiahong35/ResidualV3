# R5真实机制案例

数据身份：fresh_test。固定字典序选例，类别可重叠；不按最大效果选。旧题描述与新题确认分开。完整词表账本在各状态token_contributions_path；表仅展示预定正/负各最多8个token。

## common_eval_gain

### 1. alpaca:1473459bd85d08300fe2 / anchor1 / i15

题面：Name a sport which is played indoors.

完整C：

```text
<|im_start|>user
Name a sport which is played indoors.<|im_end|>
<|im_start|>assistant
<think>

</think>

One
```

fit不同H数2，eval状态5；M=0.967841879，固定a=M/2；eval平均改善=0.404468751，损伤比例=0。探针可改完整词表，不能当真实头合法动作。

H (eval, branch19)：

```text
 example of a sport that is played indoors is **tennis**.
```

共同前驱='.'，rawTV=0.989868347，headTV=0.947540122，head gain=0.042328225；尾部闭式剩余=0.312149479。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" It" (1084)|0.678617|0.00121706|0.0448749|0.0218289|toward_not_crossed|True|
|positive|"**" (334)|8.12936e-08|0.0429028|0.00223419|0.0203343|toward_not_crossed|True|
|positive|" a" (264)|8.84027e-09|0.0313883|6.33791e-05|0.0156625|toward_not_crossed|True|
|positive|"." (13)|4.75895e-10|0.0244453|9.7195e-06|0.0122178|toward_not_crossed|True|
|positive|".\n\n" (382)|1.91184e-09|0.0215729|9.81635e-05|0.0107373|toward_not_crossed|True|
|positive|" the" (279)|9.91851e-10|0.0202658|2.90171e-05|0.0101184|toward_not_crossed|True|
|positive|" of" (315)|4.8339e-10|0.0202658|7.1818e-05|0.010097|toward_not_crossed|True|
|positive|"," (11)|1.48894e-09|0.0178845|1.87348e-05|0.0089329|toward_not_crossed|True|
|negative|" **" (3070)|5.6429e-07|0.0101903|0.453221|-0.221515|reverse_or_depart_equal|True|
|negative|" The" (576)|0.000481937|0.00188502|0.0448749|-0.021495|reverse_or_depart_equal|True|
|negative|"  \n" (2303)|0.000129711|0.00310786|0.0421561|-0.0195241|reverse_or_depart_equal|True|
|negative|" Among" (21658)|9.90359e-07|0.000890418|0.0328312|-0.0159704|reverse_or_depart_equal|True|
|negative|" \n\n" (4710)|0.00168212|0.00227376|0.025569|-0.0116476|reverse_or_depart_equal|True|
|negative|" A" (362)|3.27962e-05|0.00242041|0.025569|-0.0115743|reverse_or_depart_equal|True|
|negative|"  \n\n" (18611)|0.000257962|0.000574897|0.0199131|-0.00966912|reverse_or_depart_equal|True|
|negative|" Here" (5692)|2.37575e-06|0.000476607|0.0155084|-0.00751588|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_1473459bd85d08300fe2/anchor1/i15_TOKEN_B.npy`，行7。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:2e82b22f416b041405c8 / anchor1 / i8

题面：Calculate the cube root of 125.

完整C：

```text
<|im_start|>user
Calculate the cube root of 125.<|im_end|>
<|im_start|>assistant
<think>

</think>

To
```

fit不同H数2，eval状态11；M=0.274481326，固定a=M/2；eval平均改善=0.137240663，损伤比例=0。探针可改完整词表，不能当真实头合法动作。

H (eval, branch16)：

```text
 calculate the **cube root of 
```

共同前驱=' '，rawTV=0.274481326，headTV=0.000043107，head gain=0.274438220；尾部闭式剩余=0.000000000。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"1" (16)|1|0.725519|0.999957|0.137219|toward_not_crossed|True|
|positive|"2" (17)|1.54281e-13|0.18344|5.42201e-06|0.0917172|toward_not_crossed|True|
|positive|" " (220)|3.99096e-18|0.0525564|3.61728e-09|0.0262782|toward_not_crossed|True|
|positive|" **" (3070)|2.18227e-19|0.0248259|7.19382e-09|0.0124129|toward_not_crossed|True|
|positive|" of" (315)|1.71186e-23|0.00335982|1.03637e-09|0.00167991|toward_not_crossed|True|
|positive|" the" (279)|4.6338e-25|0.00179838|9.73578e-10|0.00089919|toward_not_crossed|True|
|positive|" $" (400)|1.45371e-19|0.00140058|2.48612e-09|0.000700289|toward_not_crossed|True|
|positive|"**" (334)|7.6812e-15|0.000962604|1.1253e-07|0.000481246|toward_not_crossed|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_2e82b22f416b041405c8/anchor1/i8_TOKEN_B.npy`，行16。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## fit_gain_eval_damage

### 1. alpaca:a8555c17eb458137e4bf / anchor65 / i8

题面：Describe a place where you feel most at peace

完整C：

```text
<|im_start|>user
Describe a place where you feel most at peace<|im_end|>
<|im_start|>assistant
<think>

</think>

The place where I feel most at peace is a quiet corner of my home—my small, sunlit reading nook. It’s tucked away in a corner of the living room, surrounded by floor-to-ceiling bookshelves that stretch from the floor to the ceiling, filled with stories, poetry, and philosophy. A
```

fit不同H数2，eval状态2；M=0.75407755，固定a=M/2；eval平均改善=0.154468114，损伤比例=0.5。探针可改完整词表，不能当真实头合法动作。

H (eval, branch23)：

```text
 large window on one side lets in
```

共同前驱=' in'，rawTV=0.784837320，headTV=0.871600724，head gain=-0.086763404；尾部闭式剩余=0.081607801。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"," (11)|7.64791e-11|0.0645662|0.00244358|0.0310613|toward_not_crossed|True|
|positive|" and" (323)|3.88393e-10|0.0269152|0.00122871|0.0128432|toward_not_crossed|True|
|positive|" of" (315)|5.9562e-11|0.0237526|1.36497e-05|0.0118695|toward_not_crossed|True|
|positive|" with" (448)|2.33248e-11|0.0135338|0.0020258|0.00575401|toward_not_crossed|True|
|positive|" glow" (35966)|5.20447e-12|0.0127138|0.00178776|0.00546304|toward_not_crossed|True|
|positive|" from" (504)|1.17284e-11|0.0105401|0.000114288|0.00521293|toward_not_crossed|True|
|positive|" is" (374)|5.20447e-12|0.0105401|0.000121659|0.00520924|toward_not_crossed|True|
|positive|" window" (3241)|1.52097e-10|0.00990155|0.00139231|0.00425462|toward_not_crossed|True|
|negative|" the" (279)|0.0420455|0.0829047|0.561698|-0.239397|reverse_or_depart_equal|True|
|negative|" a" (264)|0.00730641|0.0502842|0.142019|-0.0458676|reverse_or_depart_equal|True|
|negative|" soft" (8413)|0.844506|0.154886|0.0760175|-0.0394344|reverse_or_depart_equal|True|
|negative|" my" (847)|1.34225e-10|0.00564173|0.0359081|-0.0151332|reverse_or_depart_equal|True|
|negative|" warm" (8205)|0.00938161|0.00529992|0.00148211|-0.00190891|reverse_or_depart_equal|True|
|negative|" gentle" (21700)|0.00730641|0.00342188|0.000481169|-0.00147036|reverse_or_depart_equal|True|
|negative|" one" (825)|6.74926e-11|0.00125884|0.00402878|-0.00138497|reverse_or_depart_equal|True|
|negative|" your" (697)|8.66622e-11|0.00104362|0.00333998|-0.00114818|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_a8555c17eb458137e4bf/anchor65/i8_TOKEN_B.npy`，行23。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:a8555c17eb458137e4bf / anchor65 / i8

题面：Describe a place where you feel most at peace

完整C：

```text
<|im_start|>user
Describe a place where you feel most at peace<|im_end|>
<|im_start|>assistant
<think>

</think>

The place where I feel most at peace is a quiet corner of my home—my small, sunlit reading nook. It’s tucked away in a corner of the living room, surrounded by floor-to-ceiling bookshelves that stretch from the floor to the ceiling, filled with stories, poetry, and philosophy. A
```

fit不同H数2，eval状态1；M=0.010595638，固定a=M/2；eval平均改善=-7.19536747e-05，损伤比例=1。探针可改完整词表，不能当真实头合法动作。

H (eval, branch19)：

```text
 velvet armchair sits by large,
```

共同前驱=','，rawTV=0.976801498，headTV=0.980687545，head gain=-0.003886048；尾部闭式剩余=0.336314192。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"," (11)|1.05397e-10|0.0645662|6.0497e-05|0.0322529|toward_not_crossed|True|
|positive|" soft" (8413)|0.00246753|0.154886|0.102754|0.0260661|toward_not_crossed|True|
|positive|" of" (315)|2.66084e-08|0.0237526|0.00137691|0.0111878|toward_not_crossed|True|
|positive|" light" (3100)|2.25009e-06|0.0209616|0.00480587|0.00807786|toward_not_crossed|True|
|positive|" that" (429)|1.2062e-09|0.0153358|0.000888997|0.00722341|toward_not_crossed|True|
|positive|" the" (279)|2.18415e-09|0.0829047|0.0706218|0.00614145|toward_not_crossed|True|
|positive|" window" (3241)|0.000801088|0.00990155|0.000650404|0.00447489|crossed_still_gain|True|
|positive|" is" (374)|1.1092e-08|0.0105401|0.00188201|0.00432907|toward_not_crossed|True|
|negative|" and" (323)|2.37158e-07|0.0269152|0.169413|-0.0712488|reverse_or_depart_equal|True|
|negative|" where" (1380)|3.24645e-10|0.0025035|0.0215384|-0.00951746|reverse_or_depart_equal|True|
|negative|" casting" (24172)|3.3495e-10|0.00118257|0.0167741|-0.00779579|reverse_or_depart_equal|True|
|negative|" draped" (89880)|9.56763e-05|0.000558607|0.0139063|-0.00667382|reverse_or_depart_equal|True|
|negative|" lined" (31293)|5.2395e-09|0.000865188|0.0115287|-0.00533175|reverse_or_depart_equal|True|
|negative|" it" (432)|4.57822e-10|0.0025035|0.0130637|-0.00528011|reverse_or_depart_equal|True|
|negative|" my" (847)|4.39359e-11|0.00564173|0.0157578|-0.00505806|reverse_or_depart_equal|True|
|negative|" a" (264)|1.042e-08|0.0502842|0.0585475|-0.00413164|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_a8555c17eb458137e4bf/anchor65/i8_TOKEN_B.npy`，行19。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## actual_head_mixed_sameZ

### 1. alpaca:2e82b22f416b041405c8 / anchor1 / i3

题面：Calculate the cube root of 125.

完整C：

```text
<|im_start|>user
Calculate the cube root of 125.<|im_end|>
<|im_start|>assistant
<think>

</think>

To
```

H (eval, branch16)：

```text
 calculate the
```

共同前驱=' the'，rawTV=0.104512112，headTV=0.025555617，head gain=0.078956495；尾部闭式剩余=0.000000001。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" cube" (23739)|0.268941|0.164429|0.294097|0.0396781|crossed_still_gain|True|
|positive|" **" (3070)|0.731057|0.835038|0.705501|0.0392127|crossed_still_gain|True|
|positive|" the" (279)|5.72786e-13|7.08264e-05|7.07629e-07|3.50594e-05|toward_not_crossed|True|
|positive|" $" (400)|1.21259e-12|1.58035e-05|3.37591e-06|6.21381e-06|toward_not_crossed|True|
|positive|" three" (2326)|5.72786e-13|2.02921e-05|8.09838e-06|6.09687e-06|toward_not_crossed|True|
|positive|" " (220)|6.97796e-12|1.08616e-05|1.69751e-06|4.58204e-06|toward_not_crossed|True|
|positive|"cube" (45411)|3.61468e-09|9.58532e-06|2.97923e-06|3.30305e-06|toward_not_crossed|True|
|positive|" calculate" (11047)|1.06114e-16|7.94651e-06|1.40729e-06|3.26961e-06|toward_not_crossed|True|
|negative|" volume" (8123)|9.44365e-13|1.39466e-05|3.62944e-05|-1.11739e-05|reverse_or_depart_equal|True|
|negative|" root" (3704)|7.82906e-13|3.3456e-05|5.28081e-05|-9.67601e-06|reverse_or_depart_equal|True|
|negative|" square" (9334)|6.28136e-10|2.95249e-05|4.1127e-05|-5.80105e-06|reverse_or_depart_equal|True|
|negative|" value" (897)|1.76431e-12|2.02921e-05|2.82661e-05|-3.987e-06|reverse_or_depart_equal|True|
|negative|" cubic" (40899)|8.67116e-09|2.2994e-05|2.82661e-05|-2.63607e-06|reverse_or_depart_equal|True|
|negative|" shape" (6083)|7.67534e-15|1.46996e-06|3.8254e-06|-1.17772e-06|reverse_or_depart_equal|True|
|negative|" edge" (6821)|1.84122e-14|5.81379e-06|8.09838e-06|-1.1423e-06|reverse_or_depart_equal|True|
|negative|" quotient" (74762)|2.22093e-14|2.92336e-06|4.91192e-06|-9.94279e-07|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_2e82b22f416b041405c8/anchor1/i3_TOKEN_B.npy`，行16。

H (fit, branch11)：

```text
 find the
```

共同前驱=' the'，rawTV=0.089097441，headTV=0.218638433，head gain=-0.129540992；尾部闭式剩余=0.000000002。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" the" (279)|6.0625e-12|7.08264e-05|7.07629e-07|3.50594e-05|toward_not_crossed|True|
|positive|" $" (400)|3.45431e-12|1.58035e-05|3.37591e-06|6.21381e-06|toward_not_crossed|True|
|positive|" three" (2326)|3.67709e-12|2.02921e-05|8.09838e-06|6.09687e-06|toward_not_crossed|True|
|positive|" " (220)|3.95325e-11|1.08616e-05|1.69751e-06|4.58204e-06|toward_not_crossed|True|
|positive|"cube" (45411)|4.03242e-09|9.58532e-06|2.97923e-06|3.30305e-06|toward_not_crossed|True|
|positive|" calculate" (11047)|3.38648e-14|7.94651e-06|1.40729e-06|3.26961e-06|toward_not_crossed|True|
|positive|" of" (315)|8.2047e-13|6.58789e-06|1.22967e-07|3.23246e-06|toward_not_crossed|True|
|positive|" compute" (12564)|2.95908e-15|9.58532e-06|3.8254e-06|2.87996e-06|toward_not_crossed|True|
|negative|" cube" (23739)|0.0758575|0.164429|0.294097|-0.064834|reverse_or_depart_equal|True|
|negative|" **" (3070)|0.924133|0.835038|0.705501|-0.0647683|reverse_or_depart_equal|True|
|negative|" volume" (8123)|6.86971e-12|1.39466e-05|3.62944e-05|-1.11739e-05|reverse_or_depart_equal|True|
|negative|" root" (3704)|5.02599e-12|3.3456e-05|5.28081e-05|-9.67601e-06|reverse_or_depart_equal|True|
|negative|" square" (9334)|2.77144e-09|2.95249e-05|4.1127e-05|-5.80105e-06|reverse_or_depart_equal|True|
|negative|" value" (897)|8.36902e-11|2.02921e-05|2.82661e-05|-3.987e-06|reverse_or_depart_equal|True|
|negative|" cubic" (40899)|2.04783e-08|2.2994e-05|2.82661e-05|-2.63607e-06|reverse_or_depart_equal|True|
|negative|" *" (353)|9.36155e-06|7.01277e-06|2.62916e-06|-2.1918e-06|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_2e82b22f416b041405c8/anchor1/i3_TOKEN_B.npy`，行11。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:61191b0966779357a4c8 / anchor65 / i15

题面：Compare the diets of a vegan and a carnivore.

完整C：

```text
<|im_start|>user
Compare the diets of a vegan and a carnivore.<|im_end|>
<|im_start|>assistant
<think>

</think>

The diets of a **vegan** and a **carnivore** differ significantly in terms of food sources, nutritional composition, and environmental impact. Here's a comprehensive comparison:

---

### **1. Food Sources**
| Aspect               | **Vegan Diet**                          | **Carnivore Diet**                     |

```

H (fit, branch1)：

```text
|----------------------|-----------------------------------------|----------------------------------------|
| **Primary Sources
```

共同前驱=' Sources'，rawTV=0.389020643，headTV=0.016686687，head gain=0.372333956；尾部闭式剩余=0.000000173。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"**" (334)|0.99998|0.61096|0.983294|0.186167|toward_not_crossed|True|
|positive|" \|" (760)|7.05274e-11|0.0390573|0.000616256|0.0192205|toward_not_crossed|True|
|positive|" (" (320)|3.65341e-08|0.0209059|0.00312961|0.00888814|toward_not_crossed|True|
|positive|" Sources" (47032)|1.98097e-13|0.0143684|9.96086e-06|0.00717922|toward_not_crossed|True|
|positive|"          " (1843)|1.47833e-11|0.0134979|0.000791289|0.00635328|toward_not_crossed|True|
|positive|" Source" (8748)|1.87949e-12|0.0119118|7.75752e-06|0.00595203|toward_not_crossed|True|
|positive|"      " (981)|5.21131e-10|0.0111901|0.00101603|0.00508704|toward_not_crossed|True|
|positive|"        " (260)|5.49268e-11|0.00871487|0.000373778|0.00417055|toward_not_crossed|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_61191b0966779357a4c8/anchor65/i15_TOKEN_B.npy`，行1。

H (eval, branch30)：

```text
|----------------------|-----------------------------------------|----------------------------------------|
| Primary Food Sources
```

共同前驱=' Sources'，rawTV=0.956975715，headTV=0.999083708，head gain=-0.042107993；尾部闭式剩余=0.000000466。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" (" (320)|1.71863e-08|0.0209059|0.00312961|0.00888814|toward_not_crossed|True|
|positive|" Sources" (47032)|2.96152e-13|0.0143684|9.96086e-06|0.00717922|toward_not_crossed|True|
|positive|"          " (1843)|3.45196e-07|0.0134979|0.000791289|0.00635328|toward_not_crossed|True|
|positive|" Source" (8748)|2.10003e-13|0.0119118|7.75752e-06|0.00595203|toward_not_crossed|True|
|positive|"      " (981)|1.00881e-05|0.0111901|0.00101603|0.00508704|toward_not_crossed|True|
|positive|"        " (260)|9.38339e-07|0.00871487|0.000373778|0.00417055|toward_not_crossed|True|
|positive|" of" (315)|4.00195e-11|0.00871487|0.00147832|0.00361827|toward_not_crossed|True|
|positive|"             " (1060)|7.70236e-08|0.00769085|0.000479941|0.00360545|toward_not_crossed|True|
|negative|"**" (334)|2.05261e-09|0.61096|0.983294|-0.186167|reverse_or_depart_equal|True|
|negative|" \|" (760)|0.995854|0.0390573|0.000616256|-0.0192205|reverse_or_depart_equal|True|
|negative|" " (220)|0.00359161|0.00341279|7.83482e-05|-0.00166722|reverse_or_depart_equal|True|
|negative|"\|" (91)|0.000334072|0.000593054|1.96141e-06|-3.6564e-05|crossed_damage|True|
|negative|" \|/" (97845)|6.73024e-09|5.59979e-09|2.53141e-11|-2.78724e-09|reverse_or_depart_equal|False|
|negative|" \|=" (8662)|5.93942e-09|3.61991e-09|1.6364e-11|-1.80177e-09|reverse_or_depart_equal|False|
|negative|"\|RF" (98578)|2.16343e-10|2.99106e-10|1.35212e-12|-6.61145e-11|crossed_damage|False|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_61191b0966779357a4c8/anchor65/i15_TOKEN_B.npy`，行30。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## tail_remaining_limit

### 1. alpaca:1473459bd85d08300fe2 / anchor1 / i8

题面：Name a sport which is played indoors.

完整C：

```text
<|im_start|>user
Name a sport which is played indoors.<|im_end|>
<|im_start|>assistant
<think>

</think>

One
```

H (eval, branch16)：

```text
 example of a sport played indoors is
```

共同前驱=' is'，rawTV=0.936676048，headTV=0.781662546，head gain=0.155013502；尾部闭式剩余=0.000018992。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" is" (374)|2.34425e-14|0.177525|0.00218273|0.0876712|toward_not_crossed|True|
|positive|" **" (3070)|0.990818|0.0541421|0.209156|0.0775068|toward_not_crossed|True|
|positive|"." (13)|3.04007e-13|0.0950223|0.0142332|0.0403946|toward_not_crossed|True|
|positive|" basketball" (19240)|0.00857226|0.0508618|0.0133709|0.0187455|toward_not_crossed|True|
|positive|"," (11)|5.01224e-13|0.0255749|0.00383082|0.010872|toward_not_crossed|True|
|positive|"**" (334)|1.19876e-06|0.015512|0.00169992|0.00690603|toward_not_crossed|True|
|positive|"ball" (3959)|4.0737e-15|0.0120807|0.000968583|0.00555608|toward_not_crossed|True|
|positive|" and" (323)|6.43584e-13|0.0113488|0.00140928|0.00496976|toward_not_crossed|True|
|negative|" a" (264)|1.22645e-10|0.015512|0.111953|-0.0482205|reverse_or_depart_equal|True|
|negative|" called" (2598)|1.61118e-14|0.00536079|0.076944|-0.0357916|reverse_or_depart_equal|True|
|negative|" the" (279)|6.49997e-12|0.0136893|0.0496788|-0.0179948|reverse_or_depart_equal|True|
|negative|":\n\n" (1447)|9.6468e-10|0.00732734|0.0363458|-0.0145092|reverse_or_depart_equal|True|
|negative|" also" (1083)|4.37964e-14|0.00144284|0.0249801|-0.0117686|reverse_or_depart_equal|True|
|negative|" known" (3881)|4.33643e-15|0.00253226|0.0220448|-0.00975629|reverse_or_depart_equal|True|
|negative|" one" (825)|9.2717e-14|0.00197213|0.0194545|-0.00874119|reverse_or_depart_equal|True|
|negative|" to" (311)|9.86968e-14|0.00883845|0.0234666|-0.00731408|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_1473459bd85d08300fe2/anchor1/i8_TOKEN_B.npy`，行16。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:1473459bd85d08300fe2 / anchor1 / i8

题面：Name a sport which is played indoors.

完整C：

```text
<|im_start|>user
Name a sport which is played indoors.<|im_end|>
<|im_start|>assistant
<think>

</think>

One
```

H (fit, branch2)：

```text
 sport that is played indoors is **
```

共同前驱=' **'，rawTV=0.989900222，headTV=0.941267315，head gain=0.048632907；尾部闭式剩余=0.383414332。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" is" (374)|1.12493e-15|0.177525|0.00107135|0.0882269|toward_not_crossed|True|
|positive|"." (13)|4.32395e-10|0.0950223|0.00165934|0.0466815|toward_not_crossed|True|
|positive|" **" (3070)|1.0105e-14|0.0541421|2.0888e-05|0.0270606|toward_not_crossed|True|
|positive|"," (11)|1.01689e-11|0.0255749|0.000610436|0.0124822|toward_not_crossed|True|
|positive|"basket" (59764)|0.36818|0.00269558|0.0189901|0.00814727|toward_not_crossed|True|
|positive|"**" (334)|3.36749e-10|0.015512|6.84891e-05|0.00772174|toward_not_crossed|True|
|positive|" a" (264)|2.41898e-15|0.015512|0.000127954|0.00769201|toward_not_crossed|True|
|positive|" the" (279)|1.59549e-16|0.0136893|1.57671e-05|0.00683675|toward_not_crossed|True|
|negative|"Table" (2556)|0.023537|0.000772295|0.159002|-0.0563503|crossed_damage|True|
|negative|"ball" (3959)|3.06153e-07|0.0120807|0.0751074|-0.0315133|reverse_or_depart_equal|True|
|negative|"B" (33)|0.000139956|0.00269558|0.0622662|-0.0297853|reverse_or_depart_equal|True|
|negative|"V" (53)|4.54371e-05|0.00269558|0.0549497|-0.0261271|reverse_or_depart_equal|True|
|negative|"\"" (1)|4.9978e-08|0.000681548|0.0354782|-0.0173983|reverse_or_depart_equal|True|
|negative|" Basketball" (46489)|7.34422e-07|0.00163495|0.0229065|-0.0106358|reverse_or_depart_equal|True|
|negative|"T" (51)|0.000179707|0.00174039|0.0229065|-0.010583|reverse_or_depart_equal|True|
|negative|"Sports" (40979)|6.35399e-09|0.00185264|0.0215186|-0.00983299|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/fresh_test/alpaca_1473459bd85d08300fe2/anchor1/i8_TOKEN_B.npy`，行2。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

