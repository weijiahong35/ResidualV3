# R5真实机制案例

数据身份：dev。固定字典序选例，类别可重叠；不按最大效果选。旧题描述与新题确认分开。完整词表账本在各状态token_contributions_path；表仅展示预定正/负各最多8个token。

## common_eval_gain

### 1. alpaca:227521df7e4c533d1fe5 / anchor1 / i8

题面：Encode a text into a numerical representation.

Input:
Text: The cat is running.

完整C：

```text
<|im_start|>user
Encode a text into a numerical representation.

Input:
Text: The cat is running.<|im_end|>
<|im_start|>assistant
<think>

</think>

To
```

fit不同H数2，eval状态4；M=0.999999253，固定a=M/2；eval平均改善=0.499999627，损伤比例=0。探针可改完整词表，不能当真实头合法动作。

H (eval, branch16)：

```text
 encode the text "The cat is
```

共同前驱=' is'，rawTV=0.999999253，headTV=0.999990982，head gain=0.000008271；尾部闭式剩余=0.999721946。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"," (11)|7.14463e-16|0.699128|0.0644826|0.317323|toward_not_crossed|True|
|positive|" representation" (13042)|5.0028e-22|0.137666|0.00872677|0.0644698|toward_not_crossed|True|
|positive|" using" (1667)|3.82425e-16|0.0736875|0.000383427|0.036652|toward_not_crossed|True|
|positive|" we" (582)|1.01913e-17|0.0145099|0.000462502|0.00702371|toward_not_crossed|True|
|positive|" you" (498)|1.68026e-17|0.0113003|0.000317872|0.00549124|toward_not_crossed|True|
|positive|" can" (646)|7.00435e-18|0.00880072|4.87473e-05|0.00437599|toward_not_crossed|True|
|positive|" of" (315)|3.13913e-17|0.0032376|0.000192799|0.0015224|toward_not_crossed|True|
|positive|" in" (304)|2.82576e-15|0.00285717|0.000170145|0.00134351|toward_not_crossed|True|
|negative|" a" (264)|1.36153e-13|0.000871389|0.420479|-0.209804|reverse_or_depart_equal|True|
|negative|" one" (825)|8.53305e-17|0.00471069|0.154686|-0.0749875|reverse_or_depart_equal|True|
|negative|" numerical" (34776)|2.3497e-21|0.012805|0.0827973|-0.0349961|reverse_or_depart_equal|True|
|negative|" represented" (15251)|6.86683e-20|0.000194433|0.0569057|-0.0283556|reverse_or_depart_equal|True|
|negative|" One" (3776)|9.49364e-22|2.32217e-05|0.034515|-0.0172459|reverse_or_depart_equal|True|
|negative|" encoded" (20498)|2.15168e-22|0.00014225|0.0163037|-0.00808075|reverse_or_depart_equal|True|
|negative|" **" (3070)|3.97896e-12|0.00162797|0.0163037|-0.00733789|reverse_or_depart_equal|True|
|negative|"\"," (497)|4.65889e-15|0.000363249|0.0126974|-0.00616706|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_227521df7e4c533d1fe5/anchor1/i8_TOKEN_B.npy`，行16。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:227521df7e4c533d1fe5 / anchor65 / i8

题面：Encode a text into a numerical representation.

Input:
Text: The cat is running.

完整C：

```text
<|im_start|>user
Encode a text into a numerical representation.

Input:
Text: The cat is running.<|im_end|>
<|im_start|>assistant
<think>

</think>

To encode the text **"The cat is running."** into a numerical representation, one common approach is to use **ASCII encoding**, where each character is converted into its corresponding ASCII value. Here's how it works:

---

### Step-by-step: Convert Each Character to ASCII

**Text:**  
`The cat is running
```

fit不同H数2，eval状态2；M=0.879559089，固定a=M/2；eval平均改善=0.383540498，损伤比例=0。探针可改完整词表，不能当真实头合法动作。

H (eval, branch28)：

```text
.`

**Breakdown:**


```

共同前驱='**\n\n'，rawTV=0.958303248，headTV=0.670512883，head gain=0.287790365；尾部闭式剩余=0.000031943。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"\|" (91)|0.959526|0.0114621|0.289028|0.138783|toward_not_crossed|True|
|positive|" ASCII" (39316)|6.76742e-11|0.10216|0.00529373|0.0484332|toward_not_crossed|True|
|positive|":" (25)|4.41291e-10|0.0795624|0.000150171|0.0397061|toward_not_crossed|True|
|positive|" Character" (15633)|6.90296e-09|0.048257|9.10836e-05|0.024083|toward_not_crossed|True|
|positive|" character" (3668)|9.43522e-09|0.0425867|0.000124496|0.0212311|toward_not_crossed|True|
|positive|" of" (315)|1.43482e-13|0.0331665|0.000159857|0.0165033|toward_not_crossed|True|
|positive|":\n\n" (1447)|5.55504e-12|0.0292694|2.45148e-05|0.0146224|toward_not_crossed|True|
|positive|"  \n" (2303)|7.66849e-11|0.0258301|1.90922e-05|0.0129055|toward_not_crossed|True|
|negative|"1" (16)|0.00392136|0.0425867|0.255066|-0.10624|reverse_or_depart_equal|True|
|negative|"-" (12)|0.0328332|0.022795|0.198646|-0.0778871|crossed_damage|True|
|negative|"``" (13874)|0.00269511|0.00740046|0.106327|-0.0494634|reverse_or_depart_equal|True|
|negative|"The" (785)|1.6891e-06|0.00613519|0.0644908|-0.0291778|reverse_or_depart_equal|True|
|negative|"Let" (10061)|0.000772162|0.000570662|0.0163058|-0.00766608|crossed_damage|True|
|negative|"We" (1654)|8.13853e-05|0.000607466|0.00988998|-0.00464126|reverse_or_depart_equal|True|
|negative|"####" (820)|8.40952e-08|0.000325153|0.00872788|-0.00420136|reverse_or_depart_equal|True|
|negative|"For" (2461)|7.97873e-07|0.000305453|0.00250058|-0.00109756|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_227521df7e4c533d1fe5/anchor65/i8_TOKEN_B.npy`，行28。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## fit_gain_eval_damage

### 1. alpaca:227521df7e4c533d1fe5 / anchor65 / i15

题面：Encode a text into a numerical representation.

Input:
Text: The cat is running.

完整C：

```text
<|im_start|>user
Encode a text into a numerical representation.

Input:
Text: The cat is running.<|im_end|>
<|im_start|>assistant
<think>

</think>

To encode the text **"The cat is running."** into a numerical representation, one common approach is to use **ASCII encoding**, where each character is converted into its corresponding ASCII value. Here's how it works:

---

### Step-by-step: Convert Each Character to ASCII

**Text:**  
`The cat is running
```

fit不同H数2，eval状态1；M=0.971301447，固定a=M/2；eval平均改善=-0.00913611906，损伤比例=1。探针可改完整词表，不能当真实头合法动作。

H (eval, branch27)：

```text
.`

**Step 1:** Break the text into individual characters:
```

共同前驱=':'，rawTV=0.981604009，headTV=0.935983295，head gain=0.045620714；尾部闭式剩余=0.000000639。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"1" (16)|5.03809e-13|0.0585471|0.00188869|0.0283292|toward_not_crossed|True|
|positive|"  \n" (2303)|0.995929|0.0178559|0.0625449|0.0223445|toward_not_crossed|True|
|positive|"**" (334)|2.35635e-14|0.0355106|0.00101094|0.0172498|toward_not_crossed|True|
|positive|"2" (17)|2.03557e-13|0.031338|0.000508334|0.0154148|toward_not_crossed|True|
|positive|"-" (12)|1.3422e-16|0.0276557|1.85161e-05|0.0138186|toward_not_crossed|True|
|positive|"`" (63)|1.45782e-12|0.0202333|0.000187006|0.0100232|toward_not_crossed|True|
|positive|" →" (11397)|5.41646e-12|0.0215383|0.00274803|0.00939513|toward_not_crossed|True|
|positive|" ASCII" (39316)|4.54197e-14|0.0333591|0.0148557|0.00925172|toward_not_crossed|True|
|negative|" " (220)|7.95942e-11|0.0455965|0.317629|-0.136016|reverse_or_depart_equal|True|
|negative|" `" (1565)|3.35104e-10|0.0259801|0.218303|-0.0961614|reverse_or_depart_equal|True|
|negative|"**\n" (1019)|1.00194e-12|0.0115286|0.0625449|-0.0255081|reverse_or_depart_equal|True|
|negative|" Convert" (7169)|1.18484e-14|0.00744343|0.0334779|-0.0130172|reverse_or_depart_equal|True|
|negative|"**\n\n" (56177)|1.15809e-10|0.0108301|0.0334779|-0.0113239|reverse_or_depart_equal|True|
|negative|" '" (364)|8.3064e-13|0.00374279|0.0260726|-0.0111649|reverse_or_depart_equal|True|
|negative|" `'" (72911)|3.31798e-11|0.00227012|0.0179194|-0.00782464|reverse_or_depart_equal|True|
|negative|" T" (350)|9.50618e-12|0.00273829|0.0158138|-0.00653777|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_227521df7e4c533d1fe5/anchor65/i15_TOKEN_B.npy`，行27。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. gsm8k:069f6cd31d89a97c36ab / anchor1 / i15

题面：A clothing store has 40 white shirts and 50 floral shirts. Half of the white shirts have collars, and 20 of the floral shirts have buttons. How many more floral shirts with no buttons are there than white shirts with no collars?
Please reason step by step, and put your final answer within \boxed{}.

完整C：

```text
<|im_start|>user
A clothing store has 40 white shirts and 50 floral shirts. Half of the white shirts have collars, and 20 of the floral shirts have buttons. How many more floral shirts with no buttons are there than white shirts with no collars?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>

</think>

Let
```

fit不同H数4，eval状态4；M=0.648482325，固定a=M/2；eval平均改善=0.24312404，损伤比例=0.25。探针可改完整词表，不能当真实头合法动作。

H (eval, branch22)：

```text
's break down the problem step by step.

---

### **Given:
```

共同前驱=':'，rawTV=0.999300674，headTV=0.995529767，head gain=0.003770907；尾部闭式剩余=0.000000013。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|":" (25)|7.28053e-19|0.954505|1.62646e-05|0.477244|toward_not_crossed|True|
|positive|"1" (16)|2.50852e-13|0.0370102|7.2893e-05|0.0184686|toward_not_crossed|True|
|positive|"**\n\n" (56177)|0.817574|0.000527922|0.0029117|0.00119189|toward_not_crossed|True|
|positive|" " (220)|9.92139e-13|0.00162611|7.75942e-05|0.00077426|toward_not_crossed|True|
|positive|"-" (12)|4.46925e-17|0.00143504|5.22825e-07|0.000717259|toward_not_crossed|True|
|positive|"**\n" (1019)|0.182426|0.000171391|0.00155852|0.000693565|toward_not_crossed|True|
|positive|" of" (315)|1.06826e-14|0.000142088|1.13069e-07|7.09876e-05|toward_not_crossed|True|
|positive|"." (13)|3.42669e-17|0.00013348|1.5788e-08|6.67319e-05|toward_not_crossed|True|
|negative|" Total" (10657)|8.30708e-12|0.00111761|0.554872|-0.276877|reverse_or_depart_equal|True|
|negative|" Understand" (70894)|2.90382e-14|0.000598214|0.297001|-0.148202|reverse_or_depart_equal|True|
|negative|" Determine" (29901)|6.08674e-15|7.14464e-05|0.0401948|-0.0200617|reverse_or_depart_equal|True|
|negative|" Initial" (4127)|3.50268e-14|6.71177e-05|0.0354718|-0.0177023|reverse_or_depart_equal|True|
|negative|" Count" (4504)|5.20627e-15|4.33345e-05|0.0243794|-0.012168|reverse_or_depart_equal|True|
|negative|" Number" (5624)|9.52127e-14|0.00013348|0.011516|-0.00569126|reverse_or_depart_equal|True|
|negative|" Identify" (64547)|4.34086e-16|1.59419e-05|0.00791482|-0.00394944|reverse_or_depart_equal|True|
|negative|" Define" (18614)|3.36142e-15|9.0834e-06|0.00543977|-0.00271534|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/gsm8k_069f6cd31d89a97c36ab/anchor1/i15_TOKEN_B.npy`，行22。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## actual_head_mixed_sameZ

### 1. alpaca:a2322985e7acd2d87d8b / anchor65 / i3

题面：Suggest a theme for the given type of event.

Input:
A dinner party for 40 people

完整C：

```text
<|im_start|>user
Suggest a theme for the given type of event.

Input:
A dinner party for 40 people<|im_end|>
<|im_start|>assistant
<think>

</think>

**Theme Suggestion:** *"Around the World in 40 Courses"*

**Theme Description:**  
Transform your dinner party into a global culinary journey, featuring dishes from different countries around the world. With 40 guests, you can create a multi-course meal that takes guests on a tour of various cuisines, from
```

H (eval, branch17)：

```text
 Asia to
```

共同前驱=' to'，rawTV=0.938186406，headTV=0.879758304，head gain=0.058428102；尾部闭式剩余=0.000001979。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" to" (311)|3.45305e-14|0.218037|0.000539836|0.108749|toward_not_crossed|True|
|positive|" South" (4882)|0.163747|0.0190519|0.0485944|0.0147713|toward_not_crossed|True|
|positive|" of" (315)|1.13221e-13|0.0139387|9.38094e-05|0.00692243|toward_not_crossed|True|
|positive|" Europe" (4505)|0.733863|0.00845423|0.0215637|0.00655471|toward_not_crossed|True|
|positive|" Mediterranean" (37685)|1.0061e-06|0.0403329|0.0333984|0.00346722|toward_not_crossed|True|
|positive|"," (11)|1.01918e-11|0.00512775|2.42302e-06|0.00256266|toward_not_crossed|True|
|positive|" countries" (5837)|9.38637e-14|0.0190519|0.0148205|0.00211572|toward_not_crossed|True|
|positive|" through" (1526)|3.67575e-14|0.0039935|0.000271446|0.00186102|toward_not_crossed|True|
|negative|" the" (279)|0.028455|0.0157946|0.0966416|-0.0277631|crossed_damage|True|
|negative|" Italy" (15344)|0.000246184|0.0277204|0.0517285|-0.0120041|reverse_or_depart_equal|True|
|negative|" Asia" (13622)|6.24326e-11|0.0295082|0.0456503|-0.00807105|reverse_or_depart_equal|True|
|negative|" global" (3644)|7.16658e-16|0.0108554|0.0260107|-0.00757764|reverse_or_depart_equal|True|
|negative|" international" (6489)|1.63111e-14|0.0215886|0.0355525|-0.00698191|reverse_or_depart_equal|True|
|negative|" exotic" (38318)|1.78873e-11|0.0058105|0.0148205|-0.00450498|reverse_or_depart_equal|True|
|negative|" new" (501)|3.91282e-14|0.00156388|0.0101859|-0.00431104|reverse_or_depart_equal|True|
|negative|" Italian" (14811)|2.00689e-12|0.00845423|0.0157763|-0.00366104|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_a2322985e7acd2d87d8b/anchor65/i3_TOKEN_B.npy`，行17。

H (eval, branch23)：

```text
 Asian to
```

共同前驱=' to'，rawTV=0.920216282，headTV=0.920527551，head gain=-0.000311269；尾部闭式剩余=0.000010730。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" to" (311)|3.44555e-13|0.218037|0.000539836|0.108749|toward_not_crossed|True|
|positive|" of" (315)|1.98278e-12|0.0139387|9.38094e-05|0.00692243|toward_not_crossed|True|
|positive|"," (11)|4.23937e-11|0.00512775|2.42302e-06|0.00256266|toward_not_crossed|True|
|positive|" countries" (5837)|1.23558e-14|0.0190519|0.0148205|0.00211572|toward_not_crossed|True|
|positive|" through" (1526)|4.96379e-14|0.0039935|0.000271446|0.00186102|toward_not_crossed|True|
|positive|" Americas" (51148)|1.41787e-08|0.00794201|0.00424614|0.00184794|toward_not_crossed|True|
|positive|" borders" (23806)|2.00134e-15|0.00845423|0.0048115|0.00182136|toward_not_crossed|True|
|positive|" European" (7513)|0.498315|0.00375154|0.00700069|0.00162458|toward_not_crossed|True|
|negative|" the" (279)|1.27633e-06|0.0157946|0.0966416|-0.0404235|reverse_or_depart_equal|True|
|negative|" South" (4882)|0.0117193|0.0190519|0.0485944|-0.0147713|reverse_or_depart_equal|True|
|negative|" Italy" (15344)|5.44346e-11|0.0277204|0.0517285|-0.0120041|reverse_or_depart_equal|True|
|negative|" Asia" (13622)|4.83862e-15|0.0295082|0.0456503|-0.00807105|reverse_or_depart_equal|True|
|negative|" global" (3644)|1.2146e-11|0.0108554|0.0260107|-0.00757764|reverse_or_depart_equal|True|
|negative|" international" (6489)|3.33452e-10|0.0215886|0.0355525|-0.00698191|reverse_or_depart_equal|True|
|negative|" Europe" (4505)|3.85418e-08|0.00845423|0.0215637|-0.00655471|reverse_or_depart_equal|True|
|negative|" exotic" (38318)|8.59984e-09|0.0058105|0.0148205|-0.00450498|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_a2322985e7acd2d87d8b/anchor65/i3_TOKEN_B.npy`，行23。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. gsm8k:069f6cd31d89a97c36ab / anchor1 / i8

题面：A clothing store has 40 white shirts and 50 floral shirts. Half of the white shirts have collars, and 20 of the floral shirts have buttons. How many more floral shirts with no buttons are there than white shirts with no collars?
Please reason step by step, and put your final answer within \boxed{}.

完整C：

```text
<|im_start|>user
A clothing store has 40 white shirts and 50 floral shirts. Half of the white shirts have collars, and 20 of the floral shirts have buttons. How many more floral shirts with no buttons are there than white shirts with no collars?
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>assistant
<think>

</think>

Let
```

H (eval, branch16)：

```text
's solve the problem step by step
```

共同前驱=' step'，rawTV=0.045787188，headTV=0.034210146，head gain=0.011577042；尾部闭式剩余=0.000000004。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" step" (3019)|9.58697e-14|0.0417651|6.57003e-05|0.0208497|toward_not_crossed|True|
|positive|"###" (14374)|4.97366e-14|0.000409451|1.73358e-07|0.000204639|toward_not_crossed|True|
|positive|"---\n\n" (44364)|1.17957e-11|0.000409451|1.06195e-06|0.000204194|toward_not_crossed|True|
|positive|" and" (323)|4.95125e-05|0.001113|0.000800393|0.000156305|toward_not_crossed|True|
|positive|"\n\n" (271)|4.75873e-09|0.000193411|1.1417e-05|9.09969e-05|toward_not_crossed|True|
|positive|".\n" (624)|6.44023e-10|0.000463969|0.000294448|8.47602e-05|toward_not_crossed|True|
|positive|"-step" (29208)|3.99466e-17|0.000132929|4.42685e-07|6.62433e-05|toward_not_crossed|True|
|positive|" Step" (14822)|7.91335e-17|6.27913e-05|6.31456e-09|3.13925e-05|toward_not_crossed|True|
|negative|".\n\n" (382)|0.962438|0.95057|0.994607|-0.0101507|crossed_damage|True|
|negative|":\n\n" (1447)|0.0373178|0.00342829|0.00316562|-0.000131336|reverse_or_depart_equal|True|
|negative|" to" (311)|8.16323e-05|0.00067507|0.000906964|-0.000115947|reverse_or_depart_equal|True|
|negative|" using" (1667)|9.25015e-05|6.27913e-05|3.51668e-05|-1.38122e-05|reverse_or_depart_equal|True|
|negative|" by" (553)|8.60398e-06|1.09115e-05|3.51668e-05|-1.21277e-05|reverse_or_depart_equal|True|
|negative|" into" (1119)|8.62992e-12|2.75886e-06|2.88667e-06|-6.39054e-08|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/gsm8k_069f6cd31d89a97c36ab/anchor1/i8_TOKEN_B.npy`，行16。

H (fit, branch13)：

```text
's break this down step by step
```

共同前驱=' step'，rawTV=0.173424557，headTV=0.173865167，head gain=-0.000440611；尾部闭式剩余=0.000000390。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" step" (3019)|9.07358e-13|0.0417651|6.57003e-05|0.0208497|toward_not_crossed|True|
|positive|"###" (14374)|2.02459e-13|0.000409451|1.73358e-07|0.000204639|toward_not_crossed|True|
|positive|"---\n\n" (44364)|6.98627e-11|0.000409451|1.06195e-06|0.000204194|toward_not_crossed|True|
|positive|" to" (311)|0.0524975|0.00067507|0.000906964|0.000115947|toward_not_crossed|True|
|positive|"\n\n" (271)|1.10373e-08|0.000193411|1.1417e-05|9.09969e-05|toward_not_crossed|True|
|positive|".\n" (624)|2.46275e-09|0.000463969|0.000294448|8.47602e-05|toward_not_crossed|True|
|positive|"-step" (29208)|2.05555e-16|0.000132929|4.42685e-07|6.62433e-05|toward_not_crossed|True|
|positive|" Step" (14822)|4.83564e-16|6.27913e-05|6.31456e-09|3.13925e-05|toward_not_crossed|True|
|negative|".\n\n" (382)|0.8212|0.95057|0.994607|-0.0220187|reverse_or_depart_equal|True|
|negative|" and" (323)|0.0132735|0.001113|0.000800393|-0.000156305|reverse_or_depart_equal|True|
|negative|":\n\n" (1447)|0.111137|0.00342829|0.00316562|-0.000131336|reverse_or_depart_equal|True|
|negative|"," (11)|0.000101344|5.54131e-05|1.46597e-05|-2.03767e-05|reverse_or_depart_equal|True|
|negative|" using" (1667)|0.00158529|6.27913e-05|3.51668e-05|-1.38122e-05|reverse_or_depart_equal|True|
|negative|" by" (553)|2.21689e-07|1.09115e-05|3.51668e-05|-1.21277e-05|reverse_or_depart_equal|True|
|negative|"." (13)|6.96527e-05|2.96605e-05|6.92476e-06|-1.13679e-05|reverse_or_depart_equal|True|
|negative|" based" (3118)|0.000130128|6.21718e-06|1.36357e-06|-2.42681e-06|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/gsm8k_069f6cd31d89a97c36ab/anchor1/i8_TOKEN_B.npy`，行13。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

## tail_remaining_limit

### 1. alpaca:227521df7e4c533d1fe5 / anchor1 / i8

题面：Encode a text into a numerical representation.

Input:
Text: The cat is running.

完整C：

```text
<|im_start|>user
Encode a text into a numerical representation.

Input:
Text: The cat is running.<|im_end|>
<|im_start|>assistant
<think>

</think>

To
```

H (eval, branch16)：

```text
 encode the text "The cat is
```

共同前驱=' is'，rawTV=0.999999253，headTV=0.999990982，head gain=0.000008271；尾部闭式剩余=0.999721946。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|"," (11)|7.14463e-16|0.699128|0.0644826|0.317323|toward_not_crossed|True|
|positive|" representation" (13042)|5.0028e-22|0.137666|0.00872677|0.0644698|toward_not_crossed|True|
|positive|" using" (1667)|3.82425e-16|0.0736875|0.000383427|0.036652|toward_not_crossed|True|
|positive|" we" (582)|1.01913e-17|0.0145099|0.000462502|0.00702371|toward_not_crossed|True|
|positive|" you" (498)|1.68026e-17|0.0113003|0.000317872|0.00549124|toward_not_crossed|True|
|positive|" can" (646)|7.00435e-18|0.00880072|4.87473e-05|0.00437599|toward_not_crossed|True|
|positive|" of" (315)|3.13913e-17|0.0032376|0.000192799|0.0015224|toward_not_crossed|True|
|positive|" in" (304)|2.82576e-15|0.00285717|0.000170145|0.00134351|toward_not_crossed|True|
|negative|" a" (264)|1.36153e-13|0.000871389|0.420479|-0.209804|reverse_or_depart_equal|True|
|negative|" one" (825)|8.53305e-17|0.00471069|0.154686|-0.0749875|reverse_or_depart_equal|True|
|negative|" numerical" (34776)|2.3497e-21|0.012805|0.0827973|-0.0349961|reverse_or_depart_equal|True|
|negative|" represented" (15251)|6.86683e-20|0.000194433|0.0569057|-0.0283556|reverse_or_depart_equal|True|
|negative|" One" (3776)|9.49364e-22|2.32217e-05|0.034515|-0.0172459|reverse_or_depart_equal|True|
|negative|" encoded" (20498)|2.15168e-22|0.00014225|0.0163037|-0.00808075|reverse_or_depart_equal|True|
|negative|" **" (3070)|3.97896e-12|0.00162797|0.0163037|-0.00733789|reverse_or_depart_equal|True|
|negative|"\"," (497)|4.65889e-15|0.000363249|0.0126974|-0.00616706|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_227521df7e4c533d1fe5/anchor1/i8_TOKEN_B.npy`，行16。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

### 2. alpaca:227521df7e4c533d1fe5 / anchor1 / i15

题面：Encode a text into a numerical representation.

Input:
Text: The cat is running.

完整C：

```text
<|im_start|>user
Encode a text into a numerical representation.

Input:
Text: The cat is running.<|im_end|>
<|im_start|>assistant
<think>

</think>

To
```

H (eval, branch16)：

```text
 encode the text "The cat is running." into a numerical representation,
```

共同前驱=','，rawTV=0.979233676，headTV=0.626781246，head gain=0.352452430；尾部闭式剩余=0.051190265。

|贡献|token|p|raw q|head q|b|类别|S内|
|---|---|---:|---:|---:|---:|---|---|
|positive|" we" (582)|0.800201|0.000705534|0.34845|0.173872|toward_not_crossed|True|
|positive|"**" (334)|6.74047e-12|0.0766079|9.10357e-05|0.0382584|toward_not_crossed|True|
|positive|" **" (3070)|3.56158e-07|0.0676062|0.00599542|0.0308054|toward_not_crossed|True|
|positive|" of" (315)|1.12239e-10|0.0300001|2.58248e-06|0.0149987|toward_not_crossed|True|
|positive|" as" (438)|9.39764e-10|0.0410052|0.0112009|0.0149022|toward_not_crossed|True|
|positive|" using" (1667)|1.54941e-09|0.036187|0.00988478|0.0131511|toward_not_crossed|True|
|positive|" into" (1119)|4.63265e-12|0.0219485|2.9555e-05|0.0109595|toward_not_crossed|True|
|positive|" (" (320)|5.24948e-12|0.018196|0.000192723|0.00900162|toward_not_crossed|True|
|negative|" you" (498)|0.0069231|0.00102655|0.128188|-0.0576841|crossed_damage|True|
|negative|" follow" (1795)|7.94695e-08|0.000905924|0.113125|-0.0561097|reverse_or_depart_equal|True|
|negative|" including" (2670)|3.42309e-11|0.00102655|0.0471576|-0.0230655|reverse_or_depart_equal|True|
|negative|" which" (892)|4.48336e-09|0.000799475|0.0222757|-0.0107381|reverse_or_depart_equal|True|
|negative|" and" (323)|3.2801e-09|0.00521323|0.0252417|-0.0100142|reverse_or_depart_equal|True|
|negative|" where" (1380)|1.96985e-10|0.00180164|0.0196582|-0.00892829|reverse_or_depart_equal|True|
|negative|" use" (990)|4.72543e-10|0.00521323|0.0209261|-0.00785642|reverse_or_depart_equal|True|
|negative|" follows" (11017)|2.69247e-10|0.00140312|0.0143823|-0.00648957|reverse_or_depart_equal|True|

完整逐token贡献：`/data/weijiahong/speculators-main-lora/residualV3_main_v1/research/mismatch_theory_round5_20261015_v1/runs/20261015T030446Z_mismatch_r5/analysis/dev/alpaca_227521df7e4c533d1fe5/anchor1/i15_TOKEN_B.npy`，行16。

此账本定位固定状态的概率移动；不识别容量、训练或优化成因。尾部下限只约束raw Top256操作；不从M=0或当前失败推出不可修。

