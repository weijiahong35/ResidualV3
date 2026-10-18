# GSM8K 重生成

原始数据：`openai/gsm8k`，配置 `main`。默认 `train` 7,473 条，thinking 开启。
`--split test` 为评测集 1,319 条；两种 split 的输出文件分开保存。
这是小规模数学训练来源，不是 EAGLE-3 / DFlash / DSpark 原论文的大规模训练混合语料。

```bash
cd /data/weijiahong/speculators-main-lora/regeneration_standalone/gsm8k_regen
HF_ENDPOINT=https://huggingface.co bash download.sh
bash run.sh --dry-run
bash run.sh --run --gpu 0
# 若准备评测数据：
bash run.sh --split test --run --gpu 0
```

下载写入 Hugging Face 默认缓存，生成结果写入本目录 `regenerated/`。
测试问题追加逐步解答与 boxed 答案提示，原始 answer 不进入生成请求。
`--thinking off` 可生成另一份非 thinking 数据。
缓存、采样、续跑和输出格式说明见 [总说明](../README.md)。
