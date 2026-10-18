# MT-Bench 双轮重生成

原始数据：`HuggingFaceH4/mt_bench_prompts`，HF split 名为 `train`，实际作为评测集。
没有独立训练 split；原论文没有用这 80 组作为训练语料。
共 80 组双轮问题，默认 thinking 开启。
第二轮使用本次生成的第一轮回答作为历史，一轮保存一条 boundary sample。
因此 `--max-samples 80` 对应 80 个完成会话，正常全量输出为 160 行。

```bash
cd /data/weijiahong/speculators-main-lora/regeneration_standalone/mt_bench_regen
HF_ENDPOINT=https://huggingface.co bash download.sh
bash run.sh --dry-run
bash run.sh --run --gpu 0
```

下载写入 Hugging Face 默认缓存，生成结果写入本目录 `regenerated/`。
本入口只生成回答，不执行模型裁判评分。
`--thinking off` 可生成另一份非 thinking 数据。
缓存、采样、续跑和输出格式说明见 [总说明](../README.md)。
