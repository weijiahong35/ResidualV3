# HumanEval 重生成

原始数据：`openai/openai_humaneval`，`test` 164 条。默认 thinking 开启。
这是评测数据，没有官方训练 split；原论文没有用这 164 条作为训练语料。
问题包装为 Python 代码生成请求；参考解答和测试代码不发送给目标模型。
`task_id` 保留为生成结果的 `primary_id`，便于关联原始题目。

```bash
cd /data/weijiahong/speculators-main-lora/regeneration_standalone/humaneval_regen
HF_ENDPOINT=https://huggingface.co bash download.sh
bash run.sh --dry-run
bash run.sh --run --gpu 0
```

下载写入 Hugging Face 默认缓存，生成结果写入本目录 `regenerated/`。
本入口只生成数据，不执行 HumanEval 单元测试或计算 pass@1。
`--thinking off` 可生成另一份非 thinking 数据。
缓存、采样、续跑和输出格式说明见 [总说明](../README.md)。
