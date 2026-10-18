"""Configuration registries for data generation pipeline."""

import os
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for loading a dataset"""

    name: str
    hf_path: str
    subset: str | None = None
    split: str
    filter_fn: Callable[[dict], bool] | None = None
    normalize_fn: Callable[[dict], dict] | None = None
    # Bare user-prompt column, used when a row has no conversation.
    prompt_field: str | None = None


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example

def _normalize_opencodeinstruct(example: dict) -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": example["input"],
            },
            {
                "role": "assistant",
                "content": example["output"],
            },
        ]
    }
def _normalize_openmathinstruct2(example: dict) -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": example["problem"],
            },
            {
                "role": "assistant",
                "content": example["generated_solution"],
            },
        ]
    }
def _normalize_gsm8k(example: dict) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def _normalize_gsm8k_test(example: dict) -> dict:
    normalized = _normalize_gsm8k(example)
    normalized["conversations"][0]["content"] += (
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    return normalized


def _normalize_humaneval(example: dict) -> dict:
    """Use only the problem; canonical_solution and tests are references."""
    return {
        "conversations": [
            {
                "role": "user",
                "content": (
                    "Write a solution to the following problem and make sure that it "
                    "passes the tests:\n```python\n" + example["prompt"] + "\n```"
                ),
            },
        ],
    }


def _normalize_mt_bench(example: dict) -> dict:
    """Keep both user turns; regeneration supplies the intervening answer."""
    prompts = example["prompt"]
    if not isinstance(prompts, list) or not prompts or any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
    ):
        raise ValueError("MT-Bench prompt must be a nonempty list of user turns")
    return {
        "conversations": [
            {"role": "user", "content": prompt} for prompt in prompts
        ],
    }


def _normalize_open_perfectblend(example: dict) -> dict:
    """Accept both the published ShareGPT schema and prepared pool prompts."""
    roles = {"human": "user", "gpt": "assistant", "chatgpt": "assistant",
             "bing": "assistant", "bard": "assistant"}
    messages = example.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Open-PerfectBlend needs a nonempty conversations list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Open-PerfectBlend conversation messages must be objects")
        role = message.get("role") or message.get("from")
        role = roles.get(role, role)
        content = message.get("content", message.get("value"))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported Open-PerfectBlend role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Open-PerfectBlend messages need nonempty text")
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("Open-PerfectBlend conversation has no user prompt")
    return {"conversations": normalized}


def _normalize_cnn_dailymail(example: dict) -> dict:
    return {
        "conversations": [
            {
                "role": "user",
                "content": f"Summarize: {example['article']}",
            },
            {
                "role": "assistant",
                "content": example["highlights"],
            },
        ]
    }

def _normalize_nemotron(example: dict) -> dict:
    """Build a conversation from Nemotron ``input`` turns plus ``output``."""
    return {
        "conversations": [
            *example["input"],
            {"role": "assistant", "content": example["output"]},
        ]
    }
def _nq_annotation_items(example: dict) -> list[dict]:
    """Normalize NQ annotations to list[dict].

    HuggingFace may expose Sequence[dict] either as list-of-dicts or
    dict-of-lists depending on the representation/version.
    """
    annotations = example.get("annotations") or []

    if isinstance(annotations, list):
        return [ann for ann in annotations if isinstance(ann, dict)]

    if isinstance(annotations, dict):
        list_lengths = [
            len(value)
            for value in annotations.values()
            if isinstance(value, list)
        ]
        if not list_lengths:
            return [annotations]

        num_annotations = max(list_lengths)
        result = []

        for i in range(num_annotations):
            ann = {}
            for key, value in annotations.items():
                if isinstance(value, list):
                    ann[key] = value[i] if i < len(value) else None
                else:
                    ann[key] = value
            result.append(ann)

        return result

    return []


def _nq_short_answer_texts(short_answers) -> list[str]:
    """Extract text spans from an NQ short_answers object."""
    if short_answers is None:
        return []

    if isinstance(short_answers, str):
        text = short_answers.strip()
        return [text] if text else []

    if isinstance(short_answers, list):
        texts = []
        for item in short_answers:
            texts.extend(_nq_short_answer_texts(item))
        return texts

    if isinstance(short_answers, dict):
        if "text" in short_answers:
            return _nq_short_answer_texts(short_answers["text"])

    return []


def _nq_reference_answers(example: dict) -> list[str]:
    """Extract usable short / yes-no reference answers from Natural Questions."""
    references = []

    for annotation in _nq_annotation_items(example):
        short_answers = _nq_short_answer_texts(
            annotation.get("short_answers")
        )

        if short_answers:
            # Multiple spans can jointly constitute one NQ short answer.
            answer = " ".join(short_answers).strip()
            if answer:
                references.append(answer)
            continue

        yes_no = annotation.get("yes_no_answer")

        if yes_no in (1, "YES", "yes"):
            references.append("yes")
        elif yes_no in (0, "NO", "no"):
            references.append("no")

    # validation has multiple annotations, so deduplicate while preserving order
    return list(dict.fromkeys(references))


def _filter_natural_questions(example: dict) -> bool:
    """Keep examples with a usable short or yes/no answer."""
    return bool(_nq_reference_answers(example))


def _normalize_natural_questions(example: dict) -> dict:
    references = _nq_reference_answers(example)

    question = example["question"]["text"]

    return {
        "conversations": [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": references[0],
            },
        ],
        # Keep all references for later NQ evaluation.
        "reference_answers": references,
    }


def get_coco_dir():
    return os.getenv("COCO_DIR") or "coco/"


def _parse_sharegpt4v_part(part: str, image_path: str):
    if part == "<image>":
        return {"type": "image", "path": image_path}

    return {"type": "text", "text": part}


def _parse_sharegpt4v_user_content(content: str, image_path: str):
    return [_parse_sharegpt4v_part(part, image_path) for part in content.split("\n")]


def _parse_sharegpt4v_assistant_content(content: str):
    return [{"type": "text", "text": content}]


def _filter_sharegpt4v_coco(example: dict) -> bool:
    return example["image"].startswith("coco/")


def _normalize_sharegpt4v_coco(example: dict) -> dict:
    coco_dir = get_coco_dir()
    image_path = os.path.join(coco_dir, example["image"].removeprefix("coco/"))

    if not os.path.exists(image_path):
        state_str = "set to" if os.getenv("COCO_DIR") else "default"

        raise ValueError(
            f"No image found at <{image_path}>. "
            f"Please download COCO 2017 Train Images from "
            f"<http://images.cocodataset.org/zips/train2017.zip> and place the "
            f"extracted folder under `COCO_DIR` ({state_str}: `{coco_dir}`)."
        )

    messages = [
        (
            turn
            | {
                "value": (
                    _parse_sharegpt4v_user_content(turn["value"], image_path)
                    if turn["from"] in ("human", "user")
                    else _parse_sharegpt4v_assistant_content(turn["value"])
                )
            }
        )
        for turn in example["conversations"]
    ]

    return {"conversations": messages}


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        normalize_fn=_normalize_ultrachat,
        prompt_field="prompt",
    ),
    "openmathinstruct2": DatasetConfig(
        name="openmathinstruct2",
        hf_path="nvidia/OpenMathInstruct-2",
        split="train_1M",
        prompt_field="problem",
        normalize_fn=_normalize_openmathinstruct2,
    ),
    "natural_questions": DatasetConfig(
        name="natural_questions",
        hf_path="google-research-datasets/natural_questions",
        subset="default",
        split="train",
        filter_fn=_filter_natural_questions,
        normalize_fn=_normalize_natural_questions,
    ),
    "cnn_dailymail": DatasetConfig(
        name="cnn_dailymail",
        hf_path="abisee/cnn_dailymail",
        subset="3.0.0",
        split="train",
        prompt_field="article",
        normalize_fn=_normalize_cnn_dailymail,
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        hf_path="openai/gsm8k",
        subset="main",
        split="train",
        normalize_fn=_normalize_gsm8k,
        prompt_field="question",
    ),
    # Benchmark presets are evaluation data. Regenerate before tokenizing the
    # prompt-only HumanEval / MT-Bench presets as training data.
    "gsm8k_test": DatasetConfig(
        name="gsm8k_test",
        hf_path="openai/gsm8k",
        subset="main",
        split="test",
        normalize_fn=_normalize_gsm8k_test,
        prompt_field="question",
    ),
    "humaneval": DatasetConfig(
        name="humaneval",
        hf_path="openai/openai_humaneval",
        split="test",
        normalize_fn=_normalize_humaneval,
        prompt_field="prompt",
    ),
    "mt_bench": DatasetConfig(
        name="mt_bench",
        hf_path="HuggingFaceH4/mt_bench_prompts",
        split="train",  # HF calls this train, but the 80 prompts are a benchmark.
        normalize_fn=_normalize_mt_bench,
    ),
    "magpie": DatasetConfig(
        name="magpie",
        hf_path="Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        split="train",
        prompt_field="instruction",
    ),
    "nemotron": DatasetConfig(
        name="nemotron",
        hf_path="nvidia/Llama-Nemotron-Post-Training-Dataset",
        subset="SFT",
        split="chat",
        normalize_fn=_normalize_nemotron,
    ),
    "opencodeinstruct": DatasetConfig(
        name="opencodeinstruct",
        hf_path="nvidia/OpenCodeInstruct",
        split="train",
        prompt_field="input",
        normalize_fn=_normalize_opencodeinstruct,
    ),
    # NOTE: You need to serve vLLM with `--allowed-local-media-path /path/to/coco`
    "sharegpt4v_coco": DatasetConfig(
        name="sharegpt4v_coco",
        hf_path="Lin-Chen/ShareGPT4V",
        subset="ShareGPT4V",
        split="train",
        filter_fn=_filter_sharegpt4v_coco,
        normalize_fn=_normalize_sharegpt4v_coco,
    ),
    "open-perfectblend": DatasetConfig(
        name="open-perfectblend",
        hf_path="mlabonne/open-perfectblend",
        split="train",
        normalize_fn=_normalize_open_perfectblend,
    ),
    # Multi-turn function-calling SFT
    "hermes-fc": DatasetConfig(
        name="hermes-fc",
        hf_path="NousResearch/hermes-function-calling-v1",
        subset="func_calling",
        split="train",
    ),
}
