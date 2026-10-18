#!/usr/bin/env python3
"""Evaluate a DFlash speculator, optionally after merging a PEFT LoRA adapter.

Designed for the vllm-project/speculators workflow and the SpecLoRA fork used in
this project.

Subcommands
-----------
inspect-lora
    Inspect adapter_config.json and adapter_model.safetensors. Reports the exact
    modules that are present in the adapter file.

merge
    Load a dense/pretrained DFlash speculator, merge only the LoRA A/B pairs
    actually present in the adapter safetensors, and save a normal standalone
    speculator checkpoint that vLLM can serve.

evaluate
    Select an exact slice of a local/Hugging Face dataset, send it through the
    vLLM OpenAI chat endpoint, and calculate speculative-decoding acceptance
    metrics from the server's Prometheus /metrics counters.

Important
---------
* LoRA merge is driven by the adapter file itself. If an adapter contains only
  gate/up/down projections, only those are merged. If a newer adapter contains
  q/k/v/o + gate/up/down, all pairs actually present are merged.
* Standard LoRA is supported. DoRA/QALoRA/modules_to_save are rejected rather
  than being silently merged incorrectly.
* Acceptance length follows the official speculators evaluator:
      1 + num_accepted_tokens / num_drafts
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# Make the script work when copied to <repo>/scripts/evaluate/ without requiring
# an editable install of this exact checkout. Installed speculators still works.
_SCRIPT_PATH = Path(__file__).resolve()
for _parent in [_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents]:
    _src = _parent / "src"
    if (_src / "speculators").is_dir():
        sys.path.insert(0, str(_src))
        break


_LORA_KEY_RE = re.compile(
    r"^(?P<module>.+)\.lora_(?P<kind>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
_PROM_SIMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$")
_PROM_LABELED_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$"
)
_SPEC_PREFIX = "vllm:spec_decode"


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _locate_adapter_files(lora_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    p = Path(lora_path).expanduser().resolve()
    if p.is_file():
        if p.name == "adapter_model.safetensors":
            weights = p
            config = p.parent / "adapter_config.json"
        elif p.name == "adapter_config.json":
            config = p
            weights = p.parent / "adapter_model.safetensors"
        else:
            raise ValueError(
                "--lora-path must be an adapter directory, adapter_config.json, "
                "or adapter_model.safetensors"
            )
    else:
        config = p / "adapter_config.json"
        weights = p / "adapter_model.safetensors"

    if not config.is_file():
        raise FileNotFoundError(f"LoRA config not found: {config}")
    if not weights.is_file():
        raise FileNotFoundError(f"LoRA weights not found: {weights}")
    return config, weights


def _parse_lora_pairs(weights_path: Path) -> tuple[dict[tuple[str, str | None], dict[str, str]], list[str]]:
    from safetensors import safe_open

    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    pairs: dict[tuple[str, str | None], dict[str, str]] = defaultdict(dict)
    unexpected: list[str] = []
    for key in keys:
        m = _LORA_KEY_RE.match(key)
        if not m:
            unexpected.append(key)
            continue
        pairs[(m.group("module"), m.group("adapter"))][m.group("kind")] = key

    incomplete = {
        f"{module} [{adapter or 'unnamed'}]": sorted({"A", "B"} - set(parts))
        for (module, adapter), parts in pairs.items()
        if set(parts) != {"A", "B"}
    }
    if incomplete:
        raise RuntimeError(f"Incomplete LoRA A/B pairs: {incomplete}")
    return dict(pairs), unexpected


def _target_module_declaration(config: dict[str, Any]) -> list[str] | str | None:
    target = config.get("target_modules")
    if isinstance(target, list):
        return [str(x) for x in target]
    if isinstance(target, str):
        return target
    return None


def inspect_lora(lora_path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path, weights_path = _locate_adapter_files(lora_path)
    config = _read_json(config_path)
    pairs, unexpected = _parse_lora_pairs(weights_path)

    module_names = sorted({module for module, _adapter in pairs})
    by_leaf = Counter(name.rsplit(".", 1)[-1] for name in module_names)
    adapter_names = sorted({adapter for _module, adapter in pairs if adapter is not None})

    summary = {
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules_declared": _target_module_declaration(config),
        "actual_module_count": len(module_names),
        "actual_modules_by_leaf": dict(sorted(by_leaf.items())),
        "actual_modules": module_names,
        "named_adapters_in_weights": adapter_names,
        "unexpected_tensor_keys": unexpected,
    }
    return summary


def _validate_mergeable_config(config: dict[str, Any], unexpected_keys: list[str]) -> None:
    peft_type = str(config.get("peft_type", "LORA")).upper()
    if peft_type != "LORA":
        raise ValueError(f"Only PEFT LORA adapters are supported, got peft_type={peft_type!r}")
    if config.get("use_dora"):
        raise ValueError("DoRA adapter detected; direct standard-LoRA merge is intentionally refused.")
    if config.get("use_qalora"):
        raise ValueError("QALoRA adapter detected; direct standard-LoRA merge is intentionally refused.")
    if config.get("modules_to_save"):
        raise ValueError(
            "modules_to_save is not empty. This evaluator only merges LoRA A/B tensors."
        )
    if config.get("lora_bias"):
        raise ValueError("lora_bias=True is not supported by this merge path.")
    if str(config.get("bias", "none")) != "none":
        raise ValueError(f"bias={config.get('bias')!r} is not supported; expected 'none'.")
    if unexpected_keys:
        raise ValueError(
            "Adapter contains tensors other than standard lora_A/lora_B weights: "
            + ", ".join(unexpected_keys[:10])
        )


def _pattern_value(pattern: Any, module_name: str, default: float | int) -> float | int:
    if not isinstance(pattern, dict) or not pattern:
        return default
    matches: list[tuple[int, Any]] = []
    for key, value in pattern.items():
        key = str(key)
        if module_name == key or module_name.endswith("." + key) or module_name.endswith(key):
            matches.append((len(key), value))
    if not matches:
        return default
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def _module_candidates(raw_name: str) -> list[str]:
    candidates = [raw_name]
    prefixes = (
        "base_model.model.",
        "base_model.",
        "model.model.",
    )
    for prefix in prefixes:
        if raw_name.startswith(prefix):
            candidates.append(raw_name[len(prefix) :])
    if raw_name.startswith("model."):
        candidates.append(raw_name[len("model.") :])
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _resolve_model_module(model: Any, raw_name: str) -> tuple[str, Any]:
    errors: list[str] = []
    for candidate in _module_candidates(raw_name):
        try:
            return candidate, model.get_submodule(candidate)
        except Exception as exc:  # noqa: BLE001 - include candidates in final diagnostic
            errors.append(f"{candidate}: {exc}")
    raise KeyError(
        f"Could not map adapter module {raw_name!r} to the DFlash model. Tried: "
        + " | ".join(errors)
    )


def _declared_target_allows(config: dict[str, Any], raw_module_name: str) -> bool:
    declared = _target_module_declaration(config)
    leaf = raw_module_name.rsplit(".", 1)[-1]
    if declared is None:
        return True
    if isinstance(declared, list):
        return leaf in declared or raw_module_name in declared
    # PEFT also permits a regex string for target_modules.
    try:
        return re.fullmatch(declared, raw_module_name) is not None or re.fullmatch(declared, leaf) is not None
    except re.error:
        return True


def merge_lora_into_dflash(
    draft_model: str,
    lora_path: str,
    output_dir: str,
    *,
    dtype: str = "bfloat16",
    overwrite: bool = False,
    local_files_only: bool = False,
    verifier: str | None = None,
) -> dict[str, Any]:
    """Merge only adapter modules actually present in safetensors into DFlash."""
    import torch
    from safetensors import safe_open

    from speculators.model import SpeculatorModel

    config_path, weights_path = _locate_adapter_files(lora_path)
    adapter_config = _read_json(config_path)
    pairs, unexpected = _parse_lora_pairs(weights_path)
    _validate_mergeable_config(adapter_config, unexpected)

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Merge output already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported --dtype {dtype!r}; choose bfloat16/float16/float32")
    torch_dtype = dtype_map[dtype.lower()]

    print(f"[merge] Loading pretrained DFlash: {draft_model}")
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    if verifier:
        load_kwargs["verifier"] = verifier
    model = SpeculatorModel.from_pretrained(draft_model, **load_kwargs)
    model.eval()

    declared = _target_module_declaration(adapter_config)
    rank_pattern = adapter_config.get("rank_pattern") or {}
    alpha_pattern = adapter_config.get("alpha_pattern") or {}
    default_alpha = float(adapter_config.get("lora_alpha", 1.0))
    default_r = int(adapter_config.get("r", 1))
    use_rslora = bool(adapter_config.get("use_rslora", False))
    fan_in_fan_out = bool(adapter_config.get("fan_in_fan_out", False))

    merged: list[dict[str, Any]] = []
    by_leaf: Counter[str] = Counter()

    with safe_open(str(weights_path), framework="pt", device="cpu") as sf:
        for (raw_module_name, adapter_name), part_keys in sorted(pairs.items()):
            if not _declared_target_allows(adapter_config, raw_module_name):
                raise RuntimeError(
                    f"Adapter weight {raw_module_name!r} is not allowed by "
                    f"adapter_config target_modules={declared!r}. Refusing ambiguous merge."
                )

            resolved_name, module = _resolve_model_module(model, raw_module_name)
            if not hasattr(module, "weight"):
                raise TypeError(f"Target module {resolved_name!r} has no .weight parameter")

            A = sf.get_tensor(part_keys["A"]).to(torch.float32)
            B = sf.get_tensor(part_keys["B"]).to(torch.float32)
            if A.ndim != 2 or B.ndim != 2:
                raise ValueError(
                    f"Expected 2D LoRA matrices for {raw_module_name}, got A{tuple(A.shape)} B{tuple(B.shape)}"
                )
            if B.shape[1] != A.shape[0]:
                raise ValueError(
                    f"LoRA rank mismatch for {raw_module_name}: A{tuple(A.shape)} B{tuple(B.shape)}"
                )

            actual_r = int(A.shape[0])
            configured_r = int(_pattern_value(rank_pattern, raw_module_name, default_r))
            if configured_r != actual_r:
                # The tensor is the source of truth for what was actually trained.
                print(
                    f"[merge] WARNING: {raw_module_name} config rank={configured_r}, "
                    f"tensor rank={actual_r}; using tensor rank."
                )
            alpha = float(_pattern_value(alpha_pattern, raw_module_name, default_alpha))
            scale = alpha / (math.sqrt(actual_r) if use_rslora else actual_r)

            delta = B @ A
            if fan_in_fan_out:
                delta = delta.T

            weight = module.weight
            if tuple(delta.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Merged delta shape mismatch for {resolved_name}: "
                    f"delta={tuple(delta.shape)}, weight={tuple(weight.shape)}"
                )

            with torch.no_grad():
                weight.add_(delta.to(device=weight.device, dtype=weight.dtype), alpha=scale)

            leaf = resolved_name.rsplit(".", 1)[-1]
            by_leaf[leaf] += 1
            merged.append(
                {
                    "adapter_module": raw_module_name,
                    "resolved_module": resolved_name,
                    "adapter_name_in_tensor": adapter_name,
                    "rank": actual_r,
                    "alpha": alpha,
                    "scale": scale,
                    "A_shape": list(A.shape),
                    "B_shape": list(B.shape),
                }
            )

    if not merged:
        raise RuntimeError("No LoRA modules were merged.")

    print(f"[merge] Merged {len(merged)} modules: {dict(sorted(by_leaf.items()))}")
    print(f"[merge] Saving standalone speculator checkpoint to: {output}")
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")

    manifest = {
        "base_draft_model": draft_model,
        "lora_path": str(Path(lora_path).expanduser().resolve()),
        "adapter_config": str(config_path),
        "adapter_weights": str(weights_path),
        "declared_target_modules": declared,
        "merged_module_count": len(merged),
        "merged_modules_by_leaf": dict(sorted(by_leaf.items())),
        "dtype": dtype,
        "use_rslora": use_rslora,
        "fan_in_fan_out": fan_in_fan_out,
        "merged_modules": merged,
    }
    _json_dump(manifest, output / "lora_merge_manifest.json")
    return manifest


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _http_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _server_base(endpoint: str) -> tuple[str, str]:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint[:-3], endpoint
    return endpoint, endpoint + "/v1"


def _fetch_model_id(v1: str) -> str:
    models = _http_json(v1 + "/models", timeout=30)
    data = models.get("data") or []
    if not data:
        raise RuntimeError(f"No model returned by {v1}/models")
    return str(data[0]["id"])


def _parse_spec_metrics(text: str) -> tuple[dict[str, float], dict[int, float]]:
    counters: dict[str, float] = defaultdict(float)
    per_pos: dict[int, float] = defaultdict(float)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_SIMPLE_RE.match(line) or _PROM_LABELED_RE.match(line)
        if not m:
            continue
        groups = m.groups()
        name = groups[0].replace("_total", "")
        if _SPEC_PREFIX not in name:
            continue
        value = float(groups[-1])
        labels = groups[1] if len(groups) == 3 else None
        if "per_pos" in name and labels:
            pm = re.search(r'position="(\d+)"', labels)
            if pm:
                per_pos[int(pm.group(1))] += value
        elif "per_pos" not in name:
            counters[name] += value
    return dict(counters), dict(per_pos)


def _metrics_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before, before_pos = _parse_spec_metrics(before_text)
    after, after_pos = _parse_spec_metrics(after_text)

    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    drafts = d("vllm:spec_decode_num_drafts")
    draft_tokens = d("vllm:spec_decode_num_draft_tokens")
    accepted = d("vllm:spec_decode_num_accepted_tokens")
    positions = sorted(set(before_pos) | set(after_pos))
    pos_delta = {p: after_pos.get(p, 0.0) - before_pos.get(p, 0.0) for p in positions}

    result: dict[str, Any] = {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "acceptance_length": (1.0 + accepted / drafts) if drafts > 0 else 0.0,
    }
    for p in positions:
        result[f"acceptance_at_pos_{p}"] = pos_delta[p] / drafts if drafts > 0 else 0.0
    return result


def _iter_local_json(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise TypeError(f"Expected JSON object per line in {path}")
                    yield obj
        return

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                raise TypeError(f"Expected list of JSON objects in {path}")
            yield row
    elif isinstance(obj, dict):
        # Accept common wrappers like {"data": [...]}.
        for key in ("data", "rows", "examples"):
            if isinstance(obj.get(key), list):
                for row in obj[key]:
                    if not isinstance(row, dict):
                        raise TypeError(f"Expected objects inside {key!r} in {path}")
                    yield row
                return
        yield obj
    else:
        raise TypeError(f"Unsupported JSON root in {path}: {type(obj).__name__}")


def _dataset_iter(
    data: str,
    *,
    data_file: str | None,
    dataset_config: str | None,
    split: str,
) -> Iterable[dict[str, Any]]:
    path = Path(data).expanduser()
    if path.exists():
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            return _iter_local_json(path)

        from datasets import Dataset, load_dataset, load_from_disk

        if path.is_file() and path.suffix.lower() == ".arrow":
            return iter(Dataset.from_file(str(path)))
        if path.is_file() and path.suffix.lower() in {".parquet", ".pq"}:
            return iter(load_dataset("parquet", data_files=str(path), split="train", streaming=True))
        if path.is_dir():
            ds = load_from_disk(str(path))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
            return iter(ds)
        raise ValueError(f"Unsupported local data source: {path}")

    # HF dataset. For speculator_benchmarks and similar repos, --data-file lets
    # us choose an exact JSONL file (e.g. HumanEval.jsonl) without relying on a
    # dataset builder/config whose schemas may differ across files.
    if data_file:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=data, filename=data_file, repo_type="dataset")
        return _iter_local_json(Path(local))

    from datasets import load_dataset

    ds = load_dataset(data, dataset_config, split=split, streaming=True)
    return iter(ds)


def _select_rows(
    data: str,
    *,
    data_file: str | None,
    dataset_config: str | None,
    split: str,
    start_index: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    rows = _dataset_iter(
        data,
        data_file=data_file,
        dataset_config=dataset_config,
        split=split,
    )
    selected = list(itertools.islice(rows, start_index, start_index + num_samples))
    if not selected:
        raise RuntimeError(
            f"No rows selected from data={data!r} at start_index={start_index}."
        )
    if len(selected) < num_samples:
        print(
            f"[eval] WARNING: requested {num_samples} rows but only {len(selected)} "
            "were available from the chosen start position."
        )
    return selected


def _normalize_message_list(value: list[Any]) -> list[dict[str, Any]] | None:
    if not value:
        return []
    if not all(isinstance(x, dict) for x in value):
        return None
    messages: list[dict[str, Any]] = []
    role_map = {"human": "user", "gpt": "assistant"}
    for item in value:
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role is None or content is None:
            return None
        messages.append({"role": role_map.get(str(role), str(role)), "content": content})
    return messages


def _row_to_messages(
    row: dict[str, Any], prompt_column: str, system_prompt: str | None
) -> list[dict[str, Any]]:
    if prompt_column not in row:
        # Helpful fallback for common chat datasets.
        for candidate in ("messages", "conversations", "input", "text"):
            if candidate in row:
                value = row[candidate]
                break
        else:
            raise KeyError(
                f"Prompt column {prompt_column!r} not found. Available columns: {sorted(row)}"
            )
    else:
        value = row[prompt_column]

    if isinstance(value, list):
        messages = _normalize_message_list(value)
        if messages is None:
            raise TypeError(
                f"Column {prompt_column!r} is a list but not a recognized chat message list"
            )
    elif isinstance(value, str):
        messages = [{"role": "user", "content": value}]
    elif isinstance(value, dict) and isinstance(value.get("text"), str):
        # Natural Questions:
        # question = {"text": "...", "tokens": [...]}
        messages = [
            {
                "role": "user",
                "content": value["text"],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": json.dumps(value, ensure_ascii=False),
            }
        ]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages

def _request_one(
    *,
    v1: str,
    model_id: str,
    index: int,
    messages: list[dict[str, Any]],
    max_new_tokens: int,
    timeout: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    enable_thinking: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if seed is not None:
        payload["seed"] = seed + index
    if enable_thinking != "auto":
        payload["chat_template_kwargs"] = {
            "enable_thinking": enable_thinking == "true"
        }

    started = time.perf_counter()
    try:
        result = _http_json(v1 + "/chat/completions", payload=payload, timeout=timeout)
        latency = time.perf_counter() - started
        choices = result.get("choices") or []
        content = None
        if choices:
            content = (choices[0].get("message") or {}).get("content")
        usage = result.get("usage") or {}
        return {
            "index": index,
            "ok": True,
            "latency_s": latency,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "content": content,
            "raw_id": result.get("id"),
        }
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - started
        return {
            "index": index,
            "ok": False,
            "latency_s": latency,
            "error": f"{type(exc).__name__}: {exc}",
        }


def evaluate(
    endpoint: str,
    data: str,
    *,
    data_file: str | None,
    dataset_config: str | None,
    split: str,
    prompt_column: str,
    start_index: int,
    num_samples: int,
    output_dir: str,
    max_new_tokens: int,
    concurrency: int,
    request_timeout: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    enable_thinking: str,
    system_prompt: str | None,
) -> dict[str, Any]:
    if concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    server, v1 = _server_base(endpoint)
    model_id = _fetch_model_id(v1)
    print(f"[eval] Server model: {model_id}")

    rows = _select_rows(
        data,
        data_file=data_file,
        dataset_config=dataset_config,
        split=split,
        start_index=start_index,
        num_samples=num_samples,
    )
    messages_list = [
        _row_to_messages(row, prompt_column=prompt_column, system_prompt=system_prompt)
        for row in rows
    ]
    print(
        f"[eval] Selected {len(rows)} rows: data={data!r}, data_file={data_file!r}, "
        f"split={split!r}, start={start_index}, count={len(rows)}"
    )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "selected_prompts.jsonl").open("w", encoding="utf-8") as f:
        for offset, (row, messages) in enumerate(zip(rows, messages_list)):
            f.write(
                json.dumps(
                    {
                        "dataset_index": start_index + offset,
                        "messages": messages,
                        "source_row": row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics_url = server + "/metrics"
    before_metrics = _http_text(metrics_url, timeout=30)

    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _request_one,
                v1=v1,
                model_id=model_id,
                index=start_index + i,
                messages=messages,
                max_new_tokens=max_new_tokens,
                timeout=request_timeout,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                enable_thinking=enable_thinking,
            )
            for i, messages in enumerate(messages_list)
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            status = "ok" if result["ok"] else "ERROR"
            print(f"[eval] {done:4d}/{len(futures)}  index={result['index']}  {status}")

    wall_s = time.perf_counter() - wall_start
    results.sort(key=lambda x: x["index"])

    # Let async metric updates flush before scraping the final counters.
    time.sleep(1.0)
    after_metrics = _http_text(metrics_url, timeout=30)
    acceptance = _metrics_delta(before_metrics, after_metrics)

    successes = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]
    completion_tokens = sum(
        int(r.get("completion_tokens") or 0) for r in successes
    )
    latency_values = [float(r["latency_s"]) for r in successes]

    summary: dict[str, Any] = {
        "server_model": model_id,
        "endpoint": v1,
        "data": data,
        "data_file": data_file,
        "dataset_config": dataset_config,
        "split": split,
        "prompt_column": prompt_column,
        "start_index": start_index,
        "requested_num_samples": num_samples,
        "selected_num_samples": len(rows),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "max_new_tokens": max_new_tokens,
        "concurrency": concurrency,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "wall_time_s": wall_s,
        "completion_tokens": completion_tokens,
        "output_tokens_per_s": completion_tokens / wall_s if wall_s > 0 else 0.0,
        "mean_request_latency_s": (
            sum(latency_values) / len(latency_values) if latency_values else 0.0
        ),
        **acceptance,
    }

    _json_dump(summary, out / "summary.json")
    with (out / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics_before.prom").write_text(before_metrics, encoding="utf-8")
    (out / "metrics_after.prom").write_text(after_metrics, encoding="utf-8")

    print("\n================ DFlash evaluation ================")
    print(f"model                  : {model_id}")
    print(f"samples                 : {len(successes)} success / {len(rows)} selected")
    print(f"num_drafts              : {acceptance['num_drafts']:.0f}")
    print(f"num_draft_tokens        : {acceptance['num_draft_tokens']:.0f}")
    print(f"num_accepted_tokens     : {acceptance['num_accepted_tokens']:.0f}")
    print(f"acceptance_length       : {acceptance['acceptance_length']:.6f}")
    pos_keys = sorted(
        (k for k in acceptance if k.startswith("acceptance_at_pos_")),
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    for key in pos_keys:
        zero_based = int(key.rsplit("_", 1)[-1])
        print(
            f"position {zero_based + 1:>2d} acceptance : "
            f"{100.0 * float(acceptance[key]):.3f}%"
        )
    print(f"output tokens/s         : {summary['output_tokens_per_s']:.3f}")
    print(f"results                 : {out}")
    print("===================================================")

    if failures:
        print(f"[eval] WARNING: {len(failures)} requests failed; see responses.jsonl")
    if acceptance["num_drafts"] <= 0:
        print(
            "[eval] WARNING: no speculative draft metrics increased. Check that the "
            "server was launched with DFlash speculative decoding enabled."
        )
    return summary


def _optional_float(value: str) -> float | None:
    if value.lower() in {"auto", "none", "default", ""}:
        return None
    return float(value)


def _optional_int(value: str) -> int | None:
    if value.lower() in {"auto", "none", "default", ""}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DFlash evaluation + automatic LoRA merge for SpecLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect-lora", help="Inspect exact modules stored in a LoRA adapter")
    p_inspect.add_argument("--lora-path", required=True)

    p_merge = sub.add_parser("merge", help="Merge a LoRA adapter into a pretrained DFlash checkpoint")
    p_merge.add_argument("--draft-model", required=True, help="HF id or local pretrained DFlash path")
    p_merge.add_argument("--lora-path", required=True, help="LoRA adapter directory or adapter file")
    p_merge.add_argument("--output-dir", required=True, help="Standalone merged DFlash output directory")
    p_merge.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p_merge.add_argument("--verifier", default=None, help="Optional verifier id/path for external-checkpoint conversion")
    p_merge.add_argument("--overwrite", action="store_true")
    p_merge.add_argument("--local-files-only", action="store_true")

    p_eval = sub.add_parser("evaluate", help="Evaluate a running vLLM DFlash server")
    p_eval.add_argument("--endpoint", required=True, help="Server root or /v1 URL")
    p_eval.add_argument("--data", required=True, help="Local json/jsonl/arrow/parquet/dataset-dir or HF dataset id")
    p_eval.add_argument("--data-file", default=None, help="Exact file inside an HF dataset repo, e.g. HumanEval.jsonl")
    p_eval.add_argument("--dataset-config", default=None, help="HF dataset config/subset when --data-file is not used")
    p_eval.add_argument("--split", default="train")
    p_eval.add_argument("--prompt-column", default="prompt")
    p_eval.add_argument("--start-index", type=int, default=0, help="Zero-based first dataset row to evaluate")
    p_eval.add_argument("--num-samples", type=int, default=80, help="Number of rows starting at --start-index")
    p_eval.add_argument("--output-dir", default="./eval_results")
    p_eval.add_argument("--max-new-tokens", type=int, default=512)
    p_eval.add_argument("--concurrency", type=int, default=1)
    p_eval.add_argument("--request-timeout", type=float, default=300.0)
    p_eval.add_argument("--temperature", type=_optional_float, default=None, help="Number, or auto/default to use server generation_config")
    p_eval.add_argument("--top-p", type=_optional_float, default=None, help="Number, or auto/default")
    p_eval.add_argument("--top-k", type=_optional_int, default=None, help="Integer, or auto/default")
    p_eval.add_argument("--seed", type=_optional_int, default=42, help="Integer, or none")
    p_eval.add_argument("--enable-thinking", choices=["auto", "true", "false"], default="auto", help="Qwen chat-template thinking switch; auto leaves server/model default unchanged")
    p_eval.add_argument("--system-prompt", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect-lora":
        summary = inspect_lora(args.lora_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "merge":
        manifest = merge_lora_into_dflash(
            args.draft_model,
            args.lora_path,
            args.output_dir,
            dtype=args.dtype,
            overwrite=args.overwrite,
            local_files_only=args.local_files_only,
            verifier=args.verifier,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        evaluate(
            args.endpoint,
            args.data,
            data_file=args.data_file,
            dataset_config=args.dataset_config,
            split=args.split,
            prompt_column=args.prompt_column,
            start_index=args.start_index,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            concurrency=args.concurrency,
            request_timeout=args.request_timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
            system_prompt=args.system_prompt,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
