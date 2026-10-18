#!/usr/bin/env python3
"""True end-to-end speculative-decoding evaluation for Residual DFlash.

This executes the real end-to-end speculative-decoding loop for every
decoding round:

  greedy DFlash/Residual proposal -> greedy target verification -> accept
  prefix -> greedy target correction/bonus token -> next round.

The acceptance counters use the same definition as the project's original
vLLM evaluator:

    acceptance_length = 1 + num_accepted_tokens / num_drafts

Both Draft and Target tokens use direct ``argmax``. This evaluator performs no
temperature, top-p, top-k, multinomial, or rejection sampling.

Modes
-----
base
    Original pretrained DFlash; no Predictor/Router/MoLE.
residual
    Full progressive Predictor + Router + gated MoLE correction.

Important performance note
--------------------------
This reference path evaluates the *real acceptance/rejection algorithm* and its
acceptance_length, but it is not the stock vLLM DFlash CUDA/KV-cache proposer.
The current Residual model needs previous-round verifier + drafter state and its
training contract stores base-DFlash previous-round drafter states; residual
mode therefore performs the explicit base-history replay used by
``ResidualDFlashModel.propose_runtime``. Acceptance metrics are meaningful and
comparable under the same prompts and greedy policy. Wall-time/throughput from
this Python reference path must not be presented as stock-vLLM speedup.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Sequence

# The Residual proposer passes dynamic Predictor/Router control dictionaries
# through the DFlash layer wrappers. In this environment torch.compile silently
# produces incorrect Draft logits for that runtime graph (including eval_base),
# while eager execution matches the stock vLLM DFlash proposals. Disable Dynamo
# before importing torch; the Target HF model is not torch.compile'd here.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from residual.checkpoint import load_residual_checkpoint  # noqa: E402
from residual.config import ResidualConfig  # noqa: E402
from residual.model_residual import ResidualDFlashModel  # noqa: E402
from residual.state_cache import PreviousRoundState  # noqa: E402
from speculators.model import SpeculatorModel  # noqa: E402


# Serialize the reference proposer because this evaluator shares one Draft module.
# Residual proposer forwards therefore must not overlap across request threads.  Target
# verification is stateless and remains free to overlap with draft work from other requests.
_DRAFT_FORWARD_LOCK = threading.Lock()


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_local_json(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"Expected JSON object per line in {path}")
                    yield row
        return
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                raise TypeError(f"Expected list of JSON objects in {path}")
            yield row
        return
    if isinstance(obj, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(obj.get(key), list):
                for row in obj[key]:
                    if not isinstance(row, dict):
                        raise TypeError(f"Expected objects inside {key!r} in {path}")
                    yield row
                return
        yield obj
        return
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
            return iter(
                load_dataset("parquet", data_files=str(path), split="train", streaming=True)
            )
        if path.is_dir():
            ds = load_from_disk(str(path))
            if hasattr(ds, "keys") and split in ds:
                ds = ds[split]
            return iter(ds)
        raise ValueError(f"Unsupported local data source: {path}")

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
    rows = _dataset_iter(
        data, data_file=data_file, dataset_config=dataset_config, split=split
    )
    selected = list(itertools.islice(rows, start_index, start_index + num_samples))
    if not selected:
        raise RuntimeError(
            f"No rows selected from data={data!r} at start_index={start_index}."
        )
    return selected


def _normalize_message_list(value: list[Any]) -> list[dict[str, Any]] | None:
    if not value:
        return []
    if not all(isinstance(x, dict) for x in value):
        return None
    role_map = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, Any]] = []
    for item in value:
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role is None or content is None:
            return None
        messages.append(
            {"role": role_map.get(str(role), str(role)), "content": content}
        )
    return messages


def _row_to_messages(
    row: dict[str, Any], prompt_column: str, system_prompt: str | None
) -> list[dict[str, Any]]:
    if prompt_column in row:
        value = row[prompt_column]
    else:
        for candidate in ("messages", "conversations", "input", "text"):
            if candidate in row:
                value = row[candidate]
                break
        else:
            raise KeyError(
                f"Prompt column {prompt_column!r} not found. Available={sorted(row)}"
            )
    if isinstance(value, list):
        messages = _normalize_message_list(value)
        if messages is None:
            raise TypeError(f"Unrecognized chat message list in {prompt_column!r}")
    elif isinstance(value, str):
        messages = [{"role": "user", "content": value}]
    elif isinstance(value, dict) and isinstance(value.get("text"), str):
        messages = [{"role": "user", "content": value["text"]}]
    else:
        messages = [{"role": "user", "content": json.dumps(value, ensure_ascii=False)}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def _checkpoint_payload(checkpoint: str | None) -> dict[str, Any]:
    if not checkpoint:
        return {}
    p = Path(checkpoint)
    if p.is_file():
        p = p.parent
    cfg = p / "residual_mole_config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Residual checkpoint config not found: {cfg}")
    return json.loads(cfg.read_text(encoding="utf-8"))


def _residual_config_for_runtime(mode: str, checkpoint: str | None) -> ResidualConfig:
    stage = {"base": "eval_base", "residual": "eval"}[mode]
    payload = _checkpoint_payload(checkpoint)
    raw = dict(payload.get("residual_config") or {})
    allowed = {f.name for f in fields(ResidualConfig)}
    raw = {k: v for k, v in raw.items() if k in allowed}
    raw["train_stage"] = stage
    cfg = ResidualConfig(**raw)
    cfg.validate()
    return cfg


def _load_models(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from speculators.models.attention import create_float_mask

    target_device = torch.device(args.target_device)
    draft_device = torch.device(args.draft_device)
    dtype = getattr(torch, args.dtype)

    # Keep the legacy shell parameter semantics as closely as possible:
    # GPU_MEMORY_UTILIZATION belonged to the Target/vLLM side, not the Draft.
    # In this single-process reference backend we therefore apply the fraction
    # only to a separate Target GPU.  When Target and Draft share one GPU, a
    # per-process fraction would cap *both* models and would not match the old
    # two-process setup, so no artificial fraction is installed in that case.
    if not (0.0 < float(args.gpu_memory_utilization) <= 1.0):
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if (
        target_device.type == "cuda"
        and target_device != draft_device
        and float(args.gpu_memory_utilization) < 1.0
    ):
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_memory_utilization), device=target_device
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.verifier_model)
    except Exception:
        gen_cfg = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(args.verifier_model, trust_remote_code=args.trust_remote_code)
        )

    target_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.target_attn_implementation != "auto":
        target_kwargs["attn_implementation"] = args.target_attn_implementation
    target = AutoModelForCausalLM.from_pretrained(args.verifier_model, **target_kwargs)
    target.to(target_device)
    target.eval()

    # SpeculatorModel.from_pretrained reloads the verifier embedding/lm-head
    # weights expected by the DFlash checkpoint.
    draft_base = SpeculatorModel.from_pretrained(
        args.draft_model,
        verifier=args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if draft_base.__class__.__name__ != "DFlashDraftModel":
        raise TypeError(f"Expected DFlashDraftModel, got {type(draft_base).__name__}")

    # Runtime drafting has only block_size (normally 8) query rows. A dense
    # eager mask is therefore small even at a 9k context, and avoids the broken
    # compiled FlexAttention graph described above.
    draft_base.config.transformer_layer_config._attn_implementation = "eager"
    draft_base._attn_impl = "eager"
    draft_base._create_mask_fn = create_float_mask

    cfg = _residual_config_for_runtime(args.mode, args.checkpoint)
    model = ResidualDFlashModel(draft_base, cfg)
    if args.mode != "base":
        if not args.checkpoint:
            raise ValueError(f"mode={args.mode} requires --checkpoint")
        missing, unexpected = load_residual_checkpoint(model, args.checkpoint, strict=False)
        critical = [x for x in missing if x.startswith("predictor.")]
        if args.mode == "residual":
            critical += [
                x
                for x in missing
                if x.startswith("routers.") or "residual_mole." in x
            ]
        if critical:
            raise RuntimeError(
                f"Checkpoint is incomplete for mode={args.mode}; first missing keys: {critical[:8]}"
            )
        if unexpected:
            print(f"[runtime] checkpoint unexpected keys: {unexpected[:8]}")
    model.to(draft_device)
    model.eval()
    return tokenizer, gen_cfg, target, model, target_device, draft_device


def _draft_argmax_target_ids(
    draft_logits: torch.Tensor,
    model: ResidualDFlashModel,
) -> torch.Tensor:
    """Direct Draft argmax, mapped to verifier token IDs when necessary."""
    ids = draft_logits.argmax(dim=-1)
    base = model.base_model
    if not getattr(base, "use_draft_vocab", False):
        return ids
    if base.d2t is None:
        raise RuntimeError("DFlash uses reduced draft vocab but d2t mapping is missing")
    d2t = base.d2t.to(device=ids.device, dtype=torch.long)
    # d2t stores an offset: target_id = draft_row + d2t[draft_row].
    return ids + d2t[ids]


def _selected_aux(
    outputs: Any,
    layer_ids: Sequence[int],
    *,
    to_device: torch.device,
    take: int | None = None,
) -> torch.Tensor:
    """Return [1,T,L*H] matching vLLM eagle_aux_hidden_state_layer_ids semantics.

    HF ``hidden_states[i]`` is the state at layer-boundary i; layer id
    ``num_hidden_layers`` is therefore the final decoder output, exactly matching
    the extraction convention used by SpecLoRA's launch_vllm.py.
    """
    hs = outputs.hidden_states
    if hs is None:
        raise RuntimeError("Verifier output_hidden_states=True returned no hidden states")
    picked = []
    for idx in layer_ids:
        if idx < 0 or idx >= len(hs):
            raise IndexError(f"target aux layer id {idx} outside hidden_states[0:{len(hs)}]")
        t = hs[idx]
        if take is not None:
            t = t[:, :take, :]
        picked.append(t)
    fused = torch.cat(picked, dim=-1)
    return fused.detach().to(device=to_device, non_blocking=True)


def _crop_cache(cache: Any, max_length: int) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    # Legacy tuple cache: each layer is typically (key, value, ...), with
    # sequence dimension -2 for Qwen-like attention caches.
    if isinstance(cache, (tuple, list)):
        out_layers = []
        for layer in cache:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise TypeError("Unsupported legacy past_key_values structure")
            items = list(layer)
            for i in (0, 1):
                if torch.is_tensor(items[i]):
                    items[i] = items[i][..., :max_length, :]
            out_layers.append(type(layer)(items) if isinstance(layer, tuple) else items)
        return tuple(out_layers) if isinstance(cache, tuple) else out_layers
    raise TypeError(
        f"Cannot crop cache of type {type(cache).__name__}; upgrade transformers or use a DynamicCache-capable Qwen3 build"
    )


def _eos_set(tokenizer: Any, gen_cfg: Any) -> set[int]:
    ids: list[int] = []
    for value in (getattr(gen_cfg, "eos_token_id", None), tokenizer.eos_token_id):
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            ids.extend(int(x) for x in value)
        else:
            ids.append(int(value))
    return set(ids)


def _prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: str,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        # transformers 5 defaults to BatchEncoding; the runtime needs the
        # actual [1, T] input_ids tensor.
        "return_dict": False,
    }
    if enable_thinking != "auto":
        kwargs["enable_thinking"] = enable_thinking == "true"
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, Mapping):
        ids = ids.get("input_ids")
        if ids is None:
            raise RuntimeError("Chat template result is missing input_ids")
    if not isinstance(ids, torch.Tensor):
        ids = torch.as_tensor(ids)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    return ids.long()


def _decode_one(
    *,
    tokenizer: Any,
    target: Any,
    draft: ResidualDFlashModel,
    target_device: torch.device,
    draft_device: torch.device,
    messages: list[dict[str, Any]],
    dataset_index: int,
    max_new_tokens: int,
    max_model_len: int,
    num_spec_tokens: int,
    enable_thinking: str,
    eos_ids: set[int],
    request_timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()

    def _check_timeout() -> None:
        if request_timeout > 0 and (time.perf_counter() - started) > request_timeout:
            raise TimeoutError(
                f"request exceeded REQUEST_TIMEOUT={request_timeout:.1f}s"
            )
    prompt_cpu = _prompt_ids(tokenizer, messages, enable_thinking=enable_thinking)
    prompt_len = int(prompt_cpu.shape[1])
    budget = min(max_new_tokens, max_model_len - prompt_len)
    if budget <= 0:
        return {
            "dataset_index": dataset_index,
            "ok": False,
            "error": f"prompt length {prompt_len} leaves no room under max_model_len={max_model_len}",
        }

    target_ids = prompt_cpu.to(target_device)
    draft_ids = prompt_cpu.to(draft_device)

    with torch.inference_mode():
        init = target(
            input_ids=target_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    past = init.past_key_values
    current_logits = init.logits[:, -1, :].detach()
    layer_ids = list(draft.target_layer_ids)
    draft_aux = _selected_aux(init, layer_ids, to_device=draft_device)
    del init

    generated: list[int] = []
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    num_full_block_accepted = 0
    accepted_at_pos = [0 for _ in range(num_spec_tokens)]
    attempted_at_pos = [0 for _ in range(num_spec_tokens)]
    base_history_replays = 0
    fixed_lag_history_replays = 0
    residual_correction_rounds = 0
    stopped_eos = False
    history_state = None

    while len(generated) < budget:
        _check_timeout()
        # DFlash's query anchor is the mandatory token sampled by Target. The
        # Draft predicts only the tokens *after* this anchor. This is the same
        # next_token_ids + mask-token contract used by vLLM's DFlash proposer.
        anchor = int(current_logits[0].argmax(dim=-1).item())
        generated.append(anchor)
        if anchor in eos_ids:
            stopped_eos = True
            break

        remaining = budget - len(generated)
        k = min(num_spec_tokens, remaining)
        if k <= 0:
            break

        anchor_draft = torch.tensor([[anchor]], device=draft_device, dtype=torch.long)
        proposal_input_ids = torch.cat([draft_ids, anchor_draft], dim=1)
        # V20.6 keeps the V10 lag-1 information contract. history_state is the
        # completed previous speculative round, never the current one.
        with _DRAFT_FORWARD_LOCK, torch.inference_mode():
            q_logits, proposed_state, diagnostics = draft.propose_runtime(
                draft_aux,
                proposal_input_ids,
                previous_state=history_state,
                build_base_next_state=False,
            )
        if float(diagnostics.get("runtime_base_history_replay", 0.0)) > 0:
            base_history_replays += 1
        if float(diagnostics.get("runtime_residual_correction_active", 0.0)) > 0:
            residual_correction_rounds += 1
        q_logits = q_logits[0, :k]
        proposal = _draft_argmax_target_ids(q_logits, draft).to(target_device).tolist()
        proposal_t = torch.tensor([proposal], device=target_device, dtype=torch.long)

        # Verify anchor + all K draft tokens in one Target pass. The logits at
        # row i verify proposal i; the final row samples the next round's anchor.
        prefix_len = int(target_ids.shape[1]) + len(generated)
        verify_input = torch.cat(
            [
                torch.tensor([[anchor]], device=target_device, dtype=torch.long),
                proposal_t,
            ],
            dim=1,
        )
        with torch.inference_mode():
            verified = target(
                input_ids=verify_input,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        verify_past = verified.past_key_values
        verify_aux = _selected_aux(verified, layer_ids, to_device=draft_device)

        num_drafts += 1
        num_draft_tokens += k
        for i in range(k):
            attempted_at_pos[i] += 1
        accepted_this_round = 0
        reject_pos: int | None = None
        accepted_tokens: list[int] = []

        for i, token in enumerate(proposal):
            target_token = int(verified.logits[0, i].argmax(dim=-1).item())
            if token == target_token:
                accepted_this_round += 1
                num_accepted += 1
                accepted_at_pos[i] += 1
                accepted_tokens.append(token)
                if token in eos_ids:
                    stopped_eos = True
                    break
            else:
                reject_pos = i
                accepted_tokens.append(target_token)
                break

        if accepted_this_round == k:
            num_full_block_accepted += 1

        generated.extend(accepted_tokens)
        if len(generated) > budget:
            generated = generated[:budget]

        if stopped_eos or len(generated) >= budget:
            del verified
            break

        if reject_pos is not None:
            # Commit anchor + accepted draft prefix, discard the rejected suffix,
            # then run the Target recovery token to update KV + aux state.
            accepted_prefix = reject_pos
            # prefix_len already includes the emitted anchor.
            past = _crop_cache(verify_past, prefix_len + accepted_prefix)
            committed_ids = [anchor, *proposal[:accepted_prefix]]
            draft_aux = torch.cat(
                [draft_aux, verify_aux[:, : 1 + accepted_prefix, :]], dim=1
            )
            draft_ids = torch.cat(
                [
                    draft_ids,
                    torch.tensor(
                        [committed_ids], device=draft_device, dtype=torch.long
                    ),
                ],
                dim=1,
            )
            recovery = accepted_tokens[-1]
            recovery_t = torch.tensor([[recovery]], device=target_device, dtype=torch.long)
            with torch.inference_mode():
                rec = target(
                    input_ids=recovery_t,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            past = rec.past_key_values
            current_logits = rec.logits[:, -1, :].detach()
            draft_aux = torch.cat(
                [draft_aux, _selected_aux(rec, layer_ids, to_device=draft_device)], dim=1
            )
            draft_ids = torch.cat(
                [draft_ids, torch.tensor([[recovery]], device=draft_device)], dim=1
            )
            if recovery in eos_ids:
                stopped_eos = True
            del rec
        else:
            # All K drafts accepted. The final verifier logits become the next
            # round's mandatory anchor; do not count it as a speculative token.
            past = verify_past
            draft_aux = torch.cat([draft_aux, verify_aux[:, : k + 1, :]], dim=1)
            draft_ids = torch.cat(
                [
                    draft_ids,
                    torch.tensor(
                        [[anchor, *proposal]], device=draft_device, dtype=torch.long
                    ),
                ],
                dim=1,
            )
            current_logits = verified.logits[:, -1, :].detach()

        # The current anchor's Target state is produced by this verifier pass.
        # Pair it with the current DFlash anchor states for the next round's
        # lag-1 Predictor history, regardless of accept/reject outcome.
        history_state = PreviousRoundState(
            target_layers=draft.runtime_target_state(verify_aux, token_index=0),
            drafter_layers=proposed_state.drafter_layers,
        )

        del verified
        if stopped_eos:
            break

    latency = time.perf_counter() - started
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "dataset_index": dataset_index,
        "ok": True,
        "prompt_tokens": prompt_len,
        "completion_tokens": len(generated),
        "latency_s": latency,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "num_full_block_accepted": num_full_block_accepted,
        "accepted_at_pos": accepted_at_pos,
        "attempted_at_pos": attempted_at_pos,
        "acceptance_length": (1.0 + num_accepted / num_drafts) if num_drafts else 0.0,
        "base_history_replays": base_history_replays,
        "fixed_lag_history_replays": fixed_lag_history_replays,
        "residual_correction_rounds": residual_correction_rounds,
        "history_round_stride_tokens": 0,
        "stopped_eos": stopped_eos,
        "content": text,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, gen_cfg, target, draft, target_device, draft_device = _load_models(args)
    eos_ids = _eos_set(tokenizer, gen_cfg)

    if draft.cfg.draft_mode != "greedy":
        raise ValueError("V20 evaluator requires draft_mode='greedy'")

    if args.num_spec_tokens != int(draft.block_size - 1):
        raise ValueError(
            f"Loaded DFlash block_size={draft.block_size}, sample_from_anchor=False => "
            f"{draft.block_size - 1} speculative tokens; got --num-spec-tokens={args.num_spec_tokens}"
        )

    rows = _select_rows(
        args.data,
        data_file=args.data_file,
        dataset_config=args.dataset_config,
        split=args.split,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    messages = [
        _row_to_messages(r, args.prompt_column, args.system_prompt) for r in rows
    ]
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "selected_prompts.jsonl").open("w", encoding="utf-8") as f:
        for offset, (row, msg) in enumerate(zip(rows, messages)):
            f.write(
                json.dumps(
                    {
                        "dataset_index": args.start_index + offset,
                        "messages": msg,
                        "source_row": row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[runtime] mode={args.mode} verifier={args.verifier_model} draft={args.draft_model} "
        f"samples={len(rows)} slice=[{args.start_index}:{args.start_index + len(rows)}]"
    )
    print(
        "[runtime] true greedy SD: Draft argmax + Target argmax + prefix rejection | "
        "temperature=0 top_p=1 top_k=0 (metadata only; implementation uses argmax)"
    )
    print(
        f"[runtime] target_device={target_device} draft_device={draft_device}; "
        "acceptance metrics are comparable; this reference backend's wall-time is not stock-vLLM throughput"
    )
    print("[runtime] Draft attention=eager; torch.compile disabled for correctness")

    wall0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    total_drafts = total_draft_tokens = total_accepted = 0
    total_full_block_accepted = 0
    pos_counts = [0 for _ in range(args.num_spec_tokens)]
    pos_attempts = [0 for _ in range(args.num_spec_tokens)]
    total_completion = 0
    total_replays = 0
    total_fixed_lag_replays = 0
    total_residual_correction_rounds = 0

    # CONCURRENCY is implemented as concurrent request state machines sharing the
    # read-only model weights.  The Residual proposer itself is serialized by
    # _DRAFT_FORWARD_LOCK protects the shared reference Draft module;
    # target verification from other requests may overlap with proposer work.
    token_budget_cap = max(1, int(args.max_num_batched_tokens) // max(1, int(args.max_model_len)))
    effective_concurrency = min(
        max(1, int(args.concurrency)),
        max(1, int(args.max_num_seqs)),
        token_budget_cap,
        len(messages),
    )
    if effective_concurrency < int(args.concurrency):
        print(
            f"[runtime] requested CONCURRENCY={args.concurrency} capped to {effective_concurrency} "
            f"by MAX_NUM_SEQS={args.max_num_seqs} / "
            f"MAX_NUM_BATCHED_TOKENS={args.max_num_batched_tokens} / MAX_MODEL_LEN={args.max_model_len}"
        )
    else:
        print(f"[runtime] effective concurrency={effective_concurrency}")

    def _run_request(item: tuple[int, list[dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
        offset, msg = item
        idx = args.start_index + offset
        try:
            result = _decode_one(
                tokenizer=tokenizer,
                target=target,
                draft=draft,
                target_device=target_device,
                draft_device=draft_device,
                messages=msg,
                dataset_index=idx,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                num_spec_tokens=args.num_spec_tokens,
                enable_thinking=args.enable_thinking,
                eos_ids=eos_ids,
                request_timeout=float(args.request_timeout),
            )
        except Exception as exc:  # noqa: BLE001
            result = {
                "dataset_index": idx,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return offset, result

    ordered: list[dict[str, Any] | None] = [None] * len(messages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
        future_to_offset = {
            pool.submit(_run_request, item): item[0]
            for item in enumerate(messages)
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_offset):
            offset, result = future.result()
            ordered[offset] = result
            completed += 1
            idx = args.start_index + offset
            status = "ok" if result.get("ok") else "ERROR"
            acc_len = float(result.get("acceptance_length", 0.0))
            print(
                f"[runtime] {completed:4d}/{len(messages)} idx={idx} {status} "
                f"acceptance_length={acc_len:.4f}"
            )

    # Aggregate and write in deterministic dataset order regardless of completion order.
    for result in ordered:
        assert result is not None
        results.append(result)
        if result.get("ok"):
            total_drafts += int(result["num_drafts"])
            total_draft_tokens += int(result["num_draft_tokens"])
            total_accepted += int(result["num_accepted_tokens"])
            total_full_block_accepted += int(result["num_full_block_accepted"])
            total_completion += int(result["completion_tokens"])
            total_replays += int(result["base_history_replays"])
            total_fixed_lag_replays += int(result.get("fixed_lag_history_replays", 0))
            total_residual_correction_rounds += int(
                result.get("residual_correction_rounds", 0)
            )
            for i, count in enumerate(result["accepted_at_pos"]):
                pos_counts[i] += int(count)
            for i, count in enumerate(result.get("attempted_at_pos", [])):
                pos_attempts[i] += int(count)
        with (out / "responses.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    wall_s = time.perf_counter() - wall0
    successful = sum(1 for r in results if r.get("ok"))
    summary: dict[str, Any] = {
        "backend": "hf_reference_true_specdecode",
        "true_speculative_decoding": True,
        "throughput_comparable_to_stock_vllm": False,
        "draft_attention_implementation": "eager",
        "torch_compile_enabled": False,
        "mode": args.mode,
        "verifier_model": args.verifier_model,
        "draft_model": args.draft_model,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "data_file": args.data_file,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "prompt_column": args.prompt_column,
        "start_index": args.start_index,
        "requested_num_samples": args.num_samples,
        "selected_num_samples": len(rows),
        "successful_requests": successful,
        "failed_requests": len(rows) - successful,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "concurrency_requested": args.concurrency,
        "concurrency_effective": effective_concurrency,
        "request_timeout_s": args.request_timeout,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "scheduler_parameter_semantics": "reference_threaded_backend_not_vllm_continuous_batching",
        "num_speculative_tokens": args.num_spec_tokens,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "draft_sample_method": "direct_argmax",
        "target_sample_method": "direct_argmax",
        "rejection_sample_method": "greedy_prefix_equality",
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted,
        "num_full_block_accepted": total_full_block_accepted,
        "acceptance_length": (
            1.0 + total_accepted / total_drafts if total_drafts else 0.0
        ),
        "runtime_greedy_eal": (
            total_accepted / total_drafts if total_drafts else 0.0
        ),
        "runtime_greedy_accept_rate": (
            total_accepted / total_draft_tokens
            if total_draft_tokens else 0.0
        ),
        "runtime_greedy_full_block_accept_rate": (
            total_full_block_accepted / total_drafts if total_drafts else 0.0
        ),
        "runtime_full_block_accept_rate": (
            total_full_block_accepted / total_drafts if total_drafts else 0.0
        ),
        "runtime_greedy_first_reject_mean": (
            total_accepted / total_drafts if total_drafts else 0.0
        ),
        "base_history_replays": total_replays,
        "fixed_lag_history_replays": total_fixed_lag_replays,
        "residual_correction_rounds": total_residual_correction_rounds,
        "history_round_stride_tokens": 0,
        "wall_time_s": wall_s,
        "completion_tokens": total_completion,
        "reference_output_tokens_per_s": total_completion / wall_s if wall_s else 0.0,
    }
    for i, count in enumerate(pos_counts, start=1):
        denom = pos_attempts[i - 1]
        summary[f"acceptance_at_pos_{i}"] = count / denom if denom else 0.0
        summary[f"runtime_greedy_position_accept_p{i}"] = (
            count / denom if denom else 0.0
        )
        summary[f"attempted_at_pos_{i}"] = denom

    _json_dump(summary, out / "summary.json")
    print("\n================ TRUE speculative decoding ================")
    print(f"mode                    : {args.mode}")
    print(f"samples                 : {successful}/{len(rows)}")
    print(f"num_drafts              : {total_drafts}")
    print(f"num_draft_tokens        : {total_draft_tokens}")
    print(f"num_accepted_tokens     : {total_accepted}")
    print(f"acceptance_length       : {summary['acceptance_length']:.6f}")
    print(f"runtime_greedy_eal      : {summary['runtime_greedy_eal']:.6f}")
    print(f"runtime_greedy_accept_rate : {summary['runtime_greedy_accept_rate']:.6f}")
    print(
        "runtime_full_block_accept_rate : "
        f"{summary['runtime_full_block_accept_rate']:.6f}"
    )
    for i in range(1, args.num_spec_tokens + 1):
        print(f"acceptance_at_pos_{i:<2d}    : {summary[f'acceptance_at_pos_{i}']:.6f}")
    print(f"base_history_replays    : {total_replays}")
    print(f"fixed_lag_replays       : {total_fixed_lag_replays}")
    print(f"residual_correction_rounds : {total_residual_correction_rounds}")
    print("history_stride_tokens   : 0 (V20 current-state Predictor)")
    print(
        "NOTE: runtime_greedy_eal is the primary metric; "
        "acceptance_length keeps the legacy +1 convention."
    )
    print("NOTE: reference wall-time is not stock-vLLM speed.")
    print(f"results                 : {out}")
    if successful == 0:
        first_error = next(
            (str(r.get("error")) for r in results if not r.get("ok")),
            "unknown error",
        )
        raise RuntimeError(
            f"All {len(results)} evaluation requests failed; first error: {first_error}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["base", "residual"], default="residual")
    p.add_argument("--verifier-model", default="Qwen/Qwen3-8B")
    p.add_argument("--draft-model", default="RedHatAI/Qwen3-8B-speculator.dflash")
    p.add_argument("--checkpoint")
    p.add_argument("--target-device", default="cuda:0")
    p.add_argument("--draft-device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--target-attn-implementation", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--data", default="nvidia/OpenMathInstruct-2")
    p.add_argument("--data-file")
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="train_1M")
    p.add_argument("--prompt-column", default="problem")
    p.add_argument("--start-index", type=int, default=10000)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--system-prompt", default="")

    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=9216)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--max-num-seqs", type=int, default=6)
    p.add_argument("--max-num-batched-tokens", type=int, default=55296)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    p.add_argument("--num-spec-tokens", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-thinking", choices=["true", "false", "auto"], default="true")
    p.add_argument("--output-dir", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "base":
        args.checkpoint = None
    if args.num_samples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--num-samples and --max-new-tokens must be > 0")
    evaluate(args)


if __name__ == "__main__":
    main()
