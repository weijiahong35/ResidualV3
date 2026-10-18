#!/usr/bin/env python3
"""Evaluate Residual/MoLE DFlash against the repo's vLLM hidden-state verifier.

This intentionally follows the same split used by the repository's training and
example evaluation scripts:

* vLLM/CUDA13 owns the verifier model and hidden-state extraction.
* SPEC_PYTHON owns dataset loading plus the local DFlash + Residual model.

V20.6 evaluation explicitly forces the verifier to greedy decoding (temperature=0,
top_p=1, top_k=0). The returned continuation is therefore the Target argmax
continuation used for strict block-level acceptance accounting.

Acceptance length uses the same definition as examples/evaluate/evaluate_dflash_lora.py:
    1 + num_accepted_tokens / num_drafts

This path is for acceptance/correctness comparison.  Wall-clock throughput is not
comparable to stock vLLM DFlash because hidden states cross the file/HTTP boundary
once per speculative round.
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
from typing import Any, Iterable

import openai

# The compiled local DFlash runtime graph silently produces incorrect logits in
# this environment. Keep vLLM compiled in its independent Target process, but
# run the small local Draft query block eagerly for correctness.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
HS_ROOT = PROJECT_ROOT / "hs_connectors" / "src"
for p in (HS_ROOT, SRC_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hs_connectors import FileTransfer  # noqa: E402
from residual.checkpoint import load_residual_checkpoint  # noqa: E402
from residual.config import ResidualConfig  # noqa: E402
from residual.model_residual import ResidualDFlashModel  # noqa: E402
from residual.state_cache import PreviousRoundState  # noqa: E402
from speculators.data_generation.vllm_client import generate_hidden_states  # noqa: E402
from speculators.model import SpeculatorModel  # noqa: E402

_DRAFT_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


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

    return iter(load_dataset(data, dataset_config, split=split, streaming=True))


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _dataset_iter(
        args.data,
        data_file=args.data_file,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    selected = list(itertools.islice(rows, args.start_index, args.start_index + args.num_samples))
    if not selected:
        raise RuntimeError(
            f"No rows selected from data={args.data!r} at start_index={args.start_index}."
        )
    return selected


def _normalize_message_list(value: list[Any]) -> list[dict[str, Any]] | None:
    if not value:
        return []
    if not all(isinstance(x, dict) for x in value):
        return None
    role_map = {"human": "user", "gpt": "assistant"}
    out: list[dict[str, Any]] = []
    for item in value:
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role is None or content is None:
            return None
        out.append({"role": role_map.get(str(role), str(role)), "content": content})
    return out


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


def _checkpoint_payload(checkpoint: str) -> dict[str, Any]:
    p = Path(checkpoint)
    if p.is_file():
        p = p.parent
    cfg = p / "residual_mole_config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Residual checkpoint config not found: {cfg}")
    return json.loads(cfg.read_text(encoding="utf-8"))


def _runtime_config(checkpoint: str) -> ResidualConfig:
    payload = _checkpoint_payload(checkpoint)
    raw = dict(payload.get("residual_config") or {})
    allowed = {f.name for f in fields(ResidualConfig)}
    raw = {k: v for k, v in raw.items() if k in allowed}
    raw["train_stage"] = "eval"
    cfg = ResidualConfig(**raw)
    cfg.validate()
    return cfg


def _load_draft(args: argparse.Namespace):
    from transformers import AutoTokenizer
    from speculators.models.attention import create_float_mask

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.draft_device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )
    base = SpeculatorModel.from_pretrained(
        args.draft_model,
        verifier=args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if base.__class__.__name__ != "DFlashDraftModel":
        raise TypeError(f"Expected DFlashDraftModel, got {type(base).__name__}")

    # Only block_size query rows attend over the Target context, so eager's
    # dense [8, context] mask remains small and avoids the bad compiled graph.
    base.config.transformer_layer_config._attn_implementation = "eager"
    base._attn_impl = "eager"
    base._create_mask_fn = create_float_mask

    model = ResidualDFlashModel(base, _runtime_config(args.checkpoint))
    missing, unexpected = load_residual_checkpoint(model, args.checkpoint, strict=False)
    critical = [
        k
        for k in missing
        if k.startswith("predictor.")
        or k.startswith("routers.")
        or "residual_mole." in k
    ]
    if critical:
        raise RuntimeError(f"Residual checkpoint incomplete; missing={critical[:12]}")
    if unexpected:
        print(f"[mole-eval] checkpoint unexpected keys: {unexpected[:12]}")

    model.to(device)
    model.eval()
    return tokenizer, model, device


def _prompt_ids(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: str) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        # transformers 5 defaults this to True and returns a BatchEncoding.
        # This evaluator needs the flat token-id sequence shared by the local
        # draft and the remote verifier.
        "return_dict": False,
    }
    if enable_thinking != "auto":
        kwargs["enable_thinking"] = enable_thinking == "true"
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    # Be defensive around tokenizer implementations that ignore return_dict.
    if isinstance(ids, Mapping):
        ids = ids.get("input_ids")
        if ids is None:
            raise RuntimeError("Chat template result is missing input_ids")
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    elif hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise RuntimeError(f"Expected one tokenized prompt, got batch size={len(ids)}")
        ids = ids[0]
    return [int(x) for x in ids]


def _eos_ids(tokenizer: Any) -> set[int]:
    out: set[int] = set()
    value = tokenizer.eos_token_id
    if value is not None:
        if isinstance(value, (list, tuple, set)):
            out.update(int(x) for x in value)
        else:
            out.add(int(value))
    return out


def _server_model(client: openai.OpenAI) -> str:
    models = client.models.list().data
    if not models:
        raise RuntimeError("vLLM returned no model from /v1/models")
    return str(models[0].id)


def _fetch_hidden_states(
    *,
    client: openai.OpenAI,
    model_id: str,
    transfer: FileTransfer,
    prefix_ids: list[int],
    timeout: float,
) -> torch.Tensor:
    handle = generate_hidden_states(
        client,
        model_id,
        {"input_ids": prefix_ids},
        timeout=timeout,
        max_retries=0,
    )
    try:
        sample = transfer.get_generated(handle)
        if sample is None:
            raise RuntimeError(f"Failed to read hidden states handle={handle}")
        token_ids = sample.get("token_ids")
        hs = sample.get("hidden_states")
        if token_ids is None or hs is None:
            raise RuntimeError(f"Hidden-state payload missing keys: {sorted(sample)}")
        got = [int(x) for x in token_ids.reshape(-1).tolist()]
        if got != prefix_ids:
            raise RuntimeError(
                f"Hidden-state token mismatch: expected len={len(prefix_ids)}, got len={len(got)}"
            )
        if hs.ndim != 3:
            raise RuntimeError(f"Expected hidden_states [T,L,H], got {tuple(hs.shape)}")
        return hs
    finally:
        try:
            transfer.delete(handle)
        except FileNotFoundError:
            pass


def _delete_response_hidden_states(response: Any, transfer: FileTransfer) -> None:
    """Delete the hidden-state side effect of an ordinary vLLM completion.

    ExampleHiddenStatesConnector writes a file for every request, including the
    greedy target-continuation requests whose hidden states we do not consume.
    Leaving those response handles untouched leaks one growing safetensors file
    per speculative round.
    """
    params = getattr(response, "kv_transfer_params", None)
    if not isinstance(params, Mapping):
        return
    handle = params.get("hidden_states_path") or params.get("handle")
    if not handle:
        return
    try:
        transfer.delete(str(handle))
    except FileNotFoundError:
        pass


def _target_continuation(
    *,
    client: openai.OpenAI,
    model_id: str,
    transfer: FileTransfer,
    prefix_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> list[int]:
    if max_tokens <= 0:
        return []
    res = client.completions.create(
        model=model_id,
        prompt=prefix_ids,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body={"return_token_ids": True, "top_k": 0},
        timeout=timeout,
    )
    try:
        token_ids = getattr(res.choices[0], "token_ids", None)
        if token_ids is None:
            raise RuntimeError(
                "vLLM completion response missing choice.token_ids; "
                "return_token_ids is required"
            )
        return [int(x) for x in token_ids]
    finally:
        _delete_response_hidden_states(res, transfer)


def _draft_target_ids(logits: torch.Tensor, model: ResidualDFlashModel) -> list[int]:
    ids = logits.argmax(dim=-1).reshape(-1)
    base = model.base_model
    if getattr(base, "use_draft_vocab", False):
        if base.d2t is None:
            raise RuntimeError("Reduced-vocab DFlash has no d2t map")
        offsets = base.d2t.to(device=ids.device, dtype=torch.long)
        ids = ids + offsets[ids]
    return [int(x) for x in ids.detach().cpu().tolist()]


def _decode_one(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    draft: ResidualDFlashModel,
    draft_device: torch.device,
    model_id: str,
    messages: list[dict[str, Any]],
    dataset_index: int,
    eos_ids: set[int],
) -> dict[str, Any]:
    started = time.perf_counter()
    client = openai.OpenAI(base_url=args.endpoint.rstrip("/") + "/", api_key="EMPTY", max_retries=0)
    transfer = FileTransfer(Path(args.hidden_states_path))

    prompt_ids = _prompt_ids(tokenizer, messages, args.enable_thinking)
    budget = min(args.max_new_tokens, args.max_model_len - len(prompt_ids))
    if budget <= 0:
        raise RuntimeError(
            f"prompt length={len(prompt_ids)} leaves no budget under max_model_len={args.max_model_len}"
        )

    prefix = list(prompt_ids)
    generated: list[int] = []
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    num_emitted_accepted = 0
    num_full_block_accepted = 0
    accepted_at_pos = [0] * args.num_spec_tokens
    attempted_at_pos = [0] * args.num_spec_tokens
    fixed_lag_history_replays = 0
    residual_correction_rounds = 0
    history_state = None

    # DFlash requires one Target-sampled anchor before its first proposal. After
    # that, a rejected Target token (or the all-accepted bonus token) is both an
    # emitted token and the next round's anchor. It must not be sampled again.
    initial_anchor = _target_continuation(
        client=client,
        model_id=model_id,
        transfer=transfer,
        prefix_ids=prefix,
        max_tokens=1,
        timeout=args.request_timeout,
    )
    if initial_anchor:
        prefix.append(initial_anchor[0])
        generated.append(initial_anchor[0])

    while generated and len(generated) < budget and generated[-1] not in eos_ids:
        remaining = budget - len(generated)
        # prefix already ends in the current Target anchor. Ask for K verifier
        # tokens plus one possible all-accepted bonus/next-anchor token.
        verifier_capacity = args.max_model_len - len(prefix)
        target_tokens = _target_continuation(
            client=client,
            model_id=model_id,
            transfer=transfer,
            prefix_ids=prefix,
            max_tokens=min(args.num_spec_tokens + 1, verifier_capacity),
            timeout=args.request_timeout,
        )
        if not target_tokens:
            break

        # Match native vLLM accounting: the final decode iteration still drafts
        # a complete block when verifier context capacity permits, even if only
        # part of that block can be emitted under max_new_tokens.
        k = min(args.num_spec_tokens, len(target_tokens))
        anchored_prefix = prefix
        hs = _fetch_hidden_states(
            client=client,
            model_id=model_id,
            transfer=transfer,
            prefix_ids=anchored_prefix,
            timeout=args.request_timeout,
        )

        n_aux = len(draft.target_layer_ids)
        if hs.shape[1] < n_aux + 1:
            raise RuntimeError(
                f"vLLM returned {hs.shape[1]} hidden-state layers; expected at least "
                f"{n_aux} DFlash aux layers + final verifier layer. Check TARGET_LAYER_IDS."
            )
        aux_with_anchor = hs[:, :n_aux, :].reshape(1, hs.shape[0], -1).to(
            device=draft_device, dtype=next(draft.parameters()).dtype
        )
        # Runtime contract: Target context states [0:P], input_ids [0:P+1],
        # where the final input token is the Target-sampled anchor.
        aux_context = aux_with_anchor[:, :-1, :]
        ids_t = torch.tensor([anchored_prefix], device=draft_device, dtype=torch.long)

        with _DRAFT_LOCK, torch.inference_mode():
            q_logits, proposed_state, diagnostics = draft.propose_runtime(
                aux_context,
                ids_t,
                previous_state=history_state,
                build_base_next_state=False,
            )
        if float(diagnostics.get("runtime_residual_correction_active", 0.0)) > 0:
            residual_correction_rounds += 1
        proposal = _draft_target_ids(q_logits[0, :k], draft)
        next_history_state = PreviousRoundState(
            target_layers=draft.runtime_target_state(
                aux_with_anchor, token_index=-1
            ),
            drafter_layers=proposed_state.drafter_layers,
        )
        verification_tokens = target_tokens[:k]

        num_drafts += 1
        num_draft_tokens += k
        for i in range(k):
            attempted_at_pos[i] += 1

        accepted = 0
        rejected = False
        accepted_eos = False
        for i in range(k):
            if proposal[i] != verification_tokens[i]:
                rejected = True
                break
            accepted += 1
            num_accepted += 1
            accepted_at_pos[i] += 1
            if proposal[i] in eos_ids:
                accepted_eos = True
                break

        # A context-limited/EOS round can contain fewer than NUM_SPEC_TOKENS; a
        # block is complete when every token actually proposed was accepted.
        if accepted == k:
            num_full_block_accepted += 1

        if accepted_eos:
            commit = proposal[:accepted]
        elif rejected:
            # The correction token is also the next round's DFlash anchor.
            commit = [*proposal[:accepted], verification_tokens[accepted]]
        else:
            commit = proposal[:k]
            # When room remains, Target's bonus token is the next anchor.
            if len(target_tokens) > k and len(commit) < remaining:
                commit.append(target_tokens[k])

        if len(commit) > remaining:
            commit = commit[:remaining]
        num_emitted_accepted += min(accepted, len(commit))
        prefix.extend(commit)
        generated.extend(commit)
        history_state = next_history_state
        if (
            args.progress_every_rounds > 0
            and num_drafts % args.progress_every_rounds == 0
        ):
            with _PRINT_LOCK:
                print(
                    f"[mole-eval:progress] index={dataset_index} "
                    f"generated={len(generated)}/{budget} rounds={num_drafts} "
                    f"acceptance_length={1.0 + num_accepted / num_drafts:.4f} "
                    f"residual_rounds={residual_correction_rounds}"
                )
        if any(t in eos_ids for t in commit):
            eos_pos = next(i for i, t in enumerate(generated) if t in eos_ids)
            generated = generated[: eos_pos + 1]
            break

    latency = time.perf_counter() - started
    return {
        "dataset_index": dataset_index,
        "ok": True,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(generated),
        "latency_s": latency,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "num_emitted_accepted_tokens": num_emitted_accepted,
        "num_non_draft_output_tokens": len(generated) - num_emitted_accepted,
        "num_full_block_accepted": num_full_block_accepted,
        "accepted_at_pos": accepted_at_pos,
        "attempted_at_pos": attempted_at_pos,
        "acceptance_length": 1.0 + num_accepted / num_drafts if num_drafts else 0.0,
        "fixed_lag_history_replays": fixed_lag_history_replays,
        "residual_correction_rounds": residual_correction_rounds,
        "content": tokenizer.decode(generated, skip_special_tokens=True),
    }


def evaluate(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, draft, draft_device = _load_draft(args)
    if args.num_spec_tokens != int(draft.block_size - 1):
        raise ValueError(
            f"DFlash block_size={draft.block_size}, expected NUM_SPEC_TOKENS={draft.block_size - 1}, "
            f"got {args.num_spec_tokens}"
        )

    rows = _select_rows(args)
    messages = [_row_to_messages(r, args.prompt_column, args.system_prompt) for r in rows]
    eos_ids = _eos_ids(tokenizer)
    discovery_client = openai.OpenAI(
        base_url=args.endpoint.rstrip("/") + "/", api_key="EMPTY", max_retries=0
    )
    model_id = _server_model(discovery_client)
    discovery_client.close()

    # Validate tokenizer output before launching all concurrent requests.
    _prompt_ids(tokenizer, messages[0], args.enable_thinking)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "selected_prompts.jsonl").open("w", encoding="utf-8") as f:
        for i, (row, msg) in enumerate(zip(rows, messages)):
            f.write(
                json.dumps(
                    {
                        "dataset_index": args.start_index + i,
                        "messages": msg,
                        "source_row": row,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[mole-eval] samples={len(rows)} slice=[{args.start_index}:{args.start_index + len(rows)}] "
        f"endpoint={args.endpoint} draft_device={draft_device}"
    )
    print(f"[mole-eval] Server model: {model_id}")
    print("[mole-eval] Draft=direct argmax; Target=vLLM greedy argmax; no sampling")
    print("[mole-eval] Local Draft attention=eager; torch.compile disabled for correctness")
    print(
        "[mole-eval] REQUEST_TIMEOUT applies to each vLLM HTTP request, "
        "not to the whole generated sample"
    )
    print(
        f"[mole-eval] Runtime progress is reported every "
        f"{args.progress_every_rounds} speculative rounds per active sample"
    )

    wall0 = time.perf_counter()
    ordered: list[dict[str, Any] | None] = [None] * len(messages)

    def run(item: tuple[int, list[dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
        i, msg = item
        idx = args.start_index + i
        try:
            r = _decode_one(
                args=args,
                tokenizer=tokenizer,
                draft=draft,
                draft_device=draft_device,
                model_id=model_id,
                messages=msg,
                dataset_index=idx,
                eos_ids=eos_ids,
            )
        except Exception as exc:  # noqa: BLE001
            r = {"dataset_index": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return i, r

    workers = min(max(1, args.concurrency), len(messages))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, item) for item in enumerate(messages)]
        for done, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            i, r = fut.result()
            ordered[i] = r
            status = "ok" if r.get("ok") else "ERROR"
            print(
                f"[mole-eval] {done:4d}/{len(messages)} index={args.start_index + i} "
                f"{status} acceptance_length={float(r.get('acceptance_length', 0.0)):.4f}"
            )

    results = [r for r in ordered if r is not None]
    with (out / "responses.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]
    drafts = sum(int(r.get("num_drafts", 0)) for r in ok)
    draft_tokens = sum(int(r.get("num_draft_tokens", 0)) for r in ok)
    accepted = sum(int(r.get("num_accepted_tokens", 0)) for r in ok)
    emitted_accepted = sum(int(r.get("num_emitted_accepted_tokens", 0)) for r in ok)
    non_draft_output = sum(int(r.get("num_non_draft_output_tokens", 0)) for r in ok)
    full_block_accepted = sum(int(r.get("num_full_block_accepted", 0)) for r in ok)
    completion = sum(int(r.get("completion_tokens", 0)) for r in ok)
    residual_correction_rounds = sum(
        int(r.get("residual_correction_rounds", 0)) for r in ok
    )
    pos_accept = [0] * args.num_spec_tokens
    pos_attempt = [0] * args.num_spec_tokens
    for r in ok:
        for i, v in enumerate(r.get("accepted_at_pos", [])):
            pos_accept[i] += int(v)
        for i, v in enumerate(r.get("attempted_at_pos", [])):
            pos_attempt[i] += int(v)

    wall_s = time.perf_counter() - wall0
    summary: dict[str, Any] = {
        "backend": "vllm_hidden_state_verifier_plus_local_residual_dflash",
        "mode": "mole",
        "throughput_comparable_to_stock_vllm": False,
        "draft_attention_implementation": "eager",
        "local_draft_torch_compile_enabled": False,
        "endpoint": args.endpoint,
        "verifier_model": args.verifier_model,
        "draft_model": args.draft_model,
        "checkpoint": args.checkpoint,
        "start_index": args.start_index,
        "requested_num_samples": args.num_samples,
        "selected_num_samples": len(rows),
        "successful_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "num_emitted_accepted_tokens": emitted_accepted,
        "num_non_draft_output_tokens": non_draft_output,
        "num_full_block_accepted": full_block_accepted,
        "seed": args.seed,
        "acceptance_length": 1.0 + accepted / drafts if drafts else 0.0,
        "runtime_greedy_eal": accepted / drafts if drafts else 0.0,
        "runtime_greedy_accept_rate": (
            accepted / draft_tokens if draft_tokens else 0.0
        ),
        "runtime_greedy_full_block_accept_rate": (
            full_block_accepted / drafts if drafts else 0.0
        ),
        "runtime_full_block_accept_rate": (
            full_block_accepted / drafts if drafts else 0.0
        ),
        "runtime_greedy_first_reject_mean": accepted / drafts if drafts else 0.0,
        "completion_tokens": completion,
        "residual_correction_rounds": residual_correction_rounds,
        "wall_time_s": wall_s,
        "output_tokens_per_s": completion / wall_s if wall_s > 0 else 0.0,
        "num_speculative_tokens": args.num_spec_tokens,
    }
    for i in range(args.num_spec_tokens):
        summary[f"acceptance_at_pos_{i}"] = (
            pos_accept[i] / pos_attempt[i] if pos_attempt[i] else 0.0
        )
        summary[f"runtime_greedy_position_accept_p{i + 1}"] = (
            pos_accept[i] / pos_attempt[i] if pos_attempt[i] else 0.0
        )
    _json_dump(summary, out / "summary.json")

    print("\n================ Residual MoLE evaluation ================")
    print(f"samples                 : {len(ok)} success / {len(rows)} selected")
    print(f"num_drafts              : {drafts}")
    print(f"num_draft_tokens        : {draft_tokens}")
    print(f"num_accepted_tokens     : {accepted}")
    print(f"num_non_draft_output_tokens : {non_draft_output}")
    print(f"acceptance_length       : {summary['acceptance_length']:.6f}")
    print(f"runtime_greedy_eal      : {summary['runtime_greedy_eal']:.6f}")
    print(f"runtime_greedy_accept_rate : {summary['runtime_greedy_accept_rate']:.6f}")
    print(
        "runtime_full_block_accept_rate : "
        f"{summary['runtime_full_block_accept_rate']:.6f}"
    )
    print(f"residual_correction_rounds : {residual_correction_rounds}")
    for i in range(args.num_spec_tokens):
        print(f"position {i + 1:>2d} acceptance : {100.0 * summary[f'acceptance_at_pos_{i}']:.3f}%")
    print(f"results                 : {out}")
    print(
        "NOTE: runtime_greedy_eal is the primary metric; "
        f"acceptance_length = 1 + EAL (maximum {args.num_spec_tokens + 1})."
    )
    print("==========================================================")
    if failures:
        print(f"[mole-eval] WARNING: {len(failures)} requests failed; see responses.jsonl")
    if not ok:
        raise RuntimeError(
            f"All {len(failures)} evaluation requests failed; "
            f"first error: {failures[0].get('error', 'unknown error')}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--hidden-states-path", required=True)
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--draft-device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--data", required=True)
    p.add_argument("--data-file", default=None)
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--prompt-column", default="problem")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=9216)
    p.add_argument("--num-spec-tokens", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--progress-every-rounds", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-thinking", choices=["true", "false", "auto"], default="true")
    p.add_argument("--system-prompt", default=None)
    return p


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
