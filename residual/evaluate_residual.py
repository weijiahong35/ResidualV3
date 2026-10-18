from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .mismatch import acceptance_structure, expected_accepted_length


@torch.no_grad()
def offline_greedy_metrics(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    *,
    block_size: int,
    sample_from_anchor: bool,
    block_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Strict prefix-acceptance diagnostics on a fixed reference trajectory.

    This is intentionally evaluation-only.  Both policies are decoded with a
    direct argmax; no sampling policy or soft training target is involved.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if target_logits.shape != draft_logits.shape:
        raise ValueError(
            f"target/draft logits differ: {target_logits.shape} vs {draft_logits.shape}"
        )
    if target_logits.ndim != 3:
        raise ValueError(
            f"expected logits with shape [N,T,V], got {tuple(target_logits.shape)}"
        )

    vocab_size = target_logits.shape[-1]
    if target_logits.shape[1] % block_size:
        raise ValueError(
            f"token length {target_logits.shape[1]} is not divisible by "
            f"block_size={block_size}"
        )
    target_blocks = target_logits.reshape(-1, block_size, vocab_size)
    draft_blocks = draft_logits.reshape(-1, block_size, vocab_size)
    num_blocks = target_blocks.shape[0]
    if num_blocks == 0:
        raise ValueError("cannot compute greedy metrics for zero blocks")
    valid_blocks = block_valid.to(
        device=target_logits.device, dtype=torch.bool
    ).reshape(-1)
    if valid_blocks.numel() != num_blocks:
        raise ValueError(
            f"block_valid has {valid_blocks.numel()} entries for {num_blocks} blocks"
        )

    proposal_start = 0 if sample_from_anchor else 1
    target_tokens = target_blocks[:, proposal_start:].argmax(dim=-1)
    draft_tokens = draft_blocks[:, proposal_start:].argmax(dim=-1)
    num_proposal_tokens = block_size - proposal_start

    matches = draft_tokens.eq(target_tokens)
    prefix_accept = matches.to(torch.long).cumprod(dim=-1)
    accepted_length = prefix_accept.sum(dim=-1)
    num_valid_blocks = valid_blocks.sum()
    has_valid_blocks = bool(valid_blocks.any().item())
    if has_valid_blocks:
        valid_accepted_length = accepted_length[valid_blocks]
        valid_prefix_accept = prefix_accept[valid_blocks]
        greedy_eal = valid_accepted_length.float().mean()
        accept_rate = greedy_eal / float(num_proposal_tokens)
        full_block_accept_rate = (
            valid_accepted_length.eq(num_proposal_tokens).float().mean()
        )
    else:
        zero = target_logits.new_zeros((), dtype=torch.float32)
        valid_prefix_accept = prefix_accept[valid_blocks]
        greedy_eal = zero
        accept_rate = zero
        full_block_accept_rate = zero

    metrics: dict[str, torch.Tensor] = {
        "offline_greedy_eal": greedy_eal,
        "offline_greedy_accept_rate": accept_rate,
        "offline_greedy_full_block_accept_rate": full_block_accept_rate,
        "offline_greedy_first_reject_mean": greedy_eal,
        "offline_num_valid_blocks": num_valid_blocks,
    }
    for pos in range(num_proposal_tokens):
        metrics[f"offline_greedy_position_accept_p{pos + 1}"] = (
            valid_prefix_accept[:, pos].float().mean()
            if has_valid_blocks
            else greedy_eal
        )
    return metrics


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten(); y = y.float().flatten()
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom.clamp_min(1e-12)).item())


def _ranks_with_ties(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu().flatten()
    order = torch.argsort(x, stable=True)
    sorted_x = x[order]
    ranks = torch.empty(x.numel(), dtype=torch.float32)
    i = 0
    while i < x.numel():
        j = i + 1
        while j < x.numel() and sorted_x[j] == sorted_x[i]:
            j += 1
        rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float().flatten().cpu(); target = target.float().flatten().cpu()
    if pred.numel() == 0:
        return {
            "mae": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "r2": float("nan"),
        }
    mae = (pred - target).abs().mean()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float("nan") if float(ss_tot) <= 1e-12 else float((1.0 - ss_res / ss_tot).item())
    return {
        "mae": float(mae.item()),
        "pearson": _pearson(pred, target),
        "spearman": _pearson(_ranks_with_ties(pred), _ranks_with_ties(target)),
        "r2": r2,
    }


@torch.no_grad()
def evaluate_predictor(
    trainer,
    *,
    max_batches: int = 0,
    output_json: str | Path | None = None,
) -> dict[str, float]:
    """Aggregate rejection-aware Predictor metrics for every progressive prefix.

    For a five-layer DFlash the normal keys are p2/p3/p4, corresponding to the
    decisions before Layers 3/4/5.  The function keeps the old unsuffixed p2
    metrics for compatibility, while the new prefixed metrics are the canonical
    progressive evaluation output.
    """
    loader = trainer.val_loader
    if loader is None:
        return {}
    model = trainer.model
    was_training = model.training
    model.eval()

    # prefix -> metric-name -> list[tensor]
    buckets: dict[int, dict[str, list[torch.Tensor]]] = {}
    base_metric_sums: dict[str, float] = {}
    n_batches = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        with torch.autocast(
            trainer.device_type,
            dtype=trainer.hidden_states_dtype,
            enabled=trainer.device_type == "cuda",
        ):
            out = model.forward_residual(
                **trainer._gpu_batch(batch),
                max_anchors=trainer.max_anchors,
                gamma=trainer.gamma,
            )
        predictions = out.predictions
        if predictions is None and out.prediction is not None:
            predictions = [out.prediction]
        if not predictions or out.mismatch is None:
            continue

        target = out.mismatch
        valid = target.valid_mask
        block_valid = target.first_reject_prob.sum(dim=-1) > 1e-8

        for pred in predictions:
            k_value = pred.get("prefix_layers")
            k = int(k_value.detach().item()) if torch.is_tensor(k_value) else int(k_value or 2)
            b = buckets.setdefault(
                k,
                {"pred_s": [], "true_s": [], "pred_r": [], "true_r": [],
                 "pred_fr": [], "true_fr": [], "pred_eal": [], "true_eal": []},
            )

            if valid.any():
                b["pred_s"].append(pred["severity"][valid].detach().float().cpu())
                b["true_s"].append(target.severity[valid].detach().float().cpu())

            if block_valid.any():
                b["pred_r"].append(
                    pred["repair_direction"][block_valid].detach().float().cpu()
                )
                b["true_r"].append(
                    target.repair_direction[block_valid].detach().float().cpu()
                )

            pred_overlap = torch.exp(-pred["severity"].float()).clamp(0.0, 1.0)
            _, pred_first_reject, _, _ = acceptance_structure(pred_overlap, valid)
            if valid.any():
                b["pred_fr"].append(pred_first_reject[valid].detach().cpu())
                b["true_fr"].append(
                    target.first_reject_prob[valid].detach().float().cpu()
                )

            b["pred_eal"].append(
                expected_accepted_length(pred_overlap, valid).detach().cpu()
            )
            b["true_eal"].append(
                expected_accepted_length(target.overlap, valid).detach().cpu()
            )

        for key, value in out.metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                base_metric_sums[key] = base_metric_sums.get(key, 0.0) + float(
                    value.detach().float().cpu()
                )
            elif isinstance(value, (int, float)):
                base_metric_sums[key] = base_metric_sums.get(key, 0.0) + float(value)
        n_batches += 1

    result: dict[str, float] = {}
    prefix_summaries: dict[int, dict[str, float]] = {}

    for k in sorted(buckets):
        b = buckets[k]
        summary: dict[str, float] = {}
        if b["pred_s"]:
            ps = torch.cat(b["pred_s"])
            ts = torch.cat(b["true_s"])
            for name, value in regression_metrics(ps, ts).items():
                summary[f"nla_{name}"] = value
            # RMSE is useful for calibration but is intentionally kept separate
            # from regression_metrics to preserve its old schema.
            summary["nla_rmse"] = float(((ps - ts) ** 2).mean().sqrt().item())
            summary["nla_bias"] = float((ps - ts).mean().item())

            pred_tv = 1.0 - torch.exp(-ps.clamp_min(0.0))
            true_tv = 1.0 - torch.exp(-ts.clamp_min(0.0))
            for name, value in regression_metrics(pred_tv, true_tv).items():
                summary[f"tv_{name}"] = value
            summary["tv_rmse"] = float(((pred_tv - true_tv) ** 2).mean().sqrt().item())

            pred_overlap = torch.exp(-ps.clamp_min(0.0))
            true_overlap = torch.exp(-ts.clamp_min(0.0))
            summary["overlap_mae"] = float((pred_overlap - true_overlap).abs().mean().item())
            summary["overlap_bias"] = float((pred_overlap - true_overlap).mean().item())

        if b["pred_r"]:
            pr = torch.cat(b["pred_r"])
            tr = torch.cat(b["true_r"])
            cos = F.cosine_similarity(pr, tr, dim=-1)
            summary["repair_cosine"] = float(cos.mean().item())
            summary["repair_cosine_std"] = float(cos.std(unbiased=False).item())

        if b["pred_fr"]:
            pfr = torch.cat(b["pred_fr"])
            tfr = torch.cat(b["true_fr"])
            for name, value in regression_metrics(pfr, tfr).items():
                summary[f"first_reject_{name}"] = value
            summary["first_reject_rmse"] = float(((pfr - tfr) ** 2).mean().sqrt().item())

        if b["pred_eal"]:
            pe = torch.cat(b["pred_eal"])
            te = torch.cat(b["true_eal"])
            for name, value in regression_metrics(pe, te).items():
                summary[f"expected_accept_len_{name}"] = value
            summary["expected_accept_len_rmse"] = float(((pe - te) ** 2).mean().sqrt().item())
            summary["expected_accept_len_bias"] = float((pe - te).mean().item())

        prefix_summaries[k] = summary
        for name, value in summary.items():
            result[f"p{k}_{name}"] = value

    # Compatibility: old scripts read unsuffixed metrics from the first routing
    # prefix (normally p2). New analyses should use p2_/p3_/p4_ explicitly.
    if prefix_summaries:
        first_k = min(prefix_summaries)
        for name, value in prefix_summaries[first_k].items():
            result.setdefault(name, value)

    # Directly expose whether deeper evidence improves prediction quality.
    ordered = sorted(prefix_summaries)
    for k1, k2 in zip(ordered[:-1], ordered[1:], strict=True):
        s1, s2 = prefix_summaries[k1], prefix_summaries[k2]
        if "nla_mae" in s1 and "nla_mae" in s2:
            result[f"progress_nla_mae_gain_p{k1}_p{k2}"] = s1["nla_mae"] - s2["nla_mae"]
        if "nla_rmse" in s1 and "nla_rmse" in s2:
            result[f"progress_nla_rmse_gain_p{k1}_p{k2}"] = s1["nla_rmse"] - s2["nla_rmse"]
        if "repair_cosine" in s1 and "repair_cosine" in s2:
            result[f"progress_repair_cos_gain_p{k1}_p{k2}"] = (
                s2["repair_cosine"] - s1["repair_cosine"]
            )
        if "expected_accept_len_mae" in s1 and "expected_accept_len_mae" in s2:
            result[f"progress_eal_mae_gain_p{k1}_p{k2}"] = (
                s1["expected_accept_len_mae"] - s2["expected_accept_len_mae"]
            )

    for key, value in base_metric_sums.items():
        result[f"model_{key}"] = value / max(n_batches, 1)

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if was_training:
        model.train()
    return result
