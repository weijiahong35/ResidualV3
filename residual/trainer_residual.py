from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from speculators.train.utils import normalize_counted_metrics

from .checkpoint import save_residual_checkpoint
from .evaluate_residual import offline_greedy_metrics

logger = logging.getLogger(__name__)

class _StepTimer:
    """Baseline-compatible step profiler. Synchronizes only on logged steps."""

    def __init__(self) -> None:
        self.enabled = False
        self._marks: dict[str, float] = {}

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks.clear()

    def mark_value(self, name: str, value: float) -> None:
        if self.enabled:
            self._marks[name] = value

    def mark(self, name: str) -> None:
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._marks[name] = time.perf_counter()

    def now(self) -> float:
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def profile(self, num_tokens: int) -> dict[str, float] | None:
        if not self.enabled:
            return None
        m = self._marks
        fetch_ms = (m["fetch"] - m["start"]) * 1000.0
        fwd_ms = (m["fwd"] - m["fetch"]) * 1000.0
        bwd_ms = (m["bwd"] - m["fwd"]) * 1000.0
        opt_ms = (m["opt"] - m["bwd"]) * 1000.0
        step_ms = (m["opt"] - m["start"]) * 1000.0
        tokens_per_s = num_tokens / (step_ms / 1000.0) if step_ms > 0 else 0.0
        fetch_frac = fetch_ms / step_ms if step_ms > 0 else 0.0
        return {
            "fetch_ms": fetch_ms,
            "fwd_ms": fwd_ms,
            "bwd_ms": bwd_ms,
            "opt_ms": opt_ms,
            "step_ms": step_ms,
            "tokens_per_s": tokens_per_s,
            "fetch_frac": fetch_frac,
        }



def _scalar_metrics(metrics: dict) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                out[k] = float(v.detach().float().cpu().item())
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _normalize_headline_metrics(vals: dict[str, float], total_loss: float) -> dict[str, float]:
    """Normalize DFlash *_sum/*_total metrics without losing residual total loss."""
    normalized = normalize_counted_metrics(dict(vals), world_size=1)
    # compute_metrics() exposes its own counted `loss`; preserve it as accept_loss.
    accept_loss = normalized.get("loss")
    if accept_loss is not None:
        normalized["accept_loss"] = accept_loss
    normalized["loss"] = total_loss
    return normalized


@torch.no_grad()
def _training_acceptance_metrics(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    *,
    block_size: int,
    sample_from_anchor: bool,
    block_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose the shared offline greedy metric under training-log names."""
    offline = offline_greedy_metrics(
        target_logits,
        draft_logits,
        block_size=block_size,
        sample_from_anchor=sample_from_anchor,
        block_valid=block_valid,
    )
    num_proposal_tokens = block_size - (0 if sample_from_anchor else 1)
    metrics = {
        "accept_eal": offline["offline_greedy_eal"],
        "accept_num_valid_blocks": offline["offline_num_valid_blocks"],
    }
    for pos in range(1, num_proposal_tokens + 1):
        metrics[f"accept_position_{pos}"] = offline[
            f"offline_greedy_position_accept_p{pos}"
        ]
    return metrics


class ResidualTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        *,
        base_lr: float,
        epochs: int,
        weight_decay: float,
        save_dir: str,
        residual_config: dict,
        max_steps: int | None = None,
        log_freq: int = 10,
        max_anchors: int = 3072,
        gamma: float = 4.0,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics_path = self.save_dir / "train_metrics.jsonl"
        self.validation_metrics_path = self.save_dir / "validation_metrics.jsonl"
        self.residual_config = residual_config
        self.max_steps = max_steps
        self.log_freq = log_freq
        self.max_anchors = max_anchors
        self.gamma = gamma
        self.hidden_states_dtype = hidden_states_dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = self.device.type
        self.model.to(self.device)
        if model.cfg.train_stage in {"eval", "eval_base", "eval_nomole"}:
            self.optimizer = None
        else:
            groups = model.trainable_parameter_groups(base_lr)
            self.optimizer = AdamW(groups, weight_decay=weight_decay)


    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _series(vals: dict[str, float], stem: str) -> str:
        items = []
        for key in sorted(vals):
            if key.startswith(stem):
                items.append(f"{key[len(stem):]}={vals[key]:.4f}")
        return ",".join(items) if items else "-"

    def _gpu_batch(self, batch):
        dtype = self.hidden_states_dtype
        out = {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                out[k] = v
            elif v.is_floating_point():
                out[k] = v.to(self.device, dtype=dtype, non_blocking=True)
            else:
                out[k] = v.to(self.device, non_blocking=True)
        return out

    @torch.no_grad()
    def validate(self, max_batches: int = 0) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        total_loss_sum = 0.0
        greedy_sums: dict[str, float] = {}
        greedy_metric_names: set[str] = set()
        greedy_blocks = 0
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            with torch.autocast(
                self.device_type, dtype=self.hidden_states_dtype, enabled=self.device_type == "cuda"
            ):
                out = self.model.forward_residual(
                    **self._gpu_batch(batch),
                    max_anchors=self.max_anchors,
                    gamma=self.gamma,
                )
                loss, metrics = out.loss, out.metrics
            if out.pair_valid is not None:
                block_valid = out.pair_valid
            elif out.mismatch is not None:
                block_valid = out.mismatch.valid_mask.any(dim=-1)
            else:
                raise RuntimeError("Validation output has no block-valid mask")
            batch_greedy = offline_greedy_metrics(
                out.targets,
                out.logits,
                block_size=self.model.block_size,
                sample_from_anchor=self.model.base_model.config.sample_from_anchor,
                block_valid=block_valid,
            )
            batch_valid_blocks = int(
                batch_greedy["offline_num_valid_blocks"].detach().cpu().item()
            )
            for k, v in batch_greedy.items():
                if k != "offline_num_valid_blocks":
                    greedy_metric_names.add(k)
                if k != "offline_num_valid_blocks" and batch_valid_blocks:
                    greedy_sums[k] = greedy_sums.get(k, 0.0) + float(
                        v.detach().float().cpu().item()
                    ) * batch_valid_blocks
            greedy_blocks += batch_valid_blocks
            vals = _scalar_metrics(metrics)
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            total_loss_sum += float(loss.detach().float().item())
            n += 1
        self.model.train()
        if n == 0:
            return {}
        averaged = {k: v / n for k, v in sums.items()}
        normalized = _normalize_headline_metrics(averaged, total_loss_sum / n)
        if greedy_blocks:
            normalized.update({k: v / greedy_blocks for k, v in greedy_sums.items()})
        else:
            normalized.update({k: 0.0 for k in greedy_metric_names})
        normalized["offline_num_valid_blocks"] = float(greedy_blocks)
        return {f"val_{k}": v for k, v in normalized.items()}

    def _log_evaluation(self, step: int, val: dict[str, float]) -> None:
        if not val:
            return
        num_proposal_tokens = self.model.block_size - (
            0 if self.model.base_model.config.sample_from_anchor else 1
        )
        positions = ",".join(
            f"p{pos}={val.get(f'val_offline_greedy_position_accept_p{pos}', float('nan')):.4f}"
            for pos in range(1, num_proposal_tokens + 1)
        )
        logger.info(
            "EVAL step=%d soft_eal=%.4f offline_greedy_eal=%.4f "
            "offline_greedy_accept_rate=%.4f "
            "offline_greedy_full_block_accept_rate=%.4f "
            "offline_greedy_first_reject_mean=%.4f "
            "offline_num_valid_blocks=%d offline_greedy_position_accept[%s]",
            step,
            val.get("val_soft_eal", float("nan")),
            val.get("val_offline_greedy_eal", float("nan")),
            val.get("val_offline_greedy_accept_rate", float("nan")),
            val.get("val_offline_greedy_full_block_accept_rate", float("nan")),
            val.get("val_offline_greedy_first_reject_mean", float("nan")),
            int(val.get("val_offline_num_valid_blocks", 0.0)),
            positions,
        )

    def train(self) -> None:
        stage = self.model.cfg.train_stage
        global_step = 0
        timer = _StepTimer()
        for epoch in range(self.epochs):
            self.model.train()
            t_before_fetch = time.perf_counter()
            for step, batch in enumerate(self.train_loader, 1):
                global_step += 1
                assert self.optimizer is not None
                timer.reset(global_step % self.log_freq == 0)
                timer.mark_value("start", t_before_fetch)
                if stage == "mole" and getattr(self.model, "routers", None):
                    total_steps = self.max_steps or max(1, self.epochs * len(self.train_loader))
                    frac = min(1.0, global_step / max(total_steps, 1))
                    self.model.set_training_progress(frac)
                    if frac <= self.model.cfg.router_soft_fraction:
                        mode = "soft"
                    elif frac <= self.model.cfg.router_st_fraction:
                        mode = "st_topk"
                    else:
                        mode = "topk"
                    t0, t1 = self.model.cfg.router_temperature_start, self.model.cfg.router_temperature_end
                    temp = t0 + (t1 - t0) * frac
                    for router in self.model.routers.values():
                        router.set_routing_mode(mode, temp)
                elif stage == "joint" and getattr(self.model, "routers", None):
                    for router in self.model.routers.values():
                        router.set_routing_mode("topk", self.model.cfg.router_temperature_end)
                gpu_batch = self._gpu_batch(batch)
                timer.mark("fetch")
                with torch.autocast(
                    self.device_type, dtype=self.hidden_states_dtype, enabled=self.device_type == "cuda"
                ):
                    out = self.model.forward_residual(
                        **gpu_batch, max_anchors=self.max_anchors, gamma=self.gamma
                    )
                    loss, metrics = out.loss, out.metrics
                timer.mark("fwd")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                timer.mark("bwd")
                self.optimizer.step()
                timer.mark("opt")
                t_before_fetch = timer.now()
                if global_step % self.log_freq == 0:
                    loss_value = float(loss.detach().float().item())
                    vals = _normalize_headline_metrics(_scalar_metrics(metrics), loss_value)
                    if out.pair_valid is not None:
                        block_valid = out.pair_valid
                    elif out.mismatch is not None:
                        block_valid = out.mismatch.valid_mask.any(dim=-1)
                    else:
                        raise RuntimeError("Training output has no block-valid mask")
                    vals.update(
                        _scalar_metrics(
                            _training_acceptance_metrics(
                                out.targets,
                                out.logits,
                                block_size=self.model.block_size,
                                sample_from_anchor=(
                                    self.model.base_model.config.sample_from_anchor
                                ),
                                block_valid=block_valid,
                            )
                        )
                    )
                    logger.info(
                        "stage=%s epoch=%d step=%d "
                        "loss=%.6f sev=%.6f sev_cal=%.6f sev_mean=%.6f repair=%.6f gate=%.6f "
                        "cross_upd=%.6f self_upd=%.6f ffn_upd=%.6f lat_div=%.6f "
                        "reject_mass=%.6f tv=%.6f overlap=%.6f soft_eal=%.6f "
                        "soft_eal_valid_blocks=%d tail=%.6f mole_upd=%.6f anchors=%d",
                        stage,
                        epoch + 1,
                        global_step,
                        loss_value,
                        vals.get("pred_severity", float("nan")),
                        vals.get("pred_severity_calibration", float("nan")),
                        vals.get("pred_severity_mean_match", float("nan")),
                        vals.get("pred_repair", float("nan")),
                        vals.get("fusion_gate_mean", float("nan")),
                        vals.get("cross_update_ratio", float("nan")),
                        vals.get("self_update_ratio", float("nan")),
                        vals.get("ffn_update_ratio", float("nan")),
                        vals.get("latent_diversity", float("nan")),
                        vals.get("reject_mass_mean", float("nan")),
                        vals.get("tv_mean", float("nan")),
                        vals.get("overlap_mean", float("nan")),
                        vals.get("soft_eal", float("nan")),
                        int(vals.get("soft_eal_valid_blocks", 0.0)),
                        vals.get("nla_tail_cvar", float("nan")),
                        vals.get("mole_update_ratio", float("nan")),
                        int(vals.get("valid_anchors", 0.0)),
                    )
                    logger.info(
                        "progressive step=%d pred_nla_mae[%s] pred_nla_pred_mean[%s] "
                        "pred_nla_target_mean[%s] pred_nla_bias[%s] pred_nla_rmse[%s] "
                        "online_nla_mae[%s] pred_repair_cos[%s] "
                        "route_gate[%s] route_entropy[%s] route_top1_change[%s] risk_latent_cos[%s] repair_latent_cos[%s]",
                        global_step,
                        self._series(vals, "pred_nla_mae_"),
                        self._series(vals, "pred_nla_pred_mean_"),
                        self._series(vals, "pred_nla_target_mean_"),
                        self._series(vals, "pred_nla_bias_"),
                        self._series(vals, "pred_nla_rmse_"),
                        self._series(vals, "online_pred_nla_mae_"),
                        self._series(vals, "pred_repair_cos_"),
                        self._series(vals, "route_gate_mean_"),
                        self._series(vals, "route_entropy_"),
                        self._series(vals, "route_top1_change_"),
                        self._series(vals, "risk_latent_cos_"),
                        self._series(vals, "repair_latent_cos_"),
                    )
                    num_proposal_tokens = self.model.block_size - (
                        0 if self.model.base_model.config.sample_from_anchor else 1
                    )
                    accept_probs = ",".join(
                        f"p{pos}={vals[f'accept_position_{pos}']:.4f}"
                        for pos in range(1, num_proposal_tokens + 1)
                    )
                    logger.info(
                        "acceptance step=%d accept_eal=%.4f accept_prob[%s] "
                        "accept_num_valid_blocks=%d",
                        global_step,
                        vals["accept_eal"],
                        accept_probs,
                        int(vals["accept_num_valid_blocks"]),
                    )
                    # DFlash-compatible headline metrics.  These position_i_acc values
                    # are argmax token accuracies, matching trainerlora.py exactly.
                    if stage != "predictor" and "full_acc" in vals:
                        pos_acc = ", ".join(
                            f"train/position_{pos}_acc={vals.get(f'position_{pos}_acc', float('nan')):.3f}"
                            for pos in range(1, self.model.block_size)
                        )
                        profile = timer.profile(
                            int((gpu_batch["document_ids"] != -1).sum().item())
                            if "document_ids" in gpu_batch
                            else 0
                        ) or {}
                        logger.info(
                            "train/loss=%.3f, train/full_acc=%.3f, %s, train/eal=%.3f, "
                            "profile/fetch_ms=%.3f, profile/fwd_ms=%.3f, "
                            "profile/bwd_ms=%.3f, profile/opt_ms=%.3f, "
                            "profile/step_ms=%.3f, profile/tokens_per_s=%.3g, "
                            "profile/fetch_frac=%.3f, epoch=%d, lr=%.3e, global_step=%d",
                            loss_value,
                            vals.get("full_acc", float("nan")),
                            pos_acc,
                            vals.get("eal", float("nan")),
                            profile.get("fetch_ms", float("nan")),
                            profile.get("fwd_ms", float("nan")),
                            profile.get("bwd_ms", float("nan")),
                            profile.get("opt_ms", float("nan")),
                            profile.get("step_ms", float("nan")),
                            profile.get("tokens_per_s", float("nan")),
                            profile.get("fetch_frac", float("nan")),
                            epoch + 1,
                            self.optimizer.param_groups[0]["lr"],
                            global_step,
                        )
                    self._append_jsonl(
                        self.train_metrics_path,
                        {
                            "kind": "train",
                            "stage": stage,
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": loss_value,
                            **vals,
                        },
                    )
                if self.max_steps and global_step >= self.max_steps:
                    break
            val = self.validate(max_batches=16)
            self._log_evaluation(global_step, val)
            logger.info("validation: %s", json.dumps(val, sort_keys=True))
            self._append_jsonl(
                self.validation_metrics_path,
                {
                    "kind": "validation",
                    "stage": stage,
                    "epoch": epoch + 1,
                    "step": global_step,
                    **val,
                },
            )
            save_residual_checkpoint(
                self.model,
                self.save_dir / f"epoch_{epoch + 1:03d}",
                stage=stage,
                config=self.residual_config,
                extra={"epoch": epoch + 1, "validation": val},
            )
            if self.max_steps and global_step >= self.max_steps:
                break
        save_residual_checkpoint(
            self.model,
            self.save_dir / "final",
            stage=stage,
            config=self.residual_config,
            extra={"global_step": global_step},
        )
