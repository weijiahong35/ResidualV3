from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
HS_ROOT = PROJECT_ROOT / "hs_connectors" / "src"
if str(HS_ROOT) not in sys.path:
    sys.path.insert(0, str(HS_ROOT))

from hs_connectors import HiddenStatesBackend  # noqa: E402
from scripts.trainlora import build_draft_model, parse_vocab_mappings  # noqa: E402
from speculators.model import SpeculatorModel  # noqa: E402
from speculators.train.config.schemalora import TrainConfig  # noqa: E402
from speculators.train.dataloader import create_train_val_loaders  # noqa: E402

from residual.checkpoint import load_residual_checkpoint  # noqa: E402
from residual.config import ResidualConfig  # noqa: E402
from residual.model_residual import ResidualDFlashModel  # noqa: E402
from residual.trainer_residual import ResidualTrainer  # noqa: E402

logger = logging.getLogger("residual")


def _residual_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--train-stage",
        choices=["predictor", "mole", "joint"],
        default="predictor",
    )
    p.add_argument("--mismatch-num-hidden-layers", type=int, default=5)
    p.add_argument("--mismatch-hidden-dim", type=int, default=256)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-attention-heads", type=int, default=4)
    p.add_argument("--predictor-ffn-expansion", type=int, default=4)
    p.add_argument("--mismatch-direction-dim", type=int, default=256)
    p.add_argument("--repair-position-dim", type=int, default=256)
    p.add_argument("--correction-layer-indices", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--mole-shared-rank", type=int, default=8)
    p.add_argument("--lambda-risk-runtime", type=float, default=0.0)
    p.add_argument("--lambda-risk-dist", type=float, default=0.0)
    p.add_argument("--lambda-risk-block", type=float, default=0.0)
    p.add_argument("--lambda-repair-direction", type=float, default=0.0)
    p.add_argument("--lambda-repair-magnitude", type=float, default=0.0)
    p.add_argument("--lambda-repair-runtime-aux", type=float, default=0.0)
    p.add_argument("--lambda-repair-dist-aux", type=float, default=0.0)
    p.add_argument("--router-temperature-start", type=float, default=2.0)
    p.add_argument("--router-temperature-end", type=float, default=0.5)
    p.add_argument("--router-soft-fraction", type=float, default=0.20)
    p.add_argument("--router-st-fraction", type=float, default=0.60)
    p.add_argument("--router-lr-scale-stage2", type=float, default=0.5)
    p.add_argument("--router-lr-scale-stage3", type=float, default=0.05)
    p.add_argument("--lambda-route-consistency", type=float, default=0.01)
    p.add_argument("--gate-warmup-ratio", type=float, default=0.10)
    p.add_argument("--mismatch-lambda-severity", type=float, default=1.0)
    p.add_argument("--mismatch-lambda-repair", type=float, default=1.0)
    p.add_argument("--severity-calibration-weight", type=float, default=0.25)
    p.add_argument("--severity-mean-weight", type=float, default=0.05)
    p.add_argument("--overlap-calibration-weight", type=float, default=0.25)
    p.add_argument("--mole-num-experts", type=int, default=8)
    p.add_argument("--mole-top-k", type=int, default=2, help="0 means dense routing")
    p.add_argument("--mole-rank", type=int, default=8)
    p.add_argument("--mole-alpha", type=float, default=16.0)
    p.add_argument("--mole-dropout", type=float, default=0.0)
    p.add_argument("--mole-max-relative-update", type=float, default=0.20)
    p.add_argument("--router-context-dim", type=int, default=4)
    p.add_argument("--router-gate-init-bias", type=float, default=-1.5)
    p.add_argument("--router-balance-mode", choices=["loss_free", "aux", "none"], default="none")
    p.add_argument("--router-bias-update-rate", type=float, default=0.0)
    p.add_argument("--router-load-ema", type=float, default=0.95)
    p.add_argument("--router-bias-clip", type=float, default=2.0)
    p.add_argument(
        "--accept-loss",
        choices=["nla", "tv", "lk_hybrid", "kl_div", "ce"],
        default="nla",
    )
    p.add_argument("--lambda-pred", type=float, default=0.15)
    p.add_argument("--lambda-soft-alignment", type=float, default=1.0)
    p.add_argument("--lambda-pred-online", type=float, default=0.10)
    p.add_argument("--lambda-reg", type=float, default=0.0)
    p.add_argument("--lambda-router-z", type=float, default=1e-4)
    p.add_argument("--lambda-tail-nla", type=float, default=0.02)
    p.add_argument("--tail-nla-fraction", type=float, default=0.10)
    p.add_argument("--predictor-lr-scale", type=float, default=0.1)
    p.add_argument("--residual-seed", type=int, default=42)
    p.add_argument("--predictor-checkpoint")
    p.add_argument("--stage2-checkpoint")
    p.add_argument("--residual-checkpoint")
    p.add_argument("--residual-save-path")
    p.add_argument("--sampling-temperature", type=float, default=0.0)
    p.add_argument("--sampling-top-p", type=float, default=1.0)
    p.add_argument("--sampling-top-k", type=int, default=0)
    p.add_argument("--distribution-temperature", type=float, default=1.0)
    p.add_argument("--teacher-trajectory", choices=["reference_off_policy", "on_policy_replay"], default="reference_off_policy")
    return p


def resolve_configs() -> tuple[TrainConfig, ResidualConfig]:
    parser = _residual_parser()
    residual_ns, remaining = parser.parse_known_args()
    old = sys.argv
    try:
        sys.argv = [old[0], *remaining]
        base_cfg = TrainConfig.resolve()
    finally:
        sys.argv = old

    values = vars(residual_ns)
    values["correction_layer_indices"] = tuple(values["correction_layer_indices"])
    r_cfg = ResidualConfig(**values)
    return base_cfg, r_cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_residual_model(base_cfg: TrainConfig, r_cfg: ResidualConfig):
    args = argparse.Namespace(**base_cfg.flatten())
    if args.speculator_type != "dflash":
        raise ValueError("Residual MoLE currently requires --speculator-type dflash")
    if base_cfg.lora.enable_lora:
        raise ValueError(
            "Do not pass --enable-lora to residual training. Existing PEFT LoRA "
            "remains the Single-LoRA baseline; Residual MoLE has separate experts."
        )

    d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)
    registry = SpeculatorModel.registry
    if registry is None or "dflash" not in registry:
        raise RuntimeError("DFlash is not registered in SpeculatorModel.registry")
    base = build_draft_model(
        args, registry["dflash"], t2d, d2t, draft_vocab_size
    )
    dtype = getattr(torch, args.hidden_states_dtype)
    model = ResidualDFlashModel(base, r_cfg)

    ckpt = None
    if r_cfg.train_stage in {"mole", "nomole"}:
        ckpt = r_cfg.predictor_checkpoint or r_cfg.residual_checkpoint
        if not ckpt:
            raise ValueError(f"Stage 2/{r_cfg.train_stage} requires --predictor-checkpoint from Stage 1")
    elif r_cfg.train_stage in {"joint", "nomole_joint"}:
        ckpt = r_cfg.stage2_checkpoint or r_cfg.residual_checkpoint
        if not ckpt:
            raise ValueError(f"Stage 3/{r_cfg.train_stage} requires --stage2-checkpoint from its Stage 2 branch")
    elif r_cfg.train_stage == "eval":
        ckpt = (
            r_cfg.residual_checkpoint
            or r_cfg.stage2_checkpoint
            or r_cfg.predictor_checkpoint
        )
        if not ckpt:
            raise ValueError("eval requires --residual-checkpoint")
    elif r_cfg.train_stage == "eval_nomole":
        ckpt = r_cfg.residual_checkpoint or r_cfg.stage2_checkpoint
        if not ckpt:
            raise ValueError("eval_nomole requires a trained no-MoLE dense correction checkpoint")

    if ckpt:
        try:
            missing, unexpected = load_residual_checkpoint(model, ckpt, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Residual checkpoint is incompatible with the V20 dual-Predictor / "
                "gated Router architecture. Retrain Stage 1 "
                "with the current code, then use that checkpoint for Stage 2/3."
            ) from exc
        critical_missing = [name for name in missing if name.startswith("predictor.")]
        if critical_missing:
            sample = ", ".join(critical_missing[:6])
            raise RuntimeError(
                "Checkpoint does not contain the complete V20 Risk+Repair Predictor "
                f"branches (missing {len(critical_missing)} Predictor tensors; e.g. {sample}). "
                "Old V20.5.x and earlier Predictor checkpoints are intentionally incompatible; retrain Stage 1."
            )
        if r_cfg.train_stage in {"joint", "eval"}:
            critical_stage2 = [
                name for name in missing
                if name.startswith("routers.") or "residual_mole." in name
            ]
            if critical_stage2:
                sample = ", ".join(critical_stage2[:6])
                raise RuntimeError(
                    "Stage 3/eval requires a complete V20.6 Stage 2 checkpoint with "
                    f"independent Routers and MoLE experts (missing {len(critical_stage2)} tensors; "
                    f"e.g. {sample})."
                )
        if r_cfg.train_stage in {"nomole_joint", "eval_nomole"}:
            critical_dense = [
                name for name in missing
                if name.startswith("dense_gates.") or name.startswith("base_model.layers") and (".lora_A" in name or ".lora_B" in name)
            ]
            if critical_dense:
                sample = ", ".join(critical_dense[:6])
                raise RuntimeError(
                    "No-MoLE Stage3/eval requires a complete dense Stage2 checkpoint "
                    f"(missing {len(critical_dense)} tensors; e.g. {sample})."
                )

        logger.info(
            "Loaded residual checkpoint %s (missing=%d, unexpected=%d)",
            ckpt,
            len(missing),
            len(unexpected),
        )
    model.freeze_for_stage(r_cfg.train_stage)
    if r_cfg.train_stage == "joint" and ckpt:
        # Snapshot the successfully loaded Stage2 partition before Stage3 updates it.
        model.snapshot_stage2_routers()

    return model, args, dtype


def build_loaders(model, args, dtype):
    backend_cls = HiddenStatesBackend.registry[args.hidden_states_backend]
    transfer = backend_cls.from_train_args(args, args.data_path)
    return create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        transfer=transfer,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=model.hidden_size,
        num_target_layers=model.num_target_layers,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=None,
        train_data_ratio=args.train_data_ratio,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    base_cfg, r_cfg = resolve_configs()
    set_seed(r_cfg.residual_seed)
    model, args, dtype = build_residual_model(base_cfg, r_cfg)

    logger.info("Target auxiliary layer IDs from loaded DFlash: %s", list(model.target_layer_ids))
    logger.info(
        "Progressive contract: correction indices=%s; residual is added after the complete frozen MLP; lm_head unchanged",
        model.correction_layer_indices,
    )
    if r_cfg.train_stage == "eval_base":
        logger.info("Base evaluation: original pretrained DFlash only; no Predictor/Router/MoLE")
    else:
        logger.info(
            "Predictor V20.6 Dual-V10 soft-logit training + V10 fused DFlash rollout: one shared V10 Target/Draft information stem; p2/p3/p4 each have TWO independent full V10 fusion predictors (Bias/NLA and Repair); each predictor has %d cross-attn+self-attn+SwiGLU blocks",
            r_cfg.predictor_depth,
        )
        if r_cfg.train_stage == "eval_nomole":
            logger.info("No-MoLE evaluation: Predictor + gated dense single-LoRA tail correction; no experts, no Router")
    logger.info(
        "Target hidden subset indices=%s; Drafter complete-layer indices=%s",
        model.target_hidden_indices,
        model.drafter_hidden_indices,
    )
    if r_cfg.train_stage == "predictor":
        logger.info("Stage 1 Predictor-only: Router disabled; MoLE modules=0")
        logger.info(
            "Tensor contract: prev hT [A,%d,%d], prev hD [A,5,%d], current progressive anchor prefix [A,K,%d]; independent Risk latent [A,256] + Repair latent [A,256] from the same V10 information",
            len(model.target_hidden_indices),
            model.hidden_size,
            model.hidden_size,
            model.hidden_size,
        )
    elif r_cfg.train_stage in {"mole", "joint", "eval"}:
        logger.info(
            "Configured correction layers=%s; independent per-position Routers=%d; lm_head unchanged; residual MoLE modules=%d",
            model.correction_layer_indices,
            len(model.routers),
            len(model._mole_modules),
        )
        logger.info(
            "Tensor contract: prev hT [A,%d,%d] + prev hD [A,5,%d] + current anchor prefix [A,K,%d]; routing stack [S,A,B,%d]",
            len(model.target_hidden_indices),
            model.hidden_size,
            model.hidden_size,
            model.hidden_size,
            r_cfg.mole_num_experts,
        )
    elif r_cfg.train_stage in {"nomole", "nomole_joint", "eval_nomole"}:
        logger.info(
            "No-MoLE dense branch: tail layers=%d; scalar gates=%d; DenseLoRA modules=%d; no experts/Router; lm_head unchanged",
            max(model.num_draft_layers - 2, 0),
            len(model.dense_gates),
            len(model._dense_modules),
        )

    logger.info(
        "Training policy: raw Target/Draft logits -> softmax(T=%s) -> overlap/NLA/soft-first-reject/repair geometry; NO argmax/sampling in training. Greedy runtime metadata: temperature=%s top_p=%s top_k=%s draft_mode=%s; teacher=%s",
        r_cfg.distribution_temperature,
        r_cfg.sampling_temperature,
        r_cfg.sampling_top_p,
        r_cfg.sampling_top_k,
        r_cfg.draft_mode,
        r_cfg.teacher_trajectory,
    )

    train_loader, val_loader = build_loaders(model, args, dtype)
    save_path = r_cfg.residual_save_path or os.path.join(args.save_path, "residual")
    trainer = ResidualTrainer(
        model,
        train_loader,
        val_loader,
        base_lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_dir=save_path,
        residual_config=r_cfg.to_dict(),
        max_steps=args.max_steps,
        log_freq=args.log_freq,
        max_anchors=args.max_anchors,
        gamma=args.dflash_decay_gamma,
        hidden_states_dtype=dtype,
    )

    trainer.train()


if __name__ == "__main__":
    main()
