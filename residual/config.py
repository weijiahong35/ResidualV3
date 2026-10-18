from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ResidualConfig:
    train_stage: str = "predictor"  # predictor | mole | joint | eval | eval_base
    mismatch_num_hidden_layers: int = 5
    mismatch_hidden_dim: int = 256
    predictor_depth: int = 4
    predictor_attention_heads: int = 4
    predictor_ffn_expansion: int = 4
    predictor_arch: str = "v20_6_dual_v10_softlogit_train"
    repair_position_dim: int = 256  # compatibility alias; Risk/Repair latents both use mismatch_hidden_dim
    mismatch_direction_dim: int = 256
    correction_layer_indices: tuple[int, ...] = (2, 3, 4)

    # Stage-1 / Stage-3 predictor objective.
    mismatch_lambda_severity: float = 1.0
    mismatch_lambda_repair: float = 1.0
    # Auxiliary calibration supervision; weighted V10 severity remains primary.
    severity_calibration_weight: float = 0.25
    severity_mean_weight: float = 0.05
    overlap_calibration_weight: float = 0.25

    # Training is distribution-only: these sampling fields are EVAL/RUNTIME
    # metadata and are intentionally ignored by mismatch target construction.
    sampling_policy_source: str = "eval_only_greedy"
    sampling_temperature: float = 0.0
    sampling_top_p: float = 1.0
    sampling_top_k: int = 0
    draft_mode: str = "greedy"
    distribution_temperature: float = 1.0
    teacher_trajectory: str = "reference_off_policy"

    # Progressive conditional residual correction.
    mole_num_experts: int = 8
    mole_top_k: int = 2
    mole_rank: int = 8
    mole_alpha: float = 16.0
    mole_shared_rank: int = 8
    mole_dropout: float = 0.0
    mole_max_relative_update: float = 0.20

    # Strict no-MoLE ablation: one parameter-matched dense LoRA per projection,
    # no experts and no Router. rank=32 ~= four rank-8 experts in parameter count.
    dense_rank: int = 32
    dense_alpha: float = 64.0
    dense_dropout: float = 0.0
    dense_max_relative_update: float = 0.20

    # Router: latent + calibrated mismatch summary -> top-k experts + scalar correction gate.
    router_context_dim: int = 4
    router_gate_init_bias: float = -1.5
    router_balance_mode: str = "none"  # V20: specialization first; collapse is monitored, not equalized
    router_bias_update_rate: float = 0.0
    router_load_ema: float = 0.95
    router_bias_clip: float = 2.0

    # Acceptance / tail / router objectives.
    accept_loss: str = "nla"
    lambda_pred: float = 0.15
    # All Stage2/3 correction supervision is soft Target->Draft KL.
    lambda_soft_alignment: float = 1.0
    lambda_pred_online: float = 0.10
    lambda_reg: float = 0.0
    lambda_router_z: float = 1.0e-4
    lambda_tail_nla: float = 0.02
    tail_nla_fraction: float = 0.10
    predictor_lr_scale: float = 0.1


    # Legacy V20.5 multi-view knobs retained for old config parsing only.
    # V20.6 uses mismatch_lambda_severity / mismatch_lambda_repair exactly.
    lambda_risk_runtime: float = 0.0
    lambda_risk_dist: float = 0.0
    lambda_risk_block: float = 0.0
    lambda_repair_direction: float = 0.0
    lambda_repair_magnitude: float = 0.0
    lambda_repair_runtime_aux: float = 0.0
    lambda_repair_dist_aux: float = 0.0

    # Stage2 soft -> straight-through TopK -> hard TopK schedule.
    router_temperature_start: float = 2.0
    router_temperature_end: float = 0.5
    router_soft_fraction: float = 0.20
    router_st_fraction: float = 0.60
    router_lr_scale_stage2: float = 0.5
    router_lr_scale_stage3: float = 0.05
    lambda_route_consistency: float = 0.01
    gate_warmup_ratio: float = 0.10

    residual_seed: int = 42
    residual_checkpoint: str | None = None
    predictor_checkpoint: str | None = None
    stage2_checkpoint: str | None = None
    residual_save_path: str | None = None
    eval_max_batches: int = 0

    def validate(self, num_draft_layers: int | None = None) -> None:
        if self.train_stage not in {"predictor", "mole", "joint", "eval", "eval_base"}:
            raise ValueError(f"Unknown train_stage={self.train_stage!r}")
        if self.mismatch_num_hidden_layers < 1:
            raise ValueError("mismatch_num_hidden_layers must be >= 1")
        if self.repair_position_dim < 1:
            raise ValueError("repair_position_dim must be >= 1")
        if self.mismatch_direction_dim < 1:
            raise ValueError("mismatch_direction_dim must be >= 1")
        if not self.correction_layer_indices:
            raise ValueError("correction_layer_indices must not be empty")
        if tuple(sorted(set(self.correction_layer_indices))) != tuple(self.correction_layer_indices):
            raise ValueError("correction_layer_indices must be sorted and unique")
        if self.predictor_depth < 1:
            raise ValueError("predictor_depth must be >= 1")
        if self.predictor_attention_heads < 1:
            raise ValueError("predictor_attention_heads must be >= 1")
        if self.mismatch_hidden_dim % self.predictor_attention_heads != 0:
            raise ValueError("mismatch_hidden_dim must be divisible by predictor_attention_heads")
        if self.repair_position_dim != self.mismatch_hidden_dim:
            raise ValueError(
                "Dual-V10 uses equal 256-D Bias/Repair latents: repair_position_dim must equal mismatch_hidden_dim"
            )
        if self.predictor_ffn_expansion < 1:
            raise ValueError("predictor_ffn_expansion must be >= 1")
        if self.router_context_dim != 4:
            raise ValueError(
                "V20.6 soft Risk router uses exactly 4 physical features: "
                "severity, TV, first_reject, reach_prob; router_context_dim must be 4"
            )
        for name in (
            "mismatch_lambda_severity", "mismatch_lambda_repair",
            "severity_calibration_weight", "severity_mean_weight",
            "overlap_calibration_weight", "lambda_pred", "lambda_pred_online",
            "lambda_soft_alignment",
            "lambda_reg", "lambda_router_z", "lambda_tail_nla",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.sampling_temperature != 0:
            raise ValueError("Evaluation runtime is greedy: sampling_temperature metadata must be 0")
        if self.sampling_top_p != 1.0:
            raise ValueError("Evaluation runtime is greedy: sampling_top_p metadata must be 1.0")
        if self.sampling_top_k != 0:
            raise ValueError("Evaluation runtime is greedy: sampling_top_k metadata must be 0")
        if self.draft_mode != "greedy":
            raise ValueError("V20 runtime proposal mode is fixed to greedy")
        if self.distribution_temperature <= 0:
            raise ValueError("distribution_temperature must be > 0")
        if self.teacher_trajectory not in {"reference_off_policy", "on_policy_replay"}:
            raise ValueError("unknown teacher_trajectory")
        if self.mole_num_experts < 1:
            raise ValueError("mole_num_experts must be >= 1")
        if self.mole_top_k < 0 or self.mole_top_k > self.mole_num_experts:
            raise ValueError("mole_top_k must be in [0, mole_num_experts]; 0 means dense")
        if self.mole_shared_rank < 1:
            raise ValueError("mole_shared_rank must be >= 1")
        if self.mole_rank < 1:
            raise ValueError("mole_rank must be >= 1")
        if self.dense_rank < 1:
            raise ValueError("dense_rank must be >= 1")
        if self.dense_max_relative_update < 0:
            raise ValueError("dense_max_relative_update must be >= 0")
        if self.mole_max_relative_update < 0:
            raise ValueError("mole_max_relative_update must be >= 0; 0 disables the safety cap")
        if self.router_context_dim != 4:
            raise ValueError("V20 risk router context is fixed at 4 physical summary features")
        if self.router_balance_mode not in {"loss_free", "aux", "none"}:
            raise ValueError("router_balance_mode must be loss_free, aux, or none")
        if not 0 <= self.router_load_ema < 1:
            raise ValueError("router_load_ema must be in [0,1)")
        if self.router_bias_update_rate < 0 or self.router_bias_clip < 0:
            raise ValueError("router bias settings must be non-negative")
        if not 0 < self.tail_nla_fraction <= 1:
            raise ValueError("tail_nla_fraction must be in (0,1]")
        if num_draft_layers is not None and num_draft_layers < 2:
            raise ValueError("D_t^1 + D_t^2 requires DFlash to have at least two draft layers")
        if (
            num_draft_layers is not None
            and self.train_stage in {"mole", "joint", "eval"}
            and num_draft_layers < 3
        ):
            raise ValueError("Routed correction on Layers 3..N requires at least three draft layers")
        if num_draft_layers is not None and max(self.correction_layer_indices) >= num_draft_layers:
            raise ValueError(
                f"correction_layer_indices={self.correction_layer_indices} exceed "
                f"num_hidden_layers={num_draft_layers}"
            )
        if num_draft_layers is not None and min(self.correction_layer_indices) < 2:
            raise ValueError("V20 correction layers must be at or after D3 (index >= 2)")
        if not 0 <= self.gate_warmup_ratio <= 1:
            raise ValueError("gate_warmup_ratio must be in [0,1]")
        if self.predictor_lr_scale < 0 or self.router_lr_scale_stage2 < 0 or self.router_lr_scale_stage3 < 0:
            raise ValueError("V20 learning-rate scales must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    def runtime_policy_dict(self) -> dict:
        return {
            "target_temperature": self.sampling_temperature,
            "target_top_k": self.sampling_top_k,
            "target_top_p": self.sampling_top_p,
            "draft_mode": self.draft_mode,
            "distribution_temperature": self.distribution_temperature,
        }
