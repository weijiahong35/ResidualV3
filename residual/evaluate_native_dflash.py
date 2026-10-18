#!/usr/bin/env python3
"""Shared Base/SpecMoLE native-vLLM benchmark evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from examples.evaluate.evaluate_dflash_lora import build_parser, evaluate


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "evaluate":
        raise ValueError("This shared native evaluator only supports the evaluate command")
    summary = evaluate(
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
    draft_tokens = float(summary.get("num_draft_tokens") or 0.0)
    accepted = float(summary.get("num_accepted_tokens") or 0.0)
    summary["acceptance_rate"] = accepted / draft_tokens if draft_tokens else 0.0
    summary_path = Path(args.output_dir).expanduser().resolve() / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"acceptance_rate         : {summary['acceptance_rate']:.6f}")


if __name__ == "__main__":
    main()
