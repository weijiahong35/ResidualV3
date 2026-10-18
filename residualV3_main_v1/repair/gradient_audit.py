"""Connectivity and finiteness checks; a finite zero is a valid gradient."""
import math


def audit_gradients(gradients, expected_names, fusion):
    missing = sorted(set(expected_names) - set(gradients))
    unexpected = sorted(set(gradients) - set(expected_names))
    if missing or unexpected:
        raise RuntimeError(f"gradient connectivity mismatch: missing={missing}, unexpected={unexpected}")
    for label, values in (("parameter", gradients), ("fusion", fusion)):
        invalid = {name: value for name, value in values.items()
                   if not math.isfinite(value) or value < 0}
        if invalid:
            raise RuntimeError(f"invalid {label} gradient L1 sums: {invalid}")
    return dict(zero_gradient_parameters=[name for name, value in gradients.items() if value == 0],
                zero_gradient_fusion_branches=[name for name, value in fusion.items() if value == 0])
