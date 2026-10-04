"""Cohort-balance diagnostics shared by the label builder, the validator and the tests.

Kept in one place so the label builder (Stage 2), the validation gates (Stage 4)
and the regression tests all judge case/control balance by exactly the same
definition, instead of each re-deriving it.
"""

from __future__ import annotations

import numpy as np


def standardized_mean_difference(treated: np.ndarray, control: np.ndarray) -> float:
    """Absolute standardized mean difference (SMD) between two samples.

    ``|mean(a) - mean(b)| / sqrt((var(a) + var(b)) / 2)`` — the usual balance
    statistic for matched designs, scale-free so it can be compared against a
    fixed threshold. The conventional reading is that < 0.1 counts as
    negligible imbalance.

    Returns 0.0 when both samples share the same mean with no spread, and
    ``inf`` when the means differ but neither sample varies at all.
    """
    a = np.asarray(treated, dtype=float)
    b = np.asarray(control, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")

    mean_gap = abs(float(a.mean()) - float(b.mean()))
    pooled = np.sqrt((float(a.var(ddof=0)) + float(b.var(ddof=0))) / 2.0)
    if pooled <= 0:
        return 0.0 if mean_gap == 0 else float("inf")
    return mean_gap / pooled
