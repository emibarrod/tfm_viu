"""Evaluation helpers for baseline multimodal experiments."""

from __future__ import annotations

import os
import tempfile
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ThresholdMethod = Literal["f1", "youden"]

# Prevalences the operating point is rescaled to. The cohort is a 1:1
# case-control sample, so its 50 % prevalence is a design choice, not an
# epidemiological fact: PPV, F1 and AUPRC all depend on it. Reported sepsis
# incidence among ICU admissions is roughly 5-10 %, so `ppv_at_5pct_prevalence`
# and `ppv_at_10pct_prevalence` bracket what the same sensitivity/specificity
# pair would deliver in a real unit -- a much less flattering number than the
# raw PPV, and the one an operational reading needs.
REFERENCE_PREVALENCES = (0.05, 0.10)


def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: ThresholdMethod = "f1",
) -> float:
    """Tune threshold on validation split."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if thresholds.size == 0:
            return 0.5
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        return float(thresholds[int(np.nanargmax(f1))])
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thresholds[int(np.nanargmax(j))])
    raise ValueError(f"Unsupported threshold method: {method}")


def ppv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Rescale positive predictive value to a target prevalence via Bayes' rule.

    Sensitivity and specificity are properties of the classifier and transfer
    across populations; PPV is not, because it depends on how many positives
    there are to find. With prevalence `p`,

        PPV = sens * p / (sens * p + (1 - spec) * (1 - p))

    At this cohort's 50 % prevalence a model with sensitivity 0.70 and
    specificity 0.59 reports PPV 0.63. In a unit where 5 % of admissions become
    septic, the same model yields PPV 0.08 -- twelve false alarms for every true
    one. Reporting only the former overstates operational usefulness by an order
    of magnitude, which is the whole point of this function.
    """
    if not np.isfinite(sensitivity) or not np.isfinite(specificity):
        return float("nan")
    true_positive_rate = sensitivity * prevalence
    false_positive_rate = (1.0 - specificity) * (1.0 - prevalence)
    denominator = true_positive_rate + false_positive_rate
    if denominator <= 0:
        return float("nan")
    return float(true_positive_rate / denominator)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error with equal-width bins.

    Partitions [0, 1] into `n_bins` equal-width bins, and averages
    |observed frequency - mean predicted probability| over bins, weighted by bin
    occupancy. 0 is perfect calibration; the value is on the probability scale,
    so 0.05 means "predicted probabilities are off by 5 percentage points on
    average".

    Equal-width (rather than equal-count) bins are the standard formulation and
    are what makes the number comparable with published values; the trade-off is
    that a sparsely populated bin can swing it, which is why the calibration
    curves are reported alongside. Note that, like every calibration statistic
    here, this is calibration *to this cohort's 50 % prevalence*.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=False` with the interior edges puts each probability in bin
    # [edges[i], edges[i+1]), and 1.0 in the last bin.
    bin_index = np.digitize(y_prob, edges[1:-1], right=False)
    total = y_true.size
    error = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(error)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_calibration_bins: int = 10,
) -> dict[str, float]:
    """Compute lean metric suite on a labeled split."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    prevalence = float(np.mean(y_true)) if y_true.size else np.nan

    metrics = {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred)),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden_j": float(sensitivity + specificity - 1.0),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_calibration_bins),
        "test_prevalence": prevalence,
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    for reference in REFERENCE_PREVALENCES:
        key = f"ppv_at_{int(round(reference * 100))}pct_prevalence"
        metrics[key] = ppv_at_prevalence(sensitivity, specificity, reference)
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric computed on already-predicted probabilities.

    Resamples `(y_true, y_prob)` pairs with replacement `n_resamples` times,
    recomputes `metric_fn(y_true_resampled, y_prob_resampled)` on each
    resample, and returns the `((1-ci)/2, 1-(1-ci)/2)` percentile bounds
    (e.g. 2.5th/97.5th for `ci=0.95`). Pure post-hoc computation — no
    retraining beyond the single fit already performed.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = y_true.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    scores = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        y_true_resampled = y_true[idx]
        y_prob_resampled = y_prob[idx]
        if len(np.unique(y_true_resampled)) < 2:
            scores[i] = np.nan
            continue
        scores[i] = metric_fn(y_true_resampled, y_prob_resampled)

    scores = scores[~np.isnan(scores)]
    if scores.size == 0:
        return float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(scores, 100 * alpha))
    high = float(np.percentile(scores, 100 * (1 - alpha)))
    return low, high


def save_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> None:
    """Save calibration curve PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def paired_bootstrap_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn=roc_auc_score,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for `metric(a) - metric(b)` on the same test rows.

    Replaces the criterion the memoir used to lean on -- "the confidence
    intervals overlap, therefore the models are equivalent" -- which is not a
    test: overlapping marginal CIs are compatible with a difference that is
    consistently signed, because the two models are evaluated on the *same*
    patients and their errors are correlated. Resampling the pair together
    cancels that shared variance, so the interval is on the difference itself.

    Both probability vectors must be aligned to the same `y_true` ordering
    (`posthoc_analysis.py` aligns them by stay_id before calling this).

    Returns the observed difference, its percentile CI, and a two-sided
    bootstrap p-value: twice the smaller tail mass of the resampled differences
    around zero, capped at 1. The interpretation is the usual one -- a CI that
    excludes 0 means the ordering of the two models is stable under resampling
    of the test set, not that either is clinically better.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_a = np.asarray(y_prob_a).astype(float)
    y_prob_b = np.asarray(y_prob_b).astype(float)
    if not (y_true.shape == y_prob_a.shape == y_prob_b.shape):
        raise ValueError("Paired bootstrap needs identically shaped, row-aligned inputs.")

    n = y_true.shape[0]
    observed = float(metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b))
    if n == 0:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    rng = np.random.RandomState(seed)
    differences = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        resampled_true = y_true[idx]
        if len(np.unique(resampled_true)) < 2:
            differences[i] = np.nan
            continue
        differences[i] = metric_fn(resampled_true, y_prob_a[idx]) - metric_fn(resampled_true, y_prob_b[idx])

    differences = differences[~np.isnan(differences)]
    if differences.size == 0:
        return {"difference": observed, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    alpha = (1.0 - ci) / 2.0
    tail = min(float(np.mean(differences <= 0.0)), float(np.mean(differences >= 0.0)))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(differences, 100 * alpha)),
        "ci_high": float(np.percentile(differences, 100 * (1 - alpha))),
        "p_value": float(min(1.0, 2.0 * tail)),
        "n_resamples": float(differences.size),
    }
