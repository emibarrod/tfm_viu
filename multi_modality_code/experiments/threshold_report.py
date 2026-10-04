"""Compare threshold-selection criteria post-hoc on saved test probabilities.

Motivation: the reported operating point (sensitivity / specificity / F1) depends
entirely on where the decision threshold is placed, and the pipeline places it by
maximising F1 on the validation split. At this cohort's 50 % prevalence a trivial
all-positive rule already scores F1 = 2p/(1+p) = 0.667, so for a weak model the
F1-optimal threshold collapses towards "alarm on everyone": high sensitivity,
near-zero specificity, and an F1 indistinguishable from the trivial rule.

This script re-derives thresholds from the probabilities persisted by
`run_experiments.py` / `dl/run_dl_experiments.py` (`<results_dir>/predictions/*.npz`)
under every criterion in `evaluate.pick_threshold`, and flags rows whose operating
point is within `--degenerate_margin` of the trivial rule. No retraining involved:
AUROC, AUPRC and the CIs are threshold-free and unchanged by any of this.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from multi_modality_code.experiments.evaluate import evaluate, pick_threshold


CRITERIA = ("f1", "youden")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions_dirs",
        nargs="+",
        default=["data/05_results/predictions", "data/05_results/deep_learning/predictions"],
        help="Directories holding the *.npz prediction dumps (classical and DL arms).",
    )
    parser.add_argument("--output_csv", default="data/05_results/threshold_comparison.csv")
    parser.add_argument(
        "--degenerate_margin",
        type=float,
        default=0.02,
        help="Flag an operating point as near-degenerate when its F1 is within this margin of the "
        "all-positive rule's F1 at the test-split prevalence.",
    )
    return parser.parse_args()


def trivial_all_positive_f1(prevalence: float) -> float:
    """F1 of predicting the positive class for everyone: 2p/(1+p)."""
    return 2.0 * prevalence / (1.0 + prevalence)


def evaluate_one_file(path: str, degenerate_margin: float) -> list[dict[str, float | int | str]]:
    data = np.load(path)
    y_val, val_prob = data["y_val"], data["val_prob"]
    y_test, test_prob = data["y_test"], data["test_prob"]
    prevalence = float(np.mean(y_test))
    trivial_f1 = trivial_all_positive_f1(prevalence)

    rows: list[dict[str, float | int | str]] = []
    for criterion in CRITERIA:
        threshold = pick_threshold(y_val, val_prob, method=criterion)
        metrics = evaluate(y_test, test_prob, threshold=threshold)
        rows.append(
            {
                "run": os.path.splitext(os.path.basename(path))[0],
                "threshold_method": criterion,
                "threshold": metrics["threshold"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "balanced_accuracy": 0.5 * (metrics["sensitivity"] + metrics["specificity"]),
                "youden_j": metrics["sensitivity"] + metrics["specificity"] - 1.0,
                "test_prevalence": prevalence,
                "trivial_all_positive_f1": trivial_f1,
                "near_degenerate": int(abs(metrics["f1"] - trivial_f1) <= degenerate_margin),
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    paths: list[str] = []
    for directory in args.predictions_dirs:
        paths.extend(sorted(glob.glob(os.path.join(directory, "*.npz"))))
    if not paths:
        raise SystemExit(
            f"No prediction dumps found under {args.predictions_dirs}. Run the experiment stages first "
            "(they write <output_dir>/predictions/*.npz)."
        )

    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        rows.extend(evaluate_one_file(path, args.degenerate_margin))
    report = pd.DataFrame(rows).sort_values(["run", "threshold_method"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    report.to_csv(args.output_csv, index=False)

    for criterion in CRITERIA:
        subset = report[report["threshold_method"] == criterion]
        print(
            f"[{criterion}] n={len(subset)} | sens {subset['sensitivity'].mean():.3f} "
            f"| spec {subset['specificity'].mean():.3f} | Youden J {subset['youden_j'].mean():.3f} "
            f"| near-degenerate {int(subset['near_degenerate'].sum())}/{len(subset)}"
        )
    print(f"Wrote {args.output_csv} ({len(report)} rows over {len(paths)} runs)")


if __name__ == "__main__":
    main(parse_args())
