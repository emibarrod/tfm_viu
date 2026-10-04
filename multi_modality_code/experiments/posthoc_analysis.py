"""Post-hoc analyses that need no retraining: paired model tests and clinical rules.

Everything here reads the probabilities persisted by `run_experiments.py` and
`dl/run_dl_experiments.py` (`<results_dir>/predictions/*.npz`) plus the scores
from `6_build_clinical_scores.py`, so adding an analysis costs seconds instead of
a full training sweep.

Two outputs:

`paired_tests.csv` -- bootstrap tests on the *difference* between two runs
    evaluated on the same test patients. This replaces the inference the memoir
    used to make from overlapping confidence intervals, which is a conservative
    heuristic and not a test: because both models score the same patients, their
    errors are correlated, and resampling the pair together removes that shared
    variance. Two families of comparison are emitted:
      - `models_within_ablation`: every pair of model families on one feature set
        (which answers "is HistGBT really equivalent to XGBoost?", and "does the
        GRU differ from the trees?").
      - `ablation_vs_primary`: the primary feature set against every other one,
        holding the model fixed (which answers "how much does dropping the
        temporal-position columns cost?" without the noise of a model swap).

`clinical_rules.csv` -- the bedside scores used as they are used in a ward: as
    fixed cut-offs, not as trained models. SOFA >= 2 and qSOFA >= 2 are the
    published thresholds; the surrounding cut-offs are reported so the memoir can
    show the whole trade-off rather than one point of it.

Nominal p-values, no multiplicity correction: the number of comparisons emitted
here is large by design (it is a diagnostic table), so the memoir should quote
only the comparisons it pre-specified -- multimodal vs best unimodal, logistic
regression vs trees, HistGBT vs XGBoost, ML vs DL, and the two drop-column
ablations.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from multi_modality_code.experiments.evaluate import (
    REFERENCE_PREVALENCES,
    bootstrap_ci,
    paired_bootstrap_difference,
    ppv_at_prevalence,
)


PRIMARY_ABLATION = "static_plus_vitals_plus_labs_plus_treatments"

# (column in clinical_scores.csv, cut-offs to report, whether it is the published one)
CLINICAL_RULES = (
    ("sofa_total_at_t0", (2, 3, 4, 6), 2),
    ("qsofa_at_t0", (1, 2, 3), 2),
    ("qsofa_last_at_t0", (1, 2, 3), 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", default="data/05_results")
    parser.add_argument("--dl_results_dir", default="data/05_results/deep_learning")
    parser.add_argument("--multimodal_dir", default="data/04_multimodal")
    parser.add_argument("--n_resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_run_index(results_dir: str, metrics_name: str, predictions_subdir: str = "predictions") -> pd.DataFrame:
    """Locate the prediction dump behind every reported row of a metrics file.

    The reported row is the median-AUPRC seed, so the paired test has to use that
    same seed's probabilities -- otherwise the table would compare a run nobody
    reported against another one nobody reported.
    """
    metrics_path = os.path.join(results_dir, metrics_name)
    if not os.path.exists(metrics_path):
        return pd.DataFrame()

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics.get("status", "ok") == "ok"].copy()
    if metrics.empty:
        return pd.DataFrame()

    rows = []
    for record in metrics.to_dict("records"):
        ablation = record["ablation"]
        model = record["model"]
        seed = record.get("median_seed")
        seed = int(seed) if pd.notna(seed) else None
        # The DL arm names its dumps after the cell type, not the row's model label.
        stem = record.get("cell_type") if pd.notna(record.get("cell_type", np.nan)) else model
        candidates = []
        if seed is not None:
            candidates.append(f"{ablation}__{stem}_seed{seed}.npz")
        candidates.append(f"{ablation}__{stem}.npz")
        path = None
        for candidate in candidates:
            full = os.path.join(results_dir, predictions_subdir, candidate)
            if os.path.exists(full):
                path = full
                break
        if path is None:
            print(f"[warn] no prediction dump for {ablation} | {model} (tried {candidates})")
            continue
        rows.append(
            {
                "ablation": ablation,
                "model": model,
                "median_seed": seed,
                "auroc_reported": record.get("auroc"),
                "path": path,
            }
        )
    return pd.DataFrame(rows)


def load_test_predictions(path: str) -> pd.Series:
    """Test-split probabilities as a Series indexed by stay_id (sorted)."""
    data = np.load(path)
    series = pd.Series(
        np.asarray(data["test_prob"], dtype=float),
        index=pd.Index(np.asarray(data["test_stay_ids"], dtype=np.int64), name="stay_id"),
    )
    return series.sort_index()


def load_test_labels(path: str) -> pd.Series:
    data = np.load(path)
    series = pd.Series(
        np.asarray(data["y_test"], dtype=int),
        index=pd.Index(np.asarray(data["test_stay_ids"], dtype=np.int64), name="stay_id"),
    )
    return series.sort_index()


def align(path_a: str, path_b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-align two runs on their shared test stay_ids, checking the labels agree."""
    prob_a, prob_b = load_test_predictions(path_a), load_test_predictions(path_b)
    labels_a, labels_b = load_test_labels(path_a), load_test_labels(path_b)
    shared = prob_a.index.intersection(prob_b.index)
    if len(shared) == 0:
        raise ValueError(f"No shared test stays between {path_a} and {path_b}.")
    if not labels_a.loc[shared].equals(labels_b.loc[shared]):
        raise ValueError(f"Label mismatch on shared stays between {path_a} and {path_b}.")
    return labels_a.loc[shared].to_numpy(), prob_a.loc[shared].to_numpy(), prob_b.loc[shared].to_numpy()


def comparison_rows(runs: pd.DataFrame, n_resamples: int, seed: int) -> pd.DataFrame:
    pairs: list[tuple[str, dict, dict]] = []
    lookup = {(row["ablation"], row["model"]): row for row in runs.to_dict("records")}

    for ablation, group in runs.groupby("ablation"):
        records = group.to_dict("records")
        for left, right in itertools.combinations(sorted(records, key=lambda r: r["model"]), 2):
            pairs.append(("models_within_ablation", left, right))

    for (ablation, model), record in sorted(lookup.items()):
        if ablation == PRIMARY_ABLATION:
            continue
        primary = lookup.get((PRIMARY_ABLATION, model))
        if primary is None:
            continue
        pairs.append(("ablation_vs_primary", primary, record))

    rows = []
    for family, left, right in pairs:
        y_true, prob_left, prob_right = align(left["path"], right["path"])
        auroc = paired_bootstrap_difference(
            y_true, prob_left, prob_right, roc_auc_score, n_resamples=n_resamples, seed=seed
        )
        auprc = paired_bootstrap_difference(
            y_true, prob_left, prob_right, average_precision_score, n_resamples=n_resamples, seed=seed
        )
        rows.append(
            {
                "family": family,
                "a_ablation": left["ablation"],
                "a_model": left["model"],
                "b_ablation": right["ablation"],
                "b_model": right["model"],
                "n_test": len(y_true),
                "auroc_a": float(roc_auc_score(y_true, prob_left)),
                "auroc_b": float(roc_auc_score(y_true, prob_right)),
                "auroc_diff": auroc["difference"],
                "auroc_diff_ci_low": auroc["ci_low"],
                "auroc_diff_ci_high": auroc["ci_high"],
                "auroc_p_value": auroc["p_value"],
                "auroc_significant": int(auroc["ci_low"] > 0 or auroc["ci_high"] < 0),
                "auprc_diff": auprc["difference"],
                "auprc_diff_ci_low": auprc["ci_low"],
                "auprc_diff_ci_high": auprc["ci_high"],
                "auprc_p_value": auprc["p_value"],
            }
        )
        print(
            f"[paired] {family} | {left['model']}@{left['ablation']} vs "
            f"{right['model']}@{right['ablation']} | dAUROC {auroc['difference']:+.4f} "
            f"[{auroc['ci_low']:+.4f}, {auroc['ci_high']:+.4f}] p={auroc['p_value']:.4f}"
        )
    return pd.DataFrame(rows)


def clinical_rule_rows(
    clinical_path: str,
    cohort_path: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate the bedside scores on the test split, as continuous scores and as cut-offs."""
    clinical = pd.read_csv(clinical_path)
    cohort = pd.read_csv(cohort_path, usecols=["stay_id", "split", "label"])
    test_ids = set(cohort.loc[cohort["split"].astype(str).str.lower() == "test", "stay_id"])
    subset = clinical[clinical["stay_id"].isin(test_ids)].copy()
    if subset.empty:
        raise ValueError("No test stays found in the clinical scores export.")

    y_true = subset["label"].to_numpy(dtype=int)
    rows = []
    for column, cutoffs, published in CLINICAL_RULES:
        score = subset[column].to_numpy(dtype=float)
        auroc = float(roc_auc_score(y_true, score))
        ci_low, ci_high = bootstrap_ci(y_true, score, roc_auc_score, n_resamples=n_resamples, seed=seed)
        for cutoff in cutoffs:
            predicted = (score >= cutoff).astype(int)
            tp = int(((predicted == 1) & (y_true == 1)).sum())
            fp = int(((predicted == 1) & (y_true == 0)).sum())
            tn = int(((predicted == 0) & (y_true == 0)).sum())
            fn = int(((predicted == 0) & (y_true == 1)).sum())
            sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
            specificity = tn / (tn + fp) if (tn + fp) else np.nan
            ppv = tp / (tp + fp) if (tp + fp) else np.nan
            row = {
                "score": column,
                "cutoff": cutoff,
                "is_published_cutoff": int(cutoff == published),
                "n_test": len(y_true),
                "auroc": auroc,
                "auroc_ci_low": ci_low,
                "auroc_ci_high": ci_high,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "youden_j": sensitivity + specificity - 1.0,
                "balanced_accuracy": 0.5 * (sensitivity + specificity),
                "alarm_rate": float(predicted.mean()),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
            for reference in REFERENCE_PREVALENCES:
                row[f"ppv_at_{int(round(reference * 100))}pct_prevalence"] = ppv_at_prevalence(
                    sensitivity, specificity, reference
                )
            rows.append(row)
            print(
                f"[rule] {column} >= {cutoff} | sens {sensitivity:.3f} spec {specificity:.3f} "
                f"J {row['youden_j']:+.3f} | AUROC(score) {auroc:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    classical = load_run_index(args.results_dir, "metrics.csv")
    deep = load_run_index(args.dl_results_dir, "dl_metrics.csv")
    runs = pd.concat([frame for frame in (classical, deep) if not frame.empty], ignore_index=True)
    if runs.empty:
        raise SystemExit(
            "No runs with persisted predictions found. Run the classical and/or DL experiments first."
        )
    print(f"Runs available: {len(runs)} ({len(classical)} classical, {len(deep)} deep)")

    paired = comparison_rows(runs, args.n_resamples, args.seed)
    paired_path = os.path.join(args.results_dir, "paired_tests.csv")
    paired.sort_values(["family", "a_ablation", "a_model", "b_ablation", "b_model"]).to_csv(paired_path, index=False)
    print(f"\nWrote {paired_path} ({len(paired)} comparisons)")

    clinical_path = os.path.join(args.multimodal_dir, "clinical_scores.csv")
    if os.path.exists(clinical_path):
        rules = clinical_rule_rows(
            clinical_path,
            os.path.join(args.multimodal_dir, "cohort.csv"),
            args.n_resamples,
            args.seed,
        )
        rules_path = os.path.join(args.results_dir, "clinical_rules.csv")
        rules.to_csv(rules_path, index=False)
        print(f"Wrote {rules_path} ({len(rules)} rules)")
    else:
        print(f"[skip] {clinical_path} not found; run 6_build_clinical_scores.py for the clinical rules.")


if __name__ == "__main__":
    main(parse_args())
