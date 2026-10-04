"""Run Sepsis-3 multimodal baseline experiments and modality ablations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from multi_modality_code.experiments.data_utils.splits import load_splits
from multi_modality_code.experiments.evaluate import (
    REFERENCE_PREVALENCES,
    bootstrap_ci,
    evaluate,
    pick_threshold,
    save_calibration_plot,
)
from multi_modality_code.experiments.features.aggregate import (
    FeatureBundle,
    build_feature_bundle,
    drop_zero_variance,
)
from multi_modality_code.experiments.models.baselines import make_logreg, make_tree, make_xgb


PRIMARY_ABLATION = "static_plus_vitals_plus_labs_plus_treatments"

# Where the observation window sits inside the stay. Offset-matched control
# sampling already removed the gross artifact that made
# `hours_from_icu_intime_to_prediction` the top feature by permutation
# importance, but "how deep into the admission are we?" is still a legitimate
# predictor that has nothing to do with physiology. Dropping both columns
# isolates how much of the primary result rests on it.
TEMPORAL_POSITION_FEATURES = (
    "hours_from_icu_intime_to_prediction",
    "hours_from_admission_to_prediction",
)

# Features derived from ICD codes. Those codes are assigned for *billing at
# discharge*, so they are not strictly available at t0 -- and `diagnosis_count`
# is the worst offender, since it counts every code of the admission, including
# acute codes for the deterioration being predicted. It is also, after the
# control fix, the highest-ranked feature in the tree models, which is exactly
# why the memoir needs a number for how much of the result depends on it.
BILLING_DERIVED_FEATURES = (
    "diagnosis_count",
    "charlson_comorbidity_index",
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    modalities: tuple[str, ...]
    include_antibiotics: bool = False
    # Columns removed from the design matrix after concatenating the modalities.
    # Applied with errors="ignore", so a name that a given modality set does not
    # contain is silently a no-op.
    drop_features: tuple[str, ...] = ()
    # Restrict the ablation to a subset of the model family (None = all of them).
    models: tuple[str, ...] | None = None
    # True for ablations the deep-learning arm cannot mirror: `run_dl_experiments`
    # resolves branch widths through `GridBundle.branch_input_dim()`, which knows
    # nothing about `drop_features` and has no clinical-score branch, so it skips
    # these instead of reporting a row that does not mean what it says.
    classical_only: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline and ablation experiments on multimodal Sepsis-3 data.")
    parser.add_argument("--multimodal_dir", default="data/04_multimodal")
    parser.add_argument("--abx_dir", default="data/04_multimodal_abx")
    parser.add_argument("--output_dir", default="data/05_results")
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds trained independently per (ablation, model). metrics.csv "
        "reports the median-AUPRC seed plus mean+-std across seeds, exactly as the deep-learning "
        "arm does, so the ML-vs-DL comparison is symmetric; every seed's probabilities are kept "
        "under <output_dir>/predictions.",
    )
    parser.add_argument(
        "--threshold_method",
        choices=["f1", "youden"],
        default="youden",
        help="Criterion used to place the decision threshold on the validation split. Default "
        "'youden' (maximise sensitivity + specificity - 1). 'f1' is kept for the secondary "
        "analysis: at this cohort's 50%% prevalence a trivial all-positive rule already scores "
        "F1 = 0.667, so maximising F1 collapses weak models onto that rule -- see "
        "experiments/threshold_report.py.",
    )
    parser.add_argument("--mimic_version", default="MIMIC-IV v3.1")
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Ignore cached feature matrices under <output_dir>/features and rebuild them.",
    )
    return parser.parse_args()


def ablation_configs() -> list[AblationConfig]:
    return [
        AblationConfig(name="static_only", modalities=("static",)),
        AblationConfig(name="vitals_only", modalities=("vitals",)),
        AblationConfig(name="labs_only", modalities=("labs",)),
        AblationConfig(name="treatments_no_antibiotics", modalities=("treatments",)),
        AblationConfig(name="vitals_plus_labs", modalities=("vitals", "labs")),
        AblationConfig(name="static_plus_vitals_plus_labs", modalities=("static", "vitals", "labs")),
        AblationConfig(
            name="static_plus_vitals_plus_labs_plus_treatments",
            modalities=("static", "vitals", "labs", "treatments"),
        ),
        AblationConfig(
            name="sensitivity_with_antibiotics",
            modalities=("static", "vitals", "labs", "treatments"),
            include_antibiotics=True,
        ),
        # Clinical comparator: logistic regression on SOFA and qSOFA evaluated at
        # t0 (see 6_build_clinical_scores.py). Two features, one model -- a
        # boosted forest over two integers would only add noise to a row whose
        # whole purpose is to be the bedside baseline.
        AblationConfig(
            name="clinical_scores_only",
            modalities=("clinical_scores",),
            models=("logreg",),
            classical_only=True,
        ),
        # Temporal-position ablation (the clean version of the "did the control
        # bug inflate the result?" question: same split, same features, same
        # library versions, exactly two columns removed).
        AblationConfig(
            name="primary_no_temporal_position",
            modalities=("static", "vitals", "labs", "treatments"),
            drop_features=TEMPORAL_POSITION_FEATURES,
            classical_only=True,
        ),
        # Same idea for the discharge-coded comorbidity features.
        AblationConfig(
            name="primary_no_billing_features",
            modalities=("static", "vitals", "labs", "treatments"),
            drop_features=BILLING_DERIVED_FEATURES,
            classical_only=True,
        ),
    ]


def build_design_matrix(
    bundle: FeatureBundle,
    modalities: tuple[str, ...],
    fit_index: np.ndarray | None = None,
    drop_features: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Concatenate the active modality matrices into one design matrix.

    `drop_features` removes named columns after concatenation, which is how the
    "same pipeline, same split, these columns gone" ablations are expressed --
    two of the temporal-position columns, nineteen of the ICD-derived ones.
    Dropping happens before the zero-variance filter so the two never interact.

    When `fit_index` (the training stay_ids) is given, columns with no variance
    on those rows are dropped -- see `aggregate.drop_zero_variance`.
    """
    frames = [bundle.modalities[name] for name in modalities]
    matrix = pd.concat(frames, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    matrix = matrix.sort_index()
    if drop_features:
        matrix = matrix.drop(columns=list(drop_features), errors="ignore")
    if fit_index is not None:
        matrix = drop_zero_variance(matrix, fit_index=fit_index)
    return matrix


def select_rows(matrix: pd.DataFrame, labels: pd.Series, stay_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice the design matrix to one split, keeping missing values as NaN.

    NaNs are deliberately preserved. This used to run `np.nan_to_num(x, nan=0.0)`
    here, which silently defeated the documented design: the logistic-regression
    pipeline's `SimpleImputer(strategy="median")` had nothing left to impute, and
    HistGradientBoosting/XGBoost never got to use their native missing-value
    handling -- every absent measurement was fed to all three models as a literal
    zero, indistinguishable from a real zero reading.

    Infinities are still neutralised and finite values clipped, since neither is
    a meaningful measurement and both can destabilise the linear model.
    """
    common = np.intersect1d(matrix.index.to_numpy(dtype=np.int64), stay_ids.astype(np.int64))
    # `copy=True` is required, not defensive: under copy-on-write, `to_numpy()` on a
    # frame that is already a single float64 block hands back a read-only *view* of
    # it, and the in-place cleaning below then raises "assignment destination is
    # read-only". Mixed-dtype matrices (every modality with one-hot dummies) have to
    # be consolidated and so copy implicitly, which is why this only surfaced with
    # the all-float clinical-scores matrix.
    x = matrix.loc[common].to_numpy(dtype=np.float64, copy=True)
    x[np.isposinf(x)] = np.nan
    x[np.isneginf(x)] = np.nan
    x = np.clip(x, -10_000.0, 10_000.0)
    y = labels.loc[common].to_numpy(dtype=int)
    return x, y


def save_predictions(
    path: str,
    y_val: np.ndarray,
    val_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
    val_stay_ids: np.ndarray,
    test_stay_ids: np.ndarray,
) -> None:
    """Persist one model's val/test probabilities so evaluation can be redone post-hoc.

    Without these, changing anything downstream of the fit -- the threshold criterion,
    a calibration metric, rescaling PPV to a different prevalence -- means retraining
    the whole matrix. Val probabilities are included because that is the split the
    threshold is tuned on.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        y_val=np.asarray(y_val, dtype=np.int64),
        val_prob=np.asarray(val_prob, dtype=np.float64),
        val_stay_ids=np.asarray(val_stay_ids, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        test_prob=np.asarray(test_prob, dtype=np.float64),
        test_stay_ids=np.asarray(test_stay_ids, dtype=np.int64),
    )


def run_one_model(
    model_name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold_method: str,
    bootstrap_seed: int = 42,
    predictions_path: str | None = None,
    val_stay_ids: np.ndarray | None = None,
    test_stay_ids: np.ndarray | None = None,
) -> tuple[dict[str, float | int | str], object, np.ndarray]:
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        val_prob = model.predict_proba(x_val)[:, 1]
        test_prob = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        val_scores = model.decision_function(x_val)
        test_scores = model.decision_function(x_test)
        val_prob = 1.0 / (1.0 + np.exp(-val_scores))
        test_prob = 1.0 / (1.0 + np.exp(-test_scores))
    else:
        raise ValueError(f"Model {type(model).__name__} does not expose probabilities.")

    threshold = pick_threshold(y_val, val_prob, method=threshold_method)
    metrics = evaluate(y_test, test_prob, threshold=threshold)
    if predictions_path is not None:
        save_predictions(
            predictions_path,
            y_val,
            val_prob,
            y_test,
            test_prob,
            val_stay_ids if val_stay_ids is not None else np.array([], dtype=np.int64),
            test_stay_ids if test_stay_ids is not None else np.array([], dtype=np.int64),
        )

    # The bootstrap RNG is deliberately *not* the model seed: the CI describes
    # test-set sampling variability, so holding its resamples fixed across seeds
    # keeps the interval comparable between them (and with the DL arm, which
    # does the same).
    auroc_ci_low, auroc_ci_high = bootstrap_ci(y_test, test_prob, roc_auc_score, seed=bootstrap_seed)
    auprc_ci_low, auprc_ci_high = bootstrap_ci(y_test, test_prob, average_precision_score, seed=bootstrap_seed)
    metrics["auroc_ci_low"] = auroc_ci_low
    metrics["auroc_ci_high"] = auroc_ci_high
    metrics["auprc_ci_low"] = auprc_ci_low
    metrics["auprc_ci_high"] = auprc_ci_high

    metrics["model"] = model_name
    metrics["threshold_method"] = threshold_method
    return metrics, model, test_prob


def compute_feature_importance(
    model_name: str,
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 20,
    top_k: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance (A11), scored on AUPRC, top-`top_k` features for one model."""
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=n_repeats,
        scoring="average_precision",
        random_state=seed,
    )
    order = np.argsort(result.importances_mean)[::-1][:top_k]
    rows = [
        {
            "model": model_name,
            "feature": feature_names[idx],
            "importance_mean": float(result.importances_mean[idx]),
            "importance_std": float(result.importances_std[idx]),
        }
        for idx in order
    ]
    return pd.DataFrame(rows)


def parse_seeds(seeds_arg: str) -> list[int]:
    """Parse the --seeds string (mirrors dl/run_dl_experiments.parse_seeds)."""
    seeds = [int(s.strip()) for s in str(seeds_arg).split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    return seeds


def summarise_seed_runs(seed_runs: list[dict]) -> tuple[dict, dict[str, float | str]]:
    """Pick the median-AUPRC run and summarise the spread across seeds.

    Same convention as the deep-learning arm: the reported operating point comes
    from one real run (the median by AUPRC) rather than from averaging metrics
    that are not linear in the predictions, and the seed sensitivity is reported
    separately as mean +- std. With three seeds the median is the middle one.
    """
    ordered = sorted(seed_runs, key=lambda run: run["metrics"]["auprc"])
    median_run = ordered[len(ordered) // 2]
    auroc_values = [run["metrics"]["auroc"] for run in seed_runs]
    auprc_values = [run["metrics"]["auprc"] for run in seed_runs]
    summary: dict[str, float | str] = {
        "seeds": ",".join(str(run["seed"]) for run in seed_runs),
        "median_seed": int(median_run["seed"]),
        "auroc_mean": float(np.mean(auroc_values)),
        "auroc_std": float(np.std(auroc_values, ddof=0)) if len(auroc_values) > 1 else 0.0,
        "auprc_mean": float(np.mean(auprc_values)),
        "auprc_std": float(np.std(auprc_values, ddof=0)) if len(auprc_values) > 1 else 0.0,
    }
    return median_run, summary


def library_versions() -> dict[str, str]:
    """Record the versions that produced a result set, for reproducibility."""
    import platform

    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def run_experiments(args: argparse.Namespace) -> None:
    xgboost_probe = subprocess.run(
        [sys.executable, "-c", "from xgboost import XGBClassifier; print('ok')"],
        capture_output=True,
        text=True,
    )
    xgboost_available = xgboost_probe.returncode == 0
    if not xgboost_available:
        print("[warn] XGBoost unavailable in this environment; xgboost rows will be marked as skipped.")

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "features")
    calibration_dir = os.path.join(args.output_dir, "calibration")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    bundle_noabx = build_feature_bundle(
        multimodal_dir=args.multimodal_dir,
        cache_dir=cache_dir,
        include_antibiotics=False,
        force_rebuild=args.force_rebuild,
    )
    bundle_abx = build_feature_bundle(
        multimodal_dir=args.abx_dir,
        cache_dir=cache_dir,
        include_antibiotics=True,
        force_rebuild=args.force_rebuild,
    )

    seeds = parse_seeds(args.seeds)
    split_arrays = load_splits(os.path.join(args.multimodal_dir, "cohort.csv"), seed=seeds[0])
    train_ids = split_arrays.train
    val_ids = split_arrays.val
    test_ids = split_arrays.test

    results: list[dict[str, float | int | str]] = []
    feature_importance_frames: list[pd.DataFrame] = []
    for config in ablation_configs():
        bundle = bundle_abx if config.include_antibiotics else bundle_noabx
        # Preparing one ablation's matrices must not be able to abort the sweep: the
        # primary ablation's permutation importance costs several minutes, so a
        # failure in a later config used to throw all of it away.
        try:
            matrix = build_design_matrix(
                bundle,
                config.modalities,
                fit_index=train_ids,
                drop_features=config.drop_features,
            )
            labels = bundle.labels
            if matrix.shape[1] == 0:
                raise ValueError(
                    f"Config {config.name} produced an empty design matrix. If it selects the "
                    "clinical_scores modality, run 6_build_clinical_scores.py first."
                )

            x_train, y_train = select_rows(matrix, labels, train_ids)
            x_val, y_val = select_rows(matrix, labels, val_ids)
            x_test, y_test = select_rows(matrix, labels, test_ids)
            if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
                raise ValueError(f"Empty split encountered in config {config.name}.")
        except Exception as exc:
            for model_name in config.models or ("logreg", "tree_hgbt", "xgboost"):
                results.append(
                    {
                        "ablation": config.name,
                        "model": model_name,
                        "modalities": "+".join(config.modalities),
                        "include_antibiotics": int(config.include_antibiotics),
                        "dropped_features": ",".join(config.drop_features),
                        "threshold_method": args.threshold_method,
                        "status": "failed",
                        "error": f"design matrix: {exc}",
                    }
                )
            print(f"[fail] {config.name} | design matrix | {exc}")
            continue

        negatives = max(1, int((y_train == 0).sum()))
        positives = max(1, int((y_train == 1).sum()))
        scale_pos_weight = negatives / positives

        # Builders take the seed, so each of --seeds gets its own model instance.
        model_builders = {
            "logreg": lambda seed: make_logreg(random_state=seed),
            "tree_hgbt": lambda seed: make_tree(random_state=seed),
        }
        if xgboost_available:
            model_builders["xgboost"] = lambda seed: make_xgb(
                random_state=seed,
                scale_pos_weight=scale_pos_weight,
            )

        base_row_common: dict[str, float | int | str] = {
            "ablation": config.name,
            "modalities": "+".join(config.modalities),
            "include_antibiotics": int(config.include_antibiotics),
            "dropped_features": ",".join(config.drop_features),
            "n_features": int(matrix.shape[1]),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "threshold_method": args.threshold_method,
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        if not xgboost_available and (config.models is None or "xgboost" in config.models):
            results.append(
                {
                    **base_row_common,
                    "model": "xgboost",
                    "status": "skipped",
                    "error": "xgboost unavailable (missing OpenMP/libomp).",
                }
            )

        matrix_ids = matrix.index.to_numpy(dtype=np.int64)
        val_row_ids = np.intersect1d(matrix_ids, val_ids.astype(np.int64))
        test_row_ids = np.intersect1d(matrix_ids, test_ids.astype(np.int64))

        selected_models = config.models if config.models is not None else tuple(model_builders)
        for model_name in selected_models:
            builder = model_builders.get(model_name)
            if builder is None:
                continue
            base_row = {**base_row_common, "model": model_name}

            seed_runs: list[dict] = []
            failure: str | None = None
            for seed in seeds:
                try:
                    model = builder(seed)
                except Exception as exc:
                    failure = f"model construction failed: {exc}"
                    break
                try:
                    metrics, fitted_model, test_prob = run_one_model(
                        model_name=model_name,
                        model=model,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        x_test=x_test,
                        y_test=y_test,
                        threshold_method=args.threshold_method,
                        bootstrap_seed=seeds[0],
                        predictions_path=os.path.join(
                            predictions_dir, f"{config.name}__{model_name}_seed{seed}.npz"
                        ),
                        val_stay_ids=val_row_ids,
                        test_stay_ids=test_row_ids,
                    )
                except Exception as exc:
                    failure = str(exc)
                    break
                seed_runs.append(
                    {"seed": seed, "metrics": metrics, "model": fitted_model, "test_prob": test_prob}
                )
                print(
                    f"[done] {config.name} | {model_name} | seed={seed} | "
                    f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
                    f"sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f}"
                )

            if failure is not None or not seed_runs:
                row = dict(base_row)
                row["status"] = "failed" if failure else "skipped"
                row["error"] = failure or "no seed produced a run"
                results.append(row)
                print(f"[fail] {config.name} | {model_name} | {failure}")
                continue

            median_run, summary = summarise_seed_runs(seed_runs)
            row = dict(median_run["metrics"])
            row.update(base_row)
            row.update(summary)
            row["status"] = "ok"
            results.append(row)
            if len(seeds) > 1:
                print(
                    f"[seeds] {config.name} | {model_name} | AUROC {summary['auroc_mean']:.4f}"
                    f"+-{summary['auroc_std']:.4f} | reported seed {summary['median_seed']}"
                )

            # Calibration curve and permutation importance describe *the reported
            # run*, so both come from the median seed rather than from an average.
            save_calibration_plot(
                y_test,
                median_run["test_prob"],
                os.path.join(calibration_dir, f"{config.name}__{model_name}.png"),
            )
            if config.name == PRIMARY_ABLATION:
                print(f"[feature_importance] {config.name} | {model_name} | computing permutation importance...")
                feature_importance_frames.append(
                    compute_feature_importance(
                        model_name=model_name,
                        model=median_run["model"],
                        x_test=x_test,
                        y_test=y_test,
                        feature_names=list(matrix.columns),
                        seed=int(median_run["seed"]),
                    )
                )

    metrics_df = pd.DataFrame(results).sort_values(["ablation", "model"]).reset_index(drop=True)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if feature_importance_frames:
        feature_importance_df = pd.concat(feature_importance_frames, ignore_index=True)
        feature_importance_path = os.path.join(args.output_dir, "feature_importance.csv")
        feature_importance_df.to_csv(feature_importance_path, index=False)
        print(f"Wrote feature importance: {feature_importance_path}")

    cohort = bundle_noabx.cohort
    prevalence = float(cohort["label"].mean()) if len(cohort) else np.nan
    # Lead time is a property of the positives only: controls carry no onset, so
    # including them used to drag the reported median from the real 6 h down to
    # 3 h (they contributed a 0 h "lead" each).
    lead_time_hours = np.nan
    if {"onset_time", "prediction_time", "label"} <= set(cohort.columns):
        positives = cohort[cohort["label"] == 1]
        lead_deltas = (positives["onset_time"] - positives["prediction_time"]) / 3600
        lead_deltas = lead_deltas[np.isfinite(lead_deltas)]
        if len(lead_deltas):
            lead_time_hours = float(lead_deltas.median())

    manifest = {
        "mimic_version": args.mimic_version,
        "lead_time_hours": lead_time_hours,
        "lead_time_basis": "positives only (controls have no onset_time)",
        "library_versions": library_versions(),
        "lookback_hours": 24,
        "seeds": seeds,
        "threshold_method": args.threshold_method,
        # Los tres modelos clásicos corren en CPU (`n_jobs=1` en XGBoost); el campo
        # existe para que el manifiesto registre lo mismo que el del brazo profundo,
        # que sí elige dispositivo, y para que §5.2 de la memoria sea cierta de los dos.
        "device": "cpu",
        "reference_prevalences": list(REFERENCE_PREVALENCES),
        "cohort_size": int(len(cohort)),
        "prevalence": prevalence,
        "input_paths": {
            "multimodal_dir": args.multimodal_dir,
            "abx_dir": args.abx_dir,
        },
        "outputs": {
            "metrics_csv": metrics_path,
            "predictions_dir": predictions_dir,
            "calibration_dir": calibration_dir,
            "feature_cache_dir": cache_dir,
        },
        "xgboost_available": xgboost_available,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    run_experiments(parse_args())
