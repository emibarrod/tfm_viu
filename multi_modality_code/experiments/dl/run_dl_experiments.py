"""CLI entry point for the DL (GRU/LSTM intermediate-fusion) experiments (A3/A4/A9).

Mirrors `run_experiments.py`: reuses the identical `AblationConfig` list, the
identical `subject_id`-grouped splits, and the identical threshold-tuning /
evaluation protocol, swapping only the model family for a fair ML-vs-DL
comparison. See `multi_modality_code/experiments/dl/README.md` for the exact
commands (smoke test + full 8-ablation run) and the results they produced.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from multi_modality_code.experiments.data_utils.splits import load_splits
from multi_modality_code.experiments.dl.datasets import AblationGridDataset
from multi_modality_code.experiments.dl.grid_data import GridBundle, build_grid_bundle
from multi_modality_code.experiments.dl.importance import permutation_importance_dl
from multi_modality_code.experiments.dl.models import FusionClassifier, count_parameters
from multi_modality_code.experiments.dl.train import TrainConfig, select_device, set_seed, train_model
from multi_modality_code.experiments.evaluate import bootstrap_ci, evaluate, pick_threshold, save_calibration_plot
from multi_modality_code.experiments.run_experiments import (
    PRIMARY_ABLATION,
    AblationConfig,
    ablation_configs,
    save_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRU/LSTM intermediate-fusion DL experiments on the frozen Sepsis-3 task."
    )
    parser.add_argument("--multimodal_dir", default="data/04_multimodal")
    parser.add_argument("--abx_dir", default="data/04_multimodal_abx")
    parser.add_argument("--output_dir", default="data/05_results/deep_learning")
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds trained independently per ablation; dl_metrics.csv reports mean+-std across them (A9).",
    )
    parser.add_argument("--cell_type", choices=["gru", "lstm"], default="gru")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers Apple MPS when available. Both are reproducible at a "
        "fixed seed and similarly fast for this model size, but they disagree with each other, so "
        "pin one for any reported run; the resolved device is recorded in dl_manifest.json.",
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--threshold_method",
        choices=["f1", "youden"],
        default="youden",
        help="Criterion used to place the decision threshold on the validation split; must match the "
        "classical arm for the comparison to be symmetric. Default 'youden' -- see "
        "run_experiments.py for why F1 is not the primary criterion here.",
    )
    parser.add_argument("--mimic_version", default="MIMIC-IV v3.1")
    parser.add_argument(
        "--max_stays",
        type=int,
        default=None,
        help="Optional small, label-balanced cohort subset for a fast correctness smoke test "
        "(same convention as 4_build_multimodal_dataset.py's --max_stays).",
    )
    parser.add_argument(
        "--permutation_repeats",
        type=int,
        default=20,
        help="Permutation-importance repeats on the primary ablation's median-seed model (A11), "
        "matching run_experiments.py's n_repeats. 0 skips the analysis.",
    )
    parser.add_argument(
        "--force_rebuild_cache",
        action="store_true",
        help="Ignore cached grid tensors under <output_dir>/features and rebuild them from the Stage-4 CSVs.",
    )
    return parser.parse_args()


def library_versions() -> dict[str, str]:
    """Resolved versions of the libraries that can move the numbers (mirrors run_experiments.py)."""
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def parse_seeds(seeds_arg: str) -> list[int]:
    seeds = [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed.")
    return seeds


def _subset_stay_ids(cohort_path: str, max_stays: int, seed: int) -> np.ndarray:
    """Label-balanced stay_id subset for --max_stays, mirroring load_onset()'s convention."""
    cohort = pd.read_csv(cohort_path, usecols=["stay_id", "label"])
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id", "label"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    cohort["label"] = cohort["label"].astype(int)

    rng = np.random.RandomState(seed)
    parts = [group.sample(n=min(len(group), max_stays // 2), random_state=rng) for _, group in cohort.groupby("label")]
    subset = pd.concat(parts)
    remaining = max_stays - len(subset)
    if remaining > 0:
        leftover = cohort[~cohort["stay_id"].isin(set(subset["stay_id"]))]
        subset = pd.concat([subset, leftover.sample(n=min(remaining, len(leftover)), random_state=rng)])
    return subset["stay_id"].to_numpy(dtype=np.int64)


def build_input_dims(bundle: GridBundle, modalities: tuple[str, ...]) -> dict[str, int]:
    return {name: bundle.branch_input_dim(name) for name in modalities}


def run_one_ablation(
    config: AblationConfig,
    bundle: GridBundle,
    split_ids: dict[str, np.ndarray],
    seeds: list[int],
    args: argparse.Namespace,
    calibration_dir: str,
    device: torch.device,
    predictions_dir: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame | None]:
    train_ds = AblationGridDataset(bundle, split_ids["train"], config.modalities)
    val_ds = AblationGridDataset(bundle, split_ids["val"], config.modalities)
    test_ds = AblationGridDataset(bundle, split_ids["test"], config.modalities)
    input_dims = build_input_dims(bundle, config.modalities)

    seed_runs = []
    for seed in seeds:
        # Seed *before* constructing the model: torch's default generator is seeded from OS
        # entropy at first use, so building the encoders before train_model()'s set_seed()
        # left weight initialisation uncontrolled -- identical seeds gave different runs, and
        # the multi-seed spread mixed real seed sensitivity with unseeded init noise.
        set_seed(seed)
        model = FusionClassifier(
            modalities=config.modalities,
            input_dims=input_dims,
            cell_type=args.cell_type,
            dropout=args.dropout,
        )
        train_config = TrainConfig(
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            pos_weight=args.pos_weight,
            seed=seed,
        )
        result = train_model(model, train_ds, val_ds, test_ds, train_config, device=device)
        threshold = pick_threshold(result.val_labels, result.val_probs, method=args.threshold_method)
        test_metrics = evaluate(result.test_labels, result.test_probs, threshold=threshold)
        # One file per seed (not just the median one) so the seed spread stays re-analysable.
        save_predictions(
            os.path.join(predictions_dir, f"{config.name}__{args.cell_type}_seed{seed}.npz"),
            result.val_labels,
            result.val_probs,
            result.test_labels,
            result.test_probs,
            val_ds.stay_ids,
            test_ds.stay_ids,
        )
        seed_runs.append(
            {
                "seed": seed,
                "metrics": test_metrics,
                "n_epochs_trained": result.n_epochs_trained,
                "test_probs": result.test_probs,
                "test_labels": result.test_labels,
                "n_params": count_parameters(result.model),
                "device": str(result.device),
                # Kept so permutation importance can be scored on the *reported*
                # model instead of retraining it; three small models, not a leak.
                "model": result.model,
            }
        )
        print(
            f"[done] {config.name} | seed={seed} | AUROC={test_metrics['auroc']:.4f} "
            f"AUPRC={test_metrics['auprc']:.4f} epochs={result.n_epochs_trained}"
        )

    auroc_values = [run["metrics"]["auroc"] for run in seed_runs]
    auprc_values = [run["metrics"]["auprc"] for run in seed_runs]
    median_run = sorted(seed_runs, key=lambda run: run["metrics"]["auprc"])[len(seed_runs) // 2]

    calibration_path = os.path.join(calibration_dir, f"{config.name}__{args.cell_type}.png")
    save_calibration_plot(median_run["test_labels"], median_run["test_probs"], calibration_path)

    # Bootstrap CIs (A8) computed on the median-seed run's test predictions,
    # same helper as the classical pipeline, for apples-to-apples reporting.
    auroc_ci_low, auroc_ci_high = bootstrap_ci(median_run["test_labels"], median_run["test_probs"], roc_auc_score, seed=seeds[0])
    auprc_ci_low, auprc_ci_high = bootstrap_ci(
        median_run["test_labels"], median_run["test_probs"], average_precision_score, seed=seeds[0]
    )

    row: dict[str, float | int | str] = dict(median_run["metrics"])
    row.update(
        {
            "auroc_ci_low": auroc_ci_low,
            "auroc_ci_high": auroc_ci_high,
            "auprc_ci_low": auprc_ci_low,
            "auprc_ci_high": auprc_ci_high,
            "model": f"{args.cell_type}_intermediate_fusion",
            "ablation": config.name,
            "modalities": "+".join(config.modalities),
            "include_antibiotics": int(config.include_antibiotics),
            "n_features": sum(input_dims.values()),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "threshold_method": args.threshold_method,
            "status": "ok",
            "cell_type": args.cell_type,
            "hidden_dims": json.dumps({"static": 16, "vitals": 64, "labs": 64, "treatments": 32}),
            "n_params": median_run["n_params"],
            "n_epochs_trained": median_run["n_epochs_trained"],
            "device": median_run["device"],
            "seeds": ",".join(str(seed) for seed in seeds),
            # Which seed the reported operating point comes from, so
            # posthoc_analysis.py can pair the right prediction dumps.
            "median_seed": int(median_run["seed"]),
            "auroc_mean": float(np.mean(auroc_values)),
            "auroc_std": float(np.std(auroc_values, ddof=0)) if len(auroc_values) > 1 else 0.0,
            "auprc_mean": float(np.mean(auprc_values)),
            "auprc_std": float(np.std(auprc_values, ddof=0)) if len(auprc_values) > 1 else 0.0,
        }
    )

    # Permutation importance describes *the reported run*, so it uses the median
    # seed's model -- the same convention as the calibration curve above and as
    # run_experiments.py, which also restricts the analysis to the primary
    # ablation.
    importance = None
    if config.name == PRIMARY_ABLATION and args.permutation_repeats > 0:
        print(f"[feature_importance] {config.name} | computing permutation importance...")
        importance = permutation_importance_dl(
            model=median_run["model"],
            bundle=bundle,
            dataset=test_ds,
            device=device,
            model_name=f"{args.cell_type}_intermediate_fusion",
            ablation=config.name,
            n_repeats=args.permutation_repeats,
            seed=int(median_run["seed"]),
        )
    return row, importance


def run_dl_experiments(args: argparse.Namespace) -> None:
    seeds = parse_seeds(args.seeds)
    device = select_device(args.device)
    print(f"[device] {device} (requested: {args.device})")
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "features")
    calibration_dir = os.path.join(args.output_dir, "calibration")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    split_arrays = load_splits(os.path.join(args.multimodal_dir, "cohort.csv"), seed=seeds[0])
    stay_ids_filter = None
    cache_tag_suffix = ""
    if args.max_stays is not None:
        stay_ids_filter = _subset_stay_ids(os.path.join(args.multimodal_dir, "cohort.csv"), args.max_stays, seeds[0])
        cache_tag_suffix = f"_max{args.max_stays}"

    train_ids = split_arrays.train if stay_ids_filter is None else np.intersect1d(split_arrays.train, stay_ids_filter)
    val_ids = split_arrays.val if stay_ids_filter is None else np.intersect1d(split_arrays.val, stay_ids_filter)
    test_ids = split_arrays.test if stay_ids_filter is None else np.intersect1d(split_arrays.test, stay_ids_filter)
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        if len(ids) == 0:
            raise ValueError(f"Split '{name}' is empty after applying --max_stays; increase --max_stays.")

    bundle_noabx = build_grid_bundle(
        multimodal_dir=args.multimodal_dir,
        cache_dir=cache_dir,
        train_stay_ids=train_ids,
        include_antibiotics=False,
        stay_ids_filter=stay_ids_filter,
        cache_tag=f"noabx{cache_tag_suffix}",
        force_rebuild=args.force_rebuild_cache,
    )
    bundle_abx = build_grid_bundle(
        multimodal_dir=args.abx_dir,
        cache_dir=cache_dir,
        train_stay_ids=train_ids,
        include_antibiotics=True,
        stay_ids_filter=stay_ids_filter,
        cache_tag=f"abx{cache_tag_suffix}",
        force_rebuild=args.force_rebuild_cache,
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    # Ablations flagged `classical_only` are skipped rather than run: this arm
    # resolves branch widths from `bundle.branch_input_dim()`, which has no
    # clinical-score branch and does not know about `drop_features`, so it cannot
    # reproduce them faithfully. Named in the log so the asymmetry with
    # metrics.csv is visible rather than inferred.
    configs = [config for config in ablation_configs() if not config.classical_only]
    skipped = [config.name for config in ablation_configs() if config.classical_only]
    if skipped:
        print(f"[skip] classical-only ablations not mirrored by the DL arm: {', '.join(skipped)}")

    results: list[dict[str, float | int | str]] = []
    importance_frames: list[pd.DataFrame] = []
    for config in configs:
        bundle = bundle_abx if config.include_antibiotics else bundle_noabx
        try:
            row, importance = run_one_ablation(
                config, bundle, split_ids, seeds, args, calibration_dir, device, predictions_dir
            )
            results.append(row)
            if importance is not None:
                importance_frames.append(importance)
        except Exception as exc:  # keep looping across ablations, mirroring run_experiments.py's resilience
            results.append(
                {
                    "ablation": config.name,
                    "model": f"{args.cell_type}_intermediate_fusion",
                    "modalities": "+".join(config.modalities),
                    "include_antibiotics": int(config.include_antibiotics),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[fail] {config.name} | {exc}")

    metrics_df = pd.DataFrame(results).sort_values(["ablation"]).reset_index(drop=True)
    metrics_path = os.path.join(args.output_dir, "dl_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    importance_path = None
    if importance_frames:
        importance_path = os.path.join(args.output_dir, "dl_feature_importance.csv")
        pd.concat(importance_frames, ignore_index=True).to_csv(importance_path, index=False)
        print(f"Wrote feature importance: {importance_path}")

    manifest = {
        "mimic_version": args.mimic_version,
        "lookback_hours": 24,
        "seeds": seeds,
        "cell_type": args.cell_type,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "device_requested": args.device,
        "threshold_method": args.threshold_method,
        "library_versions": library_versions(),
        "max_stays": args.max_stays,
        "permutation_repeats": args.permutation_repeats,
        "cohort_size": bundle_noabx.n_stays,
        "input_paths": {"multimodal_dir": args.multimodal_dir, "abx_dir": args.abx_dir},
        "outputs": {
            "dl_metrics_csv": metrics_path,
            "dl_feature_importance_csv": importance_path,
            "predictions_dir": predictions_dir,
            "calibration_dir": calibration_dir,
            "feature_cache_dir": cache_dir,
        },
    }
    manifest_path = os.path.join(args.output_dir, "dl_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    run_dl_experiments(parse_args())
