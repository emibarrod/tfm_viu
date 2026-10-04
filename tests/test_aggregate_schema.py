"""Feature-aggregation schema regression tests (A10): `aggregate_long_modality()`
and `build_static_matrix()` output shapes/column-counts must match what's
already recorded per ablation in `data/05_results/metrics.csv`, so a silent
schema drift in the aggregation code gets caught before it corrupts results."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from multi_modality_code.experiments.features.aggregate import (
    _group_slopes,
    _normalize_concept,
    _safe_slope,
    aggregate_long_modality,
    build_feature_bundle,
    build_static_matrix,
)
from multi_modality_code.experiments.data_utils.splits import load_splits
from multi_modality_code.experiments.run_experiments import ablation_configs, build_design_matrix

MULTIMODAL_DIR = os.path.join("data", "04_multimodal")
ABX_DIR = os.path.join("data", "04_multimodal_abx")
FEATURE_CACHE_DIR = os.path.join("data", "05_results", "features")
METRICS_PATH = os.path.join("data", "05_results", "metrics.csv")


def _require(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{path} not present; run the Stage 1-4 pipeline (and run_experiments.py) first.")


def test_build_static_matrix_shape_and_index() -> None:
    static_path = os.path.join(MULTIMODAL_DIR, "static.csv")
    _require(static_path)
    static_df = pd.read_csv(static_path)
    matrix = build_static_matrix(static_df)
    assert matrix.index.name == "stay_id"
    assert matrix.shape[1] > 0
    assert not matrix.index.duplicated().any()


def test_aggregate_long_modality_one_row_per_stay() -> None:
    vitals_path = os.path.join(MULTIMODAL_DIR, "vitals_timeseries.csv")
    _require(vitals_path)
    vitals_df = pd.read_csv(vitals_path, nrows=200_000)
    matrix = aggregate_long_modality(vitals_df)
    assert matrix.index.name == "stay_id"
    assert not matrix.index.duplicated().any()
    # Eight summary stats per concept: count/present/last/min/max/mean/std/slope.
    n_concepts = vitals_df["concept"].nunique()
    assert matrix.shape[1] <= n_concepts * 8


def test_ablation_feature_widths_match_metrics_csv() -> None:
    """Cross-check `build_design_matrix()` widths against the already-recorded
    `n_features` column in metrics.csv, so any aggregation-schema drift is
    caught relative to ground truth rather than a possibly-stale hardcoded
    expectation.

    `fit_index` is the training split, exactly as `run_experiments.py` passes it,
    so the zero-variance filter is part of what gets compared. This test
    therefore fails against a metrics.csv produced before that filter existed --
    rerun the experiment stage to refresh the ground truth."""
    _require(METRICS_PATH)
    _require(os.path.join(MULTIMODAL_DIR, "cohort.csv"))
    _require(os.path.join(ABX_DIR, "cohort.csv"))

    metrics = pd.read_csv(METRICS_PATH)
    expected_widths = metrics.drop_duplicates("ablation").set_index("ablation")["n_features"].to_dict()

    train_ids = load_splits(os.path.join(MULTIMODAL_DIR, "cohort.csv"), seed=42).train
    bundle_noabx = build_feature_bundle(multimodal_dir=MULTIMODAL_DIR, cache_dir=FEATURE_CACHE_DIR, include_antibiotics=False)
    bundle_abx = build_feature_bundle(multimodal_dir=ABX_DIR, cache_dir=FEATURE_CACHE_DIR, include_antibiotics=True)

    for config in ablation_configs():
        bundle = bundle_abx if config.include_antibiotics else bundle_noabx
        matrix = build_design_matrix(
            bundle,
            config.modalities,
            fit_index=train_ids,
            drop_features=config.drop_features,
        )
        assert config.name in expected_widths, f"{config.name} missing from metrics.csv"
        assert matrix.shape[1] == expected_widths[config.name], (
            f"{config.name}: build_design_matrix produced {matrix.shape[1]} columns, "
            f"metrics.csv recorded n_features={expected_widths[config.name]}"
        )


def test_group_slopes_matches_safe_slope_exactly() -> None:
    """`_group_slopes()` must agree with `_safe_slope()` to the last bit.

    The vectorised path stacks equal-length groups and reduces the last axis,
    relying on numpy summing a row of a C-contiguous block exactly as it sums
    the same values as a standalone 1-D array. That holds today, but it is a
    property of numpy rather than a promise, and the feature cache it feeds is
    read back into HistGradientBoosting -- where a last-digit change moves a bin
    edge and with it the reported metrics. So it is asserted, not assumed.
    """
    vitals_path = os.path.join(MULTIMODAL_DIR, "vitals_timeseries.csv")
    _require(vitals_path)
    df = pd.read_csv(vitals_path, nrows=200_000)

    work = df[["stay_id", "concept", "value", "hours_before_prediction"]].copy()
    for column in ("stay_id", "value", "hours_before_prediction"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["concept"] = _normalize_concept(work["concept"])
    work = work.dropna(subset=["stay_id", "value", "hours_before_prediction"]).copy()
    work["stay_id"] = work["stay_id"].astype("int64")

    group_cols = ["stay_id", "concept"]
    vectorised = _group_slopes(work, group_cols).set_index(group_cols)["slope"]
    reference = {
        key: _safe_slope(sub["hours_before_prediction"].to_numpy(), sub["value"].to_numpy())
        for key, sub in work.groupby(group_cols)
    }

    assert len(vectorised) == len(reference)
    for key, expected in reference.items():
        actual = vectorised.loc[key]
        if pd.isna(expected):
            assert pd.isna(actual), f"{key}: expected NaN, got {actual!r}"
        else:
            # Exact equality, not approximate: see the docstring.
            assert actual == expected, f"{key}: {actual!r} != {expected!r}"
