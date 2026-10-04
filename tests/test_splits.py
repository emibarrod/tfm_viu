"""Subject-grouped split correctness (A10): no leakage on the real cohort export,
and the grouped-split fallback produces disjoint subject sets."""

from __future__ import annotations

import os

import pytest

from multi_modality_code.experiments.data_utils.splits import (
    assert_no_subject_leakage,
    load_cohort,
    make_grouped_splits,
)

COHORT_PATH = os.path.join("data", "04_multimodal", "cohort.csv")


def _require_cohort() -> None:
    if not os.path.exists(COHORT_PATH):
        pytest.skip(f"{COHORT_PATH} not present (data/ is gitignored; run the Stage 1-4 pipeline first).")


def test_existing_cohort_has_no_subject_leakage() -> None:
    _require_cohort()
    cohort = load_cohort(COHORT_PATH)
    assert "split" in cohort.columns
    # Raises ValueError on leakage; a clean return is the assertion.
    assert_no_subject_leakage(cohort)


def test_grouped_split_fallback_produces_disjoint_subjects() -> None:
    _require_cohort()
    cohort = load_cohort(COHORT_PATH).drop(columns=["split"], errors="ignore")
    split_cohort = make_grouped_splits(cohort, seed=42)

    subjects_by_split = {
        name: set(split_cohort.loc[split_cohort["split"] == name, "subject_id"])
        for name in ("train", "val", "test")
    }
    assert subjects_by_split["train"] & subjects_by_split["val"] == set()
    assert subjects_by_split["train"] & subjects_by_split["test"] == set()
    assert subjects_by_split["val"] & subjects_by_split["test"] == set()
    for name, subjects in subjects_by_split.items():
        assert len(subjects) > 0, f"split '{name}' has no subjects"

    # assert_no_subject_leakage() already runs inside make_grouped_splits(); re-check explicitly too.
    assert_no_subject_leakage(split_cohort)


def test_grouped_split_fallback_is_deterministic_given_seed() -> None:
    _require_cohort()
    cohort = load_cohort(COHORT_PATH).drop(columns=["split"], errors="ignore")
    first = make_grouped_splits(cohort, seed=42)
    second = make_grouped_splits(cohort, seed=42)
    merged = first[["stay_id", "split"]].merge(
        second[["stay_id", "split"]], on="stay_id", suffixes=("_first", "_second")
    )
    assert (merged["split_first"] == merged["split_second"]).all()
