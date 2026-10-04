"""Split loading and leakage checks for multimodal Sepsis-3 experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


VALID_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "te": "test",
}


@dataclass(frozen=True)
class SplitArrays:
    """Container with stay ids per split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return split arrays as a dictionary."""
        return {"train": self.train, "val": self.val, "test": self.test}


def load_cohort(cohort_path: str) -> pd.DataFrame:
    """Load cohort CSV and normalize critical columns."""
    cohort = pd.read_csv(cohort_path)
    required = {"stay_id", "subject_id", "label"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"{cohort_path} missing required columns: {sorted(missing)}")

    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce").astype("Int64")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id", "subject_id", "label"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    cohort["subject_id"] = cohort["subject_id"].astype(np.int64)
    cohort["label"] = cohort["label"].astype(np.int64)
    return cohort


def _normalize_split_values(split_series: pd.Series) -> pd.Series:
    normalized = split_series.astype(str).str.lower().str.strip().map(SPLIT_ALIASES)
    if normalized.isna().any():
        bad = sorted(split_series[normalized.isna()].astype(str).str.strip().unique().tolist())
        raise ValueError(f"Unknown split values: {bad}")
    return normalized


def assert_no_subject_leakage(cohort: pd.DataFrame) -> None:
    """Raise when any subject appears in more than one split."""
    if "split" not in cohort.columns:
        raise ValueError("Cannot check leakage: 'split' column missing from cohort.")

    working = cohort.copy()
    working["split"] = _normalize_split_values(working["split"])

    grouped = working.groupby("subject_id")["split"].nunique()
    leaking = grouped[grouped > 1]
    if not leaking.empty:
        raise ValueError(
            f"Subject leakage detected: {len(leaking)} subjects in multiple splits "
            f"(first examples: {leaking.index[:10].tolist()})"
        )


def make_grouped_splits(
    cohort: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Create subject-grouped train/val/test splits when split is unavailable."""
    if cohort.empty:
        raise ValueError("Cannot create splits from an empty cohort.")

    working = cohort.copy()
    working = working.sort_values("stay_id").reset_index(drop=True)
    y = working["label"].to_numpy()
    groups = working["subject_id"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(working, y=y, groups=groups))
    split = np.full(len(working), "train", dtype=object)
    split[test_idx] = "test"

    train_val_df = working.iloc[train_val_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_idx_inner, val_idx_inner = next(
        inner.split(
            train_val_df,
            y=train_val_df["label"].to_numpy(),
            groups=train_val_df["subject_id"].to_numpy(),
        )
    )
    split[train_val_idx[val_idx_inner]] = "val"
    split[train_val_idx[train_idx_inner]] = "train"
    working["split"] = split

    assert_no_subject_leakage(working)
    return working


def load_splits(cohort_path: str, seed: int = 42) -> SplitArrays:
    """Load existing split column from cohort, fallback to grouped split creation."""
    cohort = load_cohort(cohort_path)
    if "split" in cohort.columns:
        cohort = cohort.copy()
        cohort["split"] = _normalize_split_values(cohort["split"])
        unknown = set(cohort["split"].unique()) - set(VALID_SPLITS)
        if unknown:
            raise ValueError(f"Invalid normalized split values: {sorted(unknown)}")
    else:
        cohort = make_grouped_splits(cohort, seed=seed)

    assert_no_subject_leakage(cohort)
    arrays = []
    for split_name in VALID_SPLITS:
        stay_ids = cohort.loc[cohort["split"] == split_name, "stay_id"].to_numpy(dtype=np.int64)
        if stay_ids.size == 0:
            raise ValueError(f"Split '{split_name}' is empty.")
        arrays.append(np.unique(stay_ids))

    return SplitArrays(train=arrays[0], val=arrays[1], test=arrays[2])
