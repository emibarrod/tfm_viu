"""Aggregate long-format multimodal tables into per-stay feature matrices."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


KEY_COLUMNS = {"stay_id", "subject_id", "label", "prediction_time", "onset_time"}


@dataclass(frozen=True)
class FeatureBundle:
    """Aggregated modality matrices with labels and cohort metadata."""

    cohort: pd.DataFrame
    labels: pd.Series
    modalities: dict[str, pd.DataFrame]


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y versus x; returns NaN when undefined.

    Reference definition of the `__slope` feature. `_group_slopes()` computes the
    same thing for every group at once; `tests/test_aggregate_schema.py` pins the
    two against each other.
    """
    if x.size < 2:
        return np.nan
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 0:
        return np.nan
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _group_slopes(work: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """`_safe_slope()` per (stay_id, concept), without a Python-level group loop.

    Sorting puts each group in one contiguous block, and groups that happen to
    have the same number of observations are then stacked into a single
    `(n_groups, block_length)` array and reduced along its last axis. That is
    the same summation numpy performs on a standalone 1-D array of that length,
    so the result is bit-identical to calling `_safe_slope()` group by group --
    which matters more than it looks: the feature cache is read back into
    HistGradientBoosting, whose bin edges sit at percentiles, and a last-digit
    change in a feature is enough to move an edge and with it the metrics (see
    `_load_or_build_matrix`). The equality is asserted in the schema tests
    rather than assumed.

    Iterating over the distinct block lengths (a few dozen) instead of over the
    groups (hundreds of thousands) is what makes this the fast path: it turned
    the slowest step of the feature build into a minor one.
    """
    stay_ids = work[group_cols[0]].to_numpy()
    concepts = work[group_cols[1]].to_numpy()
    # Stable, so rows keep their original order inside each group, exactly as
    # `groupby` presented them to `_safe_slope`.
    order = np.lexsort((concepts, stay_ids))
    stay_ids = stay_ids[order]
    concepts = concepts[order]
    x = work["hours_before_prediction"].to_numpy(dtype=float)[order]
    y = work["value"].to_numpy(dtype=float)[order]

    starts_mask = np.empty(len(order), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = (stay_ids[1:] != stay_ids[:-1]) | (concepts[1:] != concepts[:-1])
    starts = np.flatnonzero(starts_mask)
    lengths = np.diff(np.append(starts, len(order)))

    slopes = np.full(starts.shape, np.nan)
    for length in np.unique(lengths):
        if length < 2:  # a single observation defines no slope
            continue
        which = np.flatnonzero(lengths == length)
        block = starts[which][:, None] + np.arange(length)[None, :]
        block_x = x[block]
        block_y = y[block]
        dx = block_x - block_x.mean(axis=1, keepdims=True)
        dy = block_y - block_y.mean(axis=1, keepdims=True)
        denom = (dx * dx).sum(axis=1)
        defined = denom > 0  # all-equal timestamps (or NaN) leave the slope undefined
        values = np.full(len(which), np.nan)
        values[defined] = (dx * dy).sum(axis=1)[defined] / denom[defined]
        slopes[which] = values

    return pd.DataFrame(
        {group_cols[0]: stay_ids[starts], group_cols[1]: concepts[starts], "slope": slopes}
    )


def _normalize_concept(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        .replace("", "unknown")
    )


def _pivot_stat_frame(stats_df: pd.DataFrame, stat_name: str) -> pd.DataFrame:
    wide = stats_df.pivot(index="stay_id", columns="concept", values=stat_name)
    wide.columns = [f"{concept}__{stat_name}" for concept in wide.columns]
    return wide


def aggregate_long_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long table with (stay_id, concept, value, hours_before_prediction)."""
    required = {"stay_id", "concept", "value", "hours_before_prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Long modality missing columns: {sorted(missing)}")

    work = df[list(required)].copy()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["hours_before_prediction"] = pd.to_numeric(work["hours_before_prediction"], errors="coerce")
    work["concept"] = _normalize_concept(work["concept"])
    work = work.dropna(subset=["stay_id", "value", "hours_before_prediction"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["stay_id"] = work["stay_id"].astype(np.int64)

    group_cols = ["stay_id", "concept"]
    grouped = work.groupby(group_cols, as_index=False)
    stats = grouped["value"].agg(count="count", min="min", max="max", mean="mean", std="std")
    stats["present"] = 1.0

    idx_last = work.groupby(group_cols)["hours_before_prediction"].idxmin()
    last_values = (
        work.loc[idx_last, ["stay_id", "concept", "value"]]
        .rename(columns={"value": "last"})
        .reset_index(drop=True)
    )
    stats = stats.merge(last_values, on=group_cols, how="left")

    stats = stats.merge(_group_slopes(work, group_cols), on=group_cols, how="left")

    feature_frames = []
    for stat_name in ("count", "present", "last", "min", "max", "mean", "std", "slope"):
        feature_frames.append(_pivot_stat_frame(stats, stat_name))

    out = pd.concat(feature_frames, axis=1).sort_index()
    out.index = out.index.astype(np.int64)
    out.index.name = "stay_id"
    return out


def build_static_matrix(static_df: pd.DataFrame) -> pd.DataFrame:
    """Return one-hot encoded static matrix indexed by stay_id."""
    if "stay_id" not in static_df.columns:
        raise ValueError("static.csv missing stay_id")

    work = static_df.copy()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["stay_id"]).copy()
    work["stay_id"] = work["stay_id"].astype(np.int64)
    work = work.set_index("stay_id")

    categorical_cols = [c for c in ("gender", "unit") if c in work.columns]
    # `gender` (1/2) and `unit` (0-6) are numeric *codes* for nominal categories.
    # They are excluded from the numeric block so they enter only as one-hot
    # dummies; keeping the raw column too would both duplicate the information
    # and hand the models a fake ordering over ICU unit types.
    numeric_cols = [
        c
        for c in work.columns
        if c not in KEY_COLUMNS
        and c not in categorical_cols
        and pd.api.types.is_numeric_dtype(work[c])
    ]
    frames = []
    if numeric_cols:
        frames.append(work[numeric_cols].copy())
    if categorical_cols:
        cat_frames = []
        for column in categorical_cols:
            encoded = pd.get_dummies(
                work[column].astype("string"),
                prefix=column,
                dummy_na=True,
            )
            cat_frames.append(encoded)
        frames.append(pd.concat(cat_frames, axis=1))

    if not frames:
        return pd.DataFrame(index=work.index)
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index.name = "stay_id"
    return out


# The clinical-comparator modality holds exactly the two bedside scores and
# nothing else. Keeping it to the totals is what makes the `clinical_scores_only`
# ablation answer the question it is there for -- "does the model beat the score
# a clinician can compute at the bedside?" -- rather than quietly becoming a
# third feature block. The component columns that `6_build_clinical_scores.py`
# also exports (worst FiO2 in the window, GCS minimum, ...) stay out of the
# design matrix; they are there to make each score auditable.
CLINICAL_SCORE_COLUMNS = ("sofa_total_at_t0", "qsofa_at_t0")


def build_clinical_scores_matrix(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Return the SOFA/qSOFA-at-t0 matrix indexed by stay_id."""
    if "stay_id" not in clinical_df.columns:
        raise ValueError("clinical_scores.csv missing stay_id")
    missing = [column for column in CLINICAL_SCORE_COLUMNS if column not in clinical_df.columns]
    if missing:
        raise ValueError(f"clinical_scores.csv missing score columns: {missing}")

    work = clinical_df.copy()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce").astype("Int64")
    work = work.dropna(subset=["stay_id"]).copy()
    work["stay_id"] = work["stay_id"].astype(np.int64)
    work = work.set_index("stay_id").sort_index()
    out = work[list(CLINICAL_SCORE_COLUMNS)].astype(float)
    out.index.name = "stay_id"
    return out


def build_treatment_specific_matrix(treatments_df: pd.DataFrame) -> pd.DataFrame:
    """Build treatment-specific engineered summaries by source/concept."""
    work = treatments_df.copy()
    if work.empty:
        return pd.DataFrame()
    work["stay_id"] = pd.to_numeric(work["stay_id"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    if "observed_duration_hours" in work.columns:
        work["observed_duration_hours"] = pd.to_numeric(work["observed_duration_hours"], errors="coerce")
    else:
        work["observed_duration_hours"] = np.nan
    work["concept"] = _normalize_concept(work["concept"])
    work["source"] = _normalize_concept(work["source"])
    work = work.dropna(subset=["stay_id"]).copy()
    work["stay_id"] = work["stay_id"].astype(np.int64)

    by_stay = []
    for stay_id, sub in work.groupby("stay_id"):
        row: dict[str, float | int] = {"stay_id": int(stay_id)}
        iv = sub[sub["concept"] == "iv_fluid"]["value"]
        row["treatments__iv_fluid_sum"] = float(iv.sum()) if not iv.empty else np.nan
        uo = sub[sub["concept"] == "urine_output"]["value"]
        row["treatments__urine_output_sum"] = float(uo.sum()) if not uo.empty else np.nan
        vaso = sub[sub["concept"] == "vasopressor_rate"]["value"]
        row["treatments__vasopressor_rate_mean"] = float(vaso.mean()) if not vaso.empty else np.nan
        row["treatments__vasopressor_rate_max"] = float(vaso.max()) if not vaso.empty else np.nan
        row["treatments__observed_duration_sum"] = float(sub["observed_duration_hours"].sum(min_count=1))
        by_stay.append(row)

    out = pd.DataFrame(by_stay)
    if out.empty:
        return pd.DataFrame()
    out = out.set_index("stay_id").sort_index()
    out.index = out.index.astype(np.int64)
    return out


# Aggregates whose "absent" state is a well-defined zero rather than an unknown.
# `__count` and `__present` count observations, and the treatment `_sum` columns
# total event amounts: if nothing was recorded in the window, the honest value is
# 0, not NaN. Everything else (`__min/max/mean/std/last/slope`, vasopressor rate
# statistics) summarises observed values and is genuinely undefined without them.
STRUCTURAL_ZERO_SUFFIXES = ("__count", "__present")
STRUCTURAL_ZERO_COLUMNS = frozenset(
    {
        "treatments__iv_fluid_sum",
        "treatments__urine_output_sum",
        "treatments__observed_duration_sum",
    }
)


def fill_structural_zeros(matrix: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with 0 in count/presence/sum columns, leaving value stats NaN.

    This matters because the models are now handed real NaNs (see
    `run_experiments.select_rows`). Without it, a `__present` column is 1.0 where
    observed and NaN where not, so median imputation collapses it to a constant
    1.0 and the logistic regression loses the missingness indicator entirely --
    the very signal the ablation analysis is meant to measure. Tree models would
    still cope, so the damage would have been silent and model-specific.
    """
    if matrix.empty:
        return matrix
    targets = [
        column
        for column in matrix.columns
        if column.endswith(STRUCTURAL_ZERO_SUFFIXES) or column in STRUCTURAL_ZERO_COLUMNS
    ]
    if not targets:
        return matrix
    matrix = matrix.copy()
    matrix[targets] = matrix[targets].fillna(0.0)
    return matrix


def drop_zero_variance(matrix: pd.DataFrame, fit_index: np.ndarray | None = None) -> pd.DataFrame:
    """Drop columns with no variance, measured on `fit_index` rows only.

    Under the current cohort filters several static columns are structurally
    constant -- `adm_order` is always 1 and `re_admission` always 0 because the
    cohort keeps only each patient's first ICU stay, and the `dummy_na` columns
    are all-zero when nothing is missing. They cost width and dilute permutation
    importance without carrying any signal.

    `fit_index` should be the training stay_ids, so the decision never looks at
    validation or test rows. (A column constant across the whole cohort is
    information-free either way, but fitting on train keeps the rule uniform with
    every other preprocessing decision in the pipeline.)
    """
    if matrix.empty:
        return matrix
    fit_rows = matrix
    if fit_index is not None:
        common = matrix.index.intersection(pd.Index(np.asarray(fit_index, dtype=np.int64)))
        if len(common):
            fit_rows = matrix.loc[common]

    keep = []
    for column in matrix.columns:
        values = pd.to_numeric(fit_rows[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            keep.append(column)
    return matrix[keep]


def source_fingerprint(paths: list[str]) -> dict[str, dict[str, float]]:
    """Size + mtime of each source file, used to detect a stale feature cache."""
    fingerprint: dict[str, dict[str, float]] = {}
    for path in sorted(paths):
        try:
            stat = os.stat(path)
        except OSError:
            fingerprint[path] = {"size": -1.0, "mtime": -1.0}
            continue
        fingerprint[path] = {"size": float(stat.st_size), "mtime": float(stat.st_mtime)}
    return fingerprint


def _load_or_build_matrix(
    cache_path: str,
    build_fn: Callable[[], pd.DataFrame],
    fingerprint: dict[str, dict[str, float]] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load the cached matrix when it still matches its sources, else rebuild it.

    The cache used to be reused whenever the file merely existed, with no check
    that it corresponded to the current inputs -- so any change upstream (a new
    cohort, a fix in the aggregation code) was silently ignored and stale
    features flowed into the results. A `.meta.json` sidecar now records the
    size/mtime of the source exports; a mismatch forces a rebuild.
    """
    meta_path = f"{cache_path}.meta.json"
    if not force_rebuild and os.path.exists(cache_path):
        stale_reason = None
        if fingerprint is not None:
            if not os.path.exists(meta_path):
                stale_reason = "no fingerprint recorded"
            else:
                with open(meta_path, encoding="utf-8") as handle:
                    stored = json.load(handle)
                if stored.get("sources") != fingerprint:
                    stale_reason = "source files changed since the cache was written"

        if stale_reason is None:
            # `float_precision="round_trip"` is the other half of the `%.17g` written
            # below. pandas' default CSV float parser is fast but not correctly
            # rounding, so 17-digit text still came back altered in ~28 % of cells
            # (by up to 9e-10) -- which was enough to keep HistGradientBoosting
            # disagreeing with itself between the run that wrote the cache and the
            # runs that read it. With both halves in place the reload is exact.
            cached = pd.read_csv(cache_path, float_precision="round_trip")
            if "stay_id" not in cached.columns:
                raise ValueError(f"Cache file {cache_path} missing stay_id column.")
            cached["stay_id"] = pd.to_numeric(cached["stay_id"], errors="coerce").astype("Int64")
            cached = cached.dropna(subset=["stay_id"]).copy()
            cached["stay_id"] = cached["stay_id"].astype(np.int64)
            cached = cached.set_index("stay_id").sort_index()
            return cached
        print(f"[cache] rebuilding {os.path.basename(cache_path)}: {stale_reason}")

    matrix = build_fn()
    matrix = matrix.copy()
    matrix.index.name = "stay_id"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # 17 significant digits is the shortest decimal representation guaranteed to
    # round-trip a float64. Without it pandas' default (~12 digits for some values)
    # altered 4-5 % of the cells by up to 3e-11 -- invisible in isolation, but
    # HistGradientBoosting places its bin edges at percentiles, so a perturbation
    # that small moves an edge and changes the tree: the run that *wrote* the cache
    # and the runs that *read* it disagreed by up to 0.008 AUROC (and 0.20
    # specificity, because the tuned threshold jumped to another part of the curve).
    # Writing 17 digits is necessary but not sufficient -- see the reader above.
    matrix.reset_index().to_csv(cache_path, index=False, float_format="%.17g")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"sources": fingerprint or {}}, handle, indent=2)
    return matrix


def build_feature_bundle(
    multimodal_dir: str,
    cache_dir: str,
    include_antibiotics: bool = False,
    force_rebuild: bool = False,
) -> FeatureBundle:
    """Build or load cached modality matrices and return aligned feature bundle."""
    cohort = pd.read_csv(os.path.join(multimodal_dir, "cohort.csv"))
    required = {"stay_id", "subject_id", "label", "split"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort.csv missing required columns: {sorted(missing)}")

    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce").astype("Int64")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id", "subject_id", "label"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    cohort["subject_id"] = cohort["subject_id"].astype(np.int64)
    cohort["label"] = cohort["label"].astype(np.int64)
    cohort = cohort.sort_values("stay_id").drop_duplicates("stay_id")

    suffix = "abx" if include_antibiotics else "noabx"
    static_df = pd.read_csv(os.path.join(multimodal_dir, "static.csv"))
    clinical_path = os.path.join(multimodal_dir, "clinical_scores.csv")
    # Optional so the classical arm still runs on an export predating Stage 6;
    # the ablations that need it fail loudly instead of silently scoring 0.
    clinical_df = pd.read_csv(clinical_path) if os.path.exists(clinical_path) else None
    vitals_df = pd.read_csv(os.path.join(multimodal_dir, "vitals_timeseries.csv"))
    labs_df = pd.read_csv(os.path.join(multimodal_dir, "labs_timeseries.csv"))
    treatments_df = pd.read_csv(
        os.path.join(multimodal_dir, "treatments_timeseries.csv"),
        low_memory=False,
    )
    if not include_antibiotics and {"source", "concept"} <= set(treatments_df.columns):
        source = treatments_df["source"].astype(str).str.lower()
        concept = treatments_df["concept"].astype(str).str.lower()
        treatments_df = treatments_df[(source != "antibiotic") & (concept != "antibiotic_active")].copy()

    def _path(name: str) -> str:
        return os.path.join(multimodal_dir, name)

    static_fp = source_fingerprint([_path("static.csv")])
    clinical_fp = source_fingerprint([_path("clinical_scores.csv")])
    vitals_fp = source_fingerprint([_path("vitals_timeseries.csv")])
    labs_fp = source_fingerprint([_path("labs_timeseries.csv")])
    treatments_fp = source_fingerprint([_path("treatments_timeseries.csv")])

    static_matrix = _load_or_build_matrix(
        os.path.join(cache_dir, f"static_{suffix}.csv"),
        lambda: build_static_matrix(static_df),
        fingerprint=static_fp,
        force_rebuild=force_rebuild,
    )
    clinical_matrix = (
        _load_or_build_matrix(
            os.path.join(cache_dir, f"clinical_scores_{suffix}.csv"),
            lambda: build_clinical_scores_matrix(clinical_df),
            fingerprint=clinical_fp,
            force_rebuild=force_rebuild,
        )
        if clinical_df is not None
        else pd.DataFrame(index=pd.Index([], name="stay_id", dtype=np.int64))
    )
    vitals_matrix = _load_or_build_matrix(
        os.path.join(cache_dir, f"vitals_{suffix}.csv"),
        lambda: aggregate_long_modality(vitals_df),
        fingerprint=vitals_fp,
        force_rebuild=force_rebuild,
    )
    labs_matrix = _load_or_build_matrix(
        os.path.join(cache_dir, f"labs_{suffix}.csv"),
        lambda: aggregate_long_modality(labs_df),
        fingerprint=labs_fp,
        force_rebuild=force_rebuild,
    )
    treatments_long_matrix = _load_or_build_matrix(
        os.path.join(cache_dir, f"treatments_long_{suffix}.csv"),
        lambda: aggregate_long_modality(treatments_df),
        fingerprint=treatments_fp,
        force_rebuild=force_rebuild,
    )
    treatments_specific_matrix = _load_or_build_matrix(
        os.path.join(cache_dir, f"treatments_specific_{suffix}.csv"),
        lambda: build_treatment_specific_matrix(treatments_df),
        fingerprint=treatments_fp,
        force_rebuild=force_rebuild,
    )
    treatments_matrix = pd.concat([treatments_long_matrix, treatments_specific_matrix], axis=1)
    treatments_matrix = treatments_matrix.loc[:, ~treatments_matrix.columns.duplicated()]

    stay_index = pd.Index(cohort["stay_id"].to_numpy(dtype=np.int64), name="stay_id")
    # Reindexing onto the full cohort introduces NaN rows for stays that have no
    # events at all in a modality; `fill_structural_zeros` then restores the 0
    # semantics for count/presence/sum columns (see its docstring).
    modalities = {
        "static": static_matrix.reindex(stay_index),
        "vitals": fill_structural_zeros(vitals_matrix.reindex(stay_index)),
        "labs": fill_structural_zeros(labs_matrix.reindex(stay_index)),
        "treatments": fill_structural_zeros(treatments_matrix.reindex(stay_index)),
        # Not a modality of the multimodal architecture: the clinical comparator,
        # exposed the same way so one ablation can select it.
        "clinical_scores": clinical_matrix.reindex(stay_index),
    }

    labels = cohort.set_index("stay_id").loc[stay_index, "label"].astype(int)
    labels.index.name = "stay_id"
    return FeatureBundle(cohort=cohort, labels=labels, modalities=modalities)
