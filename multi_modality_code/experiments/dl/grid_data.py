"""Hourly-grid tensor builder for DL sequence models on the frozen Sepsis-3 task.

Reuses the existing Stage 4 exports (`data/04_multimodal[_abx]/*.csv`) unchanged
and turns them into dense `(N, 24, C, 3)` tensors — one 24-hour hourly grid per
stay, per modality, with `value` / `mask` / `delta` channels per concept. See
`multi_modality_code/experiments/dl/README.md` for the full specification of
the binning, fill, and normalization rules implemented here.

Timestep convention: index `t=0` is the oldest hour in the 24h lookback window
(`hours_before_prediction` in `[23, 24)`) and `t=23` is the most recent hour
(`hours_before_prediction` in `[0, 1)`), i.e. tensors are chronological
oldest -> newest, matching the order fed into the recurrent encoders.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from multi_modality_code.experiments.features.aggregate import build_static_matrix, drop_zero_variance


N_BINS = 24

VITALS_CONCEPTS: tuple[str, ...] = (
    "respiratory_rate",
    "map",
    "sbp_arterial",
    "dbp_arterial",
    "heart_rate",
    "spo2",
    "mechvent",
    "temp_F",
    "oxygen_flow_device",
    "richmond_ras",
    "oxygen_flow",
    "peep",
    "fio2",
    "tidal_volume",
    "minute_volume",
    "temp_C",
)

LABS_CONCEPTS: tuple[str, ...] = (
    "glucose",
    "potassium",
    "hematocrit",
    "sodium",
    "hemoglobin",
    "chloride",
    "creatinine",
    "ph_arterial",
    "platelets",
    "wbc",
    "arterial_o2_pressure",
    "arterial_base_excess",
    "arterial_co2_pressure",
    "ptt",
    "pt",
    "inr",
    "urea_nitrogen",
    "hco3",
    "lactate",
    "lactic_acid",
    "bilirubin_total",
)

TREATMENTS_CONCEPTS_PRIMARY: tuple[str, ...] = ("iv_fluid", "vasopressor_rate", "urine_output")
TREATMENTS_CONCEPTS_ABX: tuple[str, ...] = TREATMENTS_CONCEPTS_PRIMARY + ("antibiotic_active",)

# Cumulative/rate treatment-amount concepts use sum-per-bin + log1p normalization
# (A1, concept class 2); everything else (all vitals, all labs, vasopressor_rate)
# is a point-in-time/physiological-state concept using mean-per-bin + forward-fill
# + plain z-score (A1, concept class 1).
CUMULATIVE_CONCEPTS: frozenset[str] = frozenset({"iv_fluid", "urine_output", "antibiotic_active"})


@dataclass(frozen=True)
class ModalityGrid:
    """One modality's `(N, 24, C)` value/mask/delta channels, already normalized."""

    concepts: tuple[str, ...]
    value: np.ndarray
    mask: np.ndarray
    delta: np.ndarray
    norm_stats: dict[str, dict[str, float]]

    def as_array(self) -> np.ndarray:
        """Stack channels into the `(N, 24, C * 3)` tensor fed to `SequenceEncoder`.

        Channels are interleaved per concept: `[value_1, mask_1, delta_1,
        value_2, mask_2, delta_2, ...]`.
        """
        n, t, c = self.value.shape
        stacked = np.stack([self.value, self.mask, self.delta], axis=-1)  # (N, T, C, 3)
        return stacked.reshape(n, t, c * 3).astype(np.float32)


@dataclass(frozen=True)
class GridBundle:
    """Cached hourly-grid tensors for one export (primary or antibiotics-sensitivity)."""

    stay_ids: np.ndarray
    labels: np.ndarray
    static: np.ndarray
    static_columns: tuple[str, ...]
    vitals: ModalityGrid
    labs: ModalityGrid
    treatments: ModalityGrid

    @property
    def n_stays(self) -> int:
        return int(self.stay_ids.shape[0])

    def branch_input_dim(self, modality: str) -> int:
        if modality == "static":
            return self.static.shape[1]
        if modality == "vitals":
            return len(self.vitals.concepts) * 3
        if modality == "labs":
            return len(self.labs.concepts) * 3
        if modality == "treatments":
            return len(self.treatments.concepts) * 3
        raise ValueError(f"Unknown modality: {modality!r}")


def _bin_index(hours_before_prediction: np.ndarray) -> np.ndarray:
    """Map `hours_before_prediction` to bin `b in [0, 23]` (b=0 closest to prediction_time)."""
    b = np.floor(hours_before_prediction).astype(np.int64)
    return np.clip(b, 0, N_BINS - 1)


def _chrono_time_index(bin_index: np.ndarray) -> np.ndarray:
    """Map bin index (0=newest..23=oldest) to chronological timestep (0=oldest..23=newest)."""
    return (N_BINS - 1) - bin_index


def _delta_from_mask(mask: np.ndarray) -> np.ndarray:
    """Hours since the last True in `mask` along axis=1 (chronological), capped at N_BINS.

    `mask` shape `(N, N_BINS)`. Returns `N_BINS` (the cap) for timesteps with no
    prior (or current) real observation anywhere in the window, else the
    integer number of hourly steps back to the most recent real observation.
    """
    n, t_len = mask.shape
    t_idx = np.arange(t_len)
    last_true_idx = np.where(mask > 0, t_idx, -1)
    last_true_idx = np.maximum.accumulate(last_true_idx, axis=1)
    delta = np.where(
        last_true_idx < 0,
        float(N_BINS),
        np.minimum(N_BINS, t_idx[None, :] - last_true_idx),
    )
    return delta.astype(np.float64)


def _train_mean_std(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def _build_point_in_time_grid(
    long_df: pd.DataFrame,
    concepts: tuple[str, ...],
    stay_index: pd.Index,
    train_stay_id_set: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Concept class 1 (A1): mean-per-bin, forward-filled, train-split z-scored."""
    n = len(stay_index)
    n_concepts = len(concepts)
    value = np.zeros((n, N_BINS, n_concepts), dtype=np.float64)
    mask = np.zeros((n, N_BINS, n_concepts), dtype=np.float64)
    delta = np.full((n, N_BINS, n_concepts), float(N_BINS), dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}

    sub_all = long_df[long_df["concept"].isin(concepts)].copy()
    if not sub_all.empty:
        sub_all["t"] = _chrono_time_index(_bin_index(sub_all["hours_before_prediction"].to_numpy()))

    for ci, concept in enumerate(concepts):
        sub = sub_all[sub_all["concept"] == concept] if not sub_all.empty else sub_all
        train_values = sub.loc[sub["stay_id"].isin(train_stay_id_set), "value"].to_numpy(dtype=np.float64)
        train_mean, train_std = _train_mean_std(train_values)

        if sub.empty:
            filled = np.full((n, N_BINS), train_mean, dtype=np.float64)
            mask_c = np.zeros((n, N_BINS), dtype=np.float64)
        else:
            grouped = sub.groupby(["stay_id", "t"], as_index=False)["value"].mean()
            pivot = grouped.pivot(index="stay_id", columns="t", values="value")
            pivot = pivot.reindex(index=stay_index, columns=range(N_BINS))
            raw = pivot.to_numpy(dtype=np.float64)
            mask_c = (~np.isnan(raw)).astype(np.float64)
            # copy=True: desde pandas 3.0 `to_numpy()` puede devolver un buffer de solo
            # lectura, y las dos líneas siguientes escriben sobre `filled` in-place.
            filled = pd.DataFrame(raw).ffill(axis=1).to_numpy(copy=True)
            never_observed = mask_c.sum(axis=1) == 0
            filled[never_observed] = train_mean
            # Rows where the *first* real bins are still NaN (forward-fill from
            # oldest->newest has nothing earlier to carry) fall back to the
            # train mean too, per A1's fill rule.
            filled = np.where(np.isnan(filled), train_mean, filled)

        delta_c = _delta_from_mask(mask_c)

        value[:, :, ci] = (filled - train_mean) / train_std
        mask[:, :, ci] = mask_c
        delta[:, :, ci] = delta_c / float(N_BINS)
        stats[concept] = {"mean": train_mean, "std": train_std}

    return value, mask, delta, stats


def _resolve_cumulative_hours_before_end(df: pd.DataFrame) -> pd.Series:
    """Bin cumulative treatment concepts by `observed_endtime`, falling back to `event_time`."""
    if {"observed_endtime", "prediction_time"} <= set(df.columns):
        hours_before_end = (df["prediction_time"] - df["observed_endtime"]) / 3600.0
        hours_before_end = hours_before_end.fillna(df["hours_before_prediction"])
        return hours_before_end
    return df["hours_before_prediction"]


def _build_cumulative_grid(
    long_df: pd.DataFrame,
    concepts: tuple[str, ...],
    stay_index: pd.Index,
    train_stay_id_set: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Concept class 2 (A1): sum-per-bin (no fill), log1p then train-split z-scored."""
    n = len(stay_index)
    n_concepts = len(concepts)
    value = np.zeros((n, N_BINS, n_concepts), dtype=np.float64)
    mask = np.zeros((n, N_BINS, n_concepts), dtype=np.float64)
    delta = np.full((n, N_BINS, n_concepts), float(N_BINS), dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}

    sub_all = long_df[long_df["concept"].isin(concepts)].copy()
    if not sub_all.empty:
        hours_before_end = _resolve_cumulative_hours_before_end(sub_all)
        sub_all["t"] = _chrono_time_index(_bin_index(hours_before_end.to_numpy()))

    train_stay_mask_full = stay_index.to_series().isin(train_stay_id_set).to_numpy()

    for ci, concept in enumerate(concepts):
        sub = sub_all[sub_all["concept"] == concept] if not sub_all.empty else sub_all

        if sub.empty:
            raw = np.zeros((n, N_BINS), dtype=np.float64)
            mask_c = np.zeros((n, N_BINS), dtype=np.float64)
        else:
            grouped = sub.groupby(["stay_id", "t"], as_index=False)["value"].sum()
            pivot = grouped.pivot(index="stay_id", columns="t", values="value")
            pivot = pivot.reindex(index=stay_index, columns=range(N_BINS))
            pivoted = pivot.to_numpy(dtype=np.float64)
            mask_c = (~np.isnan(pivoted)).astype(np.float64)
            raw = np.nan_to_num(pivoted, nan=0.0)

        log_values = np.log1p(np.clip(raw, a_min=0.0, a_max=None))
        train_mean, train_std = _train_mean_std(log_values[train_stay_mask_full].ravel())
        delta_c = _delta_from_mask(mask_c)

        value[:, :, ci] = (log_values - train_mean) / train_std
        mask[:, :, ci] = mask_c
        delta[:, :, ci] = delta_c / float(N_BINS)
        stats[concept] = {"mean": train_mean, "std": train_std}

    return value, mask, delta, stats


def _build_modality_grid(
    long_df: pd.DataFrame,
    point_in_time_concepts: tuple[str, ...],
    cumulative_concepts: tuple[str, ...],
    all_concepts_in_order: tuple[str, ...],
    stay_index: pd.Index,
    train_stay_id_set: set[int],
) -> ModalityGrid:
    pit_value, pit_mask, pit_delta, pit_stats = _build_point_in_time_grid(
        long_df, point_in_time_concepts, stay_index, train_stay_id_set
    )
    cum_value, cum_mask, cum_delta, cum_stats = _build_cumulative_grid(
        long_df, cumulative_concepts, stay_index, train_stay_id_set
    )

    pit_lookup = {c: i for i, c in enumerate(point_in_time_concepts)}
    cum_lookup = {c: i for i, c in enumerate(cumulative_concepts)}
    n = len(stay_index)
    value = np.zeros((n, N_BINS, len(all_concepts_in_order)), dtype=np.float64)
    mask = np.zeros((n, N_BINS, len(all_concepts_in_order)), dtype=np.float64)
    delta = np.zeros((n, N_BINS, len(all_concepts_in_order)), dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}
    for out_idx, concept in enumerate(all_concepts_in_order):
        if concept in pit_lookup:
            src = pit_lookup[concept]
            value[:, :, out_idx] = pit_value[:, :, src]
            mask[:, :, out_idx] = pit_mask[:, :, src]
            delta[:, :, out_idx] = pit_delta[:, :, src]
            stats[concept] = pit_stats[concept]
        else:
            src = cum_lookup[concept]
            value[:, :, out_idx] = cum_value[:, :, src]
            mask[:, :, out_idx] = cum_mask[:, :, src]
            delta[:, :, out_idx] = cum_delta[:, :, src]
            stats[concept] = cum_stats[concept]

    _validate_modality_grid(f"{all_concepts_in_order}", value, mask, delta)
    return ModalityGrid(concepts=all_concepts_in_order, value=value, mask=mask, delta=delta, norm_stats=stats)


def _validate_modality_grid(name: str, value: np.ndarray, mask: np.ndarray, delta: np.ndarray) -> None:
    if value.shape[1] != N_BINS or mask.shape[1] != N_BINS or delta.shape[1] != N_BINS:
        raise ValueError(f"{name}: expected {N_BINS} timesteps in every channel.")
    if np.isnan(value).any():
        raise ValueError(f"{name}: NaNs remain in the value channel after fill/normalization.")
    if np.isnan(mask).any() or not np.isin(np.unique(mask), [0.0, 1.0]).all():
        raise ValueError(f"{name}: mask channel must be binary {{0, 1}}.")
    if np.isnan(delta).any() or (delta < -1e-9).any() or (delta > 1.0 + 1e-9).any():
        raise ValueError(f"{name}: delta channel must lie in [0, 1] after /24 normalization.")


def _load_long_csv(path: str, usecols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["hours_before_prediction"] = pd.to_numeric(df["hours_before_prediction"], errors="coerce")
    df = df.dropna(subset=["stay_id", "value", "hours_before_prediction"]).copy()
    df["stay_id"] = df["stay_id"].astype(np.int64)
    if (df["hours_before_prediction"] < -1e-6).any():
        raise ValueError(f"{path}: found rows with hours_before_prediction < 0 (prediction-time leakage).")
    return df


def _assert_static_matches(static_arr: np.ndarray, columns: tuple[str, ...]) -> None:
    if static_arr.shape[1] != len(columns):
        raise ValueError("Static matrix column count mismatch after alignment.")
    if np.isnan(static_arr).any():
        raise ValueError("Static matrix contains NaNs after train-mean imputation.")


def _grid_source_paths(multimodal_dir: str) -> list[str]:
    return [
        os.path.join(multimodal_dir, name)
        for name in ("cohort.csv", "static.csv", "vitals_timeseries.csv", "labs_timeseries.csv", "treatments_timeseries.csv")
    ]


def _cache_stale_reason(cached, multimodal_dir: str, treatments_concepts: tuple[str, ...]) -> str | None:
    """Why a cached grid bundle must not be reused, or None when it is still valid.

    Two independent hazards: the Stage-4 exports may have been regenerated since
    the tensors were built, and the concept tuples in this module may have
    changed. The second is the nastier one -- concepts were not stored in the
    `.npz`, so a reload rebuilt `ModalityGrid` with today's tuples over
    yesterday's channel order, silently mislabelling every feature.
    """
    for key, expected in (
        ("vitals_concepts", VITALS_CONCEPTS),
        ("labs_concepts", LABS_CONCEPTS),
        ("treatments_concepts", treatments_concepts),
    ):
        if key not in cached:
            return "cache predates concept-list validation"
        if tuple(json.loads(cached[key].item())) != tuple(expected):
            return f"{key} changed since the cache was written"

    if "source_fingerprint" not in cached:
        return "no source fingerprint recorded"
    if json.loads(cached["source_fingerprint"].item()) != _source_fingerprint(multimodal_dir):
        return "Stage-4 exports changed since the cache was written"
    return None


def _source_fingerprint(multimodal_dir: str) -> dict[str, list[float]]:
    fingerprint: dict[str, list[float]] = {}
    for path in _grid_source_paths(multimodal_dir):
        try:
            stat = os.stat(path)
            fingerprint[os.path.basename(path)] = [float(stat.st_size), float(stat.st_mtime)]
        except OSError:
            fingerprint[os.path.basename(path)] = [-1.0, -1.0]
    return fingerprint


def build_grid_bundle(
    multimodal_dir: str,
    cache_dir: str,
    train_stay_ids: np.ndarray,
    include_antibiotics: bool = False,
    stay_ids_filter: np.ndarray | None = None,
    cache_tag: str | None = None,
    force_rebuild: bool = False,
) -> GridBundle:
    """Build (or load cached) hourly-grid tensors for one Stage-4 export.

    `train_stay_ids` must come from `data_utils.splits.load_splits(...).train`
    (or a subset of it, e.g. under `--max_stays`) so that every normalization
    statistic is computed strictly from the training split, never val/test.
    """
    suffix = "abx" if include_antibiotics else "noabx"
    tag = cache_tag or suffix
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"grid_{tag}.npz")
    stats_path = os.path.join(cache_dir, f"grid_{tag}_norm_stats.json")

    treatments_concepts = TREATMENTS_CONCEPTS_ABX if include_antibiotics else TREATMENTS_CONCEPTS_PRIMARY

    if not force_rebuild and os.path.exists(npz_path):
        cached = np.load(npz_path, allow_pickle=False)
        stale_reason = _cache_stale_reason(cached, multimodal_dir, treatments_concepts)
        if stale_reason is not None:
            print(f"[cache] rebuilding {os.path.basename(npz_path)}: {stale_reason}")
            cached = None
    else:
        cached = None

    if cached is not None:
        bundle = GridBundle(
            stay_ids=cached["stay_ids"],
            labels=cached["labels"],
            static=cached["static"],
            static_columns=tuple(json.loads(cached["static_columns"].item())),
            vitals=ModalityGrid(
                concepts=VITALS_CONCEPTS,
                value=cached["vitals_value"],
                mask=cached["vitals_mask"],
                delta=cached["vitals_delta"],
                norm_stats={},
            ),
            labs=ModalityGrid(
                concepts=LABS_CONCEPTS,
                value=cached["labs_value"],
                mask=cached["labs_mask"],
                delta=cached["labs_delta"],
                norm_stats={},
            ),
            treatments=ModalityGrid(
                concepts=treatments_concepts,
                value=cached["treatments_value"],
                mask=cached["treatments_mask"],
                delta=cached["treatments_delta"],
                norm_stats={},
            ),
        )
        return bundle

    cohort = pd.read_csv(os.path.join(multimodal_dir, "cohort.csv"))
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["label"] = pd.to_numeric(cohort["label"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id", "label"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    cohort["label"] = cohort["label"].astype(np.int64)
    cohort = cohort.sort_values("stay_id").drop_duplicates("stay_id")

    if stay_ids_filter is not None:
        keep = np.isin(cohort["stay_id"].to_numpy(), np.asarray(stay_ids_filter, dtype=np.int64))
        cohort = cohort[keep]

    stay_index = pd.Index(np.sort(cohort["stay_id"].unique()), name="stay_id")
    train_stay_id_set = set(int(s) for s in train_stay_ids) & set(int(s) for s in stay_index.to_numpy())
    if not train_stay_id_set:
        raise ValueError("No training stay_ids overlap with the (possibly filtered) cohort; cannot compute norm stats.")

    labels = cohort.set_index("stay_id").loc[stay_index, "label"].to_numpy(dtype=np.int64)

    static_df = pd.read_csv(os.path.join(multimodal_dir, "static.csv"))
    static_raw = build_static_matrix(static_df).reindex(stay_index)
    # Same zero-variance filter the classical baselines apply, fitted on the same
    # training stays, so both model families really do see an identical static
    # feature set -- the comparison claims exactly that.
    static_raw = drop_zero_variance(static_raw, fit_index=np.asarray(sorted(train_stay_id_set), dtype=np.int64))
    static_columns = tuple(static_raw.columns)
    static_arr = static_raw.to_numpy(dtype=np.float64)
    train_positions = stay_index.to_series().isin(train_stay_id_set).to_numpy()
    train_static = static_arr[train_positions]
    static_means = np.nanmean(train_static, axis=0) if train_static.size else np.zeros(static_arr.shape[1])
    static_stds = np.nanstd(train_static, axis=0) if train_static.size else np.ones(static_arr.shape[1])
    static_stds = np.where((~np.isfinite(static_stds)) | (static_stds < 1e-8), 1.0, static_stds)
    static_means = np.where(np.isfinite(static_means), static_means, 0.0)
    static_arr = np.where(np.isnan(static_arr), static_means[None, :], static_arr)
    static_arr = (static_arr - static_means[None, :]) / static_stds[None, :]
    _assert_static_matches(static_arr, static_columns)

    vitals_df = _load_long_csv(
        os.path.join(multimodal_dir, "vitals_timeseries.csv"),
        usecols=["stay_id", "concept", "value", "hours_before_prediction"],
    )
    vitals_df = vitals_df[vitals_df["stay_id"].isin(stay_index)]
    vitals_grid = _build_modality_grid(
        vitals_df, VITALS_CONCEPTS, (), VITALS_CONCEPTS, stay_index, train_stay_id_set
    )

    labs_df = _load_long_csv(
        os.path.join(multimodal_dir, "labs_timeseries.csv"),
        usecols=["stay_id", "concept", "value", "hours_before_prediction"],
    )
    labs_df = labs_df[labs_df["stay_id"].isin(stay_index)]
    labs_grid = _build_modality_grid(labs_df, LABS_CONCEPTS, (), LABS_CONCEPTS, stay_index, train_stay_id_set)

    treatments_cols = ["stay_id", "concept", "value", "hours_before_prediction", "prediction_time", "observed_endtime"]
    treatments_raw = pd.read_csv(os.path.join(multimodal_dir, "treatments_timeseries.csv"), usecols=treatments_cols, low_memory=False)
    treatments_raw["stay_id"] = pd.to_numeric(treatments_raw["stay_id"], errors="coerce")
    treatments_raw["value"] = pd.to_numeric(treatments_raw["value"], errors="coerce")
    treatments_raw["hours_before_prediction"] = pd.to_numeric(treatments_raw["hours_before_prediction"], errors="coerce")
    treatments_raw["prediction_time"] = pd.to_numeric(treatments_raw["prediction_time"], errors="coerce")
    treatments_raw["observed_endtime"] = pd.to_numeric(treatments_raw["observed_endtime"], errors="coerce")
    treatments_raw = treatments_raw.dropna(subset=["stay_id", "value", "hours_before_prediction"]).copy()
    treatments_raw["stay_id"] = treatments_raw["stay_id"].astype(np.int64)
    if (treatments_raw["hours_before_prediction"] < -1e-6).any():
        raise ValueError("treatments_timeseries.csv: found rows with hours_before_prediction < 0 (leakage).")
    treatments_raw = treatments_raw[treatments_raw["stay_id"].isin(stay_index)]
    point_in_time_treatment_concepts = tuple(c for c in treatments_concepts if c not in CUMULATIVE_CONCEPTS)
    cumulative_treatment_concepts = tuple(c for c in treatments_concepts if c in CUMULATIVE_CONCEPTS)
    treatments_grid = _build_modality_grid(
        treatments_raw,
        point_in_time_treatment_concepts,
        cumulative_treatment_concepts,
        treatments_concepts,
        stay_index,
        train_stay_id_set,
    )

    bundle = GridBundle(
        stay_ids=stay_index.to_numpy(dtype=np.int64),
        labels=labels,
        static=static_arr.astype(np.float32),
        static_columns=static_columns,
        vitals=vitals_grid,
        labs=labs_grid,
        treatments=treatments_grid,
    )

    np.savez_compressed(
        npz_path,
        stay_ids=bundle.stay_ids,
        labels=bundle.labels,
        static=bundle.static,
        static_columns=np.array(json.dumps(list(static_columns))),
        vitals_concepts=np.array(json.dumps(list(VITALS_CONCEPTS))),
        labs_concepts=np.array(json.dumps(list(LABS_CONCEPTS))),
        treatments_concepts=np.array(json.dumps(list(treatments_concepts))),
        source_fingerprint=np.array(json.dumps(_source_fingerprint(multimodal_dir))),
        vitals_value=bundle.vitals.value.astype(np.float32),
        vitals_mask=bundle.vitals.mask.astype(np.float32),
        vitals_delta=bundle.vitals.delta.astype(np.float32),
        labs_value=bundle.labs.value.astype(np.float32),
        labs_mask=bundle.labs.mask.astype(np.float32),
        labs_delta=bundle.labs.delta.astype(np.float32),
        treatments_value=bundle.treatments.value.astype(np.float32),
        treatments_mask=bundle.treatments.mask.astype(np.float32),
        treatments_delta=bundle.treatments.delta.astype(np.float32),
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vitals": bundle.vitals.norm_stats,
                "labs": bundle.labs.norm_stats,
                "treatments": bundle.treatments.norm_stats,
                "static_columns": list(static_columns),
                "n_train_stay_ids_used": len(train_stay_id_set),
            },
            f,
            indent=2,
        )

    return bundle
