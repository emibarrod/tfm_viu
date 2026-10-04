"""
Build the frozen Sepsis-3 label artifact for the multimodal TFM pipeline.

This module turns the suspected-infection timing (antibiotics + cultures) plus
hourly SOFA computed from the processed ICU files into the canonical Sepsis-3
label:

    Sepsis-3 positive = suspected infection AND SOFA >= 2 within
                        [suspected_infection_time - 48h,
                         suspected_infection_time + 24h]

The predicted event (`onset_time`) is the first qualifying SOFA time inside that
association window, and `prediction_time = onset_time - lead_time`.

It writes `processed_files/sepsis3_onset.csv`, the primary source of `label`,
`onset_time`, and `prediction_time` for `4_build_multimodal_dataset.py`.

Documented simplifications (see SEPSIS3_TASK_DECISIONS.md):
- Baseline SOFA is assumed 0 (MIT-LCP sepsis3.sql convention).
- Missing SOFA components are scored 0.
- The cardiovascular component is approximate: `vaso.csv` only stores a single
  norepinephrine-equivalent rate, so the drug-specific SOFA tiers cannot be
  reproduced; we use MAP < 70 plus the norepinephrine-equivalent rate.
- Respiration uses PaO2/FiO2 when available, otherwise an SpO2/FiO2 fallback.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from multi_modality_code.utils.diagnostics import standardized_mean_difference
from multi_modality_code.utils.pipeline_io import resolve_input
# The SOFA component scorers live in `utils.scores` because two stages evaluate
# them at different instants: here on the hourly grid that *defines* onset, and
# in `6_build_clinical_scores.py` at the prediction time t0, where they act as
# the clinical comparator. One copy, one set of thresholds.
from multi_modality_code.utils.scores import (
    _isnan,
    score_cardiovascular,
    score_cns,
    score_coagulation,
    score_liver,
    score_renal,
    score_respiration,
)


HOUR = 3600.0
DAY = 24 * HOUR

# Sepsis-3 association window around suspected infection.
ASSOCIATION_BEFORE = 48 * HOUR
ASSOCIATION_AFTER = 24 * HOUR

# Trailing window used to take the worst value of each SOFA component, hour by
# hour, following the 24h convention in MIT-LCP sofa.sql.
SOFA_WORST_WINDOW = 24 * HOUR

# Concepts read from the processed files for SOFA, resolved to itemids through
# the local measurement mapping so they stay in sync with preprocessing.
CHARTEVENT_CONCEPTS = ("map", "spo2", "fio2")
LAB_CONCEPTS = ("arterial_o2_pressure", "platelets", "bilirubin_total", "creatinine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Sepsis-3 label artifact.")
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Directory with raw extraction tables.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Directory with onset and processed helper files.")
    parser.add_argument("--output_dir", default="data/03_labels", help="Output directory for the Sepsis-3 label artifact.")
    parser.add_argument(
        "--mapping_file",
        default="multi_modality_code/reference_files/measurement_mappings.json",
        help="JSON file mapping MIMIC itemids to clinical concepts.",
    )
    parser.add_argument("--output_file", default="sepsis3_onset.csv", help="Output filename inside output_dir.")
    parser.add_argument(
        "--lead_time",
        type=float,
        default=6.0,
        help="Hours before onset used as prediction cutoff. This is the single source of "
        "truth for the task's lead time: it is written into sepsis3_onset.csv and every "
        "later stage inherits it from there.",
    )
    parser.add_argument("--sofa_threshold", type=int, default=2, help="SOFA total threshold for organ dysfunction.")
    parser.add_argument("--min_los_hours", type=float, default=24.0, help="Minimum ICU length of stay in hours.")
    parser.add_argument("--min_age", type=float, default=18.0, help="Minimum age (adult cohort).")
    parser.add_argument("--control_ratio", type=float, default=1.0, help="Control-to-positive ratio.")
    parser.add_argument("--abx_before_culture_h", type=float, default=24.0, help="Antibiotic-before-culture window.")
    parser.add_argument("--culture_before_abx_h", type=float, default=72.0, help="Culture-before-antibiotic window.")
    parser.add_argument("--chunk_size", type=int, default=2_000_000, help="Rows per chunk for large event files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for control sampling and splits.")
    parser.add_argument("--train_frac", type=float, default=0.70, help="Training fraction (grouped by subject).")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction (grouped by subject).")
    return parser.parse_args()


def timed(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{name}")
            print("-" * len(name))
            start = time.time()
            result = func(*args, **kwargs)
            print(f"Done in {time.time() - start:.1f}s")
            return result

        return wrapper

    return decorator


def read_pipe_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", **kwargs)


def _find_optional(filename: str, search_dirs):
    """Return the first existing path for an optional file, or None."""
    try:
        return resolve_input(filename, search_dirs)
    except FileNotFoundError:
        return None


def to_int_stay(series: pd.Series) -> pd.Series:
    """Normalize stay_id values that may be stored as floats like ``30000484.0``."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_concept_codes(mapping_file: str, concepts) -> dict[str, set[int]]:
    with open(mapping_file) as file:
        mapping = json.load(file)
    codes: dict[str, set[int]] = {}
    for concept in concepts:
        if concept not in mapping:
            raise KeyError(f"Concept '{concept}' missing from mapping file {mapping_file}")
        codes[concept] = {int(code) for code in mapping[concept]["codes"]}
    return codes


# ---------------------------------------------------------------------------
# Suspected infection
# ---------------------------------------------------------------------------
@timed("Find suspected infection times")
def find_suspected_infection(
    abx: pd.DataFrame, bacterio: pd.DataFrame, abx_before_culture_h: float, culture_before_abx_h: float
) -> pd.DataFrame:
    """Replicate the antibiotics/culture suspected-infection rule, keeping the
    antibiotic and culture timestamps that triggered the suspicion.

    The rule is per antibiotic administration, scanning them oldest-first within
    a stay and stopping at the first one that pairs with a culture:

    * a culture drawn in `[t_abx, t_abx + abx_before_culture_h]` -> suspicion at
      the antibiotic time (the *nearest* such culture is the one recorded);
    * failing that, a culture drawn in `[t_abx - culture_before_abx_h, t_abx]`
      -> suspicion at the culture time (again the nearest one).

    "Nearest culture on each side of the antibiotic" is exactly what
    `merge_asof` computes, so the whole scan is two ordered joins plus a
    first-match-per-stay reduction, instead of a Python loop over stays x
    antibiotic administrations. Both branches are evaluated for every
    administration and the `after` branch is given precedence per row, which
    reproduces the sequential `break` because the two windows overlap only at
    `diff == 0`, where the original also takes the `after` branch.
    """
    abx = abx.dropna(subset=["stay_id", "starttime"]).copy()
    bacterio = bacterio.dropna(subset=["stay_id", "charttime"]).copy()
    abx["stay_id"] = to_int_stay(abx["stay_id"])
    bacterio["stay_id"] = to_int_stay(bacterio["stay_id"])
    abx = abx.dropna(subset=["stay_id"])
    bacterio = bacterio.dropna(subset=["stay_id"])

    abx = pd.DataFrame(
        {
            "stay_id": abx["stay_id"].astype(np.int64),
            "antibiotic_time": pd.to_numeric(abx["starttime"], errors="coerce").astype(float),
        }
    ).dropna(subset=["antibiotic_time"])
    cultures = pd.DataFrame(
        {
            "stay_id": bacterio["stay_id"].astype(np.int64),
            "subject_id": bacterio["subject_id"],
            "culture_time": pd.to_numeric(bacterio["charttime"], errors="coerce").astype(float),
        }
    ).dropna(subset=["culture_time"])
    if abx.empty or cultures.empty:
        return pd.DataFrame(
            columns=["subject_id", "stay_id", "suspected_infection_time", "antibiotic_time", "culture_time"]
        )

    # `merge_asof` needs the join key globally ordered, not just within `by`.
    abx = abx.sort_values("antibiotic_time", kind="stable", ignore_index=True)
    cultures = cultures.sort_values("culture_time", kind="stable", ignore_index=True)
    culture_keys = cultures[["stay_id", "culture_time"]]

    # Nearest culture at or after each antibiotic, and nearest at or before it.
    nearest_after = pd.merge_asof(
        abx, culture_keys, left_on="antibiotic_time", right_on="culture_time", by="stay_id", direction="forward"
    )["culture_time"]
    nearest_before = pd.merge_asof(
        abx, culture_keys, left_on="antibiotic_time", right_on="culture_time", by="stay_id", direction="backward"
    )["culture_time"]

    # Same arithmetic as the scan it replaces: the window test is applied to the
    # hour-scaled difference, so the boundary cases fall on the same side.
    diff_after_h = (nearest_after - abx["antibiotic_time"]) / HOUR
    diff_before_h = (nearest_before - abx["antibiotic_time"]) / HOUR
    match_after = (diff_after_h >= 0) & (diff_after_h <= abx_before_culture_h)
    match_before = (diff_before_h <= 0) & (diff_before_h >= -culture_before_abx_h)

    matched = abx[match_after | match_before].copy()
    matched["culture_time"] = np.where(match_after[matched.index], nearest_after[matched.index], nearest_before[matched.index])
    # Suspicion is dated by whichever event came first: the antibiotic when the
    # culture follows it, the culture when it precedes.
    matched["suspected_infection_time"] = np.where(
        match_after[matched.index], matched["antibiotic_time"], matched["culture_time"]
    )

    # `abx` is ordered by time, so the first surviving row per stay is the
    # earliest qualifying antibiotic -- where the original loop broke out.
    suspected = matched.groupby("stay_id", as_index=False, sort=True).first()

    # subject_id comes from the stay's earliest culture record, as before.
    subject_by_stay = (
        cultures.dropna(subset=["subject_id"]).groupby("stay_id", sort=False)["subject_id"].first()
    )
    subject_id = suspected["stay_id"].map(subject_by_stay)
    suspected.insert(0, "subject_id", subject_id.astype(np.int64) if subject_id.notna().all() else subject_id.astype(float))

    return suspected[
        ["subject_id", "stay_id", "suspected_infection_time", "antibiotic_time", "culture_time"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trailing-window aggregation helpers
# ---------------------------------------------------------------------------
def _trailing(ts: np.ndarray, vs: np.ndarray, grid_time: float, reducer) -> float:
    """
    This function extracts values (vs) within a trailing time window
    (SOFA_WORST_WINDOW) ending at grid_time (using their timestamps ts),
    and applies a reduction function (reducer, e.g., np.nanmin, np.nanmax)
    to summarize those values. Returns np.nan if the window is empty.
    Used to compute "worst" values over a recent time window for scoring
    in SOFA calculations.
    """
    if ts.size == 0:
        return np.nan
    lo = np.searchsorted(ts, grid_time - SOFA_WORST_WINDOW, side="right")
    hi = np.searchsorted(ts, grid_time, side="right")
    if hi <= lo:
        return np.nan
    return float(reducer(vs[lo:hi]))


def _trailing_sum_count(ts: np.ndarray, vs: np.ndarray, grid_time: float) -> tuple[float, int]:
    if ts.size == 0:
        return 0.0, 0
    lo = np.searchsorted(ts, grid_time - SOFA_WORST_WINDOW, side="right")
    hi = np.searchsorted(ts, grid_time, side="right")
    if hi <= lo:
        return 0.0, 0
    return float(np.nansum(vs[lo:hi])), int(hi - lo)


def _vaso_trailing_max(start: np.ndarray, end: np.ndarray, rate: np.ndarray, grid_time: float) -> float:
    if start.size == 0:
        return np.nan
    active = (start <= grid_time) & (end >= grid_time - SOFA_WORST_WINDOW)
    if not active.any():
        return np.nan
    return float(np.nanmax(rate[active]))


class StaySeries:
    """Per-stay component time series used for hourly SOFA."""

    __slots__ = (
        "map_t", "map_v", "spo2_t", "spo2_v", "fio2_t", "fio2_v",
        "pao2_t", "pao2_v", "plt_t", "plt_v", "bili_t", "bili_v",
        "creat_t", "creat_v", "gcs_t", "gcs_v", "vent_t", "vent_v",
        "uo_t", "uo_v", "vaso_s", "vaso_e", "vaso_r",
    )

    def __init__(self):
        empty = np.array([], dtype=float)
        for name in self.__slots__:
            setattr(self, name, empty)


def compute_sofa_onset(series: StaySeries, grid: np.ndarray, threshold: int):
    """Return (sofa_time, subscores, total) for the first hour SOFA >= threshold,
    or None if the threshold is never reached on the grid."""
    for grid_time in grid:
        pao2 = _trailing(series.pao2_t, series.pao2_v, grid_time, np.nanmin)
        fio2 = _trailing(series.fio2_t, series.fio2_v, grid_time, np.nanmax)
        spo2 = _trailing(series.spo2_t, series.spo2_v, grid_time, np.nanmin)
        vent = _trailing(series.vent_t, series.vent_v, grid_time, np.nanmax)
        vent_flag = (not _isnan(vent)) and vent >= 1

        plt_min = _trailing(series.plt_t, series.plt_v, grid_time, np.nanmin)
        bili_max = _trailing(series.bili_t, series.bili_v, grid_time, np.nanmax)
        map_min = _trailing(series.map_t, series.map_v, grid_time, np.nanmin)
        vaso_max = _vaso_trailing_max(series.vaso_s, series.vaso_e, series.vaso_r, grid_time)
        gcs_min = _trailing(series.gcs_t, series.gcs_v, grid_time, np.nanmin)
        creat_max = _trailing(series.creat_t, series.creat_v, grid_time, np.nanmax)
        uo_sum, uo_count = _trailing_sum_count(series.uo_t, series.uo_v, grid_time)

        resp = score_respiration(pao2, fio2, spo2, vent_flag)
        coag = score_coagulation(plt_min)
        liver = score_liver(bili_max)
        cardio = score_cardiovascular(map_min, vaso_max)
        cns = score_cns(gcs_min)
        renal = score_renal(creat_max, uo_sum, uo_count)
        total = resp + coag + liver + cardio + cns + renal

        if total >= threshold:
            subscores = {
                "sofa_respiration": resp,
                "sofa_coagulation": coag,
                "sofa_liver": liver,
                "sofa_cardiovascular": cardio,
                "sofa_cns": cns,
                "sofa_renal": renal,
            }
            return float(grid_time), subscores, int(total)
    return None


# ---------------------------------------------------------------------------
# Event loading (bounded to each stay's association window +/- trailing window)
# ---------------------------------------------------------------------------
def _filter_to_bounds(chunk: pd.DataFrame, bounds: pd.DataFrame, time_column: str) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["stay_id"] = to_int_stay(chunk["stay_id"])
    chunk = chunk.merge(bounds, on="stay_id", how="inner")
    chunk[time_column] = pd.to_numeric(chunk[time_column], errors="coerce")
    keep = (chunk[time_column] >= chunk["lo_time"]) & (chunk[time_column] <= chunk["hi_time"])
    return chunk[keep]


@timed("Load SOFA component events")
def load_component_events(
    search_dirs, mapping_file: str, bounds: pd.DataFrame, chunk_size: int
) -> dict[int, StaySeries]:
    chart_codes = load_concept_codes(mapping_file, CHARTEVENT_CONCEPTS)
    lab_codes = load_concept_codes(mapping_file, LAB_CONCEPTS)

    series: dict[int, StaySeries] = {int(s): StaySeries() for s in bounds["stay_id"].dropna().astype(int)}

    def assign(stay_groups, attr_t, attr_v):
        for stay_id, group in stay_groups:
            stay_id = int(stay_id)
            target = series.get(stay_id)
            if target is None:
                continue
            ordered = group.sort_values("t")
            setattr(target, attr_t, ordered["t"].to_numpy(dtype=float))
            setattr(target, attr_v, ordered["v"].to_numpy(dtype=float))

    # chartevents: map, spo2, fio2
    chart_path = resolve_input("chartevents.csv", search_dirs)
    wanted_chart = {code: concept for concept, codes in chart_codes.items() for code in codes}
    buffers = {concept: [] for concept in CHARTEVENT_CONCEPTS}
    for chunk in pd.read_csv(chart_path, sep="|", usecols=["stay_id", "charttime", "itemid", "valuenum"], chunksize=chunk_size):
        chunk = chunk[chunk["itemid"].isin(wanted_chart)]
        if chunk.empty:
            continue
        chunk = _filter_to_bounds(chunk, bounds, "charttime")
        if chunk.empty:
            continue
        chunk["concept"] = chunk["itemid"].map(wanted_chart)
        for concept in CHARTEVENT_CONCEPTS:
            part = chunk[chunk["concept"] == concept]
            if not part.empty:
                buffers[concept].append(part[["stay_id", "charttime", "valuenum"]])
    for concept, attr in (("map", ("map_t", "map_v")), ("spo2", ("spo2_t", "spo2_v")), ("fio2", ("fio2_t", "fio2_v"))):
        if buffers[concept]:
            frame = pd.concat(buffers[concept], ignore_index=True).rename(columns={"charttime": "t", "valuenum": "v"})
            frame = frame.dropna(subset=["v"])
            assign(frame.groupby("stay_id"), *attr)

    # labu: pao2, platelets, bilirubin, creatinine
    labu_path = resolve_input("labu.csv", search_dirs)
    wanted_lab = {code: concept for concept, codes in lab_codes.items() for code in codes}
    lab_buffers = {concept: [] for concept in LAB_CONCEPTS}
    for chunk in pd.read_csv(labu_path, sep="|", usecols=["stay_id", "charttime", "itemid", "valuenum"], chunksize=chunk_size):
        chunk = chunk[chunk["itemid"].isin(wanted_lab)]
        if chunk.empty:
            continue
        chunk = _filter_to_bounds(chunk, bounds, "charttime")
        if chunk.empty:
            continue
        chunk["concept"] = chunk["itemid"].map(wanted_lab)
        for concept in LAB_CONCEPTS:
            part = chunk[chunk["concept"] == concept]
            if not part.empty:
                lab_buffers[concept].append(part[["stay_id", "charttime", "valuenum"]])
    lab_attr = {
        "arterial_o2_pressure": ("pao2_t", "pao2_v"),
        "platelets": ("plt_t", "plt_v"),
        "bilirubin_total": ("bili_t", "bili_v"),
        "creatinine": ("creat_t", "creat_v"),
    }
    for concept, attr in lab_attr.items():
        if lab_buffers[concept]:
            frame = pd.concat(lab_buffers[concept], ignore_index=True).rename(columns={"charttime": "t", "valuenum": "v"})
            frame = frame.dropna(subset=["v"])
            assign(frame.groupby("stay_id"), *attr)

    # gcs
    gcs_path = _find_optional("gcs.csv", search_dirs)
    if gcs_path is not None:
        gcs = read_pipe_csv(gcs_path, usecols=["stay_id", "charttime", "gcs"])
        gcs = _filter_to_bounds(gcs.rename(columns={}), bounds, "charttime")
        gcs = gcs.rename(columns={"charttime": "t", "gcs": "v"}).dropna(subset=["v"])
        assign(gcs.groupby("stay_id"), "gcs_t", "gcs_v")
    else:
        print("WARNING: gcs.csv not found; the CNS (GCS) component will score 0 everywhere.")

    # mechvent
    mv_path = _find_optional("mechvent.csv", search_dirs)
    if mv_path is not None:
        mv = read_pipe_csv(mv_path, usecols=["stay_id", "charttime", "mechvent"])
        mv = _filter_to_bounds(mv, bounds, "charttime")
        mv = mv.rename(columns={"charttime": "t", "mechvent": "v"}).dropna(subset=["v"])
        assign(mv.groupby("stay_id"), "vent_t", "vent_v")

    # urine output
    uo_path = _find_optional("uo.csv", search_dirs)
    if uo_path is not None:
        uo = read_pipe_csv(uo_path, usecols=["stay_id", "charttime", "value"])
        uo = _filter_to_bounds(uo, bounds, "charttime")
        uo = uo.rename(columns={"charttime": "t", "value": "v"}).dropna(subset=["v"])
        uo = uo[uo["v"] >= 0]
        assign(uo.groupby("stay_id"), "uo_t", "uo_v")

    # vasopressors (interval events)
    vaso_path = _find_optional("vaso.csv", search_dirs)
    if vaso_path is not None:
        vaso = read_pipe_csv(vaso_path, usecols=["stay_id", "starttime", "endtime", "rate_std"])
        vaso["stay_id"] = to_int_stay(vaso["stay_id"])
        vaso = vaso.merge(bounds, on="stay_id", how="inner")
        vaso["starttime"] = pd.to_numeric(vaso["starttime"], errors="coerce")
        vaso["endtime"] = pd.to_numeric(vaso["endtime"], errors="coerce").fillna(vaso["starttime"])
        overlap = (vaso["starttime"] <= vaso["hi_time"]) & (vaso["endtime"] >= vaso["lo_time"])
        vaso = vaso[overlap].dropna(subset=["rate_std"])
        for stay_id, group in vaso.groupby("stay_id"):
            target = series.get(int(stay_id))
            if target is None:
                continue
            ordered = group.sort_values("starttime")
            target.vaso_s = ordered["starttime"].to_numpy(dtype=float)
            target.vaso_e = ordered["endtime"].to_numpy(dtype=float)
            target.vaso_r = ordered["rate_std"].to_numpy(dtype=float)

    return series


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------
def assign_splits(
    subject_ids: pd.Series,
    labels: pd.Series,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.Series:
    """Assign train/val/test by subject, stratified on the label.

    Grouping is by ``subject_id``, so every ICU stay of a patient lands in one
    split. Under the current cohort filters (first ICU stay per subject) that is
    a one-stay-per-subject mapping, so the grouping is trivially satisfied -- but
    it is enforced here anyway, because relaxing the first-stay filter would
    otherwise silently introduce patient leakage.

    Stratification is the change over the previous implementation, which drew one
    uniform random number per subject and bucketed it by the cumulative
    fractions. That left split prevalence to chance; the observed drift was small
    (max 0.69 pp off 50%), but it is free to remove and keeps the arms comparable
    for any seed or cohort size.

    Fractions are deliberately kept at the documented 70/15/15 rather than reusing
    ``experiments.data_utils.splits.make_grouped_splits()``, whose 5-fold
    construction would silently reshape them to roughly 64/16/20.
    """
    frame = pd.DataFrame({"subject_id": subject_ids, "label": labels})
    subject_label = (
        frame.dropna(subset=["subject_id"])
        .groupby("subject_id")["label"]
        # A subject with mixed labels (possible only if the first-stay filter is
        # relaxed) is stratified by its positive status, the scarcer class.
        .max()
    )

    rng = np.random.default_rng(seed)
    mapping: dict = {}
    for _, group in subject_label.groupby(subject_label.values):
        subjects = group.index.to_numpy()
        subjects = rng.permutation(subjects)
        n = len(subjects)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for subject in subjects[:n_train]:
            mapping[subject] = "train"
        for subject in subjects[n_train : n_train + n_val]:
            mapping[subject] = "val"
        for subject in subjects[n_train + n_val :]:
            mapping[subject] = "test"

    return subject_ids.map(mapping)


SUBSCORE_COLUMNS = [
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
]

OUTPUT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "suspected_infection_time",
    "antibiotic_time",
    "culture_time",
    "sofa_time",
    "onset_time",
    "prediction_time",
    "label",
    "sofa_total",
    *SUBSCORE_COLUMNS,
    "age",
    "los_hours",
    "adm_order",
    "is_adult",
    "is_first_icu_stay",
    "los_ge_24h",
    "prediction_in_icu",
    "split",
]


@timed("Build Sepsis-3 cohort")
def build_cohort(args: argparse.Namespace) -> pd.DataFrame:
    search_dirs = [args.onset_dir, args.extracted_dir]

    demog = read_pipe_csv(resolve_input("demog_processed.csv", search_dirs))
    demog["stay_id"] = to_int_stay(demog["stay_id"])
    demog = demog.dropna(subset=["stay_id"]).copy()
    demog["los_hours"] = pd.to_numeric(demog["los"], errors="coerce") * 24.0

    # Eligibility filters that gate the primary cohort.
    demog["is_adult"] = pd.to_numeric(demog["age"], errors="coerce") >= args.min_age
    demog["is_first_icu_stay"] = pd.to_numeric(demog["adm_order"], errors="coerce") == 1
    demog["los_ge_24h"] = demog["los_hours"] >= args.min_los_hours
    eligible = demog[demog["is_adult"] & demog["is_first_icu_stay"] & demog["los_ge_24h"]].copy()
    print(f"ICU stays in demographics:        {len(demog)}")
    print(f"Adult + first stay + LOS>=24h:    {len(eligible)}")

    abx = read_pipe_csv(resolve_input("abx_processed.csv", search_dirs))
    bacterio = read_pipe_csv(resolve_input("bacterio_processed.csv", search_dirs))
    suspected = find_suspected_infection(abx, bacterio, args.abx_before_culture_h, args.culture_before_abx_h)
    print(f"Stays with suspected infection:   {len(suspected)}")

    # Only score SOFA for eligible suspected-infection stays.
    eligible_keys = eligible[["subject_id", "stay_id", "hadm_id", "intime", "outtime", "age", "los_hours", "adm_order"]]
    suspected = suspected.merge(eligible_keys, on="stay_id", how="inner", suffixes=("", "_demog"))
    if "subject_id_demog" in suspected.columns:
        suspected["subject_id"] = suspected["subject_id"].fillna(suspected["subject_id_demog"])
        suspected = suspected.drop(columns=["subject_id_demog"])
    print(f"Eligible suspected-infection:     {len(suspected)}")

    # Bounds for event loading: 72h before suspicion (48h window + 24h trailing)
    # up to 24h after suspicion, clipped to ICU discharge.
    bounds = suspected[["stay_id", "suspected_infection_time", "intime", "outtime"]].copy()
    bounds["lo_time"] = bounds["suspected_infection_time"] - ASSOCIATION_BEFORE - SOFA_WORST_WINDOW
    bounds["hi_time"] = np.minimum(bounds["suspected_infection_time"] + ASSOCIATION_AFTER, bounds["outtime"])
    bounds = bounds[["stay_id", "lo_time", "hi_time"]]

    series = load_component_events(search_dirs, args.mapping_file, bounds, args.chunk_size)

    positives = []
    n_no_sofa = 0
    n_pred_outside = 0
    for row in suspected.itertuples(index=False):
        stay_id = int(row.stay_id)
        susp = float(row.suspected_infection_time)
        intime = float(row.intime)
        outtime = float(row.outtime)

        grid_start = max(susp - ASSOCIATION_BEFORE, intime)
        grid_end = min(susp + ASSOCIATION_AFTER, outtime)
        if grid_end < grid_start:
            n_no_sofa += 1
            continue
        grid = np.arange(grid_start, grid_end + 1.0, HOUR)
        stay_series = series.get(stay_id, StaySeries())

        result = compute_sofa_onset(stay_series, grid, args.sofa_threshold)
        if result is None:
            n_no_sofa += 1
            continue

        sofa_time, subscores, total = result
        onset_time = sofa_time
        prediction_time = onset_time - args.lead_time * HOUR
        prediction_in_icu = intime <= prediction_time <= outtime
        if not prediction_in_icu:
            n_pred_outside += 1
            continue

        positives.append(
            {
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "stay_id": stay_id,
                "suspected_infection_time": susp,
                "antibiotic_time": row.antibiotic_time,
                "culture_time": row.culture_time,
                "sofa_time": sofa_time,
                "onset_time": onset_time,
                "prediction_time": prediction_time,
                "label": 1,
                "sofa_total": total,
                **subscores,
                "age": row.age,
                "los_hours": row.los_hours,
                "adm_order": row.adm_order,
                "prediction_in_icu": True,
            }
        )

    positives_df = pd.DataFrame(positives)
    print(f"Suspected but SOFA<{args.sofa_threshold} in window: {n_no_sofa}")
    print(f"Dropped (prediction before ICU):  {n_pred_outside}")
    print(f"Sepsis-3 positives:               {len(positives_df)}")

    controls_df = build_controls(eligible, positives_df, args)
    print(f"Controls sampled:                 {len(controls_df)}")

    cohort = pd.concat([positives_df, controls_df], ignore_index=True)
    cohort = cohort.drop_duplicates(subset=["stay_id"])

    cohort["is_adult"] = True
    cohort["is_first_icu_stay"] = True
    cohort["los_ge_24h"] = True
    cohort["prediction_in_icu"] = True
    cohort["split"] = assign_splits(
        cohort["subject_id"], cohort["label"], args.train_frac, args.val_frac, args.seed
    )

    for column in OUTPUT_COLUMNS:
        if column not in cohort.columns:
            cohort[column] = np.nan

    cohort["label"] = cohort["label"].astype(int)
    cohort["stay_id"] = to_int_stay(cohort["stay_id"])
    return cohort[OUTPUT_COLUMNS]


def build_controls(
    eligible: pd.DataFrame, positives_df: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Sample controls whose prediction times are offset-matched to the positives.

    Every positive contributes its own offset ``o = prediction_time - intime``.
    A control is drawn without replacement from the eligible stays long enough
    to host that same offset plus a ``lead_time`` tail margin
    (``outtime - intime >= o + lead_time``), and is assigned exactly that offset.

    Two properties hold by construction:

    - ``hours_from_icu_intime_to_prediction`` has the same distribution in both
      arms, so where the observation window sits inside the ICU stay carries no
      information about the label.
    - Every control keeps at least ``lead_time`` hours of ICU stay after its
      prediction time, mirroring the guarantee positives get from
      ``prediction_time = onset_time - lead_time``. No prediction time is ever
      clipped to ``outtime``.

    The previous implementation sampled an offset independently of the control's
    length of stay and then clipped with ``min(pseudo_pred, outtime)``. That put
    1,499 of 4,639 controls (32.3%) exactly at their ICU discharge time, and left
    1,652 (35.6%) with under `lead_time` hours of stay remaining, against 0% of
    positives. Their observation window was therefore the pre-discharge period,
    while every positive's window preceded an acute deterioration -- which made
    time-since-ICU-admission the single most important feature for the tree
    models.

    Positives are processed from the longest offset down, so the tightest
    constraints are served first; every control feasible for a given positive is
    also feasible for all the weaker ones that follow, which makes this greedy
    pass sufficient whenever a complete matching exists at all. On the frozen
    MIMIC-IV v3.1 cohort it matches all 4,639 positives against a pool of 47,199
    eligible stays with none left over, so no offset cap is needed.
    """
    if positives_df.empty:
        return pd.DataFrame()

    positive_stays = set(positives_df["stay_id"].astype(int))
    pool = eligible[~eligible["stay_id"].astype(int).isin(positive_stays)].copy()
    pool = pool.dropna(subset=["intime", "outtime"])
    if pool.empty:
        return pd.DataFrame()

    pos_with_intime = positives_df.merge(eligible[["stay_id", "intime"]], on="stay_id", how="left")
    offsets = (pos_with_intime["prediction_time"] - pos_with_intime["intime"]).to_numpy(dtype=float)
    # `>= 0`, not `> 0`: an offset of exactly 0 is legitimate -- 390 positives in
    # this cohort have their onset 6h after ICU admission, so t0 lands on intime.
    # Dropping them left fewer offsets than controls, and the shortfall was
    # resampled from the strictly-positive tail, which pushed the control arm
    # about 5h later into the stay than the positive arm (SMD 0.068 instead of 0).
    all_positive_offsets = offsets[np.isfinite(offsets)]
    offsets = offsets[np.isfinite(offsets) & (offsets >= 0)]
    if offsets.size == 0:
        print("WARNING: no usable positive offsets; cannot build matched controls.")
        return pd.DataFrame()

    n_controls = int(min(len(pool), round(len(positives_df) * max(args.control_ratio, 0.0))))
    if n_controls <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(args.seed)
    if n_controls <= offsets.size:
        requested = rng.choice(offsets, size=n_controls, replace=False)
    else:
        extra = rng.choice(offsets, size=n_controls - offsets.size, replace=True)
        requested = np.concatenate([offsets, extra])

    # Pool sorted by ICU-stay capacity so the feasible candidates for a given
    # offset are always a contiguous suffix, found with a binary search.
    pool = pool.copy()
    pool["_capacity"] = pool["outtime"].to_numpy(dtype=float) - pool["intime"].to_numpy(dtype=float)
    pool = pool.sort_values("_capacity", kind="stable").reset_index(drop=True)
    capacities = pool["_capacity"].to_numpy(dtype=float)
    available = np.ones(len(pool), dtype=bool)

    lead_seconds = args.lead_time * HOUR
    picked_positions: list[int] = []
    picked_offsets: list[float] = []
    n_unmatched = 0

    for offset in np.sort(requested)[::-1]:  # tightest constraint first
        first_feasible = int(np.searchsorted(capacities, offset + lead_seconds, side="left"))
        candidates = np.flatnonzero(available[first_feasible:])
        if candidates.size == 0:
            n_unmatched += 1
            continue
        position = first_feasible + int(rng.choice(candidates))
        available[position] = False
        picked_positions.append(position)
        picked_offsets.append(float(offset))

    if not picked_positions:
        print("WARNING: no eligible stay can host any positive offset; no controls built.")
        return pd.DataFrame()

    chosen = pool.iloc[picked_positions].reset_index(drop=True)
    chosen_offsets = np.asarray(picked_offsets, dtype=float)
    pseudo_pred = chosen["intime"].to_numpy(dtype=float) + chosen_offsets

    _report_control_matching(
        chosen=chosen,
        chosen_offsets=chosen_offsets,
        pseudo_pred=pseudo_pred,
        # The full positive arm, not the filtered sampling pool: comparing the
        # pool against itself would report balance even when the filter had
        # silently dropped part of the arm.
        positive_offsets=all_positive_offsets,
        lead_seconds=lead_seconds,
        n_requested=n_controls,
        n_unmatched=n_unmatched,
    )

    controls = pd.DataFrame(
        {
            "subject_id": chosen["subject_id"].to_numpy(),
            "hadm_id": chosen["hadm_id"].to_numpy(),
            "stay_id": chosen["stay_id"].to_numpy(),
            "suspected_infection_time": np.nan,
            "antibiotic_time": np.nan,
            "culture_time": np.nan,
            "sofa_time": np.nan,
            # Controls have no onset. Leaving this NaN (rather than copying
            # prediction_time, as before) keeps any lead-time statistic computed
            # over the whole cohort from silently averaging a real 6 h lead with
            # a fabricated 0 h one.
            "onset_time": np.nan,
            "prediction_time": pseudo_pred,
            "label": 0,
            "sofa_total": np.nan,
            "age": chosen["age"].to_numpy(),
            "los_hours": chosen["los_hours"].to_numpy(),
            "adm_order": chosen["adm_order"].to_numpy(),
        }
    )
    for column in SUBSCORE_COLUMNS:
        controls[column] = np.nan
    return controls


def _report_control_matching(
    chosen: pd.DataFrame,
    chosen_offsets: np.ndarray,
    pseudo_pred: np.ndarray,
    positive_offsets: np.ndarray,
    lead_seconds: float,
    n_requested: int,
    n_unmatched: int,
) -> None:
    """Print the balance diagnostics that Stage 4 later enforces as hard gates."""
    outtime = chosen["outtime"].to_numpy(dtype=float)
    n_clipped = int(np.sum(pseudo_pred > outtime - lead_seconds + 1e-6))
    smd = standardized_mean_difference(positive_offsets / HOUR, chosen_offsets / HOUR)

    print(f"  controls requested:             {n_requested}")
    print(f"  controls matched:               {len(chosen)}")
    print(f"  positives with no feasible control: {n_unmatched}")
    print(f"  controls without the {lead_seconds / HOUR:.0f}h tail margin: {n_clipped} (must be 0)")
    print(f"  SMD of hours-from-ICU-admission: {smd:.4f} (must be < 0.1)")
    if n_unmatched:
        print(
            "  WARNING: some positives could not be matched. Consider capping the offset "
            "symmetrically in BOTH arms and documenting it as a cohort criterion."
        )
    if n_clipped:
        print("  WARNING: tail margin violated; control prediction times are not leakage-safe.")


def main() -> None:
    args = parse_args()
    cohort = build_cohort(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    cohort.to_csv(output_path, sep="|", index=False)

    n_pos = int((cohort["label"] == 1).sum())
    n_neg = int((cohort["label"] == 0).sum())
    print(f"\nWrote {output_path}")
    print(f"Positives: {n_pos} | Controls: {n_neg} | Total: {len(cohort)}")
    print("Split counts:")
    print(cohort.groupby(["split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
