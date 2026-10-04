"""
Evaluate SOFA and qSOFA at the prediction time t0, for every stay in the cohort.

Pipeline position: Stage 6 (post-labelling; reads the Stage 4 export)
Inputs:  data/04_multimodal/{cohort,vitals_timeseries,labs_timeseries,treatments_timeseries}.csv
         data/01_extracted/gcs.csv   (the only SOFA/qSOFA input Stage 4 does not export)
Outputs: clinical_scores.csv in each --output_dirs

Why this exists
---------------
Until now SOFA only *defined* the label (Stage 2 searches the hourly grid for the
first SOFA >= 2 and calls that instant the Sepsis-3 onset), and qSOFA did not
appear anywhere. There was therefore no clinical comparator: no answer to the
reviewer's first question, "does the machine-learning model beat the bedside
score you could compute by hand?". This script produces the two scores as of t0,
which `experiments/run_experiments.py` then uses for the `clinical_scores_only`
ablation and `experiments/posthoc_analysis.py` for the fixed-cut-off rules
(SOFA >= 2, qSOFA >= 2).

Leakage
-------
Both scores are computed from the same 24 h window the models see: the Stage 4
export contains, by construction, only events with
`lower_bound <= charttime <= prediction_time`, so every value here predates t0.
That window is also exactly the trailing window Stage 2 uses to take the worst
value of each component (SOFA_WORST_WINDOW = 24 h), so aggregating a concept over
the whole exported window *is* the trailing-window aggregate at t0 -- no grid
walk needed. GCS is read from the raw Stage 1 extraction and windowed here, with
the same closed interval.

Two differences from the Stage 2 numbers, neither of them a defect:
- Stage 2 evaluates the components at `onset_time`; these are at
  `t0 = onset_time - lead_time`, 6 h earlier and therefore usually milder.
- Stage 4 applies `BASIC_VALUE_LIMITS` to the exported measurements, so
  physiologically impossible values (a FiO2 of 3 %, a creatinine of 900) are
  already gone here, while Stage 2 saw them raw.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from multi_modality_code.utils.pipeline_io import resolve_input
from multi_modality_code.utils.scores import (
    score_cardiovascular,
    score_cns,
    score_coagulation,
    score_liver,
    score_qsofa,
    score_renal,
    score_respiration,
)


HOUR = 3600.0

SOFA_SUBSCORE_COLUMNS = (
    "sofa_respiration_at_t0",
    "sofa_coagulation_at_t0",
    "sofa_liver_at_t0",
    "sofa_cardiovascular_at_t0",
    "sofa_cns_at_t0",
    "sofa_renal_at_t0",
)

QSOFA_SUBSCORE_COLUMNS = (
    "qsofa_respiratory_at_t0",
    "qsofa_mentation_at_t0",
    "qsofa_cardiovascular_at_t0",
)

# Component inputs kept in the export. They are not model features (only the two
# totals are, see `experiments/features/aggregate.CLINICAL_SCORE_COLUMNS`), but
# they make every score auditable row by row and let the memoir quantify how
# often each component was simply never charted.
COMPONENT_COLUMNS = (
    "pao2_min_24h",
    "fio2_max_24h",
    "spo2_min_24h",
    "mechvent_24h",
    "platelets_min_24h",
    "bilirubin_max_24h",
    "map_min_24h",
    "vasopressor_rate_max_24h",
    "gcs_min_24h",
    "creatinine_max_24h",
    "urine_output_sum_24h",
    "urine_output_count_24h",
    "respiratory_rate_max_24h",
    "sbp_min_24h",
    "respiratory_rate_last",
    "sbp_last",
    "gcs_last",
)

OUTPUT_COLUMNS = (
    "stay_id",
    "subject_id",
    "label",
    "prediction_time",
    *SOFA_SUBSCORE_COLUMNS,
    "sofa_total_at_t0",
    *QSOFA_SUBSCORE_COLUMNS,
    "qsofa_at_t0",
    "qsofa_last_at_t0",
    *COMPONENT_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SOFA and qSOFA at the prediction time t0.")
    parser.add_argument(
        "--multimodal_dir",
        default="data/04_multimodal",
        help="Stage 4 export used as the source of the 24 h windows. The antibiotics variant is "
        "never needed: no antibiotic enters SOFA or qSOFA.",
    )
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Directory holding gcs.csv.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Secondary search dir for gcs.csv.")
    parser.add_argument(
        "--output_dirs",
        nargs="+",
        default=["data/04_multimodal", "data/04_multimodal_abx"],
        help="Directories to write clinical_scores.csv into. Both Stage 4 exports get a copy so "
        "the experiment runner finds one next to whichever cohort it is loading.",
    )
    parser.add_argument("--lookback_hours", type=float, default=24.0, help="Window length used by Stage 4.")
    parser.add_argument("--chunk_size", type=int, default=1_000_000, help="Rows per chunk for gcs.csv.")
    return parser.parse_args()


def load_cohort(multimodal_dir: str) -> pd.DataFrame:
    cohort = pd.read_csv(os.path.join(multimodal_dir, "cohort.csv"))
    required = {"stay_id", "subject_id", "label", "prediction_time"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort.csv missing required columns: {sorted(missing)}")
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    return cohort.sort_values("stay_id").drop_duplicates("stay_id").reset_index(drop=True)


def load_long(multimodal_dir: str, filename: str, lookback_hours: float) -> pd.DataFrame:
    path = os.path.join(multimodal_dir, filename)
    frame = pd.read_csv(
        path,
        usecols=["stay_id", "concept", "value", "hours_before_prediction"],
        low_memory=False,
    )
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["hours_before_prediction"] = pd.to_numeric(frame["hours_before_prediction"], errors="coerce")
    frame = frame.dropna(subset=["stay_id", "value", "hours_before_prediction"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(np.int64)

    # The window guarantee this whole script rests on. A negative offset would
    # mean a post-t0 measurement leaked into the export; an offset beyond the
    # lookback would mean the aggregate is not the 24 h trailing window.
    if (frame["hours_before_prediction"] < -1e-6).any():
        raise ValueError(f"{path}: rows with hours_before_prediction < 0 (post-t0 leakage).")
    if (frame["hours_before_prediction"] > lookback_hours + 1e-6).any():
        raise ValueError(f"{path}: rows older than the {lookback_hours} h lookback window.")
    return frame


def worst(frame: pd.DataFrame, concept: str, how: str) -> pd.Series:
    """Reduce one concept over its whole exported window (= the 24 h trailing window at t0)."""
    subset = frame[frame["concept"] == concept]
    if subset.empty:
        return pd.Series(dtype=float, name=concept)
    return subset.groupby("stay_id")["value"].agg(how)


def last_observed(frame: pd.DataFrame, concept: str) -> pd.Series:
    """Value of the reading closest to t0, i.e. the smallest hours_before_prediction."""
    subset = frame[frame["concept"] == concept]
    if subset.empty:
        return pd.Series(dtype=float, name=concept)
    index = subset.groupby("stay_id")["hours_before_prediction"].idxmin()
    return subset.loc[index].set_index("stay_id")["value"]


def load_gcs_window(search_dirs, cohort: pd.DataFrame, lookback_hours: float, chunk_size: int) -> pd.DataFrame:
    """Per-stay GCS minimum and last value in [t0 - lookback, t0].

    GCS is the one input both scores need that Stage 4 does not export: it is not
    in `measurement_mappings.json` (Stage 1 assembles it from the three component
    itemids into `gcs.csv`), and the Stage 4 streamers only read the mapping plus
    mechvent. Windowing it here keeps that export untouched -- adding a GCS
    concept to `vitals_timeseries.csv` would silently change the vitals feature
    block and with it every published number.
    """
    path = resolve_input("gcs.csv", search_dirs)
    bounds = cohort[["stay_id", "prediction_time"]].copy()
    bounds["lower_bound"] = bounds["prediction_time"] - lookback_hours * HOUR

    minima: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep="|", usecols=["stay_id", "charttime", "gcs"], chunksize=chunk_size):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
        chunk = chunk.merge(bounds, on="stay_id", how="inner")
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_numeric(chunk["charttime"], errors="coerce")
        chunk["gcs"] = pd.to_numeric(chunk["gcs"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "gcs"])
        inside = (chunk["charttime"] >= chunk["lower_bound"]) & (chunk["charttime"] <= chunk["prediction_time"])
        chunk = chunk[inside]
        if chunk.empty:
            continue
        chunk["hours_before_prediction"] = (chunk["prediction_time"] - chunk["charttime"]) / HOUR
        minima.append(chunk[["stay_id", "gcs", "hours_before_prediction"]])

    if not minima:
        print("WARNING: no GCS readings fell inside any prediction window; the CNS component and the "
              "qSOFA mentation point will be 0 for every stay.")
        return pd.DataFrame(columns=["gcs_min_24h", "gcs_last"]).set_index(pd.Index([], name="stay_id", dtype=np.int64))

    gcs = pd.concat(minima, ignore_index=True)
    gcs_min = gcs.groupby("stay_id")["gcs"].min().rename("gcs_min_24h")
    last_index = gcs.groupby("stay_id")["hours_before_prediction"].idxmin()
    gcs_last = gcs.loc[last_index].set_index("stay_id")["gcs"].rename("gcs_last")
    return pd.concat([gcs_min, gcs_last], axis=1)


def build_components(
    cohort: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    treatments: pd.DataFrame,
    gcs: pd.DataFrame,
) -> pd.DataFrame:
    """One row per stay with every raw input the two scores need."""
    index = pd.Index(cohort["stay_id"].to_numpy(dtype=np.int64), name="stay_id")
    urine = treatments[treatments["concept"] == "urine_output"]

    columns = {
        "pao2_min_24h": worst(labs, "arterial_o2_pressure", "min"),
        "fio2_max_24h": worst(vitals, "fio2", "max"),
        "spo2_min_24h": worst(vitals, "spo2", "min"),
        "mechvent_24h": worst(vitals, "mechvent", "max"),
        "platelets_min_24h": worst(labs, "platelets", "min"),
        "bilirubin_max_24h": worst(labs, "bilirubin_total", "max"),
        "map_min_24h": worst(vitals, "map", "min"),
        "vasopressor_rate_max_24h": worst(treatments, "vasopressor_rate", "max"),
        "creatinine_max_24h": worst(labs, "creatinine", "max"),
        "urine_output_sum_24h": urine.groupby("stay_id")["value"].sum() if not urine.empty else pd.Series(dtype=float),
        "urine_output_count_24h": urine.groupby("stay_id")["value"].size() if not urine.empty else pd.Series(dtype=float),
        "respiratory_rate_max_24h": worst(vitals, "respiratory_rate", "max"),
        "sbp_min_24h": worst(vitals, "sbp_arterial", "min"),
        "respiratory_rate_last": last_observed(vitals, "respiratory_rate"),
        "sbp_last": last_observed(vitals, "sbp_arterial"),
    }
    frame = pd.DataFrame({name: series.reindex(index) for name, series in columns.items()}, index=index)
    frame["gcs_min_24h"] = gcs["gcs_min_24h"].reindex(index) if len(gcs) else np.nan
    frame["gcs_last"] = gcs["gcs_last"].reindex(index) if len(gcs) else np.nan
    # Urine output is a counted quantity: no rows in the window means 0 mL over 0
    # readings, which is what `score_renal` expects (`uo_count == 0` disables the
    # urine tier rather than scoring it as anuria).
    frame["urine_output_sum_24h"] = frame["urine_output_sum_24h"].fillna(0.0)
    frame["urine_output_count_24h"] = frame["urine_output_count_24h"].fillna(0).astype(np.int64)
    return frame


def score_rows(cohort: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay in cohort.itertuples(index=False):
        stay_id = int(stay.stay_id)
        component = components.loc[stay_id]

        respiration = score_respiration(
            float(component["pao2_min_24h"]),
            float(component["fio2_max_24h"]),
            float(component["spo2_min_24h"]),
            bool(component["mechvent_24h"] >= 1) if pd.notna(component["mechvent_24h"]) else False,
        )
        coagulation = score_coagulation(float(component["platelets_min_24h"]))
        liver = score_liver(float(component["bilirubin_max_24h"]))
        cardiovascular = score_cardiovascular(
            float(component["map_min_24h"]), float(component["vasopressor_rate_max_24h"])
        )
        cns = score_cns(float(component["gcs_min_24h"]))
        renal = score_renal(
            float(component["creatinine_max_24h"]),
            float(component["urine_output_sum_24h"]),
            int(component["urine_output_count_24h"]),
        )
        qsofa_respiratory, qsofa_mentation, qsofa_cardiovascular, qsofa_total = score_qsofa(
            float(component["respiratory_rate_max_24h"]),
            float(component["gcs_min_24h"]),
            float(component["sbp_min_24h"]),
        )
        _, _, _, qsofa_last = score_qsofa(
            float(component["respiratory_rate_last"]),
            float(component["gcs_last"]),
            float(component["sbp_last"]),
        )

        rows.append(
            {
                "stay_id": stay_id,
                "subject_id": stay.subject_id,
                "label": int(stay.label),
                "prediction_time": float(stay.prediction_time),
                "sofa_respiration_at_t0": respiration,
                "sofa_coagulation_at_t0": coagulation,
                "sofa_liver_at_t0": liver,
                "sofa_cardiovascular_at_t0": cardiovascular,
                "sofa_cns_at_t0": cns,
                "sofa_renal_at_t0": renal,
                "sofa_total_at_t0": respiration + coagulation + liver + cardiovascular + cns + renal,
                "qsofa_respiratory_at_t0": qsofa_respiratory,
                "qsofa_mentation_at_t0": qsofa_mentation,
                "qsofa_cardiovascular_at_t0": qsofa_cardiovascular,
                "qsofa_at_t0": qsofa_total,
                "qsofa_last_at_t0": qsofa_last,
                **{name: component[name] for name in COMPONENT_COLUMNS},
            }
        )
    return pd.DataFrame(rows)[list(OUTPUT_COLUMNS)]


def report(scores: pd.DataFrame, components: pd.DataFrame) -> None:
    positives = scores[scores["label"] == 1]
    controls = scores[scores["label"] == 0]
    print(f"\nStays scored: {len(scores)} ({len(positives)} positives, {len(controls)} controls)")

    print("\nScore distribution at t0 (mean [median] by arm):")
    for column in ("sofa_total_at_t0", "qsofa_at_t0", "qsofa_last_at_t0"):
        print(
            f"  {column:20s} positives {positives[column].mean():.2f} [{positives[column].median():.0f}]"
            f"   controls {controls[column].mean():.2f} [{controls[column].median():.0f}]"
        )

    print("\nDiscrimination of the raw score over the whole cohort (AUROC):")
    for column in ("sofa_total_at_t0", "qsofa_at_t0", "qsofa_last_at_t0"):
        print(f"  {column:20s} {roc_auc_score(scores['label'], scores[column]):.4f}")

    print("\nFraction of stays with the component never charted in the 24 h window:")
    for column in COMPONENT_COLUMNS:
        if column in ("urine_output_sum_24h", "urine_output_count_24h"):
            continue
        print(f"  {column:26s} {components[column].isna().mean():.3f}")
    print(f"  {'urine_output (no readings)':26s} {(components['urine_output_count_24h'] == 0).mean():.3f}")

    print("\nqSOFA components positive (worst-in-window variant):")
    for column in QSOFA_SUBSCORE_COLUMNS:
        print(f"  {column:28s} {scores[column].mean():.3f}")


def main() -> None:
    args = parse_args()
    cohort = load_cohort(args.multimodal_dir)
    print(f"Cohort: {len(cohort)} stays from {args.multimodal_dir}")

    vitals = load_long(args.multimodal_dir, "vitals_timeseries.csv", args.lookback_hours)
    labs = load_long(args.multimodal_dir, "labs_timeseries.csv", args.lookback_hours)
    treatments = load_long(args.multimodal_dir, "treatments_timeseries.csv", args.lookback_hours)
    gcs = load_gcs_window([args.extracted_dir, args.onset_dir], cohort, args.lookback_hours, args.chunk_size)
    print(f"GCS available for {len(gcs)} of {len(cohort)} stays")

    components = build_components(cohort, vitals, labs, treatments, gcs)
    scores = score_rows(cohort, components)
    report(scores, components)

    for output_dir in args.output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clinical_scores.csv")
        scores.to_csv(output_path, index=False)
        print(f"\nWrote {output_path} ({len(scores)} rows, {len(scores.columns)} columns)")


if __name__ == "__main__":
    main()
