"""
Build a leakage-safe multimodal dataset for early sepsis prediction.

The script reads the Stage 1/2 CSV outputs produced by
`multi_modality_code/preprocess_mimic.py` and writes simple, long-format
modality files.
"""

import argparse
import json
import os
from typing import Iterable

import numpy as np
import pandas as pd

from multi_modality_code.utils.pipeline_io import resolve_input


KEY_COLUMNS = ["stay_id", "subject_id", "label", "prediction_time", "onset_time"]

STATIC_FEATURE_COLUMNS = [
    "age",
    "gender",
    "unit",
    "adm_order",
    "re_admission",
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
    "hours_from_admission_to_prediction",
    "hours_from_icu_intime_to_prediction",
]

VITAL_CATEGORIES = {"vital", "respiratory", "neurological", "hemodynamic", "demographic"}
LAB_CATEGORIES = {"laboratory", "blood_gas"}

MEASUREMENT_OUTPUT_COLUMNS = [
    "stay_id",
    "subject_id",
    "label",
    "prediction_time",
    "onset_time",
    "charttime",
    "hours_before_prediction",
    "source",
    "itemid",
    "concept",
    "display_name",
    "category",
    "unit",
    "value",
]

TREATMENT_OUTPUT_COLUMNS = [
    "stay_id",
    "subject_id",
    "label",
    "prediction_time",
    "onset_time",
    "event_time",
    "hours_before_prediction",
    "source",
    "itemid",
    "concept",
    "value",
    "value_unit",
    "starttime",
    "endtime",
    "observed_starttime",
    "observed_endtime",
    "observed_duration_hours",
    "drug",
]

BASIC_VALUE_LIMITS = {
    "heart_rate": (0, 250),
    "sbp_arterial": (0, 300),
    "dbp_arterial": (0, 200),
    "map": (0, 200),
    "respiratory_rate": (0, 80),
    "spo2": (0, 100),
    "temp_C": (25, 45),
    "temp_F": (70, 115),
    "fio2": (20, 100),
    "peep": (0, 40),
    "potassium": (1, 15),
    "sodium": (95, 178),
    "chloride": (70, 150),
    "glucose": (1, 1000),
    "creatinine": (0, 150),
    "bilirubin_total": (0, 30),
    "hemoglobin": (0, 20),
    "hematocrit": (0, 65),
    "wbc": (0, 500),
    "platelets": (0, 2000),
    "inr": (0, 20),
    "arterial_o2_pressure": (0, 700),
    "arterial_co2_pressure": (0, 200),
    "lactic_acid": (0, 30),
    "lactate": (0, 30),
    "ph_arterial": (6.7, 8.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe multimodal sepsis prediction files.")
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Directory with raw extraction tables.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Directory with onset and processed helper files.")
    parser.add_argument("--labels_dir", default="data/03_labels", help="Directory with the frozen Sepsis-3 label artifact.")
    parser.add_argument(
        "--output_dir",
        default="data/04_multimodal",
        help="Directory where multimodal CSVs will be written.",
    )
    parser.add_argument(
        "--mapping_file",
        default="multi_modality_code/reference_files/measurement_mappings.json",
        help="JSON file mapping MIMIC itemids to clinical concepts.",
    )
    parser.add_argument(
        "--onset_file",
        default="sepsis3_onset.csv",
        help="Label artifact filename inside labels_dir (frozen Sepsis-3 labels).",
    )
    parser.add_argument("--lookback_hours", type=float, default=24.0, help="Hours before prediction_time to include.")
    parser.add_argument("--chunk_size", type=int, default=500_000, help="Rows per chunk for large event files.")
    parser.add_argument("--max_stays", type=int, default=None, help="Optional small cohort limit for testing.")
    parser.add_argument(
        "--skip_timeseries",
        action="store_true",
        help="Rewrite cohort.csv and static.csv only, leaving the three time-series exports "
        "untouched. The long-format files depend on the label artifact (prediction times and the "
        "lookback window) but not on demographics, so a change confined to the static block -- "
        "adding the Charlson columns, for instance -- does not require another pass over "
        "chartevents.",
    )
    parser.add_argument(
        "--include_antibiotics",
        action="store_true",
        help="Export antibiotic interval features. Off by default: antibiotics participate in the "
        "Sepsis-3 label and are kept only for a clearly named sensitivity/leakage analysis.",
    )
    return parser.parse_args()


def require_file(path: str, explanation: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {explanation}: {path}")


def prepare_output_file(path: str, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def append_rows(path: str, rows: pd.DataFrame, columns: list[str]) -> None:
    if rows.empty:
        return
    rows.reindex(columns=columns).to_csv(path, mode="a", header=False, index=False)


def read_pipe_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", **kwargs)


def load_onset(labels_dir: str, onset_file: str, lookback_hours: float, max_stays: int | None) -> pd.DataFrame:
    onset_path = os.path.join(labels_dir, onset_file)
    require_file(
        onset_path,
        "Sepsis-3 label artifact. Run multi_modality_code/build_sepsis3_labels.py before building the multimodal dataset",
    )

    onset = read_pipe_csv(onset_path)
    required = set(KEY_COLUMNS)
    missing = required - set(onset.columns)
    if missing:
        raise ValueError(f"{onset_file} is missing required columns: {sorted(missing)}")

    # Carry optional metadata columns through to the cohort export when present.
    optional_columns = [c for c in ("hadm_id", "split", "onset_time") if c in onset.columns]
    keep_columns = KEY_COLUMNS + [c for c in optional_columns if c not in KEY_COLUMNS]
    onset = onset[keep_columns].copy()
    onset = onset.dropna(subset=["stay_id", "prediction_time", "label"])
    onset["stay_id"] = pd.to_numeric(onset["stay_id"], errors="coerce").astype("Int64")
    onset["label"] = onset["label"].astype(int)
    onset["lower_bound"] = onset["prediction_time"] - lookback_hours * 3600

    if max_stays is not None:
        full_onset = onset.sort_values(["label", "stay_id"])
        onset = full_onset.groupby("label", group_keys=False).head(max_stays // 2)
        if len(onset) < max_stays:
            extra = max_stays - len(onset)
            used = set(onset["stay_id"])
            onset = pd.concat([onset, full_onset[~full_onset["stay_id"].isin(used)].head(extra)], ignore_index=True)

    return onset.drop_duplicates(subset=["stay_id"])


def load_demographics(search_dirs) -> pd.DataFrame:
    try:
        demog_path = resolve_input("demog_processed.csv", search_dirs)
    except FileNotFoundError:
        demog_path = resolve_input("demog.csv", search_dirs)
    return read_pipe_csv(demog_path)


def build_cohort_and_static(onset: pd.DataFrame, demog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    demog_cols = [column for column in demog.columns if column not in {"subject_id"}]
    merged = onset.merge(demog[["subject_id", *demog_cols]], on=["stay_id"], how="left", suffixes=("", "_demog"))
    if "subject_id_demog" in merged.columns:
        merged["subject_id"] = merged["subject_id"].fillna(merged["subject_id_demog"])

    for column in ["admittime", "intime"]:
        if column not in merged.columns:
            merged[column] = np.nan

    merged["hours_from_admission_to_prediction"] = (merged["prediction_time"] - merged["admittime"]) / 3600
    merged["hours_from_icu_intime_to_prediction"] = (merged["prediction_time"] - merged["intime"]) / 3600

    cohort_columns = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "label",
        "split",
        "onset_time",
        "prediction_time",
        "lower_bound",
        "admittime",
        "intime",
    ]
    cohort = merged[[column for column in cohort_columns if column in merged.columns]].copy()

    static_columns = KEY_COLUMNS + [column for column in STATIC_FEATURE_COLUMNS if column in merged.columns]
    static = merged[static_columns].copy()
    return cohort, static


def load_measurement_mapping(mapping_file: str) -> pd.DataFrame:
    require_file(mapping_file, "measurement mapping file")
    with open(mapping_file) as file:
        mapping = json.load(file)

    rows = []
    for concept, info in mapping.items():
        for code in info["codes"]:
            rows.append(
                {
                    "itemid_key": str(code),
                    "concept": concept,
                    "display_name": info.get("display_name", concept),
                    "category": info.get("category", ""),
                    "unit": info.get("unit", ""),
                }
            )
    return pd.DataFrame(rows)


def normalize_itemid(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def add_cohort_window(chunk: pd.DataFrame, cohort: pd.DataFrame, time_column: str) -> pd.DataFrame:
    merged = chunk.merge(
        cohort[["stay_id", "subject_id", "label", "prediction_time", "onset_time", "lower_bound"]],
        on="stay_id",
        how="inner",
    )
    merged[time_column] = pd.to_numeric(merged[time_column], errors="coerce")
    in_window = (merged[time_column] >= merged["lower_bound"]) & (merged[time_column] <= merged["prediction_time"])
    merged = merged[in_window].copy()
    merged["hours_before_prediction"] = (merged["prediction_time"] - merged[time_column]) / 3600
    return merged


def apply_basic_value_limits(df: pd.DataFrame) -> pd.DataFrame:
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for concept, (low, high) in BASIC_VALUE_LIMITS.items():
        mask = df["concept"] == concept
        if not mask.any():
            continue
        if low is not None:
            df.loc[mask & (df["value"] < low), "value"] = np.nan
        if high is not None:
            df.loc[mask & (df["value"] > high), "value"] = np.nan
    return df.dropna(subset=["value"])


def stream_measurements(
    input_path: str,
    output_path: str,
    cohort: pd.DataFrame,
    mapping: pd.DataFrame,
    categories: set[str],
    source: str,
    time_column: str,
    chunk_size: int,
) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return 0

    rows_written = 0
    wanted_mapping = mapping[mapping["category"].isin(categories)].copy()
    wanted_codes = set(wanted_mapping["itemid_key"])

    usecols = ["stay_id", time_column, "itemid", "valuenum"]
    for chunk in pd.read_csv(input_path, sep="|", usecols=usecols, chunksize=chunk_size):
        chunk["itemid_key"] = normalize_itemid(chunk["itemid"])
        chunk = chunk[chunk["itemid_key"].isin(wanted_codes)]
        if chunk.empty:
            continue

        if time_column != "charttime":
            chunk = chunk.rename(columns={time_column: "charttime"})
            event_time_column = "charttime"
        else:
            event_time_column = time_column

        chunk = add_cohort_window(chunk, cohort, event_time_column)
        if chunk.empty:
            continue

        chunk = chunk.merge(wanted_mapping, on="itemid_key", how="left")
        chunk = chunk.rename(columns={"valuenum": "value"})
        chunk["source"] = source
        chunk = apply_basic_value_limits(chunk)
        append_rows(output_path, chunk, MEASUREMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)

    return rows_written


def stream_mechvent(input_path: str, output_path: str, cohort: pd.DataFrame, chunk_size: int) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return 0

    rows_written = 0
    for chunk in pd.read_csv(input_path, sep="|", chunksize=chunk_size):
        chunk = add_cohort_window(chunk, cohort, "charttime")
        if chunk.empty:
            continue
        chunk["source"] = "mechvent"
        chunk["itemid"] = ""
        chunk["concept"] = "mechvent"
        chunk["display_name"] = "Mechanical ventilation"
        chunk["category"] = "respiratory"
        chunk["unit"] = "binary"
        chunk["value"] = pd.to_numeric(chunk["mechvent"], errors="coerce")
        append_rows(output_path, chunk, MEASUREMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)
    return rows_written


def add_interval_window(chunk: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    merged = chunk.merge(
        cohort[["stay_id", "subject_id", "label", "prediction_time", "onset_time", "lower_bound"]],
        on="stay_id",
        how="inner",
    )
    merged["starttime"] = pd.to_numeric(merged["starttime"], errors="coerce")
    merged["endtime"] = pd.to_numeric(merged["endtime"], errors="coerce").fillna(merged["starttime"])

    overlaps = (merged["starttime"] <= merged["prediction_time"]) & (merged["endtime"] >= merged["lower_bound"])
    merged = merged[overlaps].copy()
    if merged.empty:
        return merged

    merged["observed_starttime"] = merged[["starttime", "lower_bound"]].max(axis=1)
    merged["observed_endtime"] = merged[["endtime", "prediction_time"]].min(axis=1)
    merged["observed_duration_hours"] = (merged["observed_endtime"] - merged["observed_starttime"]) / 3600
    merged["event_time"] = merged["observed_starttime"]
    merged["hours_before_prediction"] = (merged["prediction_time"] - merged["event_time"]) / 3600
    return merged[merged["observed_duration_hours"] >= 0].copy()


def stream_fluid(input_path: str, output_path: str, cohort: pd.DataFrame, chunk_size: int) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return 0

    rows_written = 0
    usecols = ["stay_id", "starttime", "endtime", "itemid", "amount", "rate", "tev"]
    for chunk in pd.read_csv(input_path, sep="|", usecols=usecols, chunksize=chunk_size):
        chunk = add_interval_window(chunk, cohort)
        if chunk.empty:
            continue

        original_duration = (chunk["endtime"] - chunk["starttime"]) / 3600
        fraction = chunk["observed_duration_hours"] / original_duration.replace(0, np.nan)
        fraction = fraction.clip(lower=0, upper=1).fillna(1)

        chunk["source"] = "fluid"
        chunk["concept"] = "iv_fluid"
        chunk["value"] = pd.to_numeric(chunk["amount"], errors="coerce") * fraction
        chunk["value_unit"] = "mL_observed"
        chunk["drug"] = ""
        append_rows(output_path, chunk, TREATMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)
    return rows_written


def stream_vasopressors(input_path: str, output_path: str, cohort: pd.DataFrame, chunk_size: int) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return 0

    rows_written = 0
    for chunk in pd.read_csv(input_path, sep="|", chunksize=chunk_size):
        chunk = add_interval_window(chunk, cohort)
        if chunk.empty:
            continue
        chunk["source"] = "vasopressor"
        chunk["concept"] = "vasopressor_rate"
        chunk["value"] = pd.to_numeric(chunk["rate_std"], errors="coerce")
        chunk["value_unit"] = "norepinephrine_equivalent_mcg_kg_min"
        chunk["drug"] = ""
        chunk = chunk.dropna(subset=["value"])
        append_rows(output_path, chunk, TREATMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)
    return rows_written


def stream_urine_output(input_path: str, output_path: str, cohort: pd.DataFrame, chunk_size: int) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return 0

    rows_written = 0
    for chunk in pd.read_csv(input_path, sep="|", chunksize=chunk_size):
        chunk = chunk.rename(columns={"charttime": "event_time"})
        chunk = add_cohort_window(chunk, cohort, "event_time")
        if chunk.empty:
            continue
        chunk["source"] = "urine_output"
        chunk["concept"] = "urine_output"
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk["value_unit"] = "mL"
        chunk["starttime"] = chunk["event_time"]
        chunk["endtime"] = chunk["event_time"]
        chunk["observed_starttime"] = chunk["event_time"]
        chunk["observed_endtime"] = chunk["event_time"]
        chunk["observed_duration_hours"] = 0.0
        chunk["drug"] = ""
        chunk = chunk.dropna(subset=["value"])
        append_rows(output_path, chunk, TREATMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)
    return rows_written


def stream_antibiotics(input_path: str, output_path: str, cohort: pd.DataFrame, chunk_size: int) -> int:
    if not os.path.exists(input_path):
        print(f"Skipping antibiotics because Stage 2 file is missing: {input_path}")
        return 0

    rows_written = 0
    usecols = ["subject_id", "hadm_id", "stay_id", "drug", "starttime", "stoptime"]
    for chunk in pd.read_csv(input_path, sep="|", usecols=usecols, chunksize=chunk_size):
        chunk = chunk.rename(columns={"stoptime": "endtime"})
        chunk = add_interval_window(chunk, cohort)
        if chunk.empty:
            continue
        chunk["source"] = "antibiotic"
        chunk["itemid"] = ""
        chunk["concept"] = "antibiotic_active"
        chunk["value"] = 1
        chunk["value_unit"] = "binary"
        append_rows(output_path, chunk, TREATMENT_OUTPUT_COLUMNS)
        rows_written += len(chunk)
    return rows_written


def build_dataset(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    search_dirs = [args.onset_dir, args.extracted_dir]

    onset = load_onset(args.labels_dir, args.onset_file, args.lookback_hours, args.max_stays)
    demog = load_demographics(search_dirs)
    cohort, static = build_cohort_and_static(onset, demog)
    mapping = load_measurement_mapping(args.mapping_file)

    cohort_path = os.path.join(args.output_dir, "cohort.csv")
    static_path = os.path.join(args.output_dir, "static.csv")
    vitals_path = os.path.join(args.output_dir, "vitals_timeseries.csv")
    labs_path = os.path.join(args.output_dir, "labs_timeseries.csv")
    treatments_path = os.path.join(args.output_dir, "treatments_timeseries.csv")

    cohort.to_csv(cohort_path, index=False)
    static.to_csv(static_path, index=False)
    print(f"Wrote cohort: {cohort_path} ({len(cohort)} stays)")
    print(f"Wrote static features: {static_path} ({len(static)} rows, {len(static.columns)} columns)")

    if args.skip_timeseries:
        # Deliberately before prepare_output_file: that call truncates each
        # time-series export to a header row, which would silently destroy them.
        for path in (vitals_path, labs_path, treatments_path):
            state = "kept" if os.path.exists(path) else "MISSING"
            print(f"Skipped time series, {state}: {path}")
        return

    prepare_output_file(vitals_path, MEASUREMENT_OUTPUT_COLUMNS)
    prepare_output_file(labs_path, MEASUREMENT_OUTPUT_COLUMNS)
    prepare_output_file(treatments_path, TREATMENT_OUTPUT_COLUMNS)

    def locate(filename: str) -> str:
        """Resolve an input file across the upstream dirs; fall back to the
        first search dir so the downstream streamers report it as missing."""
        try:
            return resolve_input(filename, search_dirs)
        except FileNotFoundError:
            return os.path.join(search_dirs[0], filename)

    chartevents = locate("chartevents.csv")
    mechvent = locate("mechvent.csv")
    labs_ce = locate("labs_ce.csv")
    labs_le = locate("labs_le.csv")
    labu = locate("labu.csv")
    fluid = locate("fluid.csv")
    vaso = locate("vaso.csv")
    uo = locate("uo.csv")
    abx_processed = locate("abx_processed.csv")

    vitals_rows = stream_measurements(
        chartevents, vitals_path, cohort, mapping, VITAL_CATEGORIES, "chartevents", "charttime", args.chunk_size
    )
    vitals_rows += stream_mechvent(mechvent, vitals_path, cohort, args.chunk_size)
    print(f"Wrote vitals/ICU monitoring events: {vitals_path} ({vitals_rows} rows)")

    if os.path.exists(labu):
        labs_rows = stream_measurements(labu, labs_path, cohort, mapping, LAB_CATEGORIES, "labu", "charttime", args.chunk_size)
    else:
        labs_rows = stream_measurements(
            labs_ce, labs_path, cohort, mapping, LAB_CATEGORIES, "labs_ce", "charttime", args.chunk_size
        )
        labs_rows += stream_measurements(
            labs_le, labs_path, cohort, mapping, LAB_CATEGORIES, "labs_le", "timestp", args.chunk_size
        )
    print(f"Wrote lab events: {labs_path} ({labs_rows} rows)")

    treatment_rows = stream_fluid(fluid, treatments_path, cohort, args.chunk_size)
    treatment_rows += stream_vasopressors(vaso, treatments_path, cohort, args.chunk_size)
    treatment_rows += stream_urine_output(uo, treatments_path, cohort, args.chunk_size)
    if args.include_antibiotics:
        treatment_rows += stream_antibiotics(abx_processed, treatments_path, cohort, args.chunk_size)
        print("WARNING: antibiotics included. This is a sensitivity/leakage analysis, not the primary feature set.")
    else:
        print("Antibiotics excluded from primary features (label-mechanism leakage). Use --include_antibiotics to add them.")
    print(f"Wrote treatment/intervention events: {treatments_path} ({treatment_rows} rows)")


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
