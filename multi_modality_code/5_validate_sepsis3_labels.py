"""
Validation gates for the frozen Sepsis-3 label artifact.

This script is read-only. It checks the timing, SOFA ranges, identifier joins,
and split integrity described in SEPSIS3_TASK_DECISIONS.md. When a multimodal
export is present, it also checks the leakage gates on the exported features.

Exit code is non-zero if any gate fails.
"""

import argparse
import os

import numpy as np
import pandas as pd

from multi_modality_code.utils.diagnostics import standardized_mean_difference


HOUR = 3600.0
SUBSCORES = [
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Sepsis-3 label artifact.")
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Directory with raw extraction tables.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Directory with onset and processed helper files.")
    parser.add_argument("--labels_dir", default="data/03_labels", help="Directory with the Sepsis-3 label artifact.")
    parser.add_argument("--label_file", default="sepsis3_onset.csv")
    parser.add_argument(
        "--lead_time",
        type=float,
        default=6.0,
        help="Lead time to re-check the labels against. Must match the value used in "
        "stage 2; this stage only verifies the invariant, it does not set it.",
    )
    parser.add_argument("--association_before_h", type=float, default=48.0)
    parser.add_argument("--association_after_h", type=float, default=24.0)
    parser.add_argument(
        "--multimodal_dir",
        default="data/04_multimodal",
        help="Optional multimodal export directory to check leakage gates.",
    )
    return parser.parse_args()


class Gate:
    def __init__(self):
        self.failures = 0

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[{status}] {name}{suffix}")
        if not passed:
            self.failures += 1


def validate_labels(args: argparse.Namespace, gate: Gate) -> None:
    label_path = os.path.join(args.labels_dir, args.label_file)
    cohort = pd.read_csv(label_path, sep="|")
    demog = pd.read_csv(os.path.join(args.onset_dir, "demog_processed.csv"), sep="|")
    demog["stay_id"] = pd.to_numeric(demog["stay_id"], errors="coerce").astype("Int64")
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    merged = cohort.merge(demog[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
    positives = merged[merged["label"] == 1]

    print(f"\nLabel artifact: {label_path}")
    print(f"Rows: {len(cohort)} | positives: {int((cohort['label'] == 1).sum())} | controls: {int((cohort['label'] == 0).sum())}")
    print("-" * 60)

    gate.check("No duplicated stay_id in final cohort", cohort["stay_id"].duplicated().sum() == 0,
               f"{int(cohort['stay_id'].duplicated().sum())} duplicates")

    diff = (positives["onset_time"] - positives["prediction_time"]) / HOUR
    gate.check("prediction_time == onset_time - lead_time",
               bool(np.allclose(diff, args.lead_time)), f"lead range [{diff.min():.3f}, {diff.max():.3f}] h")

    inside = (positives["prediction_time"] >= positives["intime"]) & (positives["prediction_time"] <= positives["outtime"])
    gate.check("prediction_time inside ICU stay (positives)", bool(inside.all()), f"{int((~inside).sum())} violations")

    in_window = (
        (positives["sofa_time"] >= positives["suspected_infection_time"] - args.association_before_h * HOUR)
        & (positives["sofa_time"] <= positives["suspected_infection_time"] + args.association_after_h * HOUR)
    )
    gate.check("SOFA association window (-48h/+24h)", bool(in_window.all()), f"{int((~in_window).sum())} violations")

    subs_ok = bool(((positives[SUBSCORES] >= 0) & (positives[SUBSCORES] <= 4)).all().all())
    gate.check("SOFA subscores within [0, 4]", subs_ok)

    total_ok = bool((positives["sofa_total"] >= 2).all() and (positives["sofa_total"] <= 24).all())
    gate.check("SOFA total within [2, 24] for positives", total_ok,
               f"[{positives['sofa_total'].min():.0f}, {positives['sofa_total'].max():.0f}]")

    sum_ok = bool((positives[SUBSCORES].sum(axis=1) == positives["sofa_total"]).all())
    gate.check("SOFA subscores sum to total", sum_ok)

    overlap = cohort.groupby("subject_id")["split"].nunique()
    gate.check("Train/val/test grouped by subject", bool((overlap == 1).all()), f"{int((overlap > 1).sum())} subjects in >1 split")

    # GCS validity from the source extraction (if present).
    gcs_path = os.path.join(args.extracted_dir, "gcs.csv")
    if os.path.exists(gcs_path):
        gcs = pd.read_csv(gcs_path, sep="|", usecols=["gcs"])
        gate.check("GCS values within [3, 15]", bool((gcs["gcs"] >= 3).all() and (gcs["gcs"] <= 15).all()),
                   f"[{gcs['gcs'].min():.0f}, {gcs['gcs'].max():.0f}]")

    validate_control_matching(merged, gate, args.lead_time)


def validate_control_matching(merged: pd.DataFrame, gate: Gate, lead_time: float) -> None:
    """Gates on how control prediction times were assigned.

    Controls have no onset, so their prediction time is imposed. If that
    assignment leaves them systematically later in the ICU stay, or bunched
    against discharge, then where the observation window sits becomes a
    predictor of the label -- a design artifact rather than physiology. These
    gates enforce the offset-matching contract in `3_build_sepsis3_labels.py`.
    """
    controls = merged[merged["label"] == 0]
    positives = merged[merged["label"] == 1]
    if controls.empty or positives.empty:
        gate.check("Control matching (needs both arms)", False, "one arm is empty")
        return

    # Exact equality, not np.isclose: on Unix-epoch seconds (~2e9) the default
    # relative tolerance spans about 5.5 hours and would flag unclipped rows.
    n_at_discharge = int((controls["prediction_time"] == controls["outtime"]).sum())
    gate.check(
        "No control prediction_time clipped to ICU discharge",
        n_at_discharge == 0,
        f"{n_at_discharge}/{len(controls)} controls sit exactly at outtime",
    )

    margin_h = (merged["outtime"] - merged["prediction_time"]) / HOUR
    short_controls = int((margin_h[merged["label"] == 0] < lead_time - 1e-6).sum())
    gate.check(
        f"Controls keep a {lead_time:.0f}h tail margin after prediction_time",
        short_controls == 0,
        f"{short_controls} controls with less than {lead_time:.0f}h of stay left",
    )

    pos_offset = ((positives["prediction_time"] - positives["intime"]) / HOUR).to_numpy(dtype=float)
    ctl_offset = ((controls["prediction_time"] - controls["intime"]) / HOUR).to_numpy(dtype=float)
    smd = standardized_mean_difference(pos_offset, ctl_offset)
    gate.check(
        "Hours-from-ICU-admission balanced across arms (SMD < 0.1)",
        bool(np.isfinite(smd) and smd < 0.1),
        f"SMD={smd:.4f}",
    )

    gate.check(
        "Controls carry no onset_time",
        bool(controls["onset_time"].isna().all()),
        f"{int(controls['onset_time'].notna().sum())} controls have a non-null onset_time",
    )


def validate_multimodal(args: argparse.Namespace, gate: Gate) -> None:
    mm_dir = args.multimodal_dir
    cohort_path = os.path.join(mm_dir, "cohort.csv")
    if not os.path.exists(cohort_path):
        print(f"\nNo multimodal export at {mm_dir}; skipping feature leakage gates.")
        return

    print(f"\nMultimodal export: {mm_dir}")
    print("-" * 60)

    treatments_path = os.path.join(mm_dir, "treatments_timeseries.csv")
    if os.path.exists(treatments_path):
        sources = set()
        leak = 0
        for chunk in pd.read_csv(treatments_path, chunksize=500_000):
            if "source" in chunk.columns:
                sources.update(chunk["source"].dropna().unique())
            after = chunk["event_time"] > chunk["prediction_time"]
            leak += int(after.sum())
        gate.check("Treatment events satisfy event_time <= prediction_time", leak == 0, f"{leak} leaking rows")
        gate.check("No antibiotic features in primary export", "antibiotic" not in sources, f"sources={sorted(sources)}")

    for name in ("vitals_timeseries.csv", "labs_timeseries.csv"):
        path = os.path.join(mm_dir, name)
        if not os.path.exists(path):
            continue
        leak = 0
        for chunk in pd.read_csv(path, chunksize=500_000):
            after = chunk["charttime"] > chunk["prediction_time"]
            leak += int(after.sum())
        gate.check(f"{name}: charttime <= prediction_time", leak == 0, f"{leak} leaking rows")


def main() -> None:
    args = parse_args()
    gate = Gate()
    validate_labels(args, gate)
    validate_multimodal(args, gate)

    print("\n" + "=" * 60)
    if gate.failures == 0:
        print("All validation gates passed.")
    else:
        print(f"{gate.failures} validation gate(s) FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
