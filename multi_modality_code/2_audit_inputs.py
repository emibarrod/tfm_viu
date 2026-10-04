"""
Audit the upstream files needed by the multimodal TFM pipeline.

This script does not modify data. It checks whether Stage 1 and Stage 2
outputs exist and whether the key columns needed by the multimodal dataset
builder are present.
"""

import argparse
import csv
import os
from typing import Iterable


STAGE1_FILES = {
    "abx.csv": ["subject_id", "hadm_id", "starttime", "stoptime"],
    "chartevents.csv": ["stay_id", "charttime", "itemid", "valuenum"],
    "culture.csv": ["subject_id", "hadm_id", "stay_id", "charttime", "itemid"],
    # `charlson_comorbidity_index` is required, not optional: Stage 4 exports the
    # static block by name, so a demog.csv predating the Charlson query would
    # silently produce a narrower feature matrix instead of failing.
    "demog.csv": ["subject_id", "hadm_id", "stay_id", "admittime", "intime", "charlson_comorbidity_index"],
    "fluid.csv": ["stay_id", "starttime", "endtime", "itemid", "amount", "rate", "tev"],
    "gcs.csv": ["stay_id", "charttime", "gcs"],
    "labs_ce.csv": ["stay_id", "charttime", "itemid", "valuenum"],
    "labs_le.csv": ["stay_id", "timestp", "itemid", "valuenum"],
    "mechvent.csv": ["stay_id", "charttime", "mechvent"],
    "microbio.csv": ["subject_id", "hadm_id", "charttime", "chartdate"],
    "uo.csv": ["stay_id", "charttime", "itemid", "value"],
    "vaso.csv": ["stay_id", "itemid", "starttime", "endtime", "rate_std"],
}

ONSET_FILES = {
    # onset.csv is a legacy artifact of Stage 1: nothing downstream reads it, since
    # Stage 2 recomputes the label into sepsis3_onset.csv. It stays in the contract
    # because a missing or truncated onset.csv is a reliable symptom of a Stage 1 run
    # that did not finish.
    "onset.csv": ["subject_id", "stay_id", "onset_time", "prediction_time", "label"],
    "abx_processed.csv": ["subject_id", "hadm_id", "stay_id", "starttime", "stoptime"],
    "bacterio_processed.csv": ["subject_id", "hadm_id", "stay_id", "charttime"],
    "demog_processed.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "intime",
        "re_admission",
        "charlson_comorbidity_index",
    ],
    "labu.csv": ["stay_id", "charttime", "itemid", "valuenum"],
}

LABEL_FILES = {
    "sepsis3_onset.csv": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "suspected_infection_time",
        "sofa_time",
        "onset_time",
        "prediction_time",
        "label",
        "sofa_total",
        "split",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit processed MIMIC-IV inputs for multimodal sepsis modeling.")
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Directory with raw extraction tables.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Directory with onset and processed helper files.")
    parser.add_argument("--labels_dir", default="data/03_labels", help="Directory with the Sepsis-3 label artifact.")
    return parser.parse_args()


def read_header(path: str) -> list[str]:
    with open(path, newline="") as file:
        reader = csv.reader(file, delimiter="|")
        return next(reader)


def missing_columns(actual: Iterable[str], expected: Iterable[str]) -> list[str]:
    actual_set = set(actual)
    return [column for column in expected if column not in actual_set]


def check_group(processed_dir: str, title: str, files: dict[str, list[str]]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    all_ok = True

    for filename, required_columns in files.items():
        path = os.path.join(processed_dir, filename)
        if not os.path.exists(path):
            print(f"[MISSING] {filename}")
            all_ok = False
            continue

        try:
            header = read_header(path)
        except Exception as exc:  # pragma: no cover - defensive CLI reporting
            print(f"[ERROR]   {filename}: could not read header ({exc})")
            all_ok = False
            continue

        missing = missing_columns(header, required_columns)
        if missing:
            print(f"[BAD]     {filename}: missing columns {missing}")
            all_ok = False
        else:
            print(f"[OK]      {filename}")

    return all_ok


def print_stage2_help() -> None:
    print("\nTo generate missing onset files, run:")
    print("uv run python multi_modality_code/1_preprocess_mimic.py \\")
    print("  --extracted_dir data/01_extracted \\")
    print("  --onset_dir data/02_onset \\")
    print("  --lead_time 6 \\")
    print("  --control_ratio 1.0 \\")
    print("  --skip_extraction")


def main() -> None:
    args = parse_args()

    print(f"Auditing extraction inputs in: {args.extracted_dir}")
    print(f"Auditing onset inputs in:      {args.onset_dir}")
    print(f"Auditing label inputs in:      {args.labels_dir}")
    for directory in (args.extracted_dir, args.onset_dir, args.labels_dir):
        if not os.path.isdir(directory):
            raise SystemExit(f"Directory not found: {directory}")

    stage1_ok = check_group(args.extracted_dir, "Stage 1 extraction outputs", STAGE1_FILES)
    onset_ok = check_group(args.onset_dir, "Stage 2a onset/cohort outputs", ONSET_FILES)
    labels_ok = check_group(args.labels_dir, "Stage 2b Sepsis-3 label artifact", LABEL_FILES)
    stage2_ok = onset_ok and labels_ok

    print("\nSummary")
    print("-------")
    print(f"Stage 1 ready: {'yes' if stage1_ok else 'no'}")
    print(f"Stage 2 ready: {'yes' if stage2_ok else 'no'}")

    if not stage2_ok:
        print_stage2_help()

    if not (stage1_ok and stage2_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
