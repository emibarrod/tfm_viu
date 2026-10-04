"""Control-construction guarantees on the real cohort export.

Controls have no onset, so their prediction time is imposed rather than
observed. `3_build_sepsis3_labels.build_controls()` assigns it by drawing an
offset from the positives and picking a control whose ICU stay can host it,
which is what keeps "where the observation window sits inside the stay" from
becoming a predictor of the label. These tests pin that contract down.

They fail against a cohort built before that fix, which is the intent: the old
sampler clipped 1,499 of 4,639 control prediction times (32.3%) to ICU
discharge and left 1,652 (35.6%) with less than the lead time of stay
remaining, against 0% of positives.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from multi_modality_code.utils.diagnostics import standardized_mean_difference

COHORT_PATH = os.path.join("data", "04_multimodal", "cohort.csv")
DEMOG_PATH = os.path.join("data", "02_onset", "demog_processed.csv")
HOUR = 3600.0
LEAD_TIME_HOURS = 6.0


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    for path in (COHORT_PATH, DEMOG_PATH):
        if not os.path.exists(path):
            pytest.skip(f"{path} not present (data/ is gitignored; run the Stage 1-4 pipeline first).")

    frame = pd.read_csv(COHORT_PATH)
    demog = pd.read_csv(DEMOG_PATH, sep="|", usecols=["stay_id", "intime", "outtime"])
    for table in (frame, demog):
        table["stay_id"] = pd.to_numeric(table["stay_id"], errors="coerce")
    merged = frame.merge(demog, on="stay_id", how="left", suffixes=("", "_demog"))
    for column in ("intime", "outtime"):
        if f"{column}_demog" in merged.columns:
            merged[column] = merged[column].fillna(merged[f"{column}_demog"])
    return merged.dropna(subset=["intime", "outtime"])


def test_no_control_prediction_time_clipped_to_discharge(cohort: pd.DataFrame) -> None:
    controls = cohort[cohort["label"] == 0]
    # Exact equality on purpose: these are Unix-epoch seconds (~2e9), where
    # np.isclose's default relative tolerance spans roughly 5.5 hours and would
    # count unclipped rows as clipped.
    at_discharge = int((controls["prediction_time"] == controls["outtime"]).sum())
    assert at_discharge == 0, (
        f"{at_discharge}/{len(controls)} controls have prediction_time exactly at ICU discharge; "
        "their 24h observation window is the pre-discharge period, not a comparable at-risk window."
    )


def test_both_arms_keep_a_lead_time_tail_margin(cohort: pd.DataFrame) -> None:
    margin_h = (cohort["outtime"] - cohort["prediction_time"]) / HOUR
    for label, arm in ((1, "positives"), (0, "controls")):
        short = int((margin_h[cohort["label"] == label] < LEAD_TIME_HOURS - 1e-6).sum())
        assert short == 0, f"{short} {arm} have less than {LEAD_TIME_HOURS:.0f}h of ICU stay after prediction_time"


def test_time_since_icu_admission_is_balanced_across_arms(cohort: pd.DataFrame) -> None:
    offsets = (cohort["prediction_time"] - cohort["intime"]) / HOUR
    smd = standardized_mean_difference(
        offsets[cohort["label"] == 1].to_numpy(dtype=float),
        offsets[cohort["label"] == 0].to_numpy(dtype=float),
    )
    assert np.isfinite(smd) and smd < 0.1, (
        f"hours-from-ICU-admission differs between arms (SMD={smd:.4f}); "
        "the position of the prediction window leaks the label."
    )


def test_controls_have_no_onset_time(cohort: pd.DataFrame) -> None:
    controls = cohort[cohort["label"] == 0]
    non_null = int(controls["onset_time"].notna().sum())
    assert non_null == 0, (
        f"{non_null} controls carry an onset_time; controls have no onset, and a fabricated one "
        "drags any cohort-wide lead-time statistic away from the real 6h."
    )
