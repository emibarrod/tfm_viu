"""Clinical-score regression tests (SOFA/qSOFA at t0).

Two layers. The scoring functions in `utils/scores.py` are pinned at their tier
boundaries, because they are shared by the stage that *defines* the Sepsis-3
label and the stage that computes the clinical comparator -- a silent drift there
would move both the labels and the baseline. Then the real Stage 6 export is
checked against its contract: one row per cohort stay, subscores that add up, and
totals inside the range each score can actually take.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from multi_modality_code.experiments.features.aggregate import (
    CLINICAL_SCORE_COLUMNS,
    build_clinical_scores_matrix,
)
from multi_modality_code.utils.scores import (
    score_cardiovascular,
    score_cns,
    score_coagulation,
    score_liver,
    score_qsofa,
    score_renal,
    score_respiration,
)

MULTIMODAL_DIR = os.path.join("data", "04_multimodal")
CLINICAL_PATH = os.path.join(MULTIMODAL_DIR, "clinical_scores.csv")
COHORT_PATH = os.path.join(MULTIMODAL_DIR, "cohort.csv")

SOFA_SUBSCORES = [
    "sofa_respiration_at_t0",
    "sofa_coagulation_at_t0",
    "sofa_liver_at_t0",
    "sofa_cardiovascular_at_t0",
    "sofa_cns_at_t0",
    "sofa_renal_at_t0",
]
QSOFA_SUBSCORES = ["qsofa_respiratory_at_t0", "qsofa_mentation_at_t0", "qsofa_cardiovascular_at_t0"]


def _require(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{path} not present; run 6_build_clinical_scores.py first.")


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def test_missing_components_score_zero() -> None:
    """A component that was never charted scores 0, in every organ system.

    This is the documented convention (MIT-LCP sepsis3.sql), and it is the reason
    a SOFA computed on a sparsely monitored stay is a lower bound rather than an
    unknown -- worth pinning, because flipping it to NaN would silently change
    every label in the cohort.
    """
    assert score_respiration(np.nan, np.nan, np.nan, False) == 0
    assert score_coagulation(np.nan) == 0
    assert score_liver(np.nan) == 0
    assert score_cardiovascular(np.nan, np.nan) == 0
    assert score_cns(np.nan) == 0
    assert score_renal(np.nan, 0.0, 0) == 0
    assert score_qsofa(np.nan, np.nan, np.nan) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "platelets, expected",
    [(200.0, 0), (149.9, 1), (99.9, 2), (49.9, 3), (19.9, 4), (150.0, 0), (100.0, 1)],
)
def test_coagulation_tiers(platelets: float, expected: int) -> None:
    assert score_coagulation(platelets) == expected


@pytest.mark.parametrize(
    "bilirubin, expected",
    [(1.0, 0), (1.2, 1), (2.0, 2), (6.0, 3), (12.0, 4)],
)
def test_liver_tiers(bilirubin: float, expected: int) -> None:
    assert score_liver(bilirubin) == expected


@pytest.mark.parametrize("gcs, expected", [(15.0, 0), (14.0, 1), (12.0, 2), (9.0, 3), (5.0, 4)])
def test_cns_tiers(gcs: float, expected: int) -> None:
    assert score_cns(gcs) == expected


def test_cardiovascular_prefers_vasopressor_over_map() -> None:
    """Any vasopressor outranks hypotension, and the rate sets the tier."""
    assert score_cardiovascular(50.0, np.nan) == 1  # MAP < 70, no drug
    assert score_cardiovascular(90.0, 0.05) == 3  # drug at a low rate
    assert score_cardiovascular(90.0, 0.5) == 4  # drug at a high rate
    assert score_cardiovascular(90.0, np.nan) == 0


def test_renal_takes_the_worse_of_creatinine_and_urine_output() -> None:
    assert score_renal(1.0, 1500.0, 12) == 0
    assert score_renal(3.6, 1500.0, 12) == 3  # creatinine drives it
    assert score_renal(1.0, 150.0, 12) == 4  # anuria drives it
    # No urine readings at all must not be scored as anuria.
    assert score_renal(1.0, 0.0, 0) == 0


def test_respiration_uses_spo2_fallback_and_ventilation() -> None:
    # PaO2/FiO2 = 90/0.5 = 180: tier 3 only while ventilated, else tier 2.
    assert score_respiration(90.0, 50.0, np.nan, True) == 3
    assert score_respiration(90.0, 50.0, np.nan, False) == 2
    # No blood gas: SpO2/FiO2 = min(99,97)/0.5 = 194 -> tier 2.
    assert score_respiration(np.nan, 50.0, 99.0, False) == 2


def test_qsofa_thresholds() -> None:
    """One point each for RR >= 22, GCS < 15 and SBP <= 100."""
    assert score_qsofa(22.0, 15.0, 101.0) == (1, 0, 0, 1)
    assert score_qsofa(21.9, 14.0, 101.0) == (0, 1, 0, 1)
    assert score_qsofa(21.9, 15.0, 100.0) == (0, 0, 1, 1)
    assert score_qsofa(30.0, 10.0, 80.0) == (1, 1, 1, 3)


# ---------------------------------------------------------------------------
# The real Stage 6 export
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clinical() -> pd.DataFrame:
    _require(CLINICAL_PATH)
    return pd.read_csv(CLINICAL_PATH)


def test_one_row_per_cohort_stay(clinical: pd.DataFrame) -> None:
    _require(COHORT_PATH)
    cohort = pd.read_csv(COHORT_PATH, usecols=["stay_id", "label"])
    assert not clinical["stay_id"].duplicated().any()
    assert set(clinical["stay_id"]) == set(cohort["stay_id"])
    merged = clinical.merge(cohort, on="stay_id", suffixes=("", "_cohort"))
    assert (merged["label"] == merged["label_cohort"]).all()


def test_subscores_sum_to_totals(clinical: pd.DataFrame) -> None:
    assert (clinical[SOFA_SUBSCORES].sum(axis=1) == clinical["sofa_total_at_t0"]).all()
    assert (clinical[QSOFA_SUBSCORES].sum(axis=1) == clinical["qsofa_at_t0"]).all()


def test_totals_within_range(clinical: pd.DataFrame) -> None:
    assert clinical[SOFA_SUBSCORES].to_numpy().min() >= 0
    assert clinical[SOFA_SUBSCORES].to_numpy().max() <= 4
    assert clinical["sofa_total_at_t0"].between(0, 24).all()
    assert clinical["qsofa_at_t0"].between(0, 3).all()
    assert clinical["qsofa_last_at_t0"].between(0, 3).all()
    assert clinical[["sofa_total_at_t0", "qsofa_at_t0"]].notna().all().all()


def test_design_matrix_exposes_only_the_two_totals(clinical: pd.DataFrame) -> None:
    """The `clinical_scores_only` ablation must be the two bedside scores, nothing else.

    If the component columns leaked into the design matrix the row would stop
    being a clinical comparator and become a third feature block -- and it would
    quietly beat the real bedside score.
    """
    matrix = build_clinical_scores_matrix(clinical)
    assert tuple(matrix.columns) == CLINICAL_SCORE_COLUMNS
    assert matrix.index.name == "stay_id"
    assert len(matrix) == len(clinical)
