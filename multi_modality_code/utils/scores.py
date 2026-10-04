"""Clinical severity scores: SOFA components and qSOFA.

These functions are shared by two callers that evaluate them at different
instants, which is the reason they live here rather than next to either one:

- `3_build_sepsis3_labels.py` walks an hourly grid inside the suspected-infection
  association window looking for the first hour with SOFA >= 2. That hour *is*
  the Sepsis-3 onset, so those scores define the label.
- `6_build_clinical_scores.py` evaluates them once per stay at the prediction
  time `t0 = onset - lead_time`, for both arms of the cohort, to give the
  experiments a clinical comparator ("does the ML model beat the bedside
  score?"). Nothing there sees data after `t0`.

Conventions and documented simplifications (see the Stage 2 module docstring and
SEPSIS3_TASK_DECISIONS.md): baseline SOFA is assumed 0, a missing component
scores 0, the cardiovascular tier uses a norepinephrine-equivalent rate instead
of drug-specific thresholds, and respiration falls back to SpO2/FiO2 when no
arterial blood gas is available.
"""

from __future__ import annotations

import numpy as np


def _isnan(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


# ---------------------------------------------------------------------------
# SOFA components (0-4 each, total 0-24)
# ---------------------------------------------------------------------------
def score_respiration(pao2: float, fio2_pct: float, spo2: float, vent: bool) -> int:
    if not _isnan(pao2) and not _isnan(fio2_pct) and fio2_pct > 0:
        ratio = pao2 / (fio2_pct / 100.0)
        if ratio < 100 and vent:
            return 4
        if ratio < 200 and vent:
            return 3
        if ratio < 300:
            return 2
        if ratio < 400:
            return 1
        return 0
    if not _isnan(spo2) and not _isnan(fio2_pct) and fio2_pct > 0:
        sf = min(spo2, 97.0) / (fio2_pct / 100.0)
        if sf <= 67:
            return 4
        if sf <= 142:
            return 3
        if sf <= 221:
            return 2
        if sf <= 302:
            return 1
        return 0
    return 0


def score_coagulation(platelets_min: float) -> int:
    if _isnan(platelets_min):
        return 0
    if platelets_min < 20:
        return 4
    if platelets_min < 50:
        return 3
    if platelets_min < 100:
        return 2
    if platelets_min < 150:
        return 1
    return 0


def score_liver(bilirubin_max: float) -> int:
    if _isnan(bilirubin_max):
        return 0
    if bilirubin_max >= 12.0:
        return 4
    if bilirubin_max >= 6.0:
        return 3
    if bilirubin_max >= 2.0:
        return 2
    if bilirubin_max >= 1.2:
        return 1
    return 0


def score_cardiovascular(map_min: float, vaso_rate_max: float) -> int:
    if not _isnan(vaso_rate_max) and vaso_rate_max > 0.1:
        return 4
    if not _isnan(vaso_rate_max) and vaso_rate_max > 0:
        return 3
    if not _isnan(map_min) and map_min < 70:
        return 1
    return 0


def score_cns(gcs_min: float) -> int:
    if _isnan(gcs_min):
        return 0
    if gcs_min < 6:
        return 4
    if gcs_min < 10:
        return 3
    if gcs_min < 13:
        return 2
    if gcs_min < 15:
        return 1
    return 0


def score_renal(creatinine_max: float, uo_sum: float, uo_count: int) -> int:
    creat_score = 0
    if not _isnan(creatinine_max):
        if creatinine_max >= 5.0:
            creat_score = 4
        elif creatinine_max >= 3.5:
            creat_score = 3
        elif creatinine_max >= 2.0:
            creat_score = 2
        elif creatinine_max >= 1.2:
            creat_score = 1

    uo_score = 0
    if uo_count > 0:
        if uo_sum < 200:
            uo_score = 4
        elif uo_sum < 500:
            uo_score = 3

    return max(creat_score, uo_score)


SOFA_COMPONENT_NAMES = (
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
)


# ---------------------------------------------------------------------------
# qSOFA (0-3): the bedside screen from Singer et al. (2016), one point each for
# respiratory rate >= 22/min, altered mentation (GCS < 15) and systolic blood
# pressure <= 100 mmHg. A score >= 2 is the published high-risk cut-off.
#
# A missing component scores 0, exactly as in the SOFA functions above: it makes
# the two scores consistent, and it is the conservative direction (a component
# that was never charted cannot raise the alarm). The cost is that qSOFA is
# systematically underestimated for stays with sparse charting, which is why the
# per-component counts are exported alongside the total.
# ---------------------------------------------------------------------------
QSOFA_RESPIRATORY_RATE_MIN = 22.0
QSOFA_GCS_MAX = 15.0
QSOFA_SBP_MAX = 100.0


def score_qsofa_respiratory(respiratory_rate: float) -> int:
    if _isnan(respiratory_rate):
        return 0
    return int(respiratory_rate >= QSOFA_RESPIRATORY_RATE_MIN)


def score_qsofa_mentation(gcs: float) -> int:
    if _isnan(gcs):
        return 0
    return int(gcs < QSOFA_GCS_MAX)


def score_qsofa_cardiovascular(sbp: float) -> int:
    if _isnan(sbp):
        return 0
    return int(sbp <= QSOFA_SBP_MAX)


def score_qsofa(respiratory_rate: float, gcs: float, sbp: float) -> tuple[int, int, int, int]:
    """Return (respiratory, mentation, cardiovascular, total) qSOFA points."""
    respiratory = score_qsofa_respiratory(respiratory_rate)
    mentation = score_qsofa_mentation(gcs)
    cardiovascular = score_qsofa_cardiovascular(sbp)
    return respiratory, mentation, cardiovascular, respiratory + mentation + cardiovascular


QSOFA_COMPONENT_NAMES = ("qsofa_respiratory", "qsofa_mentation", "qsofa_cardiovascular")
