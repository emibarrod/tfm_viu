"""Prediction-time leakage guards on the real Stage-4 exports (A10):
`hours_before_prediction >= 0` everywhere, and antibiotics only exist in the
explicitly-named sensitivity export, never in the primary one."""

from __future__ import annotations

import os

import pandas as pd
import pytest

MULTIMODAL_DIR = os.path.join("data", "04_multimodal")
ABX_DIR = os.path.join("data", "04_multimodal_abx")

LONG_MODALITY_FILES = ("vitals_timeseries.csv", "labs_timeseries.csv", "treatments_timeseries.csv")


def _require(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{path} not present (data/ is gitignored; run the Stage 1-4 pipeline first).")


@pytest.mark.parametrize("filename", LONG_MODALITY_FILES)
def test_no_future_leakage_in_primary_export(filename: str) -> None:
    path = os.path.join(MULTIMODAL_DIR, filename)
    _require(path)
    df = pd.read_csv(path, usecols=["hours_before_prediction"])
    hours = pd.to_numeric(df["hours_before_prediction"], errors="coerce").dropna()
    assert (hours >= -1e-6).all(), f"{filename}: found rows with hours_before_prediction < 0 (future leakage)."


@pytest.mark.parametrize("filename", LONG_MODALITY_FILES)
def test_no_future_leakage_in_antibiotics_sensitivity_export(filename: str) -> None:
    path = os.path.join(ABX_DIR, filename)
    _require(path)
    df = pd.read_csv(path, usecols=["hours_before_prediction"])
    hours = pd.to_numeric(df["hours_before_prediction"], errors="coerce").dropna()
    assert (hours >= -1e-6).all(), f"{filename}: found rows with hours_before_prediction < 0 (future leakage)."


def test_antibiotics_absent_from_primary_treatments_export() -> None:
    path = os.path.join(MULTIMODAL_DIR, "treatments_timeseries.csv")
    _require(path)
    df = pd.read_csv(path, usecols=["concept"])
    concepts = df["concept"].astype(str).str.lower()
    assert "antibiotic_active" not in set(concepts.unique()), (
        "antibiotic_active leaked into the primary (non-sensitivity) treatments export; "
        "antibiotics participate in the Sepsis-3 label and must stay isolated to data/04_multimodal_abx."
    )


def test_antibiotics_present_only_in_sensitivity_export() -> None:
    path = os.path.join(ABX_DIR, "treatments_timeseries.csv")
    _require(path)
    df = pd.read_csv(path, usecols=["concept"])
    concepts = set(df["concept"].astype(str).str.lower().unique())
    assert "antibiotic_active" in concepts, (
        "Expected antibiotic_active in the antibiotics-sensitivity export; "
        "if this legitimately changed, update this test alongside the pipeline."
    )
