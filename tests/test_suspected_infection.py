"""The suspected-infection rule of `3_build_sepsis3_labels.find_suspected_infection()`.

That function is the gate to the whole Sepsis-3 label: a stay it does not flag
never gets a SOFA trajectory and cannot become a positive. It was rewritten from
a per-stay Python scan into two `merge_asof` joins, verified bit-identical on the
real inputs (42 451 stays, all five columns); these tests pin the rule itself, on
hand-built cases, so the edges stay pinned independently of that one comparison.

The rule, per stay, walking the antibiotic administrations oldest-first and
stopping at the first that pairs:

* a culture drawn within `abx_before_culture_h` *after* the antibiotic ->
  suspicion is dated at the antibiotic;
* otherwise a culture drawn within `culture_before_abx_h` *before* it ->
  suspicion is dated at the culture.
"""

from __future__ import annotations

import importlib.util
import sys

import pandas as pd
import pytest


HOUR = 3600.0
ABX_BEFORE_CULTURE_H = 24.0
CULTURE_BEFORE_ABX_H = 72.0


@pytest.fixture(scope="module")
def labels_module():
    spec = importlib.util.spec_from_file_location(
        "stage3_labels", "multi_modality_code/3_build_sepsis3_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage3_labels"] = module
    spec.loader.exec_module(module)
    return module


def _run(labels_module, abx_hours, culture_hours, stay_id=1, subject_id=7):
    """Run the rule on one stay whose events are given in hours from an origin."""
    abx = pd.DataFrame({"stay_id": [stay_id] * len(abx_hours), "starttime": [h * HOUR for h in abx_hours]})
    bacterio = pd.DataFrame(
        {
            "stay_id": [stay_id] * len(culture_hours),
            "subject_id": [subject_id] * len(culture_hours),
            "charttime": [h * HOUR for h in culture_hours],
        }
    )
    return labels_module.find_suspected_infection(abx, bacterio, ABX_BEFORE_CULTURE_H, CULTURE_BEFORE_ABX_H)


def test_culture_after_antibiotic_dates_the_suspicion_at_the_antibiotic(labels_module):
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[110.0])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["antibiotic_time"] == 100.0 * HOUR
    assert row["culture_time"] == 110.0 * HOUR
    assert row["suspected_infection_time"] == 100.0 * HOUR
    assert row["subject_id"] == 7


def test_culture_before_antibiotic_dates_the_suspicion_at_the_culture(labels_module):
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[50.0])
    row = out.iloc[0]
    assert row["antibiotic_time"] == 100.0 * HOUR
    assert row["culture_time"] == 50.0 * HOUR
    assert row["suspected_infection_time"] == 50.0 * HOUR


def test_the_after_window_wins_when_both_sides_have_a_culture(labels_module):
    """A culture on each side: the rule looks forward first, so the antibiotic dates it."""
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[50.0, 110.0])
    row = out.iloc[0]
    assert row["culture_time"] == 110.0 * HOUR
    assert row["suspected_infection_time"] == 100.0 * HOUR


def test_the_nearest_culture_on_the_matching_side_is_the_one_recorded(labels_module):
    after = _run(labels_module, abx_hours=[100.0], culture_hours=[105.0, 118.0]).iloc[0]
    assert after["culture_time"] == 105.0 * HOUR
    before = _run(labels_module, abx_hours=[100.0], culture_hours=[40.0, 95.0]).iloc[0]
    assert before["culture_time"] == 95.0 * HOUR
    assert before["suspected_infection_time"] == 95.0 * HOUR


@pytest.mark.parametrize(
    "culture_hour, expected_match",
    [
        (100.0 + ABX_BEFORE_CULTURE_H, True),  # exactly on the forward boundary: inside
        (100.0 + ABX_BEFORE_CULTURE_H + 0.5, False),
        (100.0 - CULTURE_BEFORE_ABX_H, True),  # exactly on the backward boundary: inside
        (100.0 - CULTURE_BEFORE_ABX_H - 0.5, False),
    ],
)
def test_window_boundaries_are_inclusive(labels_module, culture_hour, expected_match):
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[culture_hour])
    assert (len(out) == 1) is expected_match


def test_a_culture_simultaneous_with_the_antibiotic_takes_the_after_branch(labels_module):
    """Zero difference satisfies both windows; the rule resolves it forwards."""
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[100.0]).iloc[0]
    assert out["suspected_infection_time"] == 100.0 * HOUR  # the antibiotic, not the culture


def test_the_earliest_qualifying_antibiotic_wins(labels_module):
    """Two antibiotics pair; the rule stops at the first, and input order is irrelevant."""
    ordered = _run(labels_module, abx_hours=[100.0, 200.0], culture_hours=[105.0, 205.0]).iloc[0]
    shuffled = _run(labels_module, abx_hours=[200.0, 100.0], culture_hours=[205.0, 105.0]).iloc[0]
    assert ordered["antibiotic_time"] == 100.0 * HOUR
    assert ordered["culture_time"] == 105.0 * HOUR
    assert shuffled.equals(ordered)


def test_an_antibiotic_that_does_not_pair_is_skipped_not_fatal(labels_module):
    """The first administration has no culture in range; the rule keeps walking."""
    out = _run(labels_module, abx_hours=[100.0], culture_hours=[500.0])
    assert out.empty
    out = _run(labels_module, abx_hours=[100.0, 480.0], culture_hours=[500.0]).iloc[0]
    assert out["antibiotic_time"] == 480.0 * HOUR
    assert out["culture_time"] == 500.0 * HOUR


def test_stays_are_independent_and_returned_sorted_by_stay_id(labels_module):
    abx = pd.DataFrame({"stay_id": [2, 1], "starttime": [300.0 * HOUR, 100.0 * HOUR]})
    bacterio = pd.DataFrame(
        {
            "stay_id": [2, 1],
            "subject_id": [20, 10],
            # Stay 1's culture would pair with stay 2's antibiotic if the stays leaked.
            "charttime": [310.0 * HOUR, 105.0 * HOUR],
        }
    )
    out = labels_module.find_suspected_infection(abx, bacterio, ABX_BEFORE_CULTURE_H, CULTURE_BEFORE_ABX_H)
    assert out["stay_id"].tolist() == [1, 2]
    assert out["subject_id"].tolist() == [10, 20]
    assert out["culture_time"].tolist() == [105.0 * HOUR, 310.0 * HOUR]


def test_a_stay_with_no_cultures_yields_nothing(labels_module):
    abx = pd.DataFrame({"stay_id": [1, 2], "starttime": [100.0 * HOUR, 100.0 * HOUR]})
    bacterio = pd.DataFrame({"stay_id": [2], "subject_id": [20], "charttime": [105.0 * HOUR]})
    out = labels_module.find_suspected_infection(abx, bacterio, ABX_BEFORE_CULTURE_H, CULTURE_BEFORE_ABX_H)
    assert out["stay_id"].tolist() == [2]


def test_rows_with_missing_timestamps_are_dropped_not_matched(labels_module):
    abx = pd.DataFrame({"stay_id": [1, 1], "starttime": [None, 100.0 * HOUR]})
    bacterio = pd.DataFrame({"stay_id": [1, 1], "subject_id": [10, 10], "charttime": [None, 105.0 * HOUR]})
    out = labels_module.find_suspected_infection(abx, bacterio, ABX_BEFORE_CULTURE_H, CULTURE_BEFORE_ABX_H)
    assert len(out) == 1
    assert out.iloc[0]["antibiotic_time"] == 100.0 * HOUR


def test_empty_inputs_return_the_expected_empty_schema(labels_module):
    empty_abx = pd.DataFrame({"stay_id": [], "starttime": []})
    bacterio = pd.DataFrame({"stay_id": [1], "subject_id": [10], "charttime": [100.0 * HOUR]})
    out = labels_module.find_suspected_infection(empty_abx, bacterio, ABX_BEFORE_CULTURE_H, CULTURE_BEFORE_ABX_H)
    assert out.empty
    assert list(out.columns) == [
        "subject_id",
        "stay_id",
        "suspected_infection_time",
        "antibiotic_time",
        "culture_time",
    ]
