"""Unit tests for the DL arm's permutation importance (`dl/importance.py`).

These run on a synthetic bundle, not on `data/`, so they stay fast and are
about the mechanics that are easy to get silently wrong: which columns a
feature unit maps to, and whether the in-place permutations leave the caller's
tensors as they found them.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from multi_modality_code.experiments.dl.datasets import AblationGridDataset
from multi_modality_code.experiments.dl.grid_data import GridBundle, ModalityGrid
from multi_modality_code.experiments.dl.importance import _feature_slices, permutation_importance_dl
from multi_modality_code.experiments.dl.models import FusionClassifier

N_STAYS = 40
N_BINS = 24
STATIC_COLUMNS = ("age", "diagnosis_count", "gender_1")
VITALS = ("heart_rate", "lactate", "spo2")
LABS = ("wbc", "platelets")
TREATMENTS = ("iv_fluid",)
MODALITIES = ("static", "vitals", "labs", "treatments")


def _grid(concepts: tuple[str, ...], rng: np.random.RandomState) -> ModalityGrid:
    shape = (N_STAYS, N_BINS, len(concepts))
    return ModalityGrid(
        concepts=concepts,
        value=rng.normal(size=shape).astype(np.float32),
        mask=rng.randint(0, 2, size=shape).astype(np.float32),
        delta=rng.uniform(0, N_BINS, size=shape).astype(np.float32),
        norm_stats={},
    )


def _bundle() -> GridBundle:
    rng = np.random.RandomState(0)
    return GridBundle(
        stay_ids=np.arange(N_STAYS, dtype=np.int64),
        labels=rng.randint(0, 2, size=N_STAYS).astype(np.float32),
        static=rng.normal(size=(N_STAYS, len(STATIC_COLUMNS))).astype(np.float32),
        static_columns=STATIC_COLUMNS,
        vitals=_grid(VITALS, rng),
        labs=_grid(LABS, rng),
        treatments=_grid(TREATMENTS, rng),
    )


def _model(bundle: GridBundle) -> FusionClassifier:
    torch.manual_seed(0)
    return FusionClassifier(
        modalities=MODALITIES,
        input_dims={name: bundle.branch_input_dim(name) for name in MODALITIES},
    )


def test_feature_slices_tile_each_branch_input_exactly() -> None:
    """Every input column must belong to exactly one feature unit, in order.

    A slice that is off by one would not raise anything -- it would just
    attribute one concept's importance to its neighbour, which is precisely the
    kind of error the resulting table cannot be checked against.
    """
    bundle = _bundle()
    units = _feature_slices(bundle, MODALITIES)

    for modality in MODALITIES:
        covered = [columns for name, _, columns in units if name == modality]
        starts = [columns.start for columns in covered]
        stops = [columns.stop for columns in covered]
        assert starts == sorted(starts), f"{modality}: slices out of order"
        assert starts[0] == 0
        assert stops[-1] == bundle.branch_input_dim(modality), f"{modality}: slices do not reach the end"
        assert starts[1:] == stops[:-1], f"{modality}: slices overlap or leave a gap"

    names = [(modality, feature) for modality, feature, _ in units]
    assert names[: len(STATIC_COLUMNS)] == [("static", column) for column in STATIC_COLUMNS]
    assert ("vitals", "lactate__seq") in names
    assert len(units) == len(STATIC_COLUMNS) + len(VITALS) + len(LABS) + len(TREATMENTS)


def test_sequence_slice_selects_that_concepts_three_channels() -> None:
    """The `3i:3i+3` slice must line up with `ModalityGrid.as_array()`'s layout."""
    bundle = _bundle()
    stacked = bundle.vitals.as_array()
    units = {feature: columns for modality, feature, columns in _feature_slices(bundle, ("vitals",))}

    for index, concept in enumerate(VITALS):
        columns = units[f"{concept}__seq"]
        np.testing.assert_array_equal(stacked[:, :, columns][:, :, 0], bundle.vitals.value[:, :, index])
        np.testing.assert_array_equal(stacked[:, :, columns][:, :, 1], bundle.vitals.mask[:, :, index])
        np.testing.assert_array_equal(stacked[:, :, columns][:, :, 2], bundle.vitals.delta[:, :, index])


def test_importance_is_deterministic_and_leaves_the_dataset_untouched() -> None:
    """Same seed -> same table, and the permutations are undone on the way out.

    The second half matters on a CPU device, where the working tensors would
    otherwise alias the dataset's arrays and the caller would silently be left
    holding a shuffled test set.
    """
    bundle = _bundle()
    model = _model(bundle)
    dataset = AblationGridDataset(bundle, bundle.stay_ids, MODALITIES)
    before = {name: getattr(dataset, name).copy() for name in MODALITIES}

    kwargs = dict(
        model=model,
        bundle=bundle,
        dataset=dataset,
        device=torch.device("cpu"),
        model_name="gru_intermediate_fusion",
        ablation="unit_test",
        n_repeats=3,
        seed=7,
    )
    first = permutation_importance_dl(**kwargs)
    second = permutation_importance_dl(**kwargs)

    for name in MODALITIES:
        np.testing.assert_array_equal(getattr(dataset, name), before[name])

    assert len(first) == len(STATIC_COLUMNS) + len(VITALS) + len(LABS) + len(TREATMENTS)
    assert first["feature"].is_unique
    assert first["importance_mean"].is_monotonic_decreasing
    assert first["baseline_auprc"].nunique() == 1
    assert first.equals(second)


def test_permuting_a_constant_feature_costs_nothing() -> None:
    """A column with no variation carries no information, so shuffling it is a no-op.

    This is the end-to-end check that a feature unit's slice really is what gets
    permuted: zero out one concept and its importance must be exactly zero,
    while the others' need not be.
    """
    bundle = _bundle()
    bundle.vitals.value[:, :, 1] = 0.0
    bundle.vitals.mask[:, :, 1] = 0.0
    bundle.vitals.delta[:, :, 1] = 0.0
    bundle.static[:, 2] = 1.0

    dataset = AblationGridDataset(bundle, bundle.stay_ids, MODALITIES)
    frame = permutation_importance_dl(
        model=_model(bundle),
        bundle=bundle,
        dataset=dataset,
        device=torch.device("cpu"),
        model_name="gru_intermediate_fusion",
        ablation="unit_test",
        n_repeats=2,
        seed=7,
    ).set_index("feature")

    assert frame.loc["lactate__seq", "importance_mean"] == 0.0
    assert frame.loc["lactate__seq", "importance_std"] == 0.0
    assert frame.loc["gender_1", "importance_mean"] == 0.0
    assert frame.loc["heart_rate__seq", "importance_mean"] != 0.0
