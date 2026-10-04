"""Permutation importance for the DL arm (the sequence-model counterpart of A11).

`run_experiments.compute_feature_importance()` hands scikit-learn a 2-D design
matrix and lets it shuffle one column at a time. That does not transfer to this
arm: a stay is not a row of scalars here but a `(24, C * 3)` grid per modality,
and shuffling one `(hour, channel)` cell would destroy the trajectory rather
than remove a variable.

The unit of permutation is therefore the thing the classical arm's `concept__*`
columns collectively stand for -- one clinical concept -- permuted as a whole:
its `value`/`mask`/`delta` channels across all 24 hours move together to another
stay, so the trajectory stays a real trajectory and only the *link between this
concept and this patient* is broken. Static features have no time axis and are
permuted column by column, exactly as in the classical arm.

Importance is the drop in test AUPRC, the same score and the same split the
classical arm uses, so the two tables are read the same way -- with the caveat
that a row here covers a whole concept and a row there covers one summary
statistic of it, which makes the deep arm's numbers the larger of the two by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from multi_modality_code.experiments.dl.datasets import AblationGridDataset
from multi_modality_code.experiments.dl.grid_data import GridBundle
from multi_modality_code.experiments.dl.models import FusionClassifier


SEQUENCE_MODALITIES: tuple[str, ...] = ("vitals", "labs", "treatments")


def _predict(
    model: FusionClassifier,
    tensors: dict[str, torch.Tensor],
    n_rows: int,
    batch_size: int,
) -> np.ndarray:
    """Sigmoid probabilities for `tensors`, batched, without touching any RNG."""
    outputs = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch = {name: tensor[start:stop] for name, tensor in tensors.items()}
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs)


def _feature_slices(
    bundle: GridBundle, modalities: tuple[str, ...]
) -> list[tuple[str, str, slice]]:
    """(modality, feature name, column slice) for every permutable unit.

    Sequence modalities interleave channels per concept -- `[value_1, mask_1,
    delta_1, value_2, ...]` (see `ModalityGrid.as_array`) -- so concept `i`
    occupies columns `3i:3i+3`, and permuting that slice moves all three
    channels of that concept together.
    """
    units: list[tuple[str, str, slice]] = []
    if "static" in modalities:
        for index, column in enumerate(bundle.static_columns):
            units.append(("static", column, slice(index, index + 1)))
    for modality in SEQUENCE_MODALITIES:
        if modality not in modalities:
            continue
        concepts = getattr(bundle, modality).concepts
        for index, concept in enumerate(concepts):
            units.append((modality, f"{concept}__seq", slice(3 * index, 3 * index + 3)))
    return units


def permutation_importance_dl(
    model: FusionClassifier,
    bundle: GridBundle,
    dataset: AblationGridDataset,
    device: torch.device,
    model_name: str,
    ablation: str,
    n_repeats: int = 20,
    seed: int = 42,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Drop in test AUPRC when each feature unit is permuted across stays.

    The same `n_repeats` shuffles are reused for every feature rather than drawn
    afresh per feature: the comparison between features is then paired, so the
    ranking is not partly an artefact of one feature having drawn a luckier
    permutation than another. `seed` fixes them.
    """
    model = model.to(device)
    model.eval()

    # `copy=True` is load-bearing: the permutations below are in-place, and on a
    # CPU device `from_numpy(...).to(device)` hands back a view of the dataset's
    # own array, so without it this would be shuffling the caller's test set.
    tensors: dict[str, torch.Tensor] = {}
    for modality in ("static",) + SEQUENCE_MODALITIES:
        array = getattr(dataset, modality)
        if array is not None:
            tensors[modality] = torch.from_numpy(np.array(array, copy=True)).to(device)

    n_rows = len(dataset)
    labels = dataset.labels
    baseline = float(average_precision_score(labels, _predict(model, tensors, n_rows, batch_size)))

    rng = np.random.RandomState(seed)
    shuffles = [rng.permutation(n_rows) for _ in range(n_repeats)]

    rows = []
    for modality, feature, columns in _feature_slices(bundle, dataset.modalities):
        original = tensors[modality][..., columns].clone()
        drops = np.empty(n_repeats)
        for repeat, order in enumerate(shuffles):
            tensors[modality][..., columns] = original[torch.as_tensor(order, device=device)]
            probs = _predict(model, tensors, n_rows, batch_size)
            drops[repeat] = baseline - float(average_precision_score(labels, probs))
        tensors[modality][..., columns] = original
        rows.append(
            {
                "model": model_name,
                "ablation": ablation,
                "modality": modality,
                "feature": feature,
                "importance_mean": float(drops.mean()),
                "importance_std": float(drops.std(ddof=0)),
            }
        )

    frame = pd.DataFrame(rows).sort_values("importance_mean", ascending=False, kind="stable")
    frame.insert(frame.columns.get_loc("importance_mean"), "baseline_auprc", baseline)
    return frame.reset_index(drop=True)
