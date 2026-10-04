"""Thin torch Dataset over cached hourly-grid tensors, keyed by ablation's active branches."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from multi_modality_code.experiments.dl.grid_data import GridBundle


class AblationGridDataset(Dataset):
    """Exposes only the modality tensors an ablation's active branches need.

    `bundle` holds the full cached grid for one export (primary or antibiotics
    sensitivity); `stay_ids` selects the rows for one split (train/val/test).
    """

    def __init__(self, bundle: GridBundle, stay_ids: np.ndarray, modalities: tuple[str, ...]):
        self.modalities = tuple(modalities)
        position = {int(sid): i for i, sid in enumerate(bundle.stay_ids.tolist())}
        try:
            indices = np.array([position[int(sid)] for sid in stay_ids], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"stay_id {exc} not present in cached grid bundle.") from exc

        self.stay_ids = bundle.stay_ids[indices]
        self.labels = bundle.labels[indices].astype(np.float32)
        self.static = bundle.static[indices] if "static" in self.modalities else None
        self.vitals = bundle.vitals.as_array()[indices] if "vitals" in self.modalities else None
        self.labs = bundle.labs.as_array()[indices] if "labs" in self.modalities else None
        self.treatments = bundle.treatments.as_array()[indices] if "treatments" in self.modalities else None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {"label": torch.tensor(self.labels[idx])}
        if self.static is not None:
            item["static"] = torch.from_numpy(self.static[idx])
        if self.vitals is not None:
            item["vitals"] = torch.from_numpy(self.vitals[idx])
        if self.labs is not None:
            item["labs"] = torch.from_numpy(self.labs[idx])
        if self.treatments is not None:
            item["treatments"] = torch.from_numpy(self.treatments[idx])
        return item


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}
