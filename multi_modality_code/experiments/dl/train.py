"""Training loop for the DL sequence models (A4): Adam, BCEWithLogitsLoss, early stopping."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from multi_modality_code.experiments.dl.datasets import AblationGridDataset, collate_batch
from multi_modality_code.experiments.dl.models import FusionClassifier


def set_seed(seed: int) -> None:
    """Seed every RNG the training loop can touch, including Apple's MPS backend.

    `torch.mps.manual_seed()` matters here because the MPS generator is separate
    from the CPU one: without it, weight init and dropout masks drawn on an Apple
    GPU vary run to run even with `torch.manual_seed()` set.

    Call this *before* constructing the model, not just before training: torch
    seeds its default generator from OS entropy at first use, so a model built
    ahead of the first `set_seed()` gets unseeded weights (see
    `run_dl_experiments.run_one_ablation`). With the call in the right place,
    repeated runs at a fixed seed reproduce `dl_metrics.csv` bit-identically on
    both `mps` and `cpu` — measured on this cohort, at smoke and full scale.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device(preference: str = "auto") -> torch.device:
    """Resolve the training device.

    `auto` prefers MPS when available (the historical behaviour). Both backends
    are reproducible at a fixed seed and, for a model this small, equally fast —
    but they do not agree with each other, so a reported run must pin one. The
    resolved device is recorded in `dl_manifest.json`.
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if preference != "auto":
        raise ValueError(f"Unsupported device preference: {preference!r} (expected 'auto', 'mps' or 'cpu').")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    pos_weight: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class TrainResult:
    model: FusionClassifier
    device: torch.device
    n_epochs_trained: int
    best_val_auprc: float
    val_probs: np.ndarray
    val_labels: np.ndarray
    test_probs: np.ndarray
    test_labels: np.ndarray


def _predict(model: FusionClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(
    model: FusionClassifier,
    train_ds: AblationGridDataset,
    val_ds: AblationGridDataset,
    test_ds: AblationGridDataset,
    config: TrainConfig,
    device: torch.device | None = None,
) -> TrainResult:
    """Fit `model` with early stopping on validation AUPRC, restoring the best-epoch weights."""
    set_seed(config.seed)
    device = device or select_device()
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    pos_weight_tensor = torch.tensor(config.pos_weight, dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_state = copy.deepcopy(model.state_dict())
    best_val_auprc = -np.inf
    epochs_without_improvement = 0
    n_epochs_trained = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
        n_epochs_trained = epoch

        val_probs, val_labels = _predict(model, val_loader, device)
        # Early stopping only needs AUPRC; the full metric suite (confusion matrix at
        # an arbitrary 0.5 threshold) was being recomputed every epoch to read one key.
        val_auprc = float(average_precision_score(val_labels, val_probs))

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    val_probs, val_labels = _predict(model, val_loader, device)
    test_probs, test_labels = _predict(model, test_loader, device)

    return TrainResult(
        model=model,
        device=device,
        n_epochs_trained=n_epochs_trained,
        best_val_auprc=float(best_val_auprc),
        val_probs=val_probs,
        val_labels=val_labels,
        test_probs=test_probs,
        test_labels=test_labels,
    )
