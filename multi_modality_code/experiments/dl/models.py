"""Per-modality intermediate-fusion architecture (A2): StaticEncoder, SequenceEncoder, FusionClassifier."""

from __future__ import annotations

import torch
from torch import nn


class StaticEncoder(nn.Module):
    """`Linear -> ReLU -> Dropout -> Linear` encoder for the static feature vector."""

    def __init__(self, input_dim: int, hidden: int = 32, output_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceEncoder(nn.Module):
    """GRU (default) or LSTM encoder over a `(B, 24, input_dim)` modality grid.

    This is the only module aware of recurrent-cell internals, so a future
    `GRUDEncoder` (decay-based, using the mask/delta channels intrinsically)
    or a `TFTEncoder` can be dropped in with the same constructor signature
    without changing `FusionClassifier` or the ablation-running code.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        cell_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        cell_type = cell_type.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported cell_type: {cell_type!r} (expected 'gru' or 'lstm').")
        self.cell_type = cell_type
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        last_hidden = hidden[0] if isinstance(hidden, tuple) else hidden  # LSTM returns (h_n, c_n)
        final_layer_hidden = last_hidden[-1]  # (B, hidden_dim), top layer's final timestep
        return self.project(final_layer_hidden)


_ENCODER_DIMS: dict[str, tuple[int, int]] = {
    # modality -> (hidden_dim/hidden, output_dim), per A2.
    "static": (32, 16),
    "vitals": (64, 64),
    "labs": (64, 64),
    "treatments": (32, 32),
}


class FusionClassifier(nn.Module):
    """Per-ablation intermediate-fusion classifier (A2/A3).

    Only the branches present in `modalities` are constructed, so a
    `static_only` ablation reduces to just the static MLP feeding the
    classifier head directly, while the primary ablation activates all four.
    Active branches' output vectors are concatenated and passed through a
    shared `Linear -> ReLU -> Dropout -> Linear` head producing one logit.
    """

    def __init__(
        self,
        modalities: tuple[str, ...],
        input_dims: dict[str, int],
        cell_type: str = "gru",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.branches = nn.ModuleDict()
        concat_dim = 0

        if "static" in self.modalities:
            hidden, output_dim = _ENCODER_DIMS["static"]
            self.branches["static"] = StaticEncoder(input_dims["static"], hidden=hidden, output_dim=output_dim, dropout=dropout)
            concat_dim += output_dim
        for name in ("vitals", "labs", "treatments"):
            if name in self.modalities:
                hidden, output_dim = _ENCODER_DIMS[name]
                self.branches[name] = SequenceEncoder(
                    input_dims[name], hidden_dim=hidden, output_dim=output_dim, cell_type=cell_type, dropout=dropout
                )
                concat_dim += output_dim

        if not self.branches:
            raise ValueError("FusionClassifier requires at least one active modality branch.")

        self.head = nn.Sequential(
            nn.Linear(concat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.concat_dim = concat_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.branches[name](batch[name]) for name in ("static", "vitals", "labs", "treatments") if name in self.branches]
        fused = torch.cat(parts, dim=-1)
        return self.head(fused).squeeze(-1)  # logits, BCEWithLogitsLoss applies sigmoid internally


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
