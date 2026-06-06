"""LocalGenomeNet — 1D-CNN variant pathogenicity classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 3}


def one_hot_encode(seq: str, max_len: int = 41) -> torch.Tensor:
    """Encode DNA string as (4, L) one-hot tensor."""
    s = seq.upper().replace("U", "T")[:max_len]
    arr = torch.zeros(4, max_len)
    for i, ch in enumerate(s):
        idx = NUC_TO_IDX.get(ch, 3)
        arr[idx, i] = 1.0
    return arr


class LocalGenomeNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        h = self.encoder(x).squeeze(-1)
        return self.head(h)
