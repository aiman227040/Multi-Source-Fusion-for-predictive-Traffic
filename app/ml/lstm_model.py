from __future__ import annotations

import torch
import torch.nn as nn


class JamForecasterLSTM(nn.Module):
    """Predicts probability of Level-5 (standstill) jam from fused multimodal sequences."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits
