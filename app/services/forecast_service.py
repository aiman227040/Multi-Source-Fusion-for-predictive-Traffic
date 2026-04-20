from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.config import settings
from app.ml.lstm_model import JamForecasterLSTM


FEATURE_ORDER = (
    "inflow_proxy",
    "outflow_proxy",
    "maps_congestion_norm",
    "cctv_density",
    "fused_congestion",
)


@dataclass
class HorizonForecast:
    horizon_minutes: int
    standstill_probability: float


class StandstillForecaster:
    """LSTM head: P(Level-5 jam) for 15–45 minute horizons from recent fused features."""

    def __init__(
        self,
        seq_len: int = 12,
        checkpoint: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JamForecasterLSTM(input_dim=len(FEATURE_ORDER)).to(self.device)
        path = checkpoint or (settings.artifact_dir / settings.lstm_checkpoint)
        self._has_checkpoint = path.exists()
        if self._has_checkpoint:
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def predict_from_sequence(self, features: np.ndarray) -> float:
        """features: (seq_len, 5) float32"""
        if features.shape != (self.seq_len, len(FEATURE_ORDER)):
            raise ValueError(f"Expected {(self.seq_len, len(FEATURE_ORDER))}, got {features.shape}")
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        logit = self.model(x).item()
        return float(1.0 / (1.0 + np.exp(-logit)))

    @staticmethod
    def heuristic_horizons(window: list[dict]) -> list[HorizonForecast]:
        last = window[-1]
        base = float(last["fused_congestion"]) * 0.72 + float(last["maps_congestion_norm"]) * 0.22
        return [
            HorizonForecast(15, float(np.clip(base * 0.85, 0.0, 1.0))),
            HorizonForecast(30, float(np.clip(base, 0.0, 1.0))),
            HorizonForecast(45, float(np.clip(base * 1.08, 0.0, 1.0))),
        ]

    def horizons_from_window(self, window: list[dict]) -> list[HorizonForecast]:
        """Build sequence from last seq_len feature dicts; return 15/30/45m probabilities."""
        if not window:
            return [
                HorizonForecast(15, 0.05),
                HorizonForecast(30, 0.05),
                HorizonForecast(45, 0.05),
            ]
        if not self._has_checkpoint:
            return self.heuristic_horizons(window)

        if len(window) < self.seq_len:
            pad = [window[0]] * (self.seq_len - len(window)) + window
        else:
            pad = window[-self.seq_len :]
        arr = np.array(
            [[float(row[k]) for k in FEATURE_ORDER] for row in pad],
            dtype=np.float32,
        )
        base_p = self.predict_from_sequence(arr)
        return [
            HorizonForecast(15, float(np.clip(base_p * 0.85, 0.0, 1.0))),
            HorizonForecast(30, float(np.clip(base_p, 0.0, 1.0))),
            HorizonForecast(45, float(np.clip(base_p * 1.08, 0.0, 1.0))),
        ]
