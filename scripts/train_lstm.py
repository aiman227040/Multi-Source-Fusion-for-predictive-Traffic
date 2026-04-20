"""
Train the standstill jam forecaster on synthetic fused feature sequences.
Writes artifacts/lstm_forecaster.pt for the API to load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ml.lstm_model import JamForecasterLSTM  # noqa: E402
from app.services.forecast_service import FEATURE_ORDER  # noqa: E402


def synthesize_dataset(n: int = 4000, seq_len: int = 12, feat_dim: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, seq_len, feat_dim)).astype(np.float32)
    # Label: standstill if high fused congestion trend + high inflow in last steps
    fused = X[:, :, 4]
    inflow = X[:, :, 0]
    score = fused[:, -3:].mean(axis=1) * 0.55 + inflow[:, -3:].mean(axis=1) * 0.35
    y = (score > 0.62).astype(np.float32)
    return X, y


def main() -> None:
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.artifact_dir / settings.lstm_checkpoint

    seq_len = 12
    feat_dim = len(FEATURE_ORDER)
    X, y = synthesize_dataset(seq_len=seq_len, feat_dim=feat_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JamForecasterLSTM(input_dim=feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(25):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"epoch {epoch+1:02d} loss {total / len(ds):.4f}")

    model.eval()
    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
