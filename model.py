"""Compact temporal CNN for fall classification over a feature-sequence window.

Input  : [batch, window, n_features]
Output : [batch, 2] logits  (class 0 = no_fall, 1 = fall)

Dilated 1D convolutions give a multi-frame receptive field at low parameter
count (~30k), so it trains on a modest number of staged windows and runs far
inside the 30 FPS budget.
"""
import numpy as np
import torch
import torch.nn as nn


class FallTCN(nn.Module):
    def __init__(self, n_features=6, hidden=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):                 # x: [B, T, F]
        z = self.net(x.transpose(1, 2))   # -> [B, hidden, 1]
        return self.head(z.squeeze(-1))


def save_fall_model(path, model, mean, std, window, n_features, threshold):
    torch.save({
        "state_dict": model.state_dict(),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "window": int(window),
        "n_features": int(n_features),
        "threshold": float(threshold),
    }, path)


def load_fall_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FallTCN(n_features=ck["n_features"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def standardize(x, mean, std):
    return (x - mean) / np.where(std < 1e-6, 1.0, std)
