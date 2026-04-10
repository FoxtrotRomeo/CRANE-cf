"""Shared PyTorch model definitions for the sepsis experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatiotemporalSumAttention(nn.Module):
    """Learnable temporal + spatial attention combined by summation."""

    def __init__(self, n_timesteps: int, n_features: int):
        super().__init__()
        self.w_timesteps = nn.Parameter(torch.empty(n_timesteps, n_timesteps))
        self.w_features = nn.Parameter(torch.empty(n_features, n_features))
        nn.init.normal_(self.w_timesteps)
        nn.init.normal_(self.w_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_time = torch.einsum("tj,bjf->btf", self.w_timesteps, x)
        a_time = torch.softmax(a_time, dim=1)
        x_perm = x.permute(0, 2, 1)
        a_feat = torch.einsum("fg,bgT->bfT", self.w_features, x_perm)
        a_feat = torch.softmax(a_feat, dim=1).permute(0, 2, 1)
        return (a_time + a_feat) * x


class SepsisModel(nn.Module):
    """Binary classifier over a time-series branch plus a static branch."""

    def __init__(
        self,
        n_timesteps: int = 24,
        n_ts_features: int = 53,
        n_static_features: int = 47,
        gru_units: int = 16,
        dense1_units: int = 53,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        n_features = n_ts_features + n_static_features
        self.n_timesteps = n_timesteps
        self.attention = SpatiotemporalSumAttention(n_timesteps, n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_units,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(gru_units, dense1_units)
        self.bn = nn.BatchNorm1d(dense1_units, momentum=0.01, eps=0.001)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(dense1_units, 1)

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [x_ts, x_static.unsqueeze(1).expand(-1, self.n_timesteps, -1)],
            dim=2,
        )
        x = self.attention(x)
        _, h = self.gru(x)
        h = self.dropout1(h.squeeze(0))
        h = torch.tanh(self.dense1(h))
        h = self.dropout2(self.bn(h))
        return torch.sigmoid(self.output_layer(h))
