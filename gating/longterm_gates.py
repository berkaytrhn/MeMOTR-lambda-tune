"""Long-term memory gates for MeMOTR adaptive memory gating.

Each gate produces the EMA mixing coefficient lambda used in
    long_memory <- (1 - lambda) * long_memory + lambda * output_embed
replacing the fixed scalar `self.long_memory_lambda` in QueryUpdater.

init_logit = -4.595 because sigmoid(-4.595) ~= 0.01, the released
LONG_MEMORY_LAMBDA default, so every gate starts (pre-clamp) at the
released behavior.
"""
import torch
import torch.nn as nn


_INIT_LOGIT = -4.595  # sigmoid(-4.595) ~= 0.01


class ScalarGate(nn.Module):
    """B3a: a single global learnable lambda. 1 trainable param."""

    def __init__(self, init_logit: float = _INIT_LOGIT, clip=(1e-3, 0.5)):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(init_logit)))
        self.clip_min, self.clip_max = clip

    def forward(self, output_embed: torch.Tensor, long_memory: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.theta).clamp(self.clip_min, self.clip_max)
        return lam  # scalar, broadcasts over (N, D)


class VectorGate(nn.Module):
    """B3b: a per-channel learnable lambda vector of shape (D,). D trainable params."""

    def __init__(self, dim: int, init_logit: float = _INIT_LOGIT, clip=(1e-3, 0.5)):
        super().__init__()
        self.theta = nn.Parameter(torch.full((dim,), float(init_logit)))
        self.clip_min, self.clip_max = clip

    def forward(self, output_embed: torch.Tensor, long_memory: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.theta).clamp(self.clip_min, self.clip_max)
        return lam  # (D,), broadcasts over (N, D) -> element-wise EMA


class SignalGate(nn.Module):
    """B3c: signal-driven per-track lambda.

    Consumes the stacked, normalized signal vector u_t (shape (N, in_dim))
    from gating.signals and returns a per-track lambda of shape (N, 1).
    The final linear bias is initialized so an all-zero (mean-normalized)
    signal yields lambda ~= 0.01.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 16,
                 bias_init: float = _INIT_LOGIT, clip=(1e-3, 0.5)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(bias_init))
        self.clip_min, self.clip_max = clip

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.net(signals)).clamp(self.clip_min, self.clip_max)
        return lam  # (N, 1)
