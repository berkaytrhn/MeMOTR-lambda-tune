"""Short-term symmetric fusion gate (B2, Variant 1).

Released MeMOTR fuses the current and previous output embeddings
asymmetrically: only O_t is gated by a sigmoid confidence weight, while
O_{t-1} passes through ungated, then both are concatenated and fused by an
MLP. This module replaces that with a symmetric softmax over two learned
logits so the network can weight O_t and O_{t-1} relative to each other.

It is a drop-in replacement for `short_memory` (shape (N, D)).
"""
import torch
import torch.nn as nn


class _ScalarMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, 1)


class SymmetricSoftmaxGate(nn.Module):
    """Two MLPs -> logits (l_t, l_{t-1}); softmax -> weights; weighted sum.

    The previous-frame branch bias is initialized low so the initial softmax
    puts most weight on O_t, approximating the released behavior where the
    current detection dominates. The init gap from the released fusion is not
    zero (released uses an MLP fusion, not a convex combination); callers
    should measure it on a smoke batch and log it.
    """

    def __init__(self, dim: int, hidden: int = 64, prev_bias_init: float = -4.0):
        super().__init__()
        self.cur_mlp = _ScalarMLP(dim, hidden)
        self.prev_mlp = _ScalarMLP(dim, hidden)
        # Bias the previous branch low so softmax ~ [~1, ~0] at init -> w_t dominates.
        nn.init.constant_(self.prev_mlp.net[-1].bias, float(prev_bias_init))

    def forward(self, output_embed: torch.Tensor, last_output_embed: torch.Tensor) -> torch.Tensor:
        l_t = self.cur_mlp(output_embed)            # (N, 1)
        l_prev = self.prev_mlp(last_output_embed)   # (N, 1)
        logits = torch.cat((l_t, l_prev), dim=-1)   # (N, 2)
        weights = torch.softmax(logits, dim=-1)     # (N, 2)
        w_t = weights[:, 0:1]
        w_prev = weights[:, 1:2]
        return w_t * output_embed + w_prev * last_output_embed  # (N, D)
