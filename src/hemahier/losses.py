"""Maturity losses (Eq. 4) and the staged lambda(t) ramp."""
from __future__ import annotations

import torch


def gather_on_chain(maturity: torch.Tensor, y: torch.Tensor, chain_of: torch.Tensor, pi_of: torch.Tensor):
    """Pick s_{q_i}(x_i) for on-chain samples. maturity: (B, Q). Returns (s, pi, chain) for i in Omega."""
    q = chain_of[y]
    on = q >= 0
    s = maturity[on, q[on]]
    return s, pi_of[y[on]], q[on]


def maturity_mse(s: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    if s.numel() == 0:
        return s.new_zeros(())
    return ((s - pi) ** 2).mean()


def ranking_loss(s: torch.Tensor, pi: torch.Tensor, q: torch.Tensor, margin: float, reduction: str = "mean"):
    """sum over pairs with q_i = q_j and pi_i < pi_j of max(0, margin - (s_j - s_i))."""
    if s.numel() < 2:
        return s.new_zeros(())
    valid = (q[:, None] == q[None, :]) & (pi[:, None] < pi[None, :])  # [i, j]: i earlier than j
    if not valid.any():
        return s.new_zeros(())
    hinge = torch.relu(margin - (s[None, :] - s[:, None]))[valid]
    return hinge.sum() if reduction == "sum" else hinge.mean()


def ramp(weight: float, epoch: float, start: float, length: float) -> float:
    """lambda(t) = weight * clip((epoch - start) / length, 0, 1); epoch may be fractional."""
    if length <= 0:
        return weight if epoch >= start else 0.0
    return weight * min(max((epoch - start) / length, 0.0), 1.0)
