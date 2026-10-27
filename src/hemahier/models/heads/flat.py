"""Flat baselines: linear probe on h_L, or multi-scale linear probe on [h_L; h_M]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hemahier.ontology import Ontology


class FlatHead(nn.Module):
    predicts_maturity = False

    def __init__(self, ont: Ontology, dim_l: int, dim_m: int, input: str = "L", **_):
        super().__init__()
        self.multiscale = input == "LM"
        self.fc = nn.Linear(dim_l + (dim_m if self.multiscale else 0), ont.num_fine)

    def forward(self, h_l: torch.Tensor, h_m: torch.Tensor) -> dict:
        x = torch.cat([h_l, h_m], dim=-1) if self.multiscale else h_l
        logits = self.fc(x)
        return dict(log_probs=F.log_softmax(logits.float(), dim=-1), raw_logits=logits)

    def loss(self, out: dict, y: torch.Tensor, epoch: float) -> tuple[torch.Tensor, dict]:
        l = F.nll_loss(out["log_probs"], y)
        return l, dict(ce=l.item())
