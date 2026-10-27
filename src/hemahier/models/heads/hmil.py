"""HMIL*: single-cell adaptation of the coarse/fine hierarchical head of Jin et al. (TMI 2025).

The attention-MIL aggregation degenerates for a single cell, so only the coarse (lineage) / fine heads remain,
trained with CE on both levels plus a consistency term between the coarse prediction and the fine
distribution marginalised over lineages. Fine predictions are not constrained to the predicted lineage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hemahier.ontology import Ontology


class HMILHead(nn.Module):
    predicts_maturity = False

    def __init__(self, ont: Ontology, dim_l: int, dim_m: int, hidden: int = 1024, dropout: float = 0.1,
                 coarse_weight: float = 1.0, consistency_weight: float = 0.1, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim_l), nn.Linear(dim_l, hidden), nn.GELU(), nn.Dropout(dropout))
        self.coarse = nn.Linear(hidden, ont.num_lineages)
        self.fine = nn.Linear(hidden, ont.num_fine)
        self.coarse_weight, self.consistency_weight = coarse_weight, consistency_weight
        self.register_buffer("lineage_of", torch.as_tensor(ont.lineage_of, dtype=torch.long), persistent=False)
        self.register_buffer("mask", ont.lineage_mask().float(), persistent=False)

    def forward(self, h_l: torch.Tensor, h_m: torch.Tensor) -> dict:
        z = self.proj(h_l)
        fine_logits, coarse_logits = self.fine(z), self.coarse(z)
        return dict(log_probs=F.log_softmax(fine_logits.float(), dim=-1), raw_logits=fine_logits,
                    lineage_logits=coarse_logits)

    def loss(self, out: dict, y: torch.Tensor, epoch: float) -> tuple[torch.Tensor, dict]:
        l_fine = F.nll_loss(out["log_probs"], y)
        log_pc = F.log_softmax(out["lineage_logits"].float(), dim=-1)
        l_coarse = F.nll_loss(log_pc, self.lineage_of[y])
        p_marg = out["log_probs"].exp() @ self.mask.T                       # (B, L) fine mass per lineage
        l_cons = F.kl_div(log_pc, p_marg.clamp_min(1e-8), reduction="batchmean")
        total = l_fine + self.coarse_weight * l_coarse + self.consistency_weight * l_cons
        return total, dict(fine=l_fine.item(), coarse=l_coarse.item(), cons=l_cons.item())
