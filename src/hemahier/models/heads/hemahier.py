"""HemaHier head: factorized features, coupled lineage/fine posterior (Eq. 2), chain-conditioned maturity."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hemahier.losses import gather_on_chain, maturity_mse, ramp, ranking_loss
from hemahier.ontology import Ontology


def mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out),
        nn.GELU(),
    )


def coupled_log_posterior(lineage_logits: torch.Tensor, raw_logits: torch.Tensor, mask: torch.Tensor,
                          lineage_of: torch.Tensor) -> torch.Tensor:
    """Eq. 2 in log space: log p_c(m(f)|x) + log p~(f|x) - log sum_{f' in m(f)} p~(f'|x).

    mask: (L, F) bool lineage membership, lineage_of: (F,) long.
    """
    lineage_logits, raw_logits = lineage_logits.float(), raw_logits.float()
    log_pc = F.log_softmax(lineage_logits, dim=-1)                                   # (B, L)
    within = raw_logits.unsqueeze(1).masked_fill(~mask, float("-inf"))              # (B, L, F)
    lse = torch.logsumexp(within, dim=-1)                                           # (B, L)
    return log_pc[:, lineage_of] + raw_logits - lse[:, lineage_of]


class HemaHierHead(nn.Module):
    def __init__(self, ont: Ontology, dim_l: int, dim_m: int, adapter_hidden: int = 1536, adapter_dim: int = 1024,
                 dropout: float = 0.1, chain_dim: int = 32, maturity_hidden: int = 96, factorized: bool = True,
                 couple: bool = True, maturity: bool = True, ranking: bool = True, margin: float = 0.1,
                 rank_reduction: str = "mean", lambda_raw: float = 0.1, lambda_c: float = 0.05,
                 lambda_m: float = 0.005, lambda_r: float = 0.003, start_c: float = 1, start_m: float = 3,
                 start_r: float = 5, ramp: float = 2, **_):
        super().__init__()
        self.factorized, self.couple = factorized, couple
        self.use_maturity, self.use_ranking = maturity, ranking
        self.predicts_maturity = maturity or ranking
        self.margin, self.rank_reduction = margin, rank_reduction
        self.lam = dict(raw=lambda_raw, c=lambda_c, m=lambda_m, r=lambda_r)
        self.start = dict(c=start_c, m=start_m, r=start_r)
        self.ramp_len = ramp

        if factorized:
            self.phi_c = mlp(dim_l, adapter_hidden, adapter_dim, dropout)            # lineage context z_c
            self.phi_d = mlp(dim_l + dim_m, adapter_hidden, adapter_dim, dropout)    # detail / maturity z_d
            d_fine = 2 * adapter_dim
        else:
            self.phi = mlp(dim_l + dim_m, adapter_hidden, 2 * adapter_dim, dropout)
            d_fine = 2 * adapter_dim
        self.lineage_head = nn.Linear(adapter_dim if factorized else d_fine, ont.num_lineages)
        self.fine_head = nn.Linear(d_fine, ont.num_fine)

        # one shared maturity head, queried by a learned chain embedding e_q
        self.chain_emb = nn.Embedding(max(ont.num_chains, 1), chain_dim)
        self.maturity_head = nn.Sequential(nn.Linear(d_fine + chain_dim, maturity_hidden), nn.GELU(),
                                           nn.Linear(maturity_hidden, 1))

        self.register_buffer("mask", ont.lineage_mask(), persistent=False)
        self.register_buffer("lineage_of", torch.as_tensor(ont.lineage_of, dtype=torch.long), persistent=False)
        self.register_buffer("chain_of", torch.as_tensor(ont.chain_of, dtype=torch.long), persistent=False)
        self.register_buffer("pi_of", torch.as_tensor(ont.pi_of, dtype=torch.float32).nan_to_num(0.0),
                             persistent=False)

    def forward(self, h_l: torch.Tensor, h_m: torch.Tensor) -> dict:
        if self.factorized:
            z_c = self.phi_c(h_l)
            z_d = self.phi_d(torch.cat([h_l, h_m], dim=-1))
            z = torch.cat([z_c, z_d], dim=-1)
            lineage_logits = self.lineage_head(z_c)
        else:
            z = self.phi(torch.cat([h_l, h_m], dim=-1))
            lineage_logits = self.lineage_head(z)
        raw_logits = self.fine_head(z)

        if self.couple:
            log_probs = coupled_log_posterior(lineage_logits, raw_logits, self.mask, self.lineage_of)
        else:
            log_probs = F.log_softmax(raw_logits.float(), dim=-1)

        # s_q(x) for every chain q at once: (B, Q)
        B, Q = z.shape[0], self.chain_emb.num_embeddings
        e = self.chain_emb.weight.unsqueeze(0).expand(B, Q, -1)
        zq = torch.cat([z.unsqueeze(1).expand(B, Q, -1), e], dim=-1)
        maturity = torch.sigmoid(self.maturity_head(zq).squeeze(-1).float())

        return dict(log_probs=log_probs, raw_logits=raw_logits, lineage_logits=lineage_logits, maturity=maturity)

    def loss(self, out: dict, y: torch.Tensor, epoch: float) -> tuple[torch.Tensor, dict]:
        logs = {}
        l_joint = F.nll_loss(out["log_probs"], y)
        l_raw = F.cross_entropy(out["raw_logits"].float(), y)
        total = l_joint + self.lam["raw"] * l_raw
        logs.update(joint=l_joint.item(), raw=l_raw.item())

        if self.couple:
            lam_c = ramp(self.lam["c"], epoch, self.start["c"], self.ramp_len)
            l_c = F.cross_entropy(out["lineage_logits"].float(), self.lineage_of[y])
            total = total + lam_c * l_c
            logs["lineage"] = l_c.item()

        if self.use_maturity or self.use_ranking:
            s, pi, q = gather_on_chain(out["maturity"], y, self.chain_of, self.pi_of)
            if self.use_maturity:
                l_m = maturity_mse(s, pi)
                total = total + ramp(self.lam["m"], epoch, self.start["m"], self.ramp_len) * l_m
                logs["maturity"] = l_m.item()
            if self.use_ranking:
                l_r = ranking_loss(s, pi, q, self.margin, self.rank_reduction)
                total = total + ramp(self.lam["r"], epoch, self.start["r"], self.ramp_len) * l_r
                logs["rank"] = l_r.item()
        return total, logs
