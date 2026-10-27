"""Bone-marrow ontology: fine classes, lineages m(f), healthy maturation chains and tree distance."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import yaml


@dataclass
class Ontology:
    name: str
    fine: list[str]                      # fine class names, index = class id
    lineages: list[str]
    lineage_of: np.ndarray               # (F,) lineage id of each fine class, m(f)
    chains: list[str]                    # chain names q in Q
    chain_of: np.ndarray                 # (F,) chain id, -1 for off-chain classes
    pi_of: np.ndarray                    # (F,) normalized maturity target j / n_q, nan for off-chain
    td_off_chain: float = 0.5
    td_cross_lineage: float = 1.0
    chain_members: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "Ontology":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        lineages = list(cfg["lineages"])
        fine, lineage_of = [], []
        for li, name in enumerate(lineages):
            for c in cfg["lineages"][name]:
                if c in fine:
                    raise ValueError(f"class {c} assigned to more than one lineage")
                fine.append(c)
                lineage_of.append(li)
        lineage_of = np.array(lineage_of)
        idx = {c: i for i, c in enumerate(fine)}

        chain_of = np.full(len(fine), -1)
        pi_of = np.full(len(fine), np.nan)
        chains, members = [], []
        for name, stages in cfg.get("chains", {}).items():
            if len(stages) < 2:
                raise ValueError(f"chain {name} needs >= 2 stages to define pi = j / n_q")
            ids = [idx[c] for c in stages]
            if len({lineage_of[i] for i in ids}) != 1:
                raise ValueError(f"chain {name} spans several lineages")
            q = len(chains)
            n_q = len(ids) - 1
            for j, i in enumerate(ids):
                if chain_of[i] >= 0:
                    raise ValueError(f"class {fine[i]} on more than one chain")
                chain_of[i] = q
                pi_of[i] = j / n_q
            chains.append(name)
            members.append(ids)

        td = cfg.get("td", {})
        return cls(
            name=cfg.get("name", path),
            fine=fine,
            lineages=lineages,
            lineage_of=lineage_of,
            chains=chains,
            chain_of=chain_of,
            pi_of=pi_of,
            td_off_chain=float(td.get("off_chain", 0.5)),
            td_cross_lineage=float(td.get("cross_lineage", 1.0)),
            chain_members=members,
        )

    @property
    def num_fine(self) -> int:
        return len(self.fine)

    @property
    def num_lineages(self) -> int:
        return len(self.lineages)

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    def index(self, name: str) -> int:
        return self.fine.index(name)

    def lineage_mask(self) -> torch.Tensor:
        """(L, F) bool, True where fine class f belongs to lineage l."""
        m = torch.zeros(self.num_lineages, self.num_fine, dtype=torch.bool)
        m[torch.as_tensor(self.lineage_of), torch.arange(self.num_fine)] = True
        return m

    def tree_distance(self) -> np.ndarray:
        """(F, F) maturity-aware tree distance; rows = predicted, cols = true. Evaluation only.

        Same-chain slips cost off_chain * |pi gap|, so even a full-chain slip never exceeds an off-chain
        confusion, and cross-lineage confusions cost the maximum.
        """
        F = self.num_fine
        D = np.empty((F, F))
        for p in range(F):
            for t in range(F):
                if p == t:
                    D[p, t] = 0.0
                elif self.lineage_of[p] != self.lineage_of[t]:
                    D[p, t] = self.td_cross_lineage
                elif self.chain_of[p] >= 0 and self.chain_of[p] == self.chain_of[t]:
                    D[p, t] = self.td_off_chain * abs(self.pi_of[p] - self.pi_of[t])
                else:
                    D[p, t] = self.td_off_chain
        return D

    def describe(self) -> str:
        lines = [f"Ontology {self.name}: {self.num_fine} fine, {self.num_lineages} lineages, {self.num_chains} chains"]
        for li, ln in enumerate(self.lineages):
            cls_ = [self.fine[i] for i in np.flatnonzero(self.lineage_of == li)]
            lines.append(f"  {ln}: {cls_}")
        for q, ids in enumerate(self.chain_members):
            lines.append(f"  chain {self.chains[q]}: " + " -> ".join(f"{self.fine[i]}({self.pi_of[i]:.2f})" for i in ids))
        return "\n".join(lines)
