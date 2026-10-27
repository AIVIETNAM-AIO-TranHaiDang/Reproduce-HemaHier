"""Recognition, error-severity (CLE, TD) and maturation-quality metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from hemahier.ontology import Ontology

SEVERITY = ["adjacent_stage", "same_chain_distant", "same_lineage_other", "cross_lineage"]


def recognition(y: np.ndarray, pred: np.ndarray, rare_classes: list[int]) -> dict:
    labels = np.unique(y)
    rare = [c for c in rare_classes if c in set(labels)]
    return {
        "acc": 100 * accuracy_score(y, pred),
        "bacc": 100 * balanced_accuracy_score(y, pred),
        "macro_f1": 100 * f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": 100 * f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
        "rare_f1": 100 * f1_score(y, pred, labels=rare, average="macro", zero_division=0) if rare else float("nan"),
    }


def severity(y: np.ndarray, pred: np.ndarray, ont: Ontology) -> dict:
    td = ont.tree_distance()
    return {
        "cle": 100 * np.mean(ont.lineage_of[pred] != ont.lineage_of[y]),
        "td": 100 * np.mean(td[pred, y]),
    }


def severity_breakdown(y: np.ndarray, pred: np.ndarray, ont: Ontology) -> dict:
    """Error rate (% of all cells) split by clinical severity, mildest first (Fig. 2)."""
    wrong = pred != y
    same_lin = ont.lineage_of[pred] == ont.lineage_of[y]
    same_chain = (ont.chain_of[pred] >= 0) & (ont.chain_of[pred] == ont.chain_of[y])
    n_q = np.array([len(ont.chain_members[q]) - 1 if q >= 0 else 1 for q in ont.chain_of])
    step = np.rint(np.abs(ont.pi_of[pred] - ont.pi_of[y]) * n_q[y])
    adjacent = wrong & same_chain & (step == 1)
    distant = wrong & same_chain & (step > 1)
    other = wrong & same_lin & ~same_chain
    cross = wrong & ~same_lin
    n = len(y)
    return {k: 100 * m.sum() / n for k, m in zip(SEVERITY, [adjacent, distant, other, cross])}


def maturity_quality(y: np.ndarray, maturity: np.ndarray, ont: Ontology, max_pairs: int = 200_000,
                     seed: int = 0) -> dict:
    """Spearman rho and pairwise order accuracy between s_q(x) and pi on healthy on-chain cells.

    Reads each cell under its true chain. Reports per-chain values and their cell-weighted mean.
    """
    rng = np.random.default_rng(seed)
    out, rhos, ords, ns = {}, [], [], []
    for q, name in enumerate(ont.chains):
        idx = np.flatnonzero(ont.chain_of[y] == q)
        if len(idx) < 3 or len(np.unique(y[idx])) < 2:
            continue
        s, pi = maturity[idx, q], ont.pi_of[y[idx]]
        rho = spearmanr(s, pi).statistic
        i, j = rng.integers(0, len(idx), size=(2, max_pairs))
        keep = pi[i] != pi[j]
        i, j = i[keep], j[keep]
        lo, hi = np.where(pi[i] < pi[j], i, j), np.where(pi[i] < pi[j], j, i)
        order = np.mean((s[hi] > s[lo]) + 0.5 * (s[hi] == s[lo]))
        out[f"mat_rho_{name}"], out[f"ord_{name}"] = rho, 100 * order
        rhos.append(rho); ords.append(order); ns.append(len(idx))
    if ns:
        w = np.asarray(ns) / sum(ns)
        out["mat_rho"] = float(np.dot(w, rhos))
        out["ord_acc"] = float(100 * np.dot(w, ords))
    return out


def evaluate(y: np.ndarray, pred: np.ndarray, ont: Ontology, rare_classes: list[int],
             maturity: np.ndarray | None = None) -> dict:
    m = recognition(y, pred, rare_classes)
    m.update(severity(y, pred, ont))
    m.update({f"err_{k}": v for k, v in severity_breakdown(y, pred, ont).items()})
    if maturity is not None:
        m.update(maturity_quality(y, maturity, ont))
    return {k: float(v) for k, v in m.items()}
