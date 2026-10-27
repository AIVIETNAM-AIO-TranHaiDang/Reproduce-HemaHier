"""Inference-time decoding of the hierarchy-consistent posterior."""
from __future__ import annotations

import numpy as np

from hemahier.ontology import Ontology


def product_decode(probs: np.ndarray, maturity: np.ndarray, ont: Ontology, sigma: float = 0.15) -> np.ndarray:
    """Re-weight on-chain classes by agreement between s_q(x) and pi_f, keeping each chain's total mass.

    probs: (N, F) p_f(f|x); maturity: (N, Q) s_q(x). Mass only moves *within* a chain, so lineage
    probabilities (and hence hierarchical consistency) are untouched. Returns re-weighted (N, F).
    """
    out = probs.copy()
    for q, ids in enumerate(ont.chain_members):
        ids = np.asarray(ids)
        p = probs[:, ids]                                                       # (N, |q|)
        w = np.exp(-((maturity[:, q:q + 1] - ont.pi_of[ids][None]) ** 2) / (2 * sigma ** 2))
        pw = p * w
        out[:, ids] = pw / np.clip(pw.sum(1, keepdims=True), 1e-12, None) * p.sum(1, keepdims=True)
    return out


def decode(probs: np.ndarray, maturity: np.ndarray | None, ont: Ontology, method: str = "argmax",
           sigma: float = 0.15) -> np.ndarray:
    if method == "product" and maturity is not None:
        probs = product_decode(probs, maturity, ont, sigma)
    return probs.argmax(1)
