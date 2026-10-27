"""Fig. 3: predicted maturity vs true stage along one chain (on-chain test cells, true-chain read-out)."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from hemahier.ontology import Ontology

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/frozen/hemahier/fold0")
ap.add_argument("--chain", default="neutrophil")
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = OmegaConf.load(Path(args.run) / "config.yaml")
ont = Ontology.from_yaml(cfg.ontology)
p = np.load(Path(args.run) / "predictions.npz")
q = ont.chains.index(args.chain)
ids = ont.chain_members[q]
on = np.isin(p["y"], ids)
s, y = p["maturity"][on, q], p["y"][on]
rho = spearmanr(s, ont.pi_of[y]).statistic

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 3))
for j, c in enumerate(ids):
    v = s[y == c]
    ax.scatter(j + rng.uniform(-0.2, 0.2, len(v)), v, s=2, alpha=0.3)
    ax.hlines(np.median(v), j - 0.3, j + 0.3, color="k")
ax.set_xticks(range(len(ids)), [ont.fine[c] for c in ids])
ax.set_ylabel("predicted maturity")
ax.set_title(f"{args.chain}: Spearman rho = {rho:.3f}")
fig.tight_layout()
out = args.out or str(Path(args.run) / f"fig3_{args.chain}.png")
fig.savefig(out, dpi=200)
print(f"saved {out} (rho={rho:.3f})")
