"""Fig. 2: fine-classification error rate decomposed by clinical severity (flat vs HemaHier)."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hemahier.metrics import SEVERITY

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--setting", default="frozen")
ap.add_argument("--models", nargs="+", default=["flat", "hemahier"])
ap.add_argument("--out", default="runs/fig2_errors.png")
args = ap.parse_args()

colors = ["#f6d365", "#f0a35e", "#e06c4f", "#a3324a"]
fig, ax = plt.subplots(figsize=(7, 1 + 0.5 * len(args.models)))
for row, m in enumerate(args.models):
    ms = [json.loads(f.read_text())["argmax"] for f in sorted(Path(args.runs, args.setting, m).glob("fold*/metrics.json"))]
    vals = [np.mean([r[f"err_{k}"] for r in ms]) for k in SEVERITY]
    left = 0.0
    for v, c, k in zip(vals, colors, SEVERITY):
        ax.barh(row, v, left=left, color=c, label=k.replace("_", " ") if row == 0 else None)
        left += v
    ax.text(left + 0.2, row, f"{left:.1f}%", va="center", fontsize=8)
ax.set_yticks(range(len(args.models)), args.models)
ax.set_xlabel("Fine-classification error rate (%), mild -> severe")
ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"saved {args.out}")
