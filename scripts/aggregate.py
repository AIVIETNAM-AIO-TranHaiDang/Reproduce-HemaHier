"""Collect runs/<setting>/<model>/fold*/metrics.json into Table 1 / Table 2 style markdown (mean +- std),
with paired t-tests over folds against the flat probe."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

COLS = [("acc", "Acc"), ("bacc", "B-Acc"), ("macro_f1", "M-F1"), ("weighted_f1", "W-F1"), ("rare_f1", "Rare"),
        ("cle", "CLE"), ("td", "TD"), ("mat_rho", "Mat-rho"), ("ord_acc", "Ord")]
ORDER = ["flat", "multiscale_flat", "mlp_flat", "hmil", "abl_factorized_lineage", "abl_branch_maturity", "hemahier"]

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
ap.add_argument("--decoder", default="argmax", choices=["argmax", "product"])
args = ap.parse_args()

for setting_dir in sorted(Path(args.runs).iterdir()):
    res = {}
    for mdir in setting_dir.iterdir():
        folds = sorted(mdir.glob("fold*/metrics.json"))
        if folds:
            res[mdir.name] = [json.loads(f.read_text()).get(args.decoder) for f in folds]
            res[mdir.name] = [r for r in res[mdir.name] if r is not None]
    names = sorted((m for m in res if res[m]), key=lambda m: ORDER.index(m) if m in ORDER else 99)
    print(f"\n### {setting_dir.name} ({args.decoder})\n")
    print("| Method | n | " + " | ".join(c for _, c in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for m in names:
        cells = []
        for k, _ in COLS:
            v = np.array([r[k] for r in res[m] if k in r])
            cells.append(f"{v.mean():.1f} ± {v.std():.1f}" if k not in ("mat_rho",) and len(v) else
                         (f"{v.mean():.3f}" if len(v) else "–"))
        print(f"| {m} | {len(res[m])} | " + " | ".join(cells) + " |")
    if "flat" in res and "hemahier" in res and len(res["flat"]) == len(res["hemahier"]) > 1:
        print("\npaired t-test HemaHier vs flat over folds:")
        for k in ("macro_f1", "rare_f1", "cle", "td"):
            a = [r[k] for r in res["hemahier"]]
            b = [r[k] for r in res["flat"]]
            t = ttest_rel(a, b)
            print(f"  {k}: diff {np.mean(a) - np.mean(b):+.2f}, p = {t.pvalue:.4f}")
