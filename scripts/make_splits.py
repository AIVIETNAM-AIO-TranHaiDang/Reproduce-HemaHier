"""Write shared 5-fold stratified splits (indices into the index CSV); every method reuses them."""
import argparse

import pandas as pd

from hemahier.data.splits import make_splits, save_splits

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/mll_index.csv")
ap.add_argument("--out", default="data/splits")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

labels = pd.read_csv(args.index).fine_name.astype("category").cat.codes.to_numpy()
folds = make_splits(labels, args.folds, args.val_frac, args.seed)
save_splits(folds, args.out)
for k, f in enumerate(folds):
    print(f"fold{k}: " + ", ".join(f"{n}={len(v)}" for n, v in f.items()))
