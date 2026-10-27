"""Index the extracted MLL folder into a CSV (path relative to root, fine class, lineage, chain, pi)."""
import argparse

from hemahier.data.mll import scan
from hemahier.ontology import Ontology

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/mll")
ap.add_argument("--ontology", default="configs/ontology/mll.yaml")
ap.add_argument("--out", default="data/mll_index.csv")
args = ap.parse_args()

ont = Ontology.from_yaml(args.ontology)
print(ont.describe())
df = scan(args.root, ont)
df.to_csv(args.out, index=False)
print(f"{len(df)} images -> {args.out}")
print(df.fine_name.value_counts().to_string())
