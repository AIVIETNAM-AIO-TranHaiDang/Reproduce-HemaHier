"""Train + evaluate one (model, setting, fold).

  python scripts/train.py --model hemahier --setting frozen --fold 0 [--ablation full] [key=value ...]
"""
import argparse

from hemahier.config import compose
from hemahier.engine import run

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["flat", "multiscale_flat", "mlp_flat", "hmil", "hemahier"])
ap.add_argument("--setting", default="frozen", choices=["frozen", "lora"])
ap.add_argument("--ablation", default=None)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. train.epochs=1 data.subset_frac=0.02")
args = ap.parse_args()

cfg = compose(args.model, args.setting, args.ablation, [f"fold={args.fold}", *args.overrides])
run(cfg)
