"""Cache frozen-backbone features h_L, h_M (fp16) for every indexed image, for fast frozen-head sweeps."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hemahier.config import compose
from hemahier.data.datasets import ImageDataset, build_transform
from hemahier.models.backbone import Backbone

ap = argparse.ArgumentParser()
ap.add_argument("overrides", nargs="*")
args = ap.parse_args()
cfg = compose("flat", "frozen", overrides=args.overrides)

df = pd.read_csv(cfg.data.index_csv)
ds = ImageDataset(cfg.data.root, df.path.tolist(), np.zeros(len(df), dtype=int), build_transform(False, cfg.data.img_size))
loader = DataLoader(ds, batch_size=256, num_workers=cfg.data.num_workers, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bb = Backbone(**{**cfg.backbone, "lora_rank": 0}).to(device).eval()

out = Path(cfg.data.cache_dir)
out.mkdir(parents=True, exist_ok=True)
h_l = np.lib.format.open_memmap(out / "h_l.npy", mode="w+", dtype=np.float16, shape=(len(df), bb.dim))
h_m = np.lib.format.open_memmap(out / "h_m.npy", mode="w+", dtype=np.float16, shape=(len(df), bb.dim))
i = 0
with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
    for batch in tqdm(loader):
        a, b = bb(batch["image"].to(device))
        n = len(a)
        h_l[i:i + n], h_m[i:i + n] = a.float().cpu().numpy(), b.float().cpu().numpy()
        i += n
h_l.flush(); h_m.flush()
print(f"saved {i} x {bb.dim} features to {out}")
