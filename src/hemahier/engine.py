"""Training / evaluation loop shared by every prediction head."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from hemahier.config import run_dir
from hemahier.data.datasets import FeatureDataset, ImageDataset, build_sampler, build_transform
from hemahier.data.splits import load_split
from hemahier.decode import decode
from hemahier.metrics import evaluate
from hemahier.models import build_head
from hemahier.models.backbone import Backbone
from hemahier.ontology import Ontology


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset(idx: list[int], frac: float, seed: int) -> list[int]:
    if frac >= 1:
        return idx
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(idx, size=max(1, int(len(idx) * frac)), replace=False).tolist())


class Model(torch.nn.Module):
    def __init__(self, backbone: Backbone | None, head: torch.nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def features(self, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backbone is None:
            return batch["h_l"].to(device, non_blocking=True), batch["h_m"].to(device, non_blocking=True)
        x = batch["image"].to(device, non_blocking=True)
        if self.backbone.lora and self.training:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        names |= {n for n, _ in self.head.named_buffers()}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k in names or k.startswith("head.")}


def build_loaders(cfg: DictConfig, df: pd.DataFrame, split: dict, ont: Ontology):
    labels = df.fine.to_numpy()
    d = cfg.data
    if d.use_cache:
        cache = Path(d.cache_dir)
        h_l = np.load(cache / "h_l.npy", mmap_mode="r")
        h_m = np.load(cache / "h_m.npy", mmap_mode="r")
        make = lambda idx, train: FeatureDataset(h_l[idx], h_m[idx], labels[idx])
    else:
        paths = df.path.to_numpy()
        make = lambda idx, train: ImageDataset(d.root, paths[idx].tolist(), labels[idx],
                                               build_transform(train, d.img_size))
    loaders = {}
    for name in ("train", "val", "test"):
        idx = np.asarray(split[name])
        ds = make(idx, name == "train")
        sampler = build_sampler(labels[idx], d.sampler, ont.num_fine, cfg.seed) if name == "train" else None
        loaders[name] = DataLoader(ds, batch_size=cfg.train.batch_size, sampler=sampler,
                                   shuffle=(name == "train" and sampler is None), drop_last=(name == "train"),
                                   num_workers=d.num_workers, pin_memory=True,
                                   persistent_workers=d.num_workers > 0)
    return loaders


@torch.no_grad()
def predict(model: Model, loader: DataLoader, device, amp_dtype) -> dict:
    model.eval()
    probs, mats, ys = [], [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            h_l, h_m = model.features(batch, device)
            out = model.head(h_l, h_m)
        probs.append(out["log_probs"].float().exp().cpu())
        if getattr(model.head, "predicts_maturity", False):
            mats.append(out["maturity"].float().cpu())
        ys.append(batch["y"])
    return {"probs": torch.cat(probs).numpy(), "y": torch.cat(ys).numpy(),
            "maturity": torch.cat(mats).numpy() if mats else None}


def run(cfg: DictConfig) -> dict:
    seed_everything(cfg.seed + cfg.fold)
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    ont = Ontology.from_yaml(cfg.ontology)
    df = pd.read_csv(cfg.data.index_csv)
    if not set(df.fine_name) <= set(ont.fine):
        raise ValueError("index CSV classes do not match ontology; re-run scripts/prepare_mll.py")
    df["fine"] = df.fine_name.map(ont.index)                      # ontology order may differ from the CSV
    counts = np.bincount(df.fine, minlength=ont.num_fine)
    rare = [c for c in range(ont.num_fine) if counts[c] < cfg.data.rare_threshold]

    split = load_split(cfg.data.splits_dir, cfg.fold)
    split = {k: subset(v, cfg.data.subset_frac, cfg.seed) for k, v in split.items()}
    loaders = build_loaders(cfg, df, split, ont)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if cfg.train.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if cfg.data.use_cache:
        if cfg.backbone.lora_rank > 0:
            raise ValueError("cached features only make sense for a frozen backbone")
        backbone, dim = None, int(np.load(Path(cfg.data.cache_dir) / "h_l.npy", mmap_mode="r").shape[1])
    else:
        backbone = Backbone(**OmegaConf.to_container(cfg.backbone))
        dim = backbone.dim
    head = build_head(OmegaConf.to_container(cfg.model), ont, dim, dim)
    model = Model(backbone, head).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head.parameters())
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    print(f"[run] {out} | head {n_head / 1e6:.2f}M | lora {n_lora / 1e6:.2f}M | rare classes "
          f"{[ont.fine[c] for c in rare]}")

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total, warm = cfg.train.epochs * steps_per_epoch, cfg.train.warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(cfg.train.epochs):
        model.train()
        t0, running = time.time(), {}
        for step, batch in enumerate(tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)):
            y = batch["y"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                h_l, h_m = model.features(batch, device)
                o = model.head(h_l, h_m)
            loss, logs = model.head.loss(o, y, epoch + step / steps_per_epoch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v / steps_per_epoch

        rec = {"epoch": epoch, "time": time.time() - t0, **{f"loss_{k}": v for k, v in running.items()}}
        if (epoch + 1) % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs:
            p = predict(model, loaders["val"], device, amp_dtype)
            val = evaluate(p["y"], p["probs"].argmax(1), ont, rare, p["maturity"])
            rec.update({f"val_{k}": val[k] for k in ("macro_f1", "acc", "cle", "td") if k in val})
            if val["macro_f1"] > best_f1:
                best_f1, best_state = val["macro_f1"], model.trainable_state()
                rec["best"] = True
        history.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}))

    model.load_state_dict(best_state, strict=False)
    p = predict(model, loaders["test"], device, amp_dtype)
    results = {"argmax": evaluate(p["y"], decode(p["probs"], p["maturity"], ont, "argmax"), ont, rare, p["maturity"])}
    if p["maturity"] is not None:
        pred = decode(p["probs"], p["maturity"], ont, "product", cfg.decode_sigma)
        results["product"] = evaluate(p["y"], pred, ont, rare, p["maturity"])
    results["n_test"] = int(len(p["y"]))
    results["head_params"] = n_head
    results["lora_params"] = n_lora

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    np.savez_compressed(out / "predictions.npz", y=p["y"], probs=p["probs"],
                        maturity=p["maturity"] if p["maturity"] is not None else np.zeros((0,)),
                        idx=np.asarray(split["test"]))
    torch.save(best_state, out / "best_trainable.pt")
    print(f"[test] {json.dumps({k: round(v, 3) for k, v in results['argmax'].items()})}")
    return results
