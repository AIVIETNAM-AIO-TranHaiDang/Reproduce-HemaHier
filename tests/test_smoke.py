"""End-to-end smoke test on synthetic data: fake JPGs -> index -> splits -> train/eval (random-init backbone)."""
import json

import numpy as np
import pytest
from PIL import Image

from hemahier.config import compose
from hemahier.data.mll import scan
from hemahier.data.splits import make_splits, save_splits
from hemahier.engine import run


@pytest.fixture
def fake_mll(tmp_path, ont):
    root = tmp_path / "mll"
    rng = np.random.default_rng(0)
    for c in ont.fine:
        d = root / c / "0001-1000"
        d.mkdir(parents=True)
        for k in range(10):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(d / f"{c}_{k:05d}.jpg")
    df = scan(root, ont)
    df.to_csv(tmp_path / "index.csv", index=False)
    save_splits(make_splits(df.fine.to_numpy(), n_folds=2, val_frac=0.45), tmp_path / "splits")
    return tmp_path


@pytest.mark.parametrize("model,setting", [("flat", "frozen"), ("hmil", "frozen"), ("hemahier", "frozen"),
                                           ("hemahier", "lora")])
def test_train_eval(fake_mll, model, setting):
    cfg = compose(model, setting, overrides=[
        f"data.root={fake_mll / 'mll'}", f"data.index_csv={fake_mll / 'index.csv'}",
        f"data.splits_dir={fake_mll / 'splits'}", f"out_dir={fake_mll / 'runs'}", "data.num_workers=0",
        "data.img_size=56", "backbone.timm_id=vit_small_patch14_dinov2", "backbone.pretrained=false",
        "backbone.img_size=56", "train.epochs=2", "train.batch_size=8", "train.amp=false",
    ])
    res = run(cfg)
    assert np.isfinite(res["argmax"]["macro_f1"])
    if model == "hemahier":
        assert "mat_rho" in res["argmax"] and "product" in res
    metrics = json.loads((fake_mll / "runs" / setting / model / "fold0" / "metrics.json").read_text())
    assert metrics["argmax"]["cle"] >= 0


def test_feature_cache_and_mlp_flat(fake_mll):
    cache = fake_mll / "features"
    cache.mkdir()
    import pandas as pd
    n = len(pd.read_csv(fake_mll / "index.csv"))
    rng = np.random.default_rng(0)
    np.save(cache / "h_l.npy", rng.standard_normal((n, 384)).astype(np.float16))
    np.save(cache / "h_m.npy", rng.standard_normal((n, 384)).astype(np.float16))
    for model in ("mlp_flat", "hemahier"):
        cfg = compose(model, "frozen", overrides=[
            f"data.index_csv={fake_mll / 'index.csv'}", f"data.splits_dir={fake_mll / 'splits'}",
            f"data.cache_dir={cache}", "data.use_cache=true", f"out_dir={fake_mll / 'runs'}",
            "data.num_workers=0", "train.epochs=2", "train.batch_size=8", "train.amp=false",
        ])
        res = run(cfg)
        assert np.isfinite(res["argmax"]["macro_f1"])
        assert ("mat_rho" in res["argmax"]) == (model == "hemahier")
