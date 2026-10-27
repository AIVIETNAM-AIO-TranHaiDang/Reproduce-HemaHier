"""Config composition: default.yaml <- model/<m>.yaml <- setting/<s>.yaml <- [ablation] <- key=value overrides."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def compose(model: str, setting: str, ablation: str | None = None, overrides: list[str] | None = None) -> DictConfig:
    parts = [CONFIG_DIR / "default.yaml", CONFIG_DIR / "model" / f"{model}.yaml",
             CONFIG_DIR / "setting" / f"{setting}.yaml"]
    if ablation:
        parts.append(CONFIG_DIR / "ablation" / f"{ablation}.yaml")
    cfg = OmegaConf.merge(*[OmegaConf.load(p) for p in parts], OmegaConf.from_dotlist(overrides or []))
    cfg.model_tag = cfg.get("tag") or model
    return cfg


def run_dir(cfg: DictConfig) -> Path:
    return Path(cfg.out_dir) / cfg.setting / cfg.model_tag / f"fold{cfg.fold}"
