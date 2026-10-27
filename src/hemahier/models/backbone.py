"""DinoBloom-S backbone returning a late semantic token h_L and an intermediate morphology feature h_M."""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen, B initialised to zero."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.T @ self.lora_b.T) * self.scale


def load_dinobloom_checkpoint(path: str) -> nn.Module:
    """Load an official DinoBloom Zenodo checkpoint (DINOv2 format, keys under 'teacher')."""
    from timm.models.vision_transformer import checkpoint_filter_fn

    model = timm.create_model("vit_small_patch14_dinov2", pretrained=False, img_size=224, num_classes=0)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("teacher", ckpt)
    sd = {k.removeprefix("backbone."): v for k, v in sd.items() if "dino_head" not in k and "ibot_head" not in k}
    sd = checkpoint_filter_fn(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys when loading {path}: {missing}")
    if unexpected:
        print(f"[backbone] ignored unexpected keys: {unexpected}")
    return model


class Backbone(nn.Module):
    def __init__(self, timm_id: str | None = None, checkpoint: str | None = None, mid_block: int = 7,
                 lora_rank: int = 0, lora_alpha: float = 16, grad_checkpointing: bool = False,
                 pretrained: bool = True, img_size: int | None = None, **_):
        super().__init__()
        if checkpoint:
            self.vit = load_dinobloom_checkpoint(checkpoint)
        else:
            kw = {"img_size": img_size} if img_size else {}
            self.vit = timm.create_model(timm_id, pretrained=pretrained, num_classes=0, **kw)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.mid_block = mid_block
        self.lora = lora_rank > 0
        self.grad_checkpointing = grad_checkpointing and self.lora
        if self.lora:
            for blk in self.vit.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, lora_rank, lora_alpha)
                blk.attn.proj = LoRALinear(blk.attn.proj, lora_rank, lora_alpha)
        self.dim = self.vit.embed_dim
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.vit
        x = v.patch_embed(x)
        x = v._pos_embed(x)
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        h_m = None
        for i, blk in enumerate(v.blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if i == self.mid_block:
                h_m = x[:, self.num_prefix:].mean(dim=1)       # mean patch token: local morphology
        x = v.norm(x)
        return x[:, 0], h_m                                     # h_L = final CLS token

    def train(self, mode: bool = True):
        # frozen weights stay in eval mode (no dropout / drop-path); only LoRA adapters learn
        super().train(mode)
        self.vit.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
