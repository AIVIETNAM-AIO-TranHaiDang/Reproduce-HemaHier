# Reproduction notes

## Targets (paper Table 1, MLLv1, 5-fold mean)
| Setting | Method | Acc | B-Acc | M-F1 | Rare | CLE↓ | TD↓ | Mat-ρ |
|---|---|---|---|---|---|---|---|---|
| Frozen | Flat | 81.5 | 76.5 | 63.3 | 38.6 | 12.7 | 16.1 | – |
| Frozen | HMIL* | 78.1 | 75.9 | 55.0 | 17.1 | 14.9 | 18.9 | – |
| Frozen | HemaHier | 84.2 | 75.7 | 72.2 | 59.1 | 10.6 | 13.7 | 0.947 |
| LoRA | Flat | 86.9 | 79.2 | 76.3 | 65.0 | 8.8 | 11.4 | – |
| LoRA | HMIL* | 85.9 | 79.8 | 74.2 | 60.9 | 9.6 | 12.3 | – |
| LoRA | HemaHier | 87.4 | 79.1 | 77.9 | 69.2 | 8.4 | 11.0 | 0.959 |

Sanity check đầu tiên: Flat frozen phải ra M-F1 khoảng 63. Nếu lệch nhiều thì lỗi nằm ở pipeline dữ liệu hoặc backbone, chưa liên quan đến method.

## Taken from the paper
224×224, AdamW (wd 1e-4), batch 64, 20 epochs, cosine + linear warmup, lr 3e-4, LoRA rank 16, λ_c = 0.05, λ_m = 0.005, λ_r = 0.003 ramped in after warmup, auxiliary fine CE weight increased under LoRA, model selection by validation macro-F1 under the hierarchy-consistent posterior, head ≈ 5.0M params (ours 5.17M), maturity part < 0.25M (ours 0.20M), LoRA 0.44M (ours 0.442M), backbone 21.6M (ours 21.63M).

## Assumptions (not stated in the paper); all configurable
| Item | Choice | Where |
|---|---|---|
| MLL → ontology mapping | see `configs/ontology/mll.yaml`; BLA = own lineage, off-chain | ontology yaml |
| BLA alternative | head of neutrophil chain | `mll_bla_in_chain.yaml` |
| TD constants | same chain 0.5·\|Δπ\|, same lineage 0.5, cross-lineage 1.0; reported ×100 | `td:` in ontology yaml |
| Rare classes | total count < 1000 | `data.rare_threshold` |
| h_M | mean patch token after block 8/12 | `backbone.mid_block` |
| Adapters | LN-Linear-GELU-Dropout-Linear-GELU, hidden 1536, out 1024 | `model/hemahier.yaml` |
| Maturity output | sigmoid → [0, 1] | `HemaHierHead` |
| Ranking margin δ, reduction | 0.1, mean over valid in-batch pairs | `model.margin`, `model.rank_reduction` |
| λ_raw | 0.1 frozen, 0.3 LoRA | `model/hemahier.yaml`, `setting/lora.yaml` |
| λ(t) schedule | linear ramp over 2 epochs starting at epoch 1 (lineage), 3 (maturity), 5 (ranking) | `start_*`, `ramp` |
| Validation | 10 % stratified from each fold's training part | `scripts/make_splits.py` |
| Sampler | sqrt-inverse-frequency, shared by all methods | `data.sampler` |
| Augmentation | RandomResizedCrop(0.8–1), flips, rot90, mild color jitter | `data/datasets.py` |
| Maturity read-out at test | Mat-ρ / Ord use the true chain on healthy on-chain cells, weighted mean over chains | `metrics.maturity_quality` |
| Product decoder | Gaussian likelihood N(s_q; π_f, σ=0.15), mass moved only within a chain | `decode.py` |
| HMIL* | shared MLP, coarse + fine heads, CE both levels + KL consistency | `models/heads/hmil.py` |

## Extra baseline (not in the paper)
`mlp_flat`: same ~5M-param MLP on [h_L; h_M] with a plain softmax. The paper compares a ~5M head against a linear probe, so part of the gain may be capacity rather than structure. This baseline checks that.

## Expected differences
- TD absolute values depend on the constants above; compare methods relative to each other.
- Public MLL splits differ across works; the paper re-trains all methods on its own 5-fold split, as we do. Expect ± 1–2 points on M-F1 and larger variance on Rare-F1 (ABE has 8 images).
