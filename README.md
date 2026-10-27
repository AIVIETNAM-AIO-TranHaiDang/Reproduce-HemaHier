# HemaHier reproduction (MLLv1)

Reproduce *HemaHier: Chain-Conditioned Ordinal Hierarchies for Lineage-Aware Bone-Marrow Cytology* (MICCAI) trên bộ public MLLv1/BMC. UKA1/UKA2 là dữ liệu nội bộ nên không có trong repo này.

## Cài đặt
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision            # bản CUDA phù hợp với server
uv pip install -e ".[dev]"
pytest -q                                    # 16 test, chạy CPU khoảng 15s, không cần dữ liệu thật
```

## Dữ liệu
1. Tải gói JPG khoảng 6.8GB của TCIA collection **Bone-Marrow-Cytomorphology_MLL_Helmholtz_Fraunhofer**, giải nén sao cho `data/raw/mll/<CLASS>/...jpg` (21 thư mục lớp).
2. Tạo index và split dùng chung:
```bash
python scripts/prepare_mll.py --root data/raw/mll          # -> data/mll_index.csv
python scripts/make_splits.py                               # -> data/splits/fold{0..4}.json
```
Backbone DinoBloom-S mặc định tải từ mirror timm `1aurent/vit_small_patch14_224.dinobloom` (đã kiểm tra: 21.6M tham số, LoRA r16 = 0.44M, khớp paper). Muốn dùng checkpoint Zenodo chính thức thì thêm `backbone.checkpoint=weights/DinoBloom-S.pth`.

## Chạy
```bash
# một run
python scripts/train.py --model hemahier --setting frozen --fold 0
python scripts/train.py --model hemahier --setting lora --fold 0
# toàn bộ Table 1 + ablation Table 2 (5 fold)
./scripts/run_cv.sh data.num_workers=8
# tổng hợp kết quả và hình
python scripts/aggregate.py                  # bảng mean ± std + paired t-test
python scripts/aggregate.py --decoder product
python scripts/plot_errors.py                # Fig. 2
python scripts/plot_maturity.py              # Fig. 3
```
Override bất kỳ key nào dạng `key=value`, ví dụ smoke test: `train.epochs=1 data.subset_frac=0.02`.

Sweep nhanh ở setting frozen bằng feature cache (không có augmentation, nên không dùng cho kết quả chính):
```bash
python scripts/extract_features.py           # -> data/features/h_{l,m}.npy
python scripts/train.py --model hemahier --setting frozen data.use_cache=true
```

## Bản đồ code ↔ paper
| Paper | Code |
|---|---|
| Ontology, chain C_q, π = j/n_q, off-chain, TD | `src/hemahier/ontology.py`, `configs/ontology/mll.yaml` |
| h_L, h_M từ backbone, LoRA | `src/hemahier/models/backbone.py` |
| z_c = φ_c(h_L), z_d = φ_d([h_L; h_M]) | `HemaHierHead.forward` |
| Eq. 2 posterior nhất quán | `coupled_log_posterior` (`models/heads/hemahier.py`) |
| s_q(x) = g(z_c, z_d, e_q) | `chain_emb` + `maturity_head` |
| Eq. 3–4, λ(t) theo giai đoạn | `HemaHierHead.loss`, `losses.py` |
| Bayesian product decoder | `decode.py` |
| Acc/B-Acc/M-F1/W-F1/Rare, CLE, TD, Mat-ρ, Ord | `metrics.py` |
| Flat, HMIL*, multi-scale flat | `models/heads/flat.py`, `hmil.py`, `configs/model/` |
| Ablation Table 2 | `configs/ablation/` |

Output của mỗi run nằm ở `runs/<setting>/<model>/fold<k>/` (`metrics.json`, `history.json`, `predictions.npz`, `best_trainable.pt`).

Các giả định do paper không nêu: xem `REPRODUCTION_NOTES.md`.
