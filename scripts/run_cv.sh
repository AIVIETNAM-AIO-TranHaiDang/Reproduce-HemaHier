#!/usr/bin/env bash
# Full reproduction: Table 1 (3 methods x 2 settings x 5 folds) and Table 2 ablation (frozen).
# Extra args are forwarded as overrides, e.g. ./scripts/run_cv.sh data.num_workers=4
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDS=${FOLDS:-"0 1 2 3 4"}

for setting in frozen lora; do
  for model in flat hmil hemahier; do
    for k in $FOLDS; do
      python scripts/train.py --model $model --setting $setting --fold $k "$@"
    done
  done
done

# Table 2: flat / multi-scale flat / +factorized+lineage / +branch maturity / +ranking (= full HemaHier)
for k in $FOLDS; do
  python scripts/train.py --model multiscale_flat --setting frozen --fold $k "$@"
  python scripts/train.py --model mlp_flat --setting frozen --fold $k "$@"
  for abl in factorized_lineage branch_maturity; do
    python scripts/train.py --model hemahier --setting frozen --ablation $abl --fold $k "$@"
  done
done
