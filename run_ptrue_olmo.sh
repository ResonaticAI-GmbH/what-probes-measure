#!/usr/bin/env bash
# C3' auf ein zweites Modell ausweiten: Varianten-Sweep + Konflikt-Test fuer OLMo-2-7B.
# OLMo ist der natuerliche zweite Fall, weil es die Within-Model-Dissoziation aus 30.1 zeigt
# (popqa P(True) 0.717 -> Hyp. B; webq 0.519 -> Hyp. A, rho +0.00).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
BASE=runs/features_newfam_olmo
$PY ptrue_variants.py --features "${BASE}.json" --model allenai/OLMo-2-1124-7B \
    --outdir runs/ptrue_var --tag olmo --batch-size 24
for v in de_orig de_raw en en_raw vague strict; do
  echo "######## KONFLIKT olmo/$v"
  $PY conflict_test.py --features "runs/ptrue_var/olmo_${v}.json" \
      --hidden "${BASE}.ahidden.npy" --layer 1 --probe diffmean
done
echo "[fertig] $(date '+%d.%m %H:%M')"
