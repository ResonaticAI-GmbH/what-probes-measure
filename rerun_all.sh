#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
# 1) Finding 2 mit neuen Dumps
for s in "Qwen3-4B:runs/features_final_4b:runs/final/4b_probe_s1.npy" \
         "Qwen3-8B:runs/features_final_8b:runs/final/8b_probe_s1.npy" \
         "Qwen3.6-35B:runs/features_final_36b:runs/final/36b_probe_s1.npy" \
         "Mistral-7B:runs/features_newfam_mistral:runs/newfamily/mistral_probe_s1.npy" \
         "OLMo-2-7B:runs/features_newfam_olmo:runs/newfamily/olmo_probe_s1.npy"; do
  tag=${s%%:*}; rest=${s#*:}; base=${rest%%:*}; sc=${rest#*:}
  $PY incremental_value.py --features ${base}.json --scores $sc --tag "$tag"
done > runs/incremental.log 2>&1
echo "[1/3] incremental fertig"
# 2) Konflikt-Test Slot 1 + Positionskontrolle
for s in "4b:runs/features_final_4b" "8b:runs/features_final_8b" "36b:runs/features_final_36b" \
         "mistral:runs/features_newfam_mistral" "olmo:runs/features_newfam_olmo"; do
  tag=${s%%:*}; base=${s#*:}
  echo "######## $tag ANTWORT-Slot1"
  $PY conflict_test.py --features ${base}.json --hidden ${base}.ahidden.npy --layer 1 --probe diffmean
  echo "######## $tag PTRUE-POSITION"
  $PY conflict_test.py --features ${base}.json --hidden ${base}.hidden.npy --probe diffmean
done > runs/conflict_recheck.log 2>&1
echo "[2/3] konflikt fertig"
# 3) Finding 3: Varianten (wandernd) + eingefroren
for m in olmo mistral; do
  base=runs/features_newfam_$m
  for v in de_orig en vague strict; do
    echo "######## MOVING $m/$v"
    $PY conflict_test.py --features runs/ptrue_var/${m}_${v}.json --hidden ${base}.ahidden.npy --layer 1 --probe diffmean
    echo "######## FROZEN $m/$v"
    $PY conflict_test.py --features runs/ptrue_var/${m}_${v}.json --hidden ${base}.ahidden.npy --layer 1 --probe diffmean --split-from runs/ptrue_var/${m}_de_orig.json
  done
done > runs/finding3_recheck.log 2>&1
echo "[3/3] finding3 fertig"
