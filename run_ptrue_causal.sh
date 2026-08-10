#!/usr/bin/env bash
# Kausaler Test der Kollaps-Bedingung (Run 30.1). Kontrastpaar:
#   8b      — starkes P(True) (0.797/0.826/0.671), kollabiert bei Qwen zuverlässig
#   mistral — schwaches P(True) (0.430/0.618/0.584), kollabiert nicht
# Sechs Prompt-Varianten je Modell, danach der Konflikt-Test je Variante auf
# UNVERAENDERTEN Hidden States (Slot 1) und mit derselben Probe.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
mkdir -p runs/ptrue_var

for spec in "8b:Qwen/Qwen3-8B:runs/features_final_8b" \
            "mistral:mistralai/Mistral-7B-v0.3:runs/features_newfam_mistral"; do
  tag=$(echo "$spec" | cut -d: -f1); model=$(echo "$spec" | cut -d: -f2); base=$(echo "$spec" | cut -d: -f3)
  echo "######## VARIANTEN $tag  $(date +%H:%M:%S)"
  $PY ptrue_variants.py --features "${base}.json" --model "$model" \
      --outdir runs/ptrue_var --tag "$tag" --batch-size 24
  for v in de_orig de_raw en en_raw vague strict; do
    echo "######## KONFLIKT $tag/$v"
    # Hidden States und Probe sind identisch zum Hauptlauf — nur ptrue_direct unterscheidet sich.
    $PY conflict_test.py --features "runs/ptrue_var/${tag}_${v}.json" \
        --hidden "${base}.ahidden.npy" --layer 1 --probe diffmean
  done
done
echo "[fertig] $(date '+%d.%m %H:%M')"
