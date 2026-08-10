#!/usr/bin/env bash
# Seed-Robustheit: zweite Itemstichprobe (Seed 1), sonst identisches Setup.
#
# Motiv: alle Zahlen im Paper stammen aus Seed 0. Die Bootstrap-CIs quantifizieren die
# Unsicherheit GEGEBEN diese Stichprobe, nicht die Unsicherheit UEBER Stichproben. Die
# Limitations gestehen das ein ("one extraction per configuration"); dieser Lauf misst es.
#
# Zwei Modelle genuegen als Stichprobe: 8B traegt die Positionskontrolle (9/9 Kollaps),
# Mistral traegt Finding 3 (Instrumenten-Sensitivitaet, 0.254 Spanne).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
N=${N:-6000}
SRC=popqa+webq+triviaqa
mkdir -p runs/seed1
for spec in "8b:Qwen/Qwen3-8B:24" "mistral:mistralai/Mistral-7B-v0.3:24"; do
  tag=${spec%%:*}; rest=${spec#*:}; model=${rest%:*}; bs=${rest##*:}
  echo "######## SEED1 $tag ($model) $(date +%H:%M:%S)"
  $PY extract_features.py --source "$SRC" --n "$N" --model "$model" --seed 1 --shots 5 \
      --raw-answer --no-reason --no-lora --with-hidden --with-answer-hidden --answer-pool last \
      --batch-size "$bs" --out "runs/seed1/${tag}.json" 2>&1 | tail -8
done
# Analysen: Antwort-Position Slot 1 + Positionskontrolle, wie im Hauptlauf
for tag in 8b mistral; do
  echo "######## SEED1-ANALYSE $tag ANTWORT-Slot1"
  $PY conflict_test.py --features runs/seed1/${tag}.json \
      --hidden runs/seed1/${tag}.ahidden.npy --layer 1 --probe diffmean
  echo "######## SEED1-ANALYSE $tag PTRUE-POSITION"
  $PY conflict_test.py --features runs/seed1/${tag}.json \
      --hidden runs/seed1/${tag}.hidden.npy --probe diffmean
done
echo "[fertig] $(date '+%d.%m %H:%M')"
