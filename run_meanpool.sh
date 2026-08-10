#!/usr/bin/env bash
# Pooling-Achse der Faktorstudie: dieselbe Extraktion, nur --answer-pool mean.
#
# Warum voll extrahieren statt nur den Forward-Pass zu wiederholen? Der Versuch steht in
# legacy/repool_hidden_FAILED.py: die Few-Shot-Praefixe stehen nicht im Feature-Cache, und
# ihre Rekonstruktion aus der Quelle ergibt zwar dieselbe ITEM-MENGE, aber eine andere
# REIHENFOLGE (build_source verbraucht Zufall anders). Damit ist nicht rekonstruierbar,
# welche fuenf Items je Task in welcher Reihenfolge im Praefix standen -- und ohne exakt
# denselben Kontext waere der Vergleich last-vs-mean nicht ceteris paribus.
# Die volle Extraktion ist deterministisch (Seed 0) und wird danach gegen den bestehenden
# Cache geprueft: Fragen, Antworten und Labels muessen Item fuer Item uebereinstimmen.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
N=${N:-6000}
SRC=popqa+webq+triviaqa
mkdir -p runs/meanpool
for spec in "4b:Qwen/Qwen3-4B:24:" "8b:Qwen/Qwen3-8B:24:" "36b:Qwen/Qwen3.6-35B-A3B:20:--suppress-think" \
            "mistral:mistralai/Mistral-7B-v0.3:24:" "olmo:allenai/OLMo-2-1124-7B:24:"; do
  tag=$(echo "$spec" | cut -d: -f1); model=$(echo "$spec" | cut -d: -f2)
  bs=$(echo "$spec" | cut -d: -f3);  think=$(echo "$spec" | cut -d: -f4)
  echo "######## MEANPOOL $tag ($model) $(date +%H:%M:%S)"
  $PY extract_features.py --source "$SRC" --n "$N" --model "$model" --seed 0 --shots 5 \
      --raw-answer $think --no-reason --no-lora --with-answer-hidden --answer-pool mean \
      --batch-size "$bs" --out "runs/meanpool/${tag}.json" 2>&1 | tail -6
done
echo "[fertig] $(date '+%d.%m %H:%M')"
