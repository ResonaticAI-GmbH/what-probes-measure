#!/usr/bin/env bash
# Run 27: Neuextraktion im STANDARD-Setup der Benchmarks — Roh-Fortsetzung ohne Chat-Template,
# Template und Metrik je Datensatz (PopQA "Q:/A:"+Substring, TriviaQA/WebQ "Question:/Answer:"+EM),
# 5-shot mit festen Exemplaren aus dem eigenen Pool (aus der Evaluation entfernt).
# aller Modelle nach dem Fix von Confound #5 (deutsche Antworten vs.
# englisches Gold) und mit fixem Seed. Erzeugt den durchgehend vergleichbaren Datensatz über
# drei Modellgrößen, auf dem Run 23–26 wiederholt werden.
#   Qwen3.6-35B-A3B laufe als LETZTES: die beiden kleinen Modelle liefern schon nach ~2 h ein
#   verwertbares Ergebnis, falls beim 35B etwas schiefgeht.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"

N=${N:-6000}
SRC=popqa+webq+triviaqa
mkdir -p runs/final

# tag:modell:batch — bs beim 35B-MoE kleiner (70 GB Gewichte resident)
# Letztes Feld: --suppress-think? NUR Qwen3.6 denkt in der Rohfortsetzung spontan; bei
# Qwen3-4B/8B zerstört der leere Think-Block die Antworten (results.md Run 27).
for spec in "4b:Qwen/Qwen3-4B:24:" "8b:Qwen/Qwen3-8B:24:" "36b:Qwen/Qwen3.6-35B-A3B:20:--suppress-think"; do
  tag=$(echo "$spec" | cut -d: -f1); model=$(echo "$spec" | cut -d: -f2)
  bs=$(echo "$spec" | cut -d: -f3);  think=$(echo "$spec" | cut -d: -f4)
  echo "######## $tag  ($model)  n=$N bs=$bs  $(date +%H:%M:%S)"
  $PY extract_features.py \
      --source "$SRC" --n "$N" --model "$model" --seed 0 --shots 5 --raw-answer $think \
      --no-reason --no-lora --with-hidden --with-answer-hidden --answer-pool last \
      --batch-size "$bs" --out "runs/features_final_${tag}.json" \
      2>&1 | tee "runs/final/${tag}_extract.log"
done

# Analysen (modellfrei, Minuten) — Run 25 + Run 26 auf sauberen Labels
for tag in 4b 8b 36b; do
  F=runs/features_final_${tag}.json
  AH=runs/features_final_${tag}.ahidden.npy
  for L in 0 1 2 3; do
    echo "######## $tag Schicht-Slot $L"
    $PY conflict_test.py     --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
    $PY probe_equivalence.py --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
    $PY combo_test.py        --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
  done 2>&1 | tee "runs/final/${tag}_layersweep.txt"
  # Kontrollen an der ALTEN P(True)-Position (doppelte Dissoziation, Run 26)
  { $PY conflict_test.py --features "$F" --hidden "runs/features_final_${tag}.hidden.npy" --probe diffmean
    $PY combo_test.py    --features "$F" --hidden "runs/features_final_${tag}.hidden.npy" --probe diffmean
  } 2>&1 | tee "runs/final/${tag}_ptrue_position_control.txt"
done
echo "[fertig] $(date +%H:%M:%S) — Ergebnisse in runs/final/"
