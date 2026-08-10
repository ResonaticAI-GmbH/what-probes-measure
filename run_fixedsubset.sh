#!/usr/bin/env bash
# Fixed-Subset-Kontrolle (Run 34): Konflikt-Teilmenge EINMAL aus der Baseline-Variante
# (de_orig) bestimmen, einfrieren, dann nur P(True) tauschen. Trennt Kompositions- von
# Verhaltensaenderung. Vorhersage: bei fixer Probe UND fixer Teilmenge muss die
# Konflikt-AUROC exakt konstant sein -- dann war die Bewegung in Run 32 zu 100% Komposition.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
for tag_base in "mistral:runs/features_newfam_mistral"; do
  tag=${tag_base%%:*}; base=${tag_base#*:}
  for v in de_orig en vague strict; do
    echo "######## FIXSUB $tag/$v"
    $PY conflict_test.py --features "runs/ptrue_var/${tag}_${v}.json" \
        --hidden "${base}.ahidden.npy" --layer 1 --probe diffmean \
        --split-from "runs/ptrue_var/${tag}_de_orig.json"
  done
done
echo "[fertig]"
