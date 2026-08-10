#!/usr/bin/env bash
# Faktorstudie, Pooling-Achse: last (Hauptlauf) vs. mean, Schicht-Slot 1, sonst identisch.
# Items/Antworten/Labels/P(True) sind gegen den Hauptlauf geprueft (5985/5985 bei allen fuenf).
set -euo pipefail
cd "$(dirname "$0")"; export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
for tag in 4b 8b 36b mistral olmo; do
  echo "######## MEAN $tag"
  $PY conflict_test.py --features runs/meanpool/${tag}.json \
      --hidden runs/meanpool/${tag}.ahidden.npy --layer 1 --probe diffmean
done
