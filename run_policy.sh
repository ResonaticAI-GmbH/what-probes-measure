#!/usr/bin/env bash
# Entscheidungsqualität statt Rangordnung, alle fünf Modelle bei Schicht-Slot 1.
# Schritt 1: OOF-Probe-Scores je Item dumpen. Schritt 2: Policy auf ptrue/probe/combo.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
L=${L:-1}
for spec in "4b:runs/features_final_4b:runs/final" "8b:runs/features_final_8b:runs/final" \
            "36b:runs/features_final_36b:runs/final" \
            "mistral:runs/features_newfam_mistral:runs/newfamily" \
            "olmo:runs/features_newfam_olmo:runs/newfamily"; do
  tag=$(echo "$spec" | cut -d: -f1); base=$(echo "$spec" | cut -d: -f2); out=$(echo "$spec" | cut -d: -f3)
  $PY probe_equivalence.py --features "${base}.json" --hidden "${base}.ahidden.npy" \
      --layer "$L" --probe diffmean --dump-scores "${out}/${tag}_probe_s${L}.npy" >/dev/null
  echo "######## $tag"
  $PY policy_compare.py --features "${base}.json" --scores "${out}/${tag}_probe_s${L}.npy" \
      --tag "${tag} (Slot ${L})"
done
