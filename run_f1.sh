#!/usr/bin/env bash
# Klassifikationsguete Probe vs. P(True), alle fuenf Modelle bei Schicht-Slot 1.
# Setzt die Probe-Score-Dumps aus run_policy.sh voraus (*_probe_s1.npy).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"
for spec in "4b:runs/features_final_4b:runs/final/4b_probe_s1.npy" \
            "8b:runs/features_final_8b:runs/final/8b_probe_s1.npy" \
            "36b:runs/features_final_36b:runs/final/36b_probe_s1.npy" \
            "mistral:runs/features_newfam_mistral:runs/newfamily/mistral_probe_s1.npy" \
            "olmo:runs/features_newfam_olmo:runs/newfamily/olmo_probe_s1.npy"; do
  tag=$(echo "$spec" | cut -d: -f1); base=$(echo "$spec" | cut -d: -f2); sc=$(echo "$spec" | cut -d: -f3)
  echo "######## $tag"
  $PY f1_compare.py --features "${base}.json" --scores "$sc" --tag "$tag"
done
