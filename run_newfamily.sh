#!/usr/bin/env bash
# Run 28: PRODUKTIONSLAUF Mistral-7B-v0.3 + OLMo-2-7B — die Familien-Lücke, die Qwen3.6-35B
# NICHT schliesst (results.md Run 27, letzter Absatz). Setup identisch zu run_final.sh, damit
# die Ergebnisse direkt neben 4B/8B/36B stehen können.
#
# Freigabe durch die Diagnose vom 30.07. 14:29 (runs/newfamily_smoke.log), n=385:
#   - Extraktion sauber bei beiden: leer=0/385, <think>-Reste=0, Median 1-2 Wörter.
#   - acc gesamt: Mistral 0.403, OLMo 0.522 — plausibel fürs Basismodell, kein Formatbruch.
#   - KEIN --suppress-think: beide Modelle haben kein Chat-Template und denken nicht spontan.
#
# Zwei Punkte, die aus der Diagnose mitgenommen werden müssen:
#
#  1) Die A/B/C/D-Tafel wird über den MEDIAN je Task geschwellt, nie über fix 0.5. Mistral liegt
#     mit P(True) zu 96 % über 0.5, OLMo zu 1 % darunter — eine absolute Schwelle lässt je nach
#     Modell Zelle B oder C leerlaufen und misst dann Kalibrierungs-Offset statt Konflikt. Bei
#     0.5 fiel jede der sechs Task x Modell-Zellen unter --min_cell und wäre übersprungen worden;
#     mit Median tragen alle sechs (kleinste Zelle 15-26 bei n=385, entsprechend mehr bei n=6000).
#     conflict_test.py macht das per Default (--threshold -1), extract_features.py seit 30.07. auch.
#
#  2) OLMos P(True) liegt mit AUROC 0.45-0.54 am ZUFALL (Mistral: 0.61-0.68). Das ist Rangordnung,
#     nicht Offset, und wird vom Median NICHT behoben. Für den Konflikt-Test ist das kein
#     Ausschluss — er fragt ja gerade, ob die Probe anders trennt als das Selbsturteil. Aber
#     combo_test.py hat bei OLMo wenig zu kombinieren; ein schwaches Combo-Ergebnis dort ist
#     erwartet und kein Fehler.
#
# Laufzeit-Schätzung aus der Diagnose: Mistral 226 s / OLMo 98 s bei n=385 -> bei n=6000 grob
# 60 min bzw. 25 min, plus Analysen. GPU-Peak lag bei 15.5 / 17.2 GB, bs=24 ist reichlich sicher.
#
# Jobs NICHT über pgrep-Namensmuster synchronisieren (kostete am 30.07. sieben Stunden Leerlauf:
# das Muster traf die eigenen Wächter-Prozesse, der Job wartete auf sich selbst). Wenn dieser
# Lauf auf freie GPU warten soll, dann über freien Speicher, nicht über einen Prozessnamen.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"

N=${N:-6000}
SRC=popqa+webq+triviaqa
mkdir -p runs/newfamily

for spec in "mistral:mistralai/Mistral-7B-v0.3:24" "olmo:allenai/OLMo-2-1124-7B:24"; do
  tag=$(echo "$spec" | cut -d: -f1); model=$(echo "$spec" | cut -d: -f2); bs=$(echo "$spec" | cut -d: -f3)
  echo "######## $tag  ($model)  n=$N bs=$bs  $(date +%H:%M:%S)"
  $PY extract_features.py \
      --source "$SRC" --n "$N" --model "$model" --seed 0 --shots 5 --raw-answer \
      --no-reason --no-lora --with-hidden --with-answer-hidden --answer-pool last \
      --batch-size "$bs" --out "runs/features_newfam_${tag}.json" \
      2>&1 | tee "runs/newfamily/${tag}_extract.log"
done

# Analysen (modellfrei, Minuten) — identisch zu run_final.sh
for tag in mistral olmo; do
  F=runs/features_newfam_${tag}.json
  AH=runs/features_newfam_${tag}.ahidden.npy
  for L in 0 1 2 3; do
    echo "######## $tag Schicht-Slot $L"
    $PY conflict_test.py     --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
    $PY probe_equivalence.py --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
    $PY combo_test.py        --features "$F" --hidden "$AH" --layer "$L" --probe diffmean
  done 2>&1 | tee "runs/newfamily/${tag}_layersweep.txt"
  # Kontrollen an der ALTEN P(True)-Position — bei Qwen 9/9 Zellen "folgt SELBSTURTEIL" mit
  # rho +0.97..+0.99. Reproduziert sich das familienübergreifend, ist Confound #4 als
  # allgemeine Eigenschaft der Sondierungsposition gezeigt, nicht als Qwen-Eigenart.
  { $PY conflict_test.py --features "$F" --hidden "runs/features_newfam_${tag}.hidden.npy" --probe diffmean
    $PY combo_test.py    --features "$F" --hidden "runs/features_newfam_${tag}.hidden.npy" --probe diffmean
  } 2>&1 | tee "runs/newfamily/${tag}_ptrue_position_control.txt"
done
echo "[fertig] $(date '+%d.%m %H:%M') — Ergebnisse in runs/newfamily/"
