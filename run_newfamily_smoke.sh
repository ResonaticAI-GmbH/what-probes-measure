#!/usr/bin/env bash
# DIAGNOSE, kein Produktionslauf. Prüft Mistral-7B-v0.3 und OLMo-2-7B auf die Fehlerquellen,
# die uns bei jedem neuen Modell erwischt haben (Antwortsprache, Chat-Artefakte, Thinking) —
# und zusätzlich auf eine, die nur Nicht-Qwen-Modelle betrifft:
#
#   Beide haben KEIN Chat-Template. Damit fällt `_yesno_prompts` auf "<content>\nAntwort:"
#   zurück, während Qwen den Chat-Wrapper bekommt. P(True)/P(IK) werden also anders erhoben
#   als bei den Qwen-Modellen — und die Prompts sind deutsch, was bei Mistral schwächer sitzt.
#   Wenn P(True) hier nicht diskriminiert, ist der Cross-Familien-Vergleich wertlos, egal wie
#   gut die Antworten aussehen.
#
# Startet NICHTS Großes. Ergebnis wird gelesen, dann wird entschieden.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONPATH=.
PY="${PY:-.venv-ecl/bin/python}"

SRC=popqa+webq+triviaqa
mkdir -p runs/newfamily

# (Warteschleife auf run_fix48.sh entfernt: `pgrep -f <skriptname>` trifft auch die
#  Kommandozeilen der Wächter-Prozesse, die denselben Namen enthalten -> der Job wartete
#  stundenlang auf sich selbst. Jobs nie über Namensmuster synchronisieren.)

for spec in "mistral:mistralai/Mistral-7B-v0.3:24" "olmo:allenai/OLMo-2-1124-7B:24"; do
  tag=$(echo "$spec" | cut -d: -f1); model=$(echo "$spec" | cut -d: -f2); bs=$(echo "$spec" | cut -d: -f3)
  echo "######## DIAGNOSE $tag  ($model)  $(date +%H:%M:%S)"
  $PY extract_features.py \
      --source "$SRC" --n 400 --model "$model" --seed 0 --shots 5 --raw-answer \
      --no-reason --no-lora --batch-size "$bs" \
      --out "runs/newfamily/${tag}.json" 2>&1 | tail -8

  $PY - "runs/newfamily/${tag}.json" "$tag" <<'EOF'
import json, sys, statistics as st
from collections import defaultdict
d = json.load(open(sys.argv[1])); tag = sys.argv[2]

def auroc(scores, y):
    pos = [s for s, c in zip(scores, y) if c]; neg = [s for s, c in zip(scores, y) if not c]
    if not pos or not neg: return float("nan")
    import itertools
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))

print(f"\n===== DIAGNOSE {tag} =====")
leer = sum(not r["answer_direct"].strip() for r in d)
L = [len(r["answer_direct"].split()) for r in d]
print(f"Antworten: leer={leer}/{len(d)}  median_wörter={st.median(L):.0f}  >8W={sum(l>8 for l in L)/len(L):.0%}")
print(f"           <think>-Reste={sum('<think>' in r['answer_direct'] for r in d)}")

for t in sorted({r["task_type"] for r in d}):
    ii = [r for r in d if r["task_type"] == t]
    y = [r["correct_direct"] for r in ii]
    pt = [r["ptrue_direct"] for r in ii]
    pk = [r["pik"] for r in ii]
    print(f"  {t:<9} n={len(ii):<4} acc={sum(y)/len(ii):.3f}  "
          f"P(True): mean={st.mean(pt):.3f} sd={st.pstdev(pt):.3f} AUROC={auroc(pt,y):.3f}  "
          f"P(IK) AUROC={auroc(pk,y):.3f}")

print("\nAntwortbeispiele:")
for r in d[:8]:
    print(f"  [{r['task_type']:<9}] {'OK' if r['correct_direct'] else 'X '} "
          f"{r['answer_direct'][:55]!r:<59} gold={r['gold'][:1]}")

# Die entscheidende Frage: trägt P(True) überhaupt Signal?
allpt = [r["ptrue_direct"] for r in d]
print(f"\nP(True) gesamt: sd={st.pstdev(allpt):.4f}  Anteil>0.5={sum(p>0.5 for p in allpt)/len(d):.2f}")
if st.pstdev(allpt) < 0.05:
    print("  [WARNUNG] P(True) ist praktisch konstant -> Selbstverifikation greift NICHT. "
          "Der deutsche Ja/Nein-Prompt ohne Chat-Template funktioniert bei diesem Modell nicht.")
EOF
done
echo "[diagnose fertig] $(date '+%d.%m %H:%M') — runs/newfamily/ ; KEIN Produktionslauf gestartet."
