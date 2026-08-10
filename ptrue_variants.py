#!/usr/bin/env python3
"""Kausaler Test der Kollaps-Bedingung aus Run 30.1.

Run 30 zeigt einen ZUSAMMENHANG: je stärker P(True), desto mehr kopiert die Probe es
statt Korrektheit auszulesen (15 Zellen, 5 Modelle, monoton). Beschrieben, nicht kausal
geprüft — die fünf Modelle unterscheiden sich in vielem, nicht nur in P(True).

Der Eingriff hier isoliert genau eine Variable. Entscheidend ist eine Eigenschaft des
Aufbaus: **die Hidden States an der Antwort-Position hängen von der P(True)-Formulierung
gar nicht ab.** Der Antwort-Prompt enthält keinen Ja/Nein-Teil (das war der Fix von
Confound #4). Also gilt bei allen Varianten:

    gleiche Items · gleiche Antworten · gleiche Labels · gleiche Hidden States · gleiche Probe
    einzige Änderung: die Formulierung, mit der das Modell nach seiner Sicherheit gefragt wird

Vorhersage aus 30.1: über die Varianten hinweg steigt rho(Probe, P(True)) mit der
P(True)-AUROC, und die Konflikt-AUROC fällt. Trifft das zu, ist die Bedingung nicht nur
beobachtet, sondern herbeigeführt. Trifft es nicht zu, ist 30.1 eine Korrelation über
Modelle und muss so berichtet werden.

Ausgabe: je Variante eine Kopie des Feature-JSON mit ersetztem `ptrue_direct` — damit
laufen conflict_test.py / probe_equivalence.py unverändert darauf (keine Neuimplementierung
der geprüften Statistik).

    .venv-ecl/bin/python ptrue_variants.py --features runs/features_final_8b.json \
        --model Qwen/Qwen3-8B --outdir runs/ptrue_var
"""

import argparse
import json
import os
import time

YES = ["Ja", "Yes", "ja", "yes", "True", "true"]
NO = ["Nein", "No", "nein", "no", "False", "false"]

# Die Stellschrauben: Sprache, Chat-Template, und wie scharf das Korrektheitskriterium
# benannt wird. Bewusst KEINE Variante, die die Antwort verändert — nur die Frage danach.
#   raw=True erzwingt Roh-Fortsetzung auch bei Modellen mit Chat-Template.
VARIANTS = {
    # Baseline: exakt der Prompt aus extract_features.py. Muss den gespeicherten
    # ptrue_direct reproduzieren — das ist die Kontrolle, dass die Pipeline stimmt.
    "de_orig": (False, lambda q, a: (
        f"Frage: {q}\nVorgeschlagene Antwort: {a}\n\n"
        f"Ist die vorgeschlagene Antwort korrekt? Antworte nur mit Ja oder Nein.")),
    "de_raw": (True, lambda q, a: (
        f"Frage: {q}\nVorgeschlagene Antwort: {a}\n\n"
        f"Ist die vorgeschlagene Antwort korrekt? Antworte nur mit Ja oder Nein.")),
    "en": (False, lambda q, a: (
        f"Question: {q}\nProposed answer: {a}\n\n"
        f"Is the proposed answer correct? Answer only Yes or No.")),
    "en_raw": (True, lambda q, a: (
        f"Question: {q}\nProposed answer: {a}\n\n"
        f"Is the proposed answer correct? Answer only Yes or No.")),
    # Absichtlich schwach: kein Korrektheitskriterium, nur eine vage Wertung.
    "vague": (False, lambda q, a: (
        f"Frage: {q}\nVorgeschlagene Antwort: {a}\n\n"
        f"Ist das eine gute Antwort? Antworte nur mit Ja oder Nein.")),
    # Absichtlich scharf: benennt Faktizität und Übereinstimmung explizit.
    "strict": (False, lambda q, a: (
        f"Question: {q}\nProposed answer: {a}\n\n"
        f"Is the proposed answer factually correct and does it match the expected "
        f"gold answer? Answer only Yes or No.")),
}


def auroc(scores, y):
    pos = [s for s, c in zip(scores, y) if c]
    neg = [s for s, c in zip(scores, y) if not c]
    if not pos or not neg:
        return float("nan")
    import numpy as np
    s = np.asarray(scores, dtype="float64")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    yv = np.asarray([1 if c else 0 for c in y])
    np_, nn = yv.sum(), len(yv) - yv.sum()
    return float((ranks[yv == 1].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="runs/ptrue_var")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = json.load(open(args.features))
    tag = args.tag or os.path.basename(args.features).replace("features_", "").replace(".json", "")
    os.makedirs(args.outdir, exist_ok=True)

    from ecl_config import ECLConfig
    from ecl_policy import ECLPolicy

    cfg = ECLConfig(base_model_name=args.model, device="cuda", dtype="bfloat16", use_lora=False)
    t0 = time.time()
    policy = ECLPolicy(cfg, "cuda")
    print(f"[setup] {time.time()-t0:.0f}s | {args.model} | n={len(d)}")

    y = [bool(r["correct_direct"]) for r in d]
    stored = [r["ptrue_direct"] for r in d]
    tasks = sorted({r["task_type"] for r in d})
    ti = {t: [i for i, r in enumerate(d) if r["task_type"] == t] for t in tasks}

    def per_task(p):
        """AUROC je Task. Gepoolt über Tasks wäre der Aggregations-Confound (Claim C1):
        unterschiedliche Basisraten erzeugen dann Trennschärfe, die es within-task nicht
        gibt — bei Mistral gepoolt 0.509 vs. within-task 0.43/0.62/0.58."""
        return [auroc([p[i] for i in ti[t]], [y[i] for i in ti[t]]) for t in tasks]

    hdr = "".join(f"{t[:8]:>9}" for t in tasks)
    print(f"\n{'Variante':<10}{hdr}   {'mean':>7}{'sd':>7}{'>0.5':>7}   {'Δ gespeichert':>14}")
    print(f"{'(gespeich.)':<10}" + "".join(f"{a:>9.3f}" for a in per_task(stored)) +
          f"   {sum(stored)/len(stored):>7.3f}"
          f"{(sum((p-sum(stored)/len(stored))**2 for p in stored)/len(stored))**0.5:>7.3f}"
          f"{sum(p>0.5 for p in stored)/len(d):>7.2f}")

    with policy.as_reference():
        for name in args.variants:
            raw, fn = VARIANTS[name]
            contents = [fn(r["question"], r["answer_direct"]) for r in d]
            out = []
            for i in range(0, len(contents), args.batch_size):
                out.extend(policy.yesno_prob_batch(contents[i:i + args.batch_size], YES, NO, raw=raw))
            mean = sum(out) / len(out)
            sd = (sum((p - mean) ** 2 for p in out) / len(out)) ** 0.5
            maxdiff = max(abs(a - b) for a, b in zip(out, stored))
            print(f"{name:<10}" + "".join(f"{a:>9.3f}" for a in per_task(out)) +
                  f"   {mean:>7.3f}{sd:>7.3f}"
                  f"{sum(p>0.5 for p in out)/len(out):>7.2f}   {maxdiff:>14.4f}")

            dd = [dict(r) for r in d]
            for r, p in zip(dd, out):
                r["ptrue_direct"] = p
            json.dump(dd, open(f"{args.outdir}/{tag}_{name}.json", "w"), ensure_ascii=False)

    print(f"\n[fertig] {time.time()-t0:.0f}s -> {args.outdir}/{tag}_*.json")
    print("  Kontrolle: 'de_orig' muss Δ≈0 zum gespeicherten P(True) haben. Weicht es ab,")
    print("  stimmt der Prompt-Nachbau nicht und alle Varianten sind wertlos.")


if __name__ == "__main__":
    main()
