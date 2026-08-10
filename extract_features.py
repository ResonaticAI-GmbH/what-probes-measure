#!/usr/bin/env python3
"""Feature-Extraktion für die Probe-Threshold-Policy (EIN Modell-Pass, dann cachen).

Pro Frage werden die deterministischen Größen berechnet, auf denen die Meta-Policy entscheidet:
  pik            : P(IK)  — "weißt du's?" vor dem Antworten
  correct_direct : ist die DIREKTE Antwort korrekt? (Ground Truth)
  ptrue_direct   : P(True) der direkten Antwort (Selbst-Verifikation)
  correct_reason : ist die Antwort NACH reason_more (CoT) korrekt?
  ptrue_reason   : P(True) der Reasoning-Antwort
  reason_tokens  : Token-Kosten des CoT-Schritts

Danach trainiert train_thresholds.py die Schwellen modellfrei auf diesem Cache.
Base-Modell (as_reference, eingefroren) — die Policy lernt NUR Schwellen, kein LoRA/KL.
"""

import argparse
import json
import re
import time

YES = ["Ja", "Yes", "ja", "yes", "True", "true"]
NO = ["Nein", "No", "nein", "no", "False", "false"]


# Mehrere a-priori P(IK)-Formulierungen (Phase 1: vergleichen, welche am besten diskriminiert)
PIK_PROMPTS = {
    "know": lambda q: f"Frage: {q}\n\nKennst du die korrekte Antwort sicher, ohne Hilfsmittel? Antworte nur mit Ja oder Nein.",
    "can":  lambda q: f"Frage: {q}\n\nKannst du diese Frage korrekt beantworten? Antworte nur mit Ja oder Nein.",
    "will": lambda q: f"Frage: {q}\n\nWirst du diese Frage richtig beantworten? Antworte nur mit Ja oder Nein.",
}


def _ptrue_content(q, a):
    return (f"Frage: {q}\nVorgeschlagene Antwort: {a}\n\n"
            f"Ist die vorgeschlagene Antwort korrekt? Antworte nur mit Ja oder Nein.")


def normalize_answer(text: str) -> str:
    """Nachbereitung der Generierung, analog zu den Filtern in lm-evaluation-harness
    (`take_first`, `remove_whitespace`):
      * nur die ERSTE Zeile/Antwort — Few-Shot verleitet Modelle dazu, direkt die nächste
        Frage mitzuhalluzinieren;
      * ein imitiertes "Answer:"-Präfix entfernen (Nebenwirkung des Few-Shot-Formats);
      * Markdown-Auszeichnung entfernen (Chat-Modelle setzen *…* / **…**).
    Rein kosmetisch gegenüber dem Teilstring-Verifier, aber es hält die Antwortlänge sauber
    und ist auf gespeicherten Antworten jederzeit nachvollziehbar.
    """
    t = str(text).strip()
    t = t.split("\n")[0].strip()
    # Stopp-Sequenzen wie das `until` des Harness: im Roh-Fortsetzungsmodus antwortet das
    # Basismodell korrekt und setzt danach das Few-Shot-Muster fort ("Mexico Question: ...").
    # Ohne diesen Schnitt wird eine richtige Antwort als falsch gewertet.
    for stop in ("Question:", "Answer:", "Q:", "A:"):
        t = t.split(stop)[0].strip()
    t = re.sub(r"^(?:answer|antwort)\s*[:\-]\s*", "", t, flags=re.I)
    t = t.replace("*", "").replace("`", "").strip()
    return t


def chunked(items, bs, fn):
    """Wendet fn auf Teilstücke an und konkateniert — hält den GPU-Speicher beschränkt."""
    out = []
    for i in range(0, len(items), bs):
        out.extend(fn(items[i:i + bs]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--source", type=str, default="gsm8k+popqa")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--reason-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=32, help="Chunk-Größe für Generierung (Speicher)")
    ap.add_argument("--with-hidden", action="store_true",
                    help="Hidden States an der P(True)-Position dumpen (für Phase 3b Hidden-State-Probe)")
    ap.add_argument("--with-answer-hidden", action="store_true",
                    help="Hidden States an der ANTWORT-Position dumpen (mehrere Schichten) -> "
                         "<out>.ahidden.npy (N, L, D). Der faire Probe-Ort ohne Ja/Nein-Prompt "
                         "im Kontext; behebt Confound #4 (results.md Run 24).")
    ap.add_argument("--answer-pool", type=str, default="last", choices=["last", "mean"],
                    help="Pooling über die Antwort-Tokens für --with-answer-hidden")
    ap.add_argument("--no-reason", action="store_true",
                    help="CoT-Block überspringen (~2/3 der Laufzeit). Für Analysen, die nur "
                         "direkte Antwort + P(True) + Hidden States brauchen (z.B. Konflikt-Test). "
                         "correct_reason/ptrue_reason sind dann None -> NICHT für C3/Reasoning-Analysen.")
    ap.add_argument("--raw-answer", action="store_true",
                    help="Antwort-Prompt OHNE Chat-Template erzeugen — reine Fortsetzung wie in "
                         "lm-evaluation-harness. Chat-Wrapper produzieren sonst Markdown, "
                         "imitierte 'Answer:'-Präfixe und Satzantworten, die es im Standard-Setup "
                         "nicht gibt. P(True)/P(IK) behalten das Chat-Template.")
    ap.add_argument("--suppress-think", action="store_true",
                    help="Im Rohmodus einen leeren <think></think>-Block voranstellen (das, was "
                         "enable_thinking=False im Chat-Template tut). NUR für Modelle nötig, die "
                         "in der Rohfortsetzung spontan denken (Qwen3.6). Bei Qwen3-4B/8B ist es "
                         "SCHÄDLICH — sie antworten dann gar nicht mehr bzw. kippen in den "
                         "Assistenzmodus. Vor dem Einsatz immer am Rauchtest prüfen.")
    ap.add_argument("--shots", type=int, default=5,
                    help="Few-Shot-Exemplare im ANTWORT-Prompt, pro Task aus dem EIGENEN Pool "
                         "gezogen und aus der Evaluationsmenge ausgeschlossen. So bindet die "
                         "Literatur die Antwortlänge (PopQA 2212.10511 nutzt 15-shot, "
                         "lm-eval-harness few-shot) — eine selbstgebaute Instruktion oder ein "
                         "handgeschriebenes Exemplar würde stattdessen den Antworttyp verzerren. "
                         "0 = zero-shot.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed für die Item-Ziehung. ecl_data nutzt das globale random-Modul; "
                         "ohne Seed zieht jeder Lauf andere Fragen (das erklärte den "
                         "Genauigkeitssprung zwischen zwei Rauchtests, Run 27).")
    ap.add_argument("--no-lora", action="store_true",
                    help="Base ohne PEFT-Wrapper laden. Für reine Extraktion ist der Adapter "
                         "ohnehin deaktiviert (as_reference); bei großen/MoE-Modellen spart das "
                         "Speicher und umgeht LoRA-Zielmodule, die es dort nicht gibt.")
    ap.add_argument("--out", type=str, default="runs/features.json")
    args = ap.parse_args()

    import random
    random.seed(args.seed)   # vor build_source: identische Items über alle Modelle hinweg

    import torch
    from ecl_config import ECLConfig
    from ecl_policy import ECLPolicy
    from ecl_environment import ECLState
    from ecl_data import build_source, ExampleVerifier

    cfg = ECLConfig(base_model_name=args.model, device="cuda", dtype="bfloat16",
                    use_lora=not args.no_lora)
    t0 = time.time()
    policy = ECLPolicy(cfg, "cuda")
    policy._suppress_think = bool(args.suppress_think)
    if args.suppress_think and not policy.has_think_mode:
        print("[WARNUNG] --suppress-think, aber das Modell kennt keinen Thinking-Modus — ignoriert.")
        policy._suppress_think = False
    verifier = ExampleVerifier()
    src = build_source(args.source, limit=max(args.n * 3, 256))
    # Ohne Zurücklegen ziehen: DataSource.sample() beginnt nach Erschöpfung des Pools von vorn,
    # was sonst still Duplikate erzeugt (z.B. SVAMP hat nur ~700 Items) und Teststärke vortäuscht.
    exs, seen = [], set()
    for _ in range(args.n * 20):
        if len(exs) >= args.n:
            break
        e = src.sample()
        if e.question not in seen:
            seen.add(e.question)
            exs.append(e)
    if len(exs) < args.n:
        print(f"[WARNUNG] nur {len(exs)} EINDEUTIGE Fragen für n={args.n} verfügbar "
              f"(Quelle erschöpft) — kleinere Datensätze limitieren, nicht auffüllen!")
    # Few-Shot-Exemplare: pro Task aus DEM EIGENEN Pool, und aus der Evaluationsmenge
    # entfernt (sonst steht die Antwort im Prompt). Format wie lm-evaluation-harness.
    shot_prefix = {}
    if args.shots > 0:
        by_task = {}
        for e in exs:
            by_task.setdefault(e.task_type, []).append(e)
        drop = set()
        for t, items in by_task.items():
            if len(items) < args.shots + 10:
                print(f"[WARNUNG] {t}: nur {len(items)} Items, Few-Shot übersprungen")
                shot_prefix[t] = ""
                continue
            shots = items[:args.shots]
            drop.update(id(e) for e in shots)
            qt, at = (("Q:", "A:") if items[0].meta.get("qa_style") == "short"
                      else ("Question:", "Answer:"))
            shot_prefix[t] = "".join(
                f"{qt} {e.question}\n{at} {e.answers[0]}\n\n" for e in shots)
        exs = [e for e in exs if id(e) not in drop]
        print(f"[few-shot] {args.shots} Exemplare je Task aus dem eigenen Pool, "
              f"aus der Evaluation entfernt -> n={len(exs)}")

    print(f"[setup] {time.time()-t0:.0f}s | n={len(exs)} source={args.source}"
          f"{' | OHNE Reasoning' if args.no_reason else ''}")

    qs = [e.question for e in exs]
    fewshot = [shot_prefix.get(e.task_type, "") for e in exs]
    bs = args.batch_size
    yesno = lambda contents: policy.yesno_prob_batch(contents, YES, NO)
    styles = [e.meta.get("qa_style", "long") for e in exs]
    answer = lambda states, fs=None, st=None: policy.commit_answer_batch(
        states, deterministic=True, kind="answer", max_new_tokens=32,
        return_tokens=True, fewshot=fs, styles=st, raw=args.raw_answer)
    with policy.as_reference():  # eingefrorene Base
        pik_variants = {name: chunked([fn(q) for q in qs], bs, yesno)
                        for name, fn in PIK_PROMPTS.items()}

        # Über Indizes chunken, damit die Few-Shot-Präfixe zeilenweise mitwandern
        direct = chunked(list(range(len(qs))), bs,
                         lambda ii: answer([qs[i] for i in ii], [fewshot[i] for i in ii],
                                           [styles[i] for i in ii]))
        a_direct = [normalize_answer(r[0]) for r in direct]
        cd = [verifier.verify(a, e) for a, e in zip(a_direct, exs)]
        ptd = chunked([_ptrue_content(q, a) for q, a in zip(qs, a_direct)], bs, yesno)

        if args.no_reason:
            cots = [("", 0)] * len(qs)
            reasoned = [(None, None, 0)] * len(qs)
            cr = [None] * len(qs)
            ptr = [None] * len(qs)
        else:
            cots = chunked(qs, bs, lambda s: policy.reason_batch(s, max_new_tokens=args.reason_tokens))
            cot_states = []
            for q, (txt, _n) in zip(qs, cots):
                st = ECLState(q)
                st.update("Überlegung: " + txt)
                cot_states.append(st.to_string())
            reasoned = chunked(cot_states, bs, answer)
            a_reason = [normalize_answer(r[0]) for r in reasoned]
            cr = [verifier.verify(a, e) for a, e in zip(a_reason, exs)]
            ptr = chunked([_ptrue_content(q, a) for q, a in zip(qs, a_reason)], bs, yesno)

        if args.with_hidden:
            import numpy as np
            ptrue_contents = [_ptrue_content(q, a) for q, a in zip(qs, a_direct)]
            H = np.concatenate(chunked(ptrue_contents, bs,
                                       lambda c: [policy.last_hidden_batch(c)]), axis=0)
            hidden_path = args.out.replace(".json", "") + ".hidden.npy"
            np.save(hidden_path, H.astype("float16"))
            print(f"[hidden] {H.shape} -> {hidden_path}")

        if args.with_answer_hidden:
            import numpy as np
            layer_idxs = []

            def _ah(chunk_idx):
                nonlocal layer_idxs
                A, layer_idxs = policy.answer_hidden_batch(
                    [qs[i] for i in chunk_idx], [a_direct[i] for i in chunk_idx],
                    pool=args.answer_pool, fewshot=[fewshot[i] for i in chunk_idx],
                    styles=[styles[i] for i in chunk_idx], raw=args.raw_answer)
                return [A]

            HA = np.concatenate(chunked(list(range(len(qs))), bs, _ah), axis=0)
            ah_path = args.out.replace(".json", "") + ".ahidden.npy"
            np.save(ah_path, HA.astype("float16"))
            json.dump({"layers": layer_idxs, "pool": args.answer_pool, "shape": list(HA.shape)},
                      open(ah_path.replace(".npy", ".meta.json"), "w"))
            print(f"[answer-hidden] {HA.shape} Schichten={layer_idxs} pool={args.answer_pool} -> {ah_path}")

    feats = []
    for i, e in enumerate(exs):
        row = {
            "pik": pik_variants["know"][i],        # Default (backward-compat mit train_thresholds)
            "correct_direct": bool(cd[i]), "ptrue_direct": ptd[i],
            "correct_reason": None if cr[i] is None else bool(cr[i]), "ptrue_reason": ptr[i],
            "reason_tokens": cots[i][1],           # CoT-Generierungskosten
            "direct_tokens": direct[i][2],         # direkte Antwort-Tokens
            "reason_answer_tokens": reasoned[i][2],
            "task_type": e.task_type, "question": e.question,
            # Antworttext mitschreiben: ohne ihn ist nicht unterscheidbar, ob ein neues Modell
            # schlechter ist oder nur ein Format liefert, das der Verifier nicht erkennt.
            "answer_direct": a_direct[i], "gold": list(e.answers[:5]),
        }
        for name, vals in pik_variants.items():
            row[f"pik_{name}"] = vals[i]            # alle Varianten für den Phase-2-Vergleich
        feats.append(row)
    json.dump(feats, open(args.out, "w"), ensure_ascii=False, indent=1)

    acc_d = sum(cd) / len(cd)
    print(f"[features] {len(feats)} -> {args.out}")
    print(f"  direkt korrekt:    {acc_d:.3f} ({sum(cd)}/{len(cd)})")
    if args.no_reason:
        print("  mit Reasoning:     übersprungen (--no-reason)")
    else:
        print(f"  mit Reasoning:     {sum(cr)/len(cr):.3f} ({sum(cr)}/{len(cr)})   "
              f"(Decke, die reason_more hebt)")
    # Vier-Felder-Tafel je Datensatz: zeigt sofort, ob die Konflikt-Zellen (B/C) tragfähig sind.
    # Schwelle = Median je Task, NICHT fix 0.5 — identisch zu conflict_test.py. Eine absolute
    # Schwelle ist über Modellfamilien hinweg bedeutungslos: Mistral-7B liegt mit P(True) fast
    # vollständig über 0.5, OLMo-2-7B fast vollständig darunter, wodurch B bzw. C leerläuft und
    # die Tafel Kalibrierungs-Offset statt Konflikt anzeigt.
    from collections import Counter
    from statistics import median
    print(f"  {'task':<10}{'n':>6}{'A r/s':>8}{'B r/u':>8}{'C f/s':>8}{'D f/u':>8}{'Konflikt':>10}{'thr':>8}")
    for t in sorted({e.task_type for e in exs}):
        ii = [i for i, e in enumerate(exs) if e.task_type == t]
        thr = median(ptd[i] for i in ii)
        c = Counter((bool(cd[i]), ptd[i] > thr) for i in ii)
        A, B, C, D = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
        print(f"  {t:<10}{len(ii):>6}{A:>8}{B:>8}{C:>8}{D:>8}{B+C:>10}{thr:>8.3f}")
    print(f"[done] {time.time()-t0:.0f}s | GPU peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")


if __name__ == "__main__":
    main()
