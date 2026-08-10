#!/usr/bin/env python3
"""Konflikt-Test: folgt die Hidden-State-Probe der KORREKTHEIT oder dem SELBSTURTEIL?

Hintergrund (Konflikt-Teilmengen-Design nach Lu, arXiv:2607.16799): Eine Probe schlägt P(True) —
aber warum? Zwei Erklärungen:
  (A) sie kennt die Korrektheit besser  ("privilegiertes Wissen")
  (B) sie liest dasselbe Selbsturteil nur sauberer aus als ein einzelner Ja/Nein-Token
      ("Ablese-Artefakt")

Vier-Felder-Tafel pro Frage (OC = objektive Korrektheit, SJ = Selbsturteil P(True)>θ):
      A: korrekt   & sicher     — einig
      B: korrekt   & unsicher   — KONFLIKT
      C: falsch    & sicher     — KONFLIKT
      D: falsch    & unsicher   — einig

Auf A/D sagen beide Hypothesen dieselbe Ordnung voraus -> diese Items tragen ZUR FRAGE nichts
bei, dominieren aber den Durchschnitt. Nur auf B-vs-C muss sich die Probe entscheiden:
  Probe(B) > Probe(C)  -> folgt der Korrektheit   (Hypothese A)
  Probe(C) > Probe(B)  -> folgt dem Selbsturteil  (Hypothese B)
Gemessen als AUROC auf der Konflikt-Teilmenge mit Label = Korrektheit:
  ~1.0 = reines Korrektheitswissen · ~0.5 = uninformativ · ~0.0 = reines Selbsturteil.

Die Probe wird STRIKT within-task und out-of-fold gefittet (keine Konflikt-Items im Training
ihrer eigenen Vorhersage). Vergleichsanker: P(True) selbst hat auf B-vs-C konstruktionsbedingt
AUROC ~0 — es IST das Selbsturteil. Zusätzlich wird der Anteil des Probe-Vorteils berichtet,
der nach Kontrolle für das Selbsturteil überlebt.

    .venv-ecl/bin/python conflict_test.py --features runs/features_conflict_4b.json
"""

import argparse
import json
import os
import numpy as np

from probe_equivalence import auroc, fit_probe, paired_bootstrap_delta, load_hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="runs/features_conflict_4b.json")
    ap.add_argument("--hidden", type=str, default="")
    ap.add_argument("--probe", type=str, default="logistic", choices=["diffmean", "logistic"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--layer", type=int, default=-1,
                    help="Schicht-Slot bei (N,L,D)-Dumps (answer-hidden); -1 = letzte")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="SJ-Schwelle auf P(True); <0 => Median je Task (balanciert die Zellen)")
    ap.add_argument("--min-cell", type=int, default=25,
                    help="Mindestgröße der KLEINEREN Konfliktzelle, sonst nicht auswertbar")
    ap.add_argument("--band", type=str, default="",
                    help="'lo,hi' — High-Confidence-Kriterium statt Median-Split: nur Items mit "
                         "P(True)<=lo (unsicher) oder >=hi (sicher), Mitte verworfen. Das ist die "
                         "Operationalisierung von Lu (2607.16799, tau=0.7 -> '0.3,0.7'); der "
                         "Median-Split ist unsere eigene und schwächere Variante.")
    ap.add_argument("--split-from", type=str, default="",
                    help="Konflikt-TEILMENGE aus dem P(True) DIESER Datei bestimmen, Rest aus "
                         "--features. Für die Fixed-Subset-Kontrolle (results.md Run 34): die "
                         "Teilmenge wird sonst durch dasselbe P(True) definiert, das variiert "
                         "wird — Kompositions- und Verhaltensänderung sind dann nicht trennbar.")
    args = ap.parse_args()

    d_all = json.load(open(args.features))
    if args.split_from:
        d_split = json.load(open(args.split_from))
        assert len(d_split) == len(d_all), "split-from hat andere Länge als features"
        print(f"[split-from] Konflikt-Teilmenge aus {args.split_from} (eingefroren)")
    hpath = args.hidden or (args.features.replace(".json", "") + ".hidden.npy")
    if not os.path.exists(hpath):
        raise SystemExit(f"Hidden-States fehlen: {hpath}  (extract_features.py --with-hidden)")
    H_all = load_hidden(hpath, args.layer)
    assert len(H_all) == len(d_all), f"Hidden ({len(H_all)}) != Features ({len(d_all)})"

    print(f"=== Konflikt-Test ({args.probe}) — {args.features} ===")
    print(f"    Label = Korrektheit auf der Konflikt-Teilmenge (B∪C).")
    print(f"    AUROC ~1.0 = Probe folgt der KORREKTHEIT · ~0.5 = uninformativ · "
          f"~0.0 = folgt dem SELBSTURTEIL\n")
    print(f"    {'task':<10}{'n':>6}{'B':>5}{'C':>5}  {'Probe|Konflikt':>15}  "
          f"{'90%-CI':>17}  {'Probe|alle':>11}{'P(True)|alle':>13}{'rho':>7}  Verdikt")
    print("    " + "-" * 108)

    for t in sorted({r.get("task_type", "?") for r in d_all}):
        idx = np.array([i for i, r in enumerate(d_all) if r.get("task_type") == t])
        d = [d_all[i] for i in idx]
        H = H_all[idx]
        y = np.array([1.0 if r["correct_direct"] else 0.0 for r in d], dtype="float32")
        pt = np.array([r["ptrue_direct"] for r in d], dtype="float32")
        # Signal, das die Teilmenge definiert — normalerweise dasselbe P(True), das auch
        # als Anker berichtet wird; bei --split-from ein eingefrorenes aus einer Baseline.
        pt_split = (np.array([d_split[i]["ptrue_direct"] for i in idx], dtype="float32")
                    if args.split_from else pt)

        oc = y == 1
        if args.band:
            lo, hi = (float(x) for x in args.band.split(","))
            keep = (pt_split <= lo) | (pt_split >= hi)   # Mitte verworfen
            sj = pt_split >= hi
            conflict = np.where(keep & ((oc & ~sj) | (~oc & sj)))[0]
            nB, nC = int((keep & oc & ~sj).sum()), int((keep & ~oc & sj).sum())
        else:
            thr = np.median(pt_split) if args.threshold < 0 else args.threshold
            sj = pt_split > thr                          # "Modell hält sich für sicher"
            conflict = np.where((oc & ~sj) | (~oc & sj))[0]  # B ∪ C
            nB, nC = int((oc & ~sj).sum()), int((~oc & sj).sum())

        if min(nB, nC) < args.min_cell:
            print(f"    {t:<10}{len(d):>6}{nB:>5}{nC:>5}  {'—':>15}  {'—':>17}  "
                  f"{'—':>11}{'—':>13}  ZU KLEIN (kleinere Zelle < {args.min_cell})")
            continue

        # Out-of-fold-Probe auf ALLEN Items der Task, danach auf die Konflikt-Teilmenge schauen.
        # Wichtig: die Probe sieht beim Fitten nie die Items, die sie später bewertet.
        a_conf, a_all, boots_all = [], [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(2000 + s)
            perm = rng.permutation(len(d))
            folds = np.array_split(perm, args.cv)
            oof = np.zeros(len(d))
            for f in range(args.cv):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(args.cv) if g != f])
                if min(y[tr].sum(), len(tr) - y[tr].sum()) < 2:
                    break
                # Standardisierung mit TRAIN-Momenten desselben Folds; die Momente des
                # ausgehaltenen Folds zu verwenden waere transduktiv (keine Label-Leckage,
                # aber Testverteilung in der Skalierung).
                both = fit_probe(H[tr], y[tr], np.concatenate([H[tr], H[te]]),
                                 args.probe, pca=args.pca)
                sc_tr, sc = both[:len(tr)], both[len(tr):]
                oof[te] = (sc - sc_tr.mean()) / (sc_tr.std() + 1e-9)
            a_conf.append(auroc(oof[conflict], y[conflict]))
            a_all.append(auroc(oof, y))
            _, b = paired_bootstrap_delta(oof[conflict], np.zeros(len(conflict)),
                                          y[conflict], n_boot=args.n_boot, seed=s)
            boots_all.append(b + 0.5)   # Δ gegen konstanten Score (=AUROC 0.5) -> AUROC-Verteilung
        pooled = np.concatenate(boots_all)
        lo, hi = np.percentile(pooled, [5, 95])
        m_conf, m_all = float(np.mean(a_conf)), float(np.mean(a_all))
        a_pt = auroc(pt, y)
        # Standard-Diagnose seit Confound #4: liest die Probe nur P(True) neu aus?
        # rho nahe 1 => Probe ist eine rangerhaltende Kopie des Selbsturteils.
        rho = float(np.corrcoef(np.argsort(np.argsort(oof)),
                                np.argsort(np.argsort(pt)))[0, 1])

        if lo > 0.5:
            verd = "folgt KORREKTHEIT (Hyp. A)"
        elif hi < 0.5:
            verd = "folgt SELBSTURTEIL (Hyp. B)"
        else:
            verd = "unentschieden — CI enthält 0.5"
        print(f"    {t:<10}{len(d):>6}{nB:>5}{nC:>5}  {m_conf:>15.3f}  "
              f"[{lo:>+6.3f},{hi:>+6.3f}]  {m_all:>11.3f}{a_pt:>13.3f}{rho:>+7.2f}  {verd}")

    print(f"\n    Schwelle: {'Median je Task' if args.threshold < 0 else args.threshold}"
          f" · {args.seeds} Seeds × {args.cv} Folds × {args.n_boot} Bootstraps")
    print("    Lesart: 'Probe|alle' ist der übliche (uninformative) Durchschnitt — er wird von den")
    print("            einigen Feldern A/D dominiert. Erst 'Probe|Konflikt' trennt die Hypothesen.")
    print("    rho = Spearman(Probe, P(True)). Nahe +1 => die Probe ist nur eine Neuauslesung")
    print("            des Selbsturteils (so entstand Confound #4: Sondierung an der P(True)-Position).")
    print("    P(True)|alle dient nur als Anker; auf der Konflikt-Teilmenge ist P(True)")
    print("            konstruktionsbedingt ~0 (die Teilmenge ist ja nach ihm definiert).")


if __name__ == "__main__":
    main()
