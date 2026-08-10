#!/usr/bin/env python3
"""Klassifikationsgüte (Precision/Recall/F1) Probe vs. P(True) — die dritte Sichtweise.

Bisher gemessen: AUROC (Rangordnung, schwellenfrei) und Utility (Entscheidung bei
Fehlerpreis λ). Beides beantwortet nicht die einfachste Frage: *wenn ich mich für eine
Schwelle entscheiden muss, wie gut klassifiziert das Ding dann?*

Schwelle wird auf TRAIN gewählt (F1-maximierend), auf TEST ausgewertet — sonst misst man
die Schwellenanpassung mit. Beide Signale bekommen dieselbe Behandlung.

Zusätzlich: dieselben Kennzahlen auf der KONFLIKT-Teilmenge (dort, wo P(True) falsch
liegt — genau die Items, an denen sich entscheidet, ob die Probe etwas Eigenes weiss).
"""

import argparse
import json
import numpy as np


def prf(pred, y):
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(score, y):
    """F1-maximierende Schwelle auf TRAIN. Kandidaten = die Scores selbst."""
    cand = np.unique(score)
    if len(cand) > 400:
        cand = np.quantile(score, np.linspace(0, 1, 400))
    best, bt = -1.0, cand[0]
    for t in cand:
        _, _, f = prf(score > t, y)
        if f > best:
            best, bt = f, t
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    d = json.load(open(args.features))
    probe_all = np.load(args.scores)
    print(f"=== F1 Probe vs. P(True) — {args.tag or args.features} ===")
    print(f"    Schwelle F1-maximierend auf TRAIN, ausgewertet auf TEST, "
          f"{args.seeds} × {int(args.train_frac*100)}/{100-int(args.train_frac*100)}-Splits\n")

    for t in sorted({r["task_type"] for r in d}):
        idx = np.array([i for i, r in enumerate(d) if r["task_type"] == t])
        y = np.array([1 if d[i]["correct_direct"] else 0 for i in idx])
        pt = np.array([d[i]["ptrue_direct"] for i in idx], dtype="float64")
        pr = probe_all[idx].astype("float64")
        if not np.isfinite(pr).all():
            print(f"    {t}: übersprungen (NaN-Scores)")
            continue

        # Konflikt-Teilmenge: dort liegt P(True) falsch (Median-Split wie conflict_test.py)
        sure = pt > np.median(pt)
        conflict = (y == 1) & ~sure | (y == 0) & sure

        print(f"    --- {t}  n={len(y)}  acc={y.mean():.3f}  "
              f"Konflikt={int(conflict.sum())} ({conflict.mean():.0%}) ---")
        print(f"    {'Menge':<10}{'Signal':<8}{'Prec':>7}{'Rec':>7}{'F1':>7}"
              f"{'  (immer-Ja F1)':>16}")
        for scope, mask in (("alle", np.ones(len(y), bool)), ("Konflikt", conflict)):
            base_p, base_r, base_f = prf(np.ones(int(mask.sum()), bool), y[mask])
            for name, sc in (("ptrue", pt), ("probe", pr)):
                fs = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(700 + s)
                    perm = rng.permutation(len(y))
                    k = int(len(y) * args.train_frac)
                    tr, te = perm[:k], perm[k:]
                    th = best_threshold(sc[tr], y[tr])
                    m = mask[te]
                    if m.sum() == 0 or y[te][m].sum() == 0:
                        continue
                    fs.append(prf(sc[te][m] > th, y[te][m]))
                if not fs:
                    continue
                p, r, f = (float(np.mean([x[i] for x in fs])) for i in range(3))
                print(f"    {scope:<10}{name:<8}{p:>7.3f}{r:>7.3f}{f:>7.3f}"
                      f"{base_f:>16.3f}")
        print()


if __name__ == "__main__":
    main()
