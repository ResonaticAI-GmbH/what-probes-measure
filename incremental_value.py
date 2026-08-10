#!/usr/bin/env python3
"""Inkrementeller Wert der Probe ÜBER P(True) hinaus — auf ALLEN Items.

Motivation: semantische Identifikation über die Konflikt-Teilmenge ist von deren
Operationalisierung abhängig (results.md Run 32/34/35). Die praktisch entscheidende Frage
lässt sich ohne Teilmenge stellen — als geschachtelter Modellvergleich auf allen Items:

    Basis:     Y ~ sigma(a * logit P(True) + b)          (was das Modell explizit sagt)
    Erweitert: Y ~ sigma(a * logit P(True) + c * s_probe + b)

Berichtet werden Delta LogLoss, Delta Brier und Delta AUROC, jeweils gepaart gebootstrappt.
Das beantwortet: *hilft die Probe bei epistemischen Entscheidungen zusätzlich zu dem, was das
Modell ohnehin schon sagt?* — ohne zu behaupten, eine latente Richtung repräsentiere
ontologisch "Korrektheit".

Sauberkeit: die Probe-Scores sind bereits out-of-fold (probe_equivalence.py --dump-scores).
Die Kombinationsgewichte werden auf einem TRAIN-Split gefittet und auf TEST ausgewertet, damit
das Zusatzgewicht nicht in-sample bestimmt wird.
"""

import argparse
import json
import numpy as np

EPS = 1e-6


def logit(p):
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def fit_logreg(X, y, steps=4000, lr=0.08, l2=1e-4):
    """Kleine logistische Regression per Gradientenabstieg (Spalten bereits standardisiert)."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * ((X.T @ g) / len(y) + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def predict(X, wb):
    w, b = wb
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p, y):
    return float(((p - y) ** 2).mean())


def auroc(s, y):
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype="float64")
    i = 0
    ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()

    d = json.load(open(args.features))
    probe = np.load(args.scores)
    print(f"=== Inkrementeller Wert der Probe über P(True) — {args.tag or args.features} ===")
    print(f"    Nested: Y~P(True) vs. Y~P(True)+Probe · {args.seeds} Splits · "
          f"{args.n_boot} gepaarte Bootstraps · Gewichte auf TRAIN\n")
    print(f"    {'task':<10}{'ΔLogLoss':>10}{'90%-CI':>18}{'ΔBrier':>9}{'ΔAUROC':>9}"
          f"{'  w[probe]':>10}")

    for t in sorted({r["task_type"] for r in d}):
        idx = np.array([i for i, r in enumerate(d) if r["task_type"] == t])
        y = np.array([1.0 if d[i]["correct_direct"] else 0.0 for i in idx])
        z_pt = logit(np.array([d[i]["ptrue_direct"] for i in idx], dtype="float64"))
        z_pr = probe[idx].astype("float64")
        if not np.isfinite(z_pr).all():
            print(f"    {t:<10} übersprungen (NaN)")
            continue
        dll, dbr, dau, wpr, boots = [], [], [], [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(900 + s)
            perm = rng.permutation(len(y))
            k = int(len(y) * args.train_frac)
            tr, te = perm[:k], perm[k:]
            # Standardisierung NUR auf Train: sonst gehen Momente des Testteils in die
            # Skalierung ein. Keine Label-Leckage, aber transduktiv -- und in einem Paper
            # ueber Messdisziplin nicht verteidigbar.
            def zs(v):
                m, sd = v[tr].mean(), v[tr].std() + 1e-9
                return (v - m) / sd
            zp, zr = zs(z_pt), zs(z_pr)
            X0 = zp[:, None]
            X1 = np.stack([zp, zr], 1)
            m0 = fit_logreg(X0[tr], y[tr])
            m1 = fit_logreg(X1[tr], y[tr])
            p0, p1 = predict(X0[te], m0), predict(X1[te], m1)
            dll.append(logloss(p0, y[te]) - logloss(p1, y[te]))   # >0 = Probe hilft
            dbr.append(brier(p0, y[te]) - brier(p1, y[te]))
            dau.append(auroc(p1, y[te]) - auroc(p0, y[te]))
            wpr.append(float(m1[0][1]))
            # gepaarter Bootstrap auf denselben Test-Items
            yb = y[te]
            for _ in range(args.n_boot // args.seeds):
                bi = rng.integers(0, len(yb), len(yb))
                if yb[bi].sum() in (0, len(bi)):
                    continue
                boots.append(logloss(p0[bi], yb[bi]) - logloss(p1[bi], yb[bi]))
        lo, hi = np.percentile(boots, [5, 95])
        sig = "sig+" if lo > 0 else ("sig-" if hi < 0 else "n.s.")
        print(f"    {t:<10}{np.mean(dll):>+10.4f}  [{lo:>+6.4f},{hi:>+6.4f}] {sig:<5}"
              f"{np.mean(dbr):>+9.4f}{np.mean(dau):>+9.4f}{np.mean(wpr):>+10.2f}")


if __name__ == "__main__":
    main()
