#!/usr/bin/env python3
"""Entscheidungsqualität statt Rangordnung: dieselbe Abstain-Policy auf drei Signalen.

Die Frage, die AUROC NICHT beantwortet. AUROC misst, ob ein Signal richtig/falsch gut
sortiert. Eine Policy muss sich festlegen: antworten oder abstinieren. Ein AUROC-Vorsprung
von +0.10 kann sich je nach Fehlerpreis λ ganz unterschiedlich auszahlen — oder gar nicht.

Utility (selektive Vorhersage, wie train_thresholds.py):
    commit & richtig: +1     commit & falsch: −λ     abstain: 0

Verglichen werden drei Entscheidungssignale, jeweils Platt-kalibriert auf TRAIN:
    ptrue  — die geprompte Selbstprüfung (der Baseline-Ansatz "frag das Modell")
    probe  — Out-of-Fold-Probe-Scores aus probe_equivalence.py --dump-scores
    combo  — beide, per Logistik auf TRAIN kombiniert

Gegen zwei Referenzen:
    immer   — nie abstinieren (coverage 1.0). Der Punkt, den man schlagen muss.
    oracle  — perfekte Selektion (committe genau die korrekten). Die Decke.

Kein reason_more-Zweig: die finalen Feature-Sets liefen mit --no-reason, `ptrue_reason` und
`correct_reason` sind None. Deshalb NICHT train_thresholds.py, das darauf aufbaut.

Schwelle: analytisch p* = λ/(1+λ) auf der kalibrierten Wahrscheinlichkeit. Das ist bei
sauberer Kalibrierung optimal und hat keine Freiheitsgrade, die man versehentlich am
Testsatz tunt — der Vergleich der drei Signale bleibt fair.

    .venv-ecl/bin/python policy_compare.py --features runs/features_newfam_mistral.json \
        --scores runs/newfamily/mistral_probe_s1.npy
"""

import argparse
import json
import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(z, y, steps=1500, lr=0.05):
    """Platt-Skalierung σ(a·z+b) per Gradientenabstieg auf TRAIN. Gibt (a, b) zurück."""
    a, b = 1.0, 0.0
    n = len(z)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        g = p - y
        a -= lr * float((g * z).mean())
        b -= lr * float(g.mean())
    return a, b


def apply_platt(z, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def fit_combo(Z, y, steps=3000, lr=0.05):
    """Logistik auf mehreren z-Spalten (TRAIN). Gibt (w, b) zurück."""
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g) / len(y)
        b -= lr * float(g.mean())
    return w, b


def evaluate(p, y, lam):
    """Committe gdw. kalibriertes P(correct) > p* = λ/(1+λ)."""
    p_star = lam / (1.0 + lam)
    commit = p > p_star
    n = len(y)
    if commit.sum() == 0:
        return {"util": 0.0, "cov": 0.0, "acc": float("nan")}
    corr = y[commit]
    util = float((corr * (1.0 + lam) - lam).sum()) / n
    return {"util": util, "cov": float(commit.mean()), "acc": float(corr.mean())}


def ref_always(y, lam):
    return {"util": float((y * (1.0 + lam) - lam).mean()), "cov": 1.0, "acc": float(y.mean())}


def ref_oracle(y, lam):
    return {"util": float(y.mean()), "cov": float(y.mean()), "acc": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True, help="Probe-OOF-Scores (.npy) aus probe_equivalence.py")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--seeds", type=int, default=20, help="zufällige Train/Test-Splits")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    d = json.load(open(args.features))
    probe_all = np.load(args.scores)
    assert len(probe_all) == len(d), f"Scores ({len(probe_all)}) != Features ({len(d)})"

    tag = args.tag or args.features
    print(f"=== Policy-Vergleich — {tag} ===")
    print(f"    Utility: commit&richtig +1 / commit&falsch −λ / abstain 0")
    print(f"    Schwelle analytisch p*=λ/(1+λ) auf Platt-kalibrierter Wahrscheinlichkeit")
    print(f"    {args.seeds} zufällige {int(args.train_frac*100)}/{100-int(args.train_frac*100)}-Splits, "
          f"Kalibrierung + Combo-Gewichte nur auf TRAIN\n")

    for t in sorted({r["task_type"] for r in d}):
        idx = np.array([i for i, r in enumerate(d) if r["task_type"] == t])
        y = np.array([1.0 if d[i]["correct_direct"] else 0.0 for i in idx])
        pt = np.array([d[i]["ptrue_direct"] for i in idx], dtype="float64")
        pr = probe_all[idx].astype("float64")
        if not np.isfinite(pr).all():
            print(f"    {t:<10} ÜBERSPRUNGEN — Probe-Scores enthalten NaN (degenerierte Task)")
            continue

        z_pt, z_pr = logit(pt), pr  # Probe-Scores sind bereits z-standardisiert

        print(f"    --- {t}  (n={len(y)}, acc={y.mean():.3f}) ---")
        print(f"    {'λ':>5}  {'Signal':<8}{'Utility':>9}{'Coverage':>10}{'Acc|commit':>12}"
              f"{'Δ vs immer':>12}")
        for lam in args.lambdas:
            rows = {}
            for name in ("ptrue", "probe", "combo"):
                us = []
                for s in range(args.seeds):
                    rng = np.random.default_rng(500 + s)
                    perm = rng.permutation(len(y))
                    k = int(len(y) * args.train_frac)
                    tr, te = perm[:k], perm[k:]
                    if name == "ptrue":
                        ab = fit_platt(z_pt[tr], y[tr]); p_te = apply_platt(z_pt[te], ab)
                    elif name == "probe":
                        ab = fit_platt(z_pr[tr], y[tr]); p_te = apply_platt(z_pr[te], ab)
                    else:
                        Z = np.stack([z_pt, z_pr], 1)
                        w, b = fit_combo(Z[tr], y[tr])
                        p_te = 1.0 / (1.0 + np.exp(-(Z[te] @ w + b)))
                    us.append(evaluate(p_te, y[te], lam))
                rows[name] = {k2: float(np.nanmean([u[k2] for u in us])) for k2 in ("util", "cov", "acc")}
            base = ref_always(y, lam)
            orc = ref_oracle(y, lam)
            for name in ("ptrue", "probe", "combo"):
                r = rows[name]
                print(f"    {lam:>5.1f}  {name:<8}{r['util']:>+9.3f}{r['cov']:>10.3f}"
                      f"{r['acc']:>12.3f}{r['util']-base['util']:>+12.3f}")
            print(f"    {'':>5}  {'immer':<8}{base['util']:>+9.3f}{base['cov']:>10.3f}"
                  f"{base['acc']:>12.3f}{0.0:>+12.3f}")
            print(f"    {'':>5}  {'oracle':<8}{orc['util']:>+9.3f}{orc['cov']:>10.3f}"
                  f"{orc['acc']:>12.3f}{orc['util']-base['util']:>+12.3f}")
            print()


if __name__ == "__main__":
    main()
