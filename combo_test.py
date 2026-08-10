#!/usr/bin/env python3
"""Vorhersage aus Run 25 prüfen: schlägt die KOMBINATION aus Antwort-Positions-Probe und
geprompteter P(True) beide Einzelsignale? (results.md Run 25, Kernergebnis #16)

Begründung: an der Antwort-Position ist rho(Probe, P(True)) nur 0.29–0.53 und die Probe trägt
auf der Konflikt-Teilmenge Information (AUROC sig. > 0.5) — also misst sie etwas, das P(True)
fehlt. Dann muss die Kombination beide schlagen, sonst ist die Komplementarität ein Artefakt.

Sauberkeit (dieselben Härtungen wie probe_equivalence.py):
  * Out-of-Fold auf allen n Items, mehrere Seeds.
  * GESCHACHTELTE CV für die Combo-Gewichte: der Probe-Score der TRAIN-Items wird selbst
    out-of-fold erzeugt (innere CV). Sonst sieht die Combo in-sample-Probe-Scores, die viel
    schärfer sind als out-of-sample, und lernt ein falsches Gewicht auf P(True).
  * Gepaarter Bootstrap auf der Differenz gegen JEDES Einzelsignal.

    .venv-ecl/bin/python combo_test.py --features runs/features_answerpos_4b.json \
        --hidden runs/features_answerpos_4b.ahidden.npy --layer 0
"""

import argparse
import json
import os
import numpy as np

from probe_equivalence import auroc, paired_bootstrap_delta, load_hidden, fit_probe


def _logit(p, eps=1e-4):
    p = np.clip(p.astype("float64"), eps, 1 - eps)
    return np.log(p / (1 - p))


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def oof_probe(H, y, folds, kind, pca):
    """Out-of-Fold-Probe-Scores, je Fold z-standardisiert (Skalen der diffmean-Richtung
    unterscheiden sich sonst zwischen den Folds und verfälschen die gepoolte AUROC)."""
    oof = np.zeros(len(y), dtype="float64")
    for f in range(len(folds)):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(len(folds)) if g != f])
        if min(y[tr].sum(), len(tr) - y[tr].sum()) < 2:
            return None
        oof[te] = _z(fit_probe(H[tr], y[tr], H[te], kind, pca=pca))
    return oof


def fit_logreg(Xtr, ytr, Xte, l2=1e-2, steps=1500):
    import torch
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        (bce(xt @ w + b, yt) + l2 * (w ** 2).sum()).backward()
        opt.step()
    Z = torch.tensor(Xte, dtype=torch.float32)
    return (Z @ w + b).detach().numpy(), w.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="runs/features_answerpos_4b.json")
    ap.add_argument("--hidden", type=str, default="")
    ap.add_argument("--probe", type=str, default="diffmean", choices=["diffmean", "logistic"])
    ap.add_argument("--layer", type=int, default=0, help="Schicht-Slot (0 = 25%% Tiefe, bester rho)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--inner-cv", type=int, default=3)
    ap.add_argument("--min-class", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pca", type=int, default=32)
    args = ap.parse_args()

    d_all = json.load(open(args.features))
    hpath = args.hidden or (args.features.replace(".json", "") + ".ahidden.npy")
    if not os.path.exists(hpath):
        raise SystemExit(f"Hidden-States fehlen: {hpath}")
    H_all = load_hidden(hpath, args.layer)
    assert len(H_all) == len(d_all)

    tasks = sorted({r.get("task_type", "?") for r in d_all})
    print(f"=== Kombination Probe({args.probe}, Slot {args.layer}) + P(True) — {args.features} ===")
    print(f"    {args.seeds} Seeds × {args.cv}-Fold (innere {args.inner_cv}-Fold für Combo-Gewichte), "
          f"{args.n_boot} gepaarte Bootstraps\n")
    print(f"    {'task':<10}{'n':>5}  {'Probe':>7}{'P(True)':>9}{'Combo':>8}  "
          f"{'Δ vs P(True)':>22}  {'Δ vs Probe':>22}  w[probe]/w[ptrue]")
    print("    " + "-" * 104)

    for t in tasks:
        idx = np.array([i for i, r in enumerate(d_all) if r.get("task_type") == t])
        d = [d_all[i] for i in idx]
        H = H_all[idx]
        y = np.array([1.0 if r["correct_direct"] else 0.0 for r in d], dtype="float32")
        pt = _logit(np.array([r["ptrue_direct"] for r in d]))

        if min(y.sum(), len(y) - y.sum()) < args.min_class:
            print(f"    {t:<10}{len(d):>5}  DEGENERIERT ({int(y.sum())} Positive) — nicht auswertbar")
            continue

        a_pr, a_pt, a_co, b_vs_pt, b_vs_pr, ws = [], [], [], [], [], []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            folds = np.array_split(rng.permutation(len(d)), args.cv)
            probe_oof = oof_probe(H, y, folds, args.probe, args.pca)
            if probe_oof is None:
                break
            combo = np.zeros(len(d), dtype="float64")
            for f in range(args.cv):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(args.cv) if g != f])
                # Innere CV: Probe-Scores der Train-Items out-of-sample erzeugen, damit die
                # Combo-Gewichte nicht auf überoptimistischen Probe-Scores gefittet werden.
                inner = np.array_split(rng.permutation(len(tr)), args.inner_cv)
                p_tr = np.zeros(len(tr), dtype="float64")
                for h in range(args.inner_cv):
                    ite = tr[inner[h]]
                    itr = np.concatenate([tr[inner[g]] for g in range(args.inner_cv) if g != h])
                    p_tr[inner[h]] = _z(fit_probe(H[itr], y[itr], H[ite], args.probe, pca=args.pca))
                Xtr = np.stack([p_tr, _z(pt[tr])], 1)
                Xte = np.stack([probe_oof[te], _z(pt[te])], 1)
                combo[te], w = fit_logreg(Xtr, y[tr], Xte)
                ws.append(w)
            a_pr.append(auroc(probe_oof, y)); a_pt.append(auroc(pt, y)); a_co.append(auroc(combo, y))
            b_vs_pt.append(paired_bootstrap_delta(combo, pt, y, args.n_boot, s)[1])
            b_vs_pr.append(paired_bootstrap_delta(combo, probe_oof, y, args.n_boot, s)[1])

        if not a_co:
            print(f"    {t:<10}{len(d):>5}  DEGENERIERT (Fold-Klassen zu dünn)")
            continue

        def ci(bs):
            lo, hi = np.percentile(np.concatenate(bs), [5, 95])
            mark = "sig+" if lo > 0 else ("sig-" if hi < 0 else "n.s.")
            return f"[{lo:>+6.3f},{hi:>+6.3f}] {mark:<4}"

        w = np.mean(ws, 0)
        print(f"    {t:<10}{len(d):>5}  {np.mean(a_pr):>7.3f}{np.mean(a_pt):>9.3f}"
              f"{np.mean(a_co):>8.3f}  {ci(b_vs_pt):>22}  {ci(b_vs_pr):>22}  "
              f"{w[0]:>+.2f} / {w[1]:>+.2f}")

    print("\n    Lesart: 'sig+' = Combo schlägt das Vergleichssignal (90%-CI der gepaarten Differenz > 0).")
    print("            Sig+ in BEIDEN Spalten bestätigt die Komplementaritäts-Vorhersage aus Run 25.")


if __name__ == "__main__":
    main()
