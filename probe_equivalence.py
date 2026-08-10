#!/usr/bin/env python3
"""Äquivalenztest Hidden-State-Probe vs. geprompte P(True) — statistisch gehärtete Fassung
von train_probe_head.py (siehe results.md, Run 23).

Drei Härtungen gegenüber train_probe_head.py:
  1. MEHRERE zufällige Splits (--seeds) statt eines fixen sequentiellen Splits.
  2. GEPAARTER Bootstrap auf der Differenz (Probe − P(True)) auf denselben Test-Items.
     Zwei unabhängige CIs zu vergleichen ("überlappen sie?") ist der falsche Test und
     viel zu konservativ — beide teilen sich dieselben Items.
  3. TOST-Äquivalenz gegen eine vorab gesetzte Schranke (--margin): erlaubt die Aussage
     "die Probe ist NICHT besser" statt nur "wir konnten keinen Unterschied zeigen".

Außerdem: Zellen mit zu wenig Positiven/Negativen (--min-class) werden als DEGENERIERT
ausgewiesen statt eine bedeutungslose AUROC zu drucken (z.B. math @32 Token: 1 Positive
von 118 Test-Items). AUROC-Bootstrap verwirft einklassige Resamples, statt sie als 0.5
zu zählen (das zieht CIs künstlich Richtung Zufall).

    .venv-ecl/bin/python probe_equivalence.py --features runs/features_scale_4b.json
    .venv-ecl/bin/python probe_equivalence.py --features runs/features_scale_4b.json --probe logistic
"""

import argparse
import json
import os
import numpy as np


# ----------------------------------------------------------------------
# AUROC (tie-aware, numpy) + gepaarter Bootstrap
# ----------------------------------------------------------------------
def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney-U mit Durchschnittsrängen. NaN wenn nur eine Klasse vorliegt
    (bewusst NICHT 0.5 — ein einklassiges Sample trägt keine Information)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype="float64")
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def paired_bootstrap_delta(s_a, s_b, y, n_boot=2000, seed=0):
    """Bootstrap auf der Differenz AUROC(a) − AUROC(b) über DIESELBEN resampelten Items.
    Gibt (delta_point, samples) zurück; einklassige Resamples werden verworfen."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = auroc(s_a, y) - auroc(s_b, y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        out.append(auroc(s_a[idx], yb) - auroc(s_b[idx], yb))
    return point, np.array(out)


def load_hidden(path: str, layer: int = -1):
    """Lädt einen Hidden-State-Dump. Akzeptiert (N, D) [alte last_hidden_batch-Dumps] und
    (N, L, D) [neue answer_hidden_batch-Dumps mit Schicht-Sweep]; bei 3D wird `layer` gewählt."""
    H = np.load(path).astype("float32")
    if H.ndim == 3:
        import json as _json
        import os as _os
        meta_p = path.replace(".npy", ".meta.json")
        meta = _json.load(open(meta_p)) if _os.path.exists(meta_p) else {}
        print(f"[hidden] {H.shape} Schichten={meta.get('layers')} pool={meta.get('pool')} "
              f"-> nutze Schicht-Slot {layer}")
        H = H[:, layer, :]
    return H


# ----------------------------------------------------------------------
# Probe-Designs (auf Train gefittet, auf Test angewandt)
# ----------------------------------------------------------------------
def fit_probe(Htr, ytr, Hte, kind: str, pca: int = 32, l2: float = 1e-2, steps: int = 2000):
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Htr, Hte = (Htr - mu) / sd, (Hte - mu) / sd
    if kind == "diffmean":
        w = Htr[ytr == 1].mean(0) - Htr[ytr == 0].mean(0)
        return Hte @ w
    import torch
    kc = min(pca, Htr.shape[0] - 1, Htr.shape[1])
    _, _, Vt = np.linalg.svd(Htr, full_matrices=False)
    P = Vt[:kc].T
    Ztr = torch.tensor(Htr @ P, dtype=torch.float32)
    Zte = torch.tensor(Hte @ P, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(kc, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        (bce(Ztr @ w + b, yt) + l2 * (w ** 2).sum()).backward()
        opt.step()
    return (Zte @ w + b).detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="runs/features_scale_4b.json")
    ap.add_argument("--hidden", type=str, default="")
    ap.add_argument("--probe", type=str, default="diffmean", choices=["diffmean", "logistic"])
    ap.add_argument("--seeds", type=int, default=5, help="Anzahl zufälliger Train/Test-Splits")
    ap.add_argument("--cv", type=int, default=5, help="Folds für Out-of-Fold-Scores")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Äquivalenzschranke in AUROC (TOST): |Δ| < margin => äquivalent")
    ap.add_argument("--min-class", type=int, default=30,
                    help="Mindestzahl Positiver UND Negativer im Test-Split, sonst degeneriert")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--layer", type=int, default=-1,
                    help="Schicht-Slot bei (N,L,D)-Dumps (answer-hidden); -1 = letzte")
    ap.add_argument("--dump-scores", type=str, default="",
                    help="Out-of-Fold-Probe-Scores je Item als .npy schreiben (Reihenfolge = "
                         "features.json). Für policy_compare.py: die Probe als Entscheidungs-"
                         "signal, ohne dass ein Item je von einer Probe bewertet wird, die es "
                         "im Training gesehen hat. Über die --seeds gemittelt.")
    args = ap.parse_args()

    d_all = json.load(open(args.features))
    hpath = args.hidden or (args.features.replace(".json", "") + ".hidden.npy")
    if not os.path.exists(hpath):
        raise SystemExit(f"Hidden-States fehlen: {hpath}")
    H_all = load_hidden(hpath, args.layer)
    assert len(H_all) == len(d_all), f"Hidden ({len(H_all)}) != Features ({len(d_all)})"

    tasks = sorted({r.get("task_type", "?") for r in d_all})
    print(f"=== Äquivalenz Probe({args.probe}) vs. P(True) — {args.features} ===")
    print(f"    {args.seeds} Splits × {args.n_boot} gepaarte Bootstraps, "
          f"Schranke ±{args.margin} AUROC, min-class={args.min_class}\n")
    print(f"    {'task':<10}{'n':>5}{'pos%':>6}  {'Probe':>7}{'P(True)':>9}"
          f"{'Δ':>8}  {'90%-CI(Δ)':>18}  Verdikt")
    print("    " + "-" * 78)

    summary = []
    # NaN, nicht 0: degenerierte Tasks bekommen keinen Score, und 0 wäre ein gültiger
    # z-Wert (Mittellage) — ein stiller Fehler in jeder nachgelagerten Auswertung.
    dump = np.full(len(d_all), np.nan, dtype="float64") if args.dump_scores else None
    for t in tasks:
        idx = np.array([i for i, r in enumerate(d_all) if r.get("task_type") == t])
        d = [d_all[i] for i in idx]
        H = H_all[idx]
        y = np.array([1.0 if r["correct_direct"] else 0.0 for r in d], dtype="float32")
        ptrue = np.array([r["ptrue_direct"] for r in d], dtype="float32")

        # Kreuzvalidierung mit gepoolten Out-of-Fold-Scores: JEDES Item wird genau einmal
        # von einer Probe bewertet, die es nicht gesehen hat -> Testsatz = alle n Items
        # (statt 33%). Das ist der Unterschied zwischen 20 und 60 Positiven pro Zelle.
        a_p, a_t, deltas_all, oof_seeds = [], [], [], []
        degenerate = min(y.sum(), len(y) - y.sum()) < args.min_class
        for s in range(0 if degenerate else args.seeds):
            rng = np.random.default_rng(1000 + s)
            perm = rng.permutation(len(d))
            folds = np.array_split(perm, args.cv)
            oof = np.zeros(len(d), dtype="float64")
            for f in range(args.cv):
                te = folds[f]
                tr = np.concatenate([folds[g] for g in range(args.cv) if g != f])
                if min(y[tr].sum(), len(tr) - y[tr].sum()) < 2:
                    degenerate = True
                    break
                # Scores für TRAIN und TEST desselben Folds in einem Aufruf, damit die
                # Standardisierung mit TRAIN-Momenten erfolgen kann.
                both = fit_probe(H[tr], y[tr], np.concatenate([H[tr], H[te]]),
                                 args.probe, pca=args.pca)
                sc_tr, sc = both[:len(tr)], both[len(tr):]
                # Pro Fold z-standardisieren: die diffmean-Richtung hat je Fold eine
                # andere Skala/Lage; ungepoolt-vergleichbare Scores würden die AUROC
                # über die Fold-Grenzen hinweg verfälschen. Momente aus dem TRAIN-Teil —
                # mit den Momenten des ausgehaltenen Folds waere es transduktiv.
                oof[te] = (sc - sc_tr.mean()) / (sc_tr.std() + 1e-9)
            if degenerate:
                break
            oof_seeds.append(oof.copy())
            _, boots = paired_bootstrap_delta(oof, ptrue, y, n_boot=args.n_boot, seed=s)
            a_p.append(auroc(oof, y))
            a_t.append(auroc(ptrue, y))
            deltas_all.append(boots)

        pos_frac = y.mean()
        if degenerate:
            n_pos = int(y.sum())
            print(f"    {t:<10}{len(d):>5}{pos_frac*100:>5.1f}%  "
                  f"{'—':>7}{'—':>9}{'—':>8}  {'—':>18}  DEGENERIERT "
                  f"({n_pos} Positive gesamt) — nicht auswertbar")
            summary.append((t, None))
            continue

        if dump is not None and oof_seeds:
            dump[idx] = np.mean(oof_seeds, axis=0)

        # Bootstraps über Splits poolen: mittelt Split-Varianz UND Resample-Varianz ein
        pooled = np.concatenate(deltas_all)
        lo90, hi90 = np.percentile(pooled, [5, 95])
        lo95, hi95 = np.percentile(pooled, [2.5, 97.5])
        delta = float(np.mean(a_p) - np.mean(a_t))
        # Äquivalenz und Signifikanz sind ZWEI Fragen — eine kleine Differenz kann
        # zugleich sicher-von-null-verschieden und praktisch irrelevant sein.
        equiv = lo90 > -args.margin and hi90 < args.margin
        if lo90 > 0:
            sig = "Probe sig. besser"
        elif hi90 < 0:
            sig = "P(True) sig. besser"
        else:
            sig = "n.s."
        verdikt = ("ÄQUIVALENT" if equiv else "unentschieden (CI reißt Schranke)") + f", {sig}"
        print(f"    {t:<10}{len(d):>5}{pos_frac*100:>5.1f}%  "
              f"{np.mean(a_p):>7.3f}{np.mean(a_t):>9.3f}{delta:>+8.3f}  "
              f"[{lo90:>+6.3f},{hi90:>+6.3f}]  {verdikt}")
        summary.append((t, (delta, lo90, hi90, lo95, hi95, verdikt)))

    if dump is not None:
        np.save(args.dump_scores, dump)
        n_ok = int(np.isfinite(dump).sum())
        print(f"\n    [dump] {n_ok}/{len(dump)} Out-of-Fold-Scores -> {args.dump_scores}"
              f"{'' if n_ok == len(dump) else '  (NaN = degenerierte Task)'}")

    ok = [s for s in summary if s[1]]
    print(f"\n    Auswertbare Tasks: {len(ok)}/{len(summary)}"
          f" (degeneriert: {[t for t, v in summary if not v] or 'keine'})")
    if ok:
        eq = [t for t, v in ok if v[5].startswith("ÄQUIVALENT")]
        print(f"    ÄQUIVALENT bei ±{args.margin}: {len(eq)}/{len(ok)} — {eq}")
        print("\n    Lesart: 90%-CI der gepaarten Differenz komplett innerhalb ±margin => TOST-Äquivalenz")
        print("            (α=0.05 einseitig je Grenze). Enthält das CI 0 UND reißt die Schranke,")
        print("            ist der Test schlicht unterpowert — das ist KEIN Äquivalenznachweis.")


if __name__ == "__main__":
    main()
