#!/usr/bin/env python3
"""Probe-Threshold-Policy mit Task-Success-Utility — modellfrei auf gecachten Features.

Utility (selektive Vorhersage):
    commit & richtig: +1     commit & falsch: −λ     abstain: 0     (minus reason_cost)
Entscheidung über **kalibriertes P(True) = P̂(correct)**:
    reason_more, falls P(IK) < θ_reason ;  abstain, falls P̂(correct) < θ_abstain
Analytische Optimal-Schwelle fürs Committen: p > λ/(1+λ)  (Sanity-Check ggü. gelernter θ_abstain).

Kalibrierung: Platt σ(a·logit(P(True))+b), auf TRAIN gefittet (kein Leak).
Lernt θ_reason, θ_abstain, τ via Gradientenaufstieg auf erwarteter Utility. Kein Modell, kein KL, kein Kollaps.
"""

import argparse
import json
import torch

EPS = 1e-4
LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def _logit(p):
    p = p.clamp(EPS, 1 - EPS)
    return torch.log(p / (1 - p))


def fit_platt(p, y, steps=800, lr=0.05):
    z = _logit(p)
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad(); bce(a * z + b, y).backward(); opt.step()
    return a.detach(), b.detach()


def run_lambda(lam, feats_tr, feats_te, reason_cost, steps, lr):
    """Fittet Schwellen für ein λ auf Train, wertet Greedy auf Train+Test aus."""
    pik_tr, ptd_tr, cd_tr, ptr_tr, cr_tr = feats_tr
    pik_te, ptd_te, cd_te, ptr_te, cr_te = feats_te

    th_r = torch.tensor(0.5, requires_grad=True)
    th_a = torch.tensor(0.5, requires_grad=True)
    log_tau = torch.tensor(-1.8, requires_grad=True)
    opt = torch.optim.Adam([th_r, th_a, log_tau], lr=lr)

    def expected_utility(pik, ptd, cd, ptr, cr):
        tau = log_tau.exp().clamp(min=1e-3)
        p_reason = torch.sigmoid((th_r - pik) / tau)

        def branch(ptrue, correct, rcost):
            p_ab = torch.sigmoid((th_a - ptrue) / tau)          # abstain, wenn P̂(correct) niedrig
            u_commit = correct * (1.0 + lam) - lam              # +1 richtig / −λ falsch
            return (1 - p_ab) * u_commit - rcost                # abstain -> 0
        u_no = branch(ptd, cd, 0.0)
        u_re = branch(ptr, cr, reason_cost)
        return (p_reason * u_re + (1 - p_reason) * u_no).mean()

    for _ in range(steps):
        opt.zero_grad(); (-expected_utility(pik_tr, ptd_tr, cd_tr, ptr_tr, cr_tr)).backward(); opt.step()
    thr, tha, tau = float(th_r), float(th_a), float(log_tau.exp())

    def greedy(pik, ptd, cd, ptr, cr):
        n = len(pik); committed = correct_c = reason_n = 0; util = 0.0
        for i in range(n):
            reasoned = float(pik[i]) < thr
            ptrue = float(ptr[i] if reasoned else ptd[i])
            corr = float(cr[i] if reasoned else cd[i])
            rcost = reason_cost if reasoned else 0.0
            reason_n += reasoned
            if ptrue < tha:
                util += 0.0 - rcost
            else:
                committed += 1; correct_c += int(round(corr))
                util += (1.0 if corr >= 0.5 else -lam) - rcost
        return {"coverage": committed / n,
                "acc_comm": (correct_c / committed) if committed else float("nan"),
                "reason_rate": reason_n / n, "util": util / n}

    def base(pik, ptd, cd, ptr, cr, mode):
        cc = cr if mode == "B" else cd
        rc = reason_cost if mode == "B" else 0.0
        u = float((cc * (1.0 + lam) - lam - rc).mean())
        return {"coverage": 1.0, "acc_comm": float(cc.mean()),
                "reason_rate": 1.0 if mode == "B" else 0.0, "util": u}

    p_star = lam / (1 + lam)

    def analytic(pik, ptd, cd, ptr, cr):
        """Immer reasonen, committen gdw. P̂(correct) > p* (analytisch optimal)."""
        n = len(cr); committed = correct_c = 0; util = 0.0
        for i in range(n):
            util -= reason_cost
            if float(ptr[i]) > p_star:
                committed += 1; correct_c += int(round(float(cr[i])))
                util += 1.0 if float(cr[i]) >= 0.5 else -lam
        return {"coverage": committed / n, "acc_comm": (correct_c / committed) if committed else float("nan"),
                "reason_rate": 1.0, "util": util / n}

    def oracle(pik, ptd, cd, ptr, cr):
        """Decke: perfekte Selektion (committe nur die korrekten reasoned-Antworten)."""
        return {"coverage": float(cr.mean()), "acc_comm": 1.0 if float(cr.mean()) > 0 else float("nan"),
                "reason_rate": 1.0, "util": float(cr.mean()) - reason_cost}

    out = {"thr": thr, "tha": tha, "tau": tau, "p_star": p_star}
    for split, f in (("train", feats_tr), ("test", feats_te)):
        out[split] = {"A": base(*f, "A"), "B": base(*f, "B"), "C": greedy(*f),
                      "An": analytic(*f), "Or": oracle(*f)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="runs/features.json")
    ap.add_argument("--reason-cost", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--train-frac", type=float, default=0.67)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    data = json.load(open(args.features))
    k = int(len(data) * args.train_frac)
    train, test = data[:k], data[k:]
    col = lambda rows, key: torch.tensor([float(r[key]) for r in rows])

    ptd_tr, cd_tr = col(train, "ptrue_direct"), col(train, "correct_direct")
    ptr_tr, cr_tr = col(train, "ptrue_reason"), col(train, "correct_reason")
    ptd_te, cd_te = col(test, "ptrue_direct"), col(test, "correct_direct")
    ptr_te, cr_te = col(test, "ptrue_reason"), col(test, "correct_reason")

    if not args.no_calibrate:
        a, b = fit_platt(torch.cat([ptd_tr, ptr_tr]), torch.cat([cd_tr, cr_tr]))
        cal = lambda p: torch.sigmoid(a * _logit(p) + b)
        ptd_tr, ptr_tr, ptd_te, ptr_te = cal(ptd_tr), cal(ptr_tr), cal(ptd_te), cal(ptr_te)
        print(f"[kalibrierung] Platt a={float(a):.3f} b={float(b):.3f} (P(True) -> P̂(correct))")
    else:
        print("[kalibrierung] aus")

    feats_tr = (col(train, "pik"), ptd_tr, cd_tr, ptr_tr, cr_tr)
    feats_te = (col(test, "pik"), ptd_te, cd_te, ptr_te, cr_te)

    print(f"\nλ-SWEEP (Task-Success-Utility, reason_cost={args.reason_cost}, test n={len(test)})")
    print("U_C=gelernte Schwelle, U_An=analytisch p*, U_Or=Oracle-Decke. cov_An=Coverage der analyt. Policy.")
    print("=" * 96)
    print(f"{'λ':>4} {'p*':>6} {'θ_abst':>8} {'cov_An':>7} {'acc_An':>7} | "
          f"{'U_A':>7} {'U_B':>7} {'U_C':>7} {'U_An':>7} {'U_Or':>7} {'best':>5}")
    print("-" * 96)
    for lam in LAMBDAS:
        r = run_lambda(lam, feats_tr, feats_te, args.reason_cost, args.steps, args.lr)
        t = r["test"]
        cand = {"A": t["A"]["util"], "B": t["B"]["util"], "C": t["C"]["util"], "An": t["An"]["util"]}
        best = max(cand, key=cand.get)
        print(f"{lam:>4.1f} {r['p_star']:>6.2f} {r['tha']:>8.3f} {t['An']['coverage']:>7.2f} "
              f"{t['An']['acc_comm']:>7.2f} | {t['A']['util']:>7.3f} {t['B']['util']:>7.3f} "
              f"{t['C']['util']:>7.3f} {t['An']['util']:>7.3f} {t['Or']['util']:>7.3f} {best:>5}")
    print("\nLesart: U_An>0 (analyt. Schwelle) => das kalibrierte Signal ERMÖGLICHT selektives Committen,")
    print("        und U_C≈U_An => die gelernte Schwelle ist optimal. U_C≈0<U_An => Optimierungs-Artefakt.")
    print("        U_Or = Decke bei perfekter Selektion (= reasoned-acc − reason_cost).")


if __name__ == "__main__":
    main()
