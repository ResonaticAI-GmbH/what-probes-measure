#!/usr/bin/env python3
"""Richtungs-Analyse: liest die korrektheits-gelabelte Probe die Selbsturteils-RICHTUNG aus?

Bisher wurde das indirekt über rho(Probe-Score, P(True)) belegt — eine Korrelation zwischen
SCORES. Die Behauptung ist aber eine über RICHTUNGEN im Repräsentationsraum. Die lässt sich
direkt messen.

Am Unembedding gilt für den Yes/No-Logit-Margin
    z_yes - z_no = h^T (u_yes - u_no) =: h^T u
mit u aus den lm_head-Zeilen der Ja/Nein-Tokens. An der P(True)-Position ist h genau der
Zustand, auf den lm_head angewandt wird (hidden_states[-1] ist NACH der finalen Norm,
empirisch geprüft) — dort ist der Vergleich exakt. An der Antwort-Position (Slot 1) liegt
ein roher Residual-Stream-Zustand mittlerer Tiefe vor, ohne finale Norm; der Vergleich bleibt
in derselben Basis, ist aber nicht mehr die Richtung, die dort tatsächlich ausgelesen würde.
Das wird mitberichtet, nicht überinterpretiert.

Skalierung: die Probe wird auf standardisierten Features gelernt (h~ = (h-mu)/sd), also ist
    score = h~^T w = h^T (w/sd) + const
Die im ROHEN Raum wirksame Richtung ist damit w/sd, und nur die ist mit u vergleichbar.
Beide Varianten werden ausgegeben, damit der Unterschied sichtbar bleibt.

Kontrollen: (a) Kosinus einer Probe auf PERMUTIERTEN Labels — Bodenwert; (b) Kosinus einer
zufälligen Richtung — analytischer Bodenwert ~1/sqrt(d).
"""

import argparse
import json
import os
import numpy as np

YES = ["Ja", "Yes", "ja", "yes", "True", "true"]
NO = ["Nein", "No", "nein", "no", "False", "false"]


def first_ids(tok, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            t = tok.encode(variant, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def unembed_rows(model_path, ids):
    """Liest NUR die benötigten Zeilen der Unembedding-Matrix aus den safetensors-Shards,
    statt das ganze Modell zu laden (wichtig für das 36B-MoE).

    ACHTUNG: `lm_head.weight` ist nur bei tie_word_embeddings=True identisch mit
    `model.embed_tokens.weight`. Von den fünf Modellen koppelt nur Qwen3-4B. Deshalb ALLE
    Shards nach lm_head durchsuchen, bevor auf embed_tokens zurückgefallen wird — sonst
    liest man bei vier von fünf Modellen die Eingabe- statt der Ausgabe-Matrix.
    """
    from transformers import AutoConfig
    from safetensors import safe_open
    import glob
    from huggingface_hub import snapshot_download
    cfg = AutoConfig.from_pretrained(model_path)
    tied = bool(getattr(cfg, "tie_word_embeddings", False))
    local = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(local, "*.safetensors")))

    def read(key):
        for f in shards:
            with safe_open(f, framework="pt") as fh:
                if key in set(fh.keys()):
                    W = fh.get_slice(key)
                    return np.stack([W[i, :].float().numpy() for i in ids])
        return None

    if not tied:
        rows = read("lm_head.weight")
        if rows is not None:
            return rows, "lm_head.weight"
        raise SystemExit(f"{model_path}: tie_word_embeddings=False, aber kein lm_head.weight "
                         f"gefunden — embed_tokens waere hier die falsche Matrix.")
    rows = read("model.embed_tokens.weight")
    if rows is not None:
        return rows, "model.embed_tokens.weight (tied)"
    raise SystemExit(f"Keine Unembedding-Matrix gefunden in {local}")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def probe_dir(H, y):
    """diffmean auf standardisierten Features; gibt (w_std, w_raw=w/sd) zurück."""
    mu, sd = H.mean(0), H.std(0) + 1e-6
    Ht = (H - mu) / sd
    w = Ht[y == 1].mean(0) - Ht[y == 0].mean(0)
    return w, w / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ptrue-hidden", required=True, help="(N,D) am Ja/Nein-Prompt, letzte Schicht")
    ap.add_argument("--answer-hidden", required=True, help="(N,L,D) an der Antwort-Position")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    yid, nid = first_ids(tok, YES), first_ids(tok, NO)
    rows_y, key = unembed_rows(args.model, yid)
    rows_n, _ = unembed_rows(args.model, nid)
    u = rows_y.mean(0) - rows_n.mean(0)

    d = json.load(open(args.features))
    Hp = np.load(args.ptrue_hidden).astype("float32")
    Ha = np.load(args.answer_hidden).astype("float32")[:, args.layer, :]
    y_all = np.array([1.0 if r["correct_direct"] else 0.0 for r in d])

    print(f"=== Richtungs-Kosinus zur Yes/No-Unembedding-Richtung — {args.tag or args.model} ===")
    print(f"    u aus '{key}', |yes|={len(yid)} |no|={len(nid)} Tokens, d={len(u)}")
    print(f"    Zufalls-Bodenwert ~1/sqrt(d) = {1/np.sqrt(len(u)):.4f}\n")
    print(f"    {'task':<10}{'Position':<16}{'cos(w_raw,u)':>13}{'cos(w_std,u)':>13}"
          f"{'perm-Kontrolle':>15}")

    pt = np.array([r["ptrue_direct"] for r in d], dtype="float64")
    rng = np.random.default_rng(0)
    for t in sorted({r["task_type"] for r in d}):
        idx = np.array([i for i, r in enumerate(d) if r["task_type"] == t])
        y = y_all[idx]
        for name, H in (("P(True)-Pos.", Hp[idx]), (f"Antwort/Slot{args.layer}", Ha[idx])):
            _, w_raw = probe_dir(H, y)
            w_std, _ = probe_dir(H, y)
            yp = rng.permutation(y)
            _, w_perm = probe_dir(H, yp)
            print(f"    {t:<10}{name:<16}{cos(w_raw,u):>+13.4f}{cos(w_std,u):>+13.4f}"
                  f"{cos(w_perm,u):>+15.4f}")
        print()

    # Der euklidische Kosinus ist in stark anisotropen Raeumen das FALSCHE Mass:
    # corr(h.w1, h.w2) = w1'S w2 / sqrt(w1'S w1 * w2'S w2) mit der Datenkovarianz S.
    # Zwei Richtungen koennen fast orthogonal sein und dennoch fast identische
    # Projektionen erzeugen. Die inhaltlich gemeinte Groesse ist daher die Ausrichtung
    # der PROJEKTIONEN: der tatsaechliche Yes/No-Logit-Margin m = h^T u.
    print(f"    {'-'*70}")
    print(f"    Projektionsebene: Yes/No-Logit-Margin m = h^T u (das inhaltlich gemeinte Mass)")
    print(f"    {'task':<10}{'Position':<16}{'r(Probe,m)':>12}{'rho(Probe,m)':>14}"
          f"{'AUROC(m)':>10}{'r(m,logit pT)':>15}")
    for t in sorted({r["task_type"] for r in d}):
        idx = np.array([i for i, r in enumerate(d) if r["task_type"] == t])
        y = y_all[idx]
        lpt = np.log(np.clip(pt[idx], 1e-6, 1 - 1e-6) / (1 - np.clip(pt[idx], 1e-6, 1 - 1e-6)))
        for name, H in (("P(True)-Pos.", Hp[idx]), (f"Antwort/Slot{args.layer}", Ha[idx])):
            _, w_raw = probe_dir(H, y)
            sc = H @ w_raw
            m = H @ u
            def sp(a, b):
                ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
                return float(np.corrcoef(ra, rb)[0, 1])
            npos, nneg = int(y.sum()), int(len(y) - y.sum())
            o = np.argsort(m, kind="mergesort"); rk = np.empty(len(m)); rk[o] = np.arange(1, len(m) + 1)
            au = (rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            print(f"    {t:<10}{name:<16}{np.corrcoef(sc,m)[0,1]:>+12.3f}{sp(sc,m):>+14.3f}"
                  f"{au:>10.3f}{np.corrcoef(m,lpt)[0,1]:>+15.3f}")
        print()


if __name__ == "__main__":
    main()
