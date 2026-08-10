# Reproducing the numbers in the paper

    pip install -r requirements.txt      # Python 3.12, torch is platform-dependent

Two levels: **extraction** needs a GPU and hours, **analysis** runs on CPU in minutes from the
caches shipped here. To recompute the tables you only need the second.

The shell runners default to `PY=.venv-ecl/bin/python` and a Hugging Face cache under
`$HOME/.cache/huggingface`. Both are environment overrides, e.g.
`PY=$(which python) HF_HOME=/data/hf ./run_f1.sh`.

## A. No GPU, from the checked-in caches (minutes)

Needs only `runs/features_*.json` and `runs/**/*_probe_s1.npy` — both are in the repository.

| Paper | Command |
|---|---|
| Table 1 ($F_1$) | `./run_f1.sh` → `runs/f1_s1.log` |
| Table 3 (incremental value) | `PYTHONPATH=. python incremental_value.py --features runs/features_final_8b.json --scores runs/final/8b_probe_s1.npy --tag Qwen3-8B` |
| Utility / coverage (§4) | `PYTHONPATH=. python policy_compare.py --features runs/features_final_8b.json --scores runs/final/8b_probe_s1.npy --tag Qwen3-8B` |

For all five models, the path list is in `run_f1.sh`.

## B. Needs the hidden-state dumps (i.e. an extraction run from C)

`*.hidden.npy` / `*.ahidden.npy` are ~2.2 GB and **not** checked in; they are produced in
section C.

| Paper | Command |
|---|---|
| Table 2 ($\rho$, conflict, all) | `./rerun_all.sh` |
| Table 2, column $r(s,m)$ | `python direction_cosine.py --features … --ptrue-hidden … --answer-hidden … --layer 1` |
| Figure 1 (conflict confound) | `./rerun_all.sh` → `runs/finding3_recheck.log` |
| Regenerate probe-score dumps | `./run_policy.sh` |
| Appendix A (layer sweep) | `./run_final.sh` (analysis part) → `runs/final/*_layersweep.txt` |

`direction_cosine.py` does not load a full model — only the required rows of the unembedding
matrix from the safetensors — but it does need the hidden states.

## C. Extraction (GPU, hours) — regenerates every cache

| Purpose | Command | Runtime |
|---|---|---|
| Main run, Qwen 4B/8B/36B | `./run_final.sh` | ~4 h |
| Families Mistral/OLMo | `./run_newfamily.sh` | ~1.5 h |
| Prompt variants (Finding 3) | `./run_ptrue_causal.sh`, `./run_ptrue_olmo.sh` | ~40 min |
| Mean-pooling factor study | `./run_meanpool.sh`, then `./run_pooling_study.sh` | ~2.5 h |
| Seed robustness (second item sample) | `./run_seed_robustness.sh` | ~1.5 h |

All main results use seed 0; items are identical across models. `run_seed_robustness.sh` re-extracts
Qwen3-8B and Mistral-7B with seed 1 — a different draw of 6000 questions, everything else unchanged.

## What is checked in and what is not

**Checked in** (~13 MB): the feature caches `runs/features_*.json` (questions, answers, labels,
$P(\text{True})$, $P(\text{IK})$) and the out-of-fold probe scores `runs/**/*_probe_s1.npy`. These
are enough for section A.

**Not checked in** (~2.2 GB): the hidden-state dumps, plus the caches of the two control studies
(`runs/seed1/`, `runs/meanpool/`) and the prompt-variant caches. All of them are only useful
together with their hidden states, and the runners in section C regenerate cache and analysis in one
pass.

## On the history

The three retractions and the failed attempts that led to the findings are documented in section 9
of the paper. The full chronological work log is not part of this release.
