# What Does a Correctness Probe Measure?

Code and full experiment log for the preprint *What Does a Correctness Probe Measure? Probing
Position, Incremental Value, and the Operationalization Sensitivity of Conflict-Subset
Evaluation* (Richard Eberle, resonaticAI).

Linear probes on hidden states predict whether a language model's answer is correct, and they
outperform simply asking the model. That is established since
[SAPLMA (2023)](https://arxiv.org/abs/2304.13734) and **we claim no novelty for it**. We ask what
such a probe measures, across 5 models / 3 families (Qwen3 4B/8B, Qwen3.6-35B-A3B,
Mistral-7B-v0.3, OLMo-2-7B), n=6000 questions from PopQA, TriviaQA and WebQuestions.

**Three findings**

1. **The probing position decides what the probe tracks, and the reported AUROC does not reveal
   it.** At the last token of the yes/no self-verification prompt, the correctness-labelled probe
   correlates $0.91$–$0.99$ with the model's own yes/no logit margin — it *is* the self-judgement
   readout. At the answer position it tracks correctness instead. Aggregate AUROC is nearly
   identical either way.
2. **The probe adds decision-relevant information over $P(\text{True})$.** Nested comparison on
   *all* items, no subset selection: adding the probe improves held-out log loss in 15/15
   model × task cells.
3. **Conflict-subset evaluation is operationalization-sensitive.** The self-judgement instrument
   defines both the construct and the evaluated population. Holding probe, data and labels fixed
   and swapping one of four instruments moves conflict AUROC by up to $0.279$ and, for one model,
   reverses the qualitative verdict. Freezing the subset removes the effect entirely.

**Three retractions.** Section 9 of the paper documents them — including a "causal confirmation"
that turned out to be an artifact of the very metric we were criticising. The correction is the
contribution; a paper about measurement discipline that hid its own errors would be self-refuting.

> Code comments occasionally cite `results.md`, our internal chronological work log. It is not part
> of this release; the substance of every decision it records is in the paper.

## Getting started

    pip install -r requirements.txt

See [`REPRODUCE.md`](REPRODUCE.md). Tables 1 and 3 and the utility numbers run **on CPU in
minutes** from the caches included here; anything needing hidden-state dumps (~2.2 GB) requires
one extraction pass on a GPU, and those analyses' outputs ship as `.txt`/`.log` under `runs/`.

The paper source is in [`paper/`](paper/) and builds with `tectonic main.tex`.

## Repository layout

| | |
|---|---|
| `extract_features.py` | generation, $P(\text{True})$/$P(\text{IK})$, hidden-state dumps |
| `probe_equivalence.py` | difference-of-means probe, out-of-fold scoring, TOST |
| `conflict_test.py` | A/B/C/D conflict-subset test, incl. frozen-subset control |
| `incremental_value.py` | nested model comparison on all items (Finding 2) |
| `f1_compare.py`, `policy_compare.py` | classification and selective-prediction views |
| `direction_cosine.py` | probe direction vs. yes/no unembedding direction (Finding 1) |
| `ptrue_variants.py` | the four self-judgement instruments (Finding 3) |

Licensed MIT. Contact: richi@resonatic.ch

Repository: <https://github.com/ResonaticAI-GmbH/what-probes-measure>
