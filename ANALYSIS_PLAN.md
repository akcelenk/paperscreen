# Analysis plan

Written on 2026-10-13, before any real labels exist. Edit it only with a dated entry at the bottom;
do not change a hypothesis after seeing results that bear on it.

## Question

For one user's stream of paper judgements, which ranking methods give better daily digests, and how
many labels does it take for learned methods to beat simple baselines?

## Unit of analysis and primary metric

* Unit: a weekly evaluation window (`evaluation.step_days = 7`). Each window's model is trained on
  labels made before the window and scored on labels made inside it.
* Primary metric: average precision (AP) per window over **all human labels**, averaged over windows
  with at least one positive and one negative. We use windows rather than days because a day holds too
  few positives for AP to mean anything.
* Bias check (required): the same analysis restricted to items first shown as uniformly random picks.
  A confirmatory claim needs the all-labels and random-slice differences to point the same way. The
  random slice is expected to be underpowered; say so rather than dropping it.

## Fixed settings

Model hyperparameters are the defaults in `default_config.yaml` (`C=1.0`, `S=1.0`, `V=0.5`,
`n_random=1000`, TF-IDF `C=0.1`) and are **not tuned on the evaluation windows**. Grids over them
are exploratory. The embedder for primary comparisons is the config default (`bge-small`).

## Primary comparisons (Holm-corrected as one family)

| id | comparison | hypothesis |
|---|---|---|
| P1 | `emb_logreg` vs `tfidf_svm` | A frozen embedding with logistic regression ranks better than TF-IDF + SVM. Direction not assumed: prior work finds TF-IDF competitive on some metrics. |
| P2 | `emb_logreg` vs `zeroshot` | Learning from labels beats ranking by interest descriptions alone. |
| P3 | `per_interest_logreg` vs `emb_logreg` | Separate per-interest models beat one global model for a multi-interest user. |

Decision rule: a comparison is called supported only if the Holm-adjusted p-value is below 0.05
and the 95% paired-bootstrap interval for the AP difference excludes 0. Otherwise report the
interval and write "not distinguishable with this data". Report effect sizes before p-values.

## When confirmatory runs happen

Once each, at 300, 1000 and 3000 human labels (whichever are reached). Intermediate reports are
exploratory monitoring. Before 300 labels (about 30 positives) no comparison is confirmatory.

## Exploratory (never headline results)

All other pairwise comparisons in the report, per-interest tables, learning-curve crossings,
`V`/`S`/`C` grids, other embedders (`minilm`, `specter2-base`, `scincl`), input-format variants,
fusion, prototype `k`. Report with uncorrected intervals and the label "exploratory".

## Exclusions and safeguards

* LLM-produced labels never enter evaluation. If used for training later, they carry
  `origin != 'human'`, are excluded by `labels_frame()` by default, and their effect is measured on
  human-labelled test items only.
* Per-interest results are only interpreted for interests with at least 20 test positives.
* Every report records the config hash and git commit.
* Seeds are fixed; learned models with seed variance (SetFit, fine-tuning; not in v1) need at least
  five seeds, with seed variance reported separately from bootstrap variance.

## Log

* 2026-10-13: plan written.
* 2026-10-13 (before any real labels): the random-negative pool rule was corrected. It used to exclude
  every paper ever labelled, which leaks future information into past windows; it now excludes only
  training labels and test-window items. No hypothesis or setting changed.
