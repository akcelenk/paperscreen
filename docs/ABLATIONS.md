# Ablation roadmap

Which axes to vary, in what order, and what each label budget can support. "v1" means implemented
and runnable today; the others are planned. Compute costs are engineering estimates for 2 CPU cores
and about 10k-100k candidate papers, not measurements. Pre-registered primary comparisons are in
[`../ANALYSIS_PLAN.md`](../ANALYSIS_PLAN.md); everything below is exploratory unless listed there.

Every ablation uses the same harness: weekly expanding windows on label time, paired cluster
bootstrap, an all-labels view and a random-slice view.

## Axes

| # | axis | options | hypothesis | primary metric | cost | order | status |
|---|---|---|---|---|---|---|---|
| A0 | baselines | random, recency, category, keyword, TF-IDF + SVM | Every learned model must beat these; TF-IDF is expected to be strong. | window AP | minutes | 1 | v1 |
| A1 | embedding model | TF-IDF, `bge-small`, `minilm`, `specter2-base`, `scincl`; later larger general models | The best model may change with the metric and with label count; domain models may win with few labels. | window AP | one embedding pass per model (small models: minutes to hours on CPU) | 2 | v1 (`evaluation.extra_embedders`) |
| A2 | input format | title; title + abstract; SPECTER-style `[SEP]`; category as a separate feature | The abstract adds the most; categories help cross-listed interests. | window AP, per-interest AP | re-embed for 1-2 models | 3 | v1 (`embedding.input_format`); category feature not yet |
| A3 | classifier | logistic regression (C grid), linear SVM, kNN/max-sim, gradient boosting, small MLP | Linear is enough below ~1000 labels. | window AP, Brier | seconds | 2 | LR, SVM, prototype in v1 |
| A4 | negatives / weak PU | explicit negatives only; plus random unlabeled with weights (`S`, `V`) | Random negatives regularise at low label counts; heavy random weight trades precision against recall of unseen interests. | window AP, hard-negative specificity | minutes | 3 | v1 |
| A5 | multi-interest | one global model; per-interest models + max; prototypes (k centroids); global + quota | Multi-interest helps small, distant interests more than it helps global AP. | per-interest AP | minutes | 4 | v1 (only interests with >= 20 positives are interpretable) |
| A6 | zero-shot cold start | description only; + anchor papers; instruction-aware prompts; blend with learned score | Zero-shot beats supervised for the first 50-100 labels, then crosses over. | learning curve at N = 0..300 | minutes | 5 | description + anchors in v1; blending not yet |
| A7 | active learning | exploit, uncertainty, stratified, random | In screening, certainty sampling finds positives fastest; in a daily stream the gap is small. | WSS@95, recall@10% (simulation) | minutes per simulation | 6 | not yet |
| A8 | training window | all history; last 90/180 days; time-decay weights | If taste drifts, recent windows win. | prequential weekly AP | minutes | 7 | not yet (needs about 3 months of labels) |
| A9 | SetFit | small body, 5+ seeds | May edge past frozen + LR at 1000+ labels; seed variance can exceed the gain. | AP +- seed SD | hours on CPU | v2 | not yet |
| A10 | LLM weak labels | none; low-weight training labels; cold start only | Helps early, neutral or harmful later. Never used for evaluation. | AP on human-labelled test | API cost | v2 | not yet |
| A11 | LLM re-ranker | re-rank top-50 | Might raise P@10; hard to reproduce when model versions change. | P@10, nDCG@20 | 50 calls/day | v3 | not yet |
| A12 | full fine-tuning | SPECTER2 / SciNCL | No expected gain below ~3000 labels. | AP +- seed SD | GPU | v3 | not yet |
| A13 | sparse + dense fusion | rank-average of TF-IDF and dense scores | Complementary errors; used in production systems. | window AP | seconds | 3 | v1 (`fusion_tfidf_emb`) |

## What each label budget supports (assuming about 10% positives; update after the first two weeks)

| labels | ~positives | do | do not |
|---|---|---|---|
| 300 | 30 | A0, A1 with at most 4 representations, A3 (LR vs SVM vs prototype), learning curve; at most one confirmatory hypothesis | per-interest conclusions, SetFit, wide grids |
| 1000 | 100 | + A2, A4, A5 (interests with >= 20 positives), A13, SetFit as exploratory; 3-4 primary hypotheses | full fine-tuning, firm LLM re-ranker conclusions |
| 3000 | 300 | + A8, A10, A11, A12, meaningful per-interest intervals, prequential drift series | - |

## Version plan

* **v1 (this repo):** sources, storage, blind digest with random slice, baselines + five learned
  models, leakage-safe evaluation, label-efficiency curve, tests, CI.
* **v2:** finish A1/A2/A6, calibration (Platt), threshold-based digests, SetFit (A9), a
  systematic-review simulation module (ASReview-style WSS@95 and recall curves, plus a stopping
  rule such as SAFE) for literature reviews, LLM weak labels (A10).
* **v3:** drift (A8), LLM re-ranking (A11), fine-tuning (A12), reference-graph features (respecting
  their licences), automatic sub-interest discovery, external validity on public multi-user vote data.

## Pitfalls to remember

* The metric chosen decides which embedding "wins": TF-IDF can lead on nDCG against random negatives
  while dense models lead on AUC against explicit negatives. Fix the primary metric first.
* Tuning on the test windows, or picking the best of many configurations and reporting it as the
  result, invalidates the interval. Grids are exploratory.
* With about 30 positives, one reordered positive moves AP by several points. Report the interval.
* Writing interest descriptions after looking at test papers is human leakage: freeze descriptions
  by date.
