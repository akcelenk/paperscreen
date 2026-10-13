# Changelog

## 0.4.0 (2026-10-13)

Fixed
- arXiv coverage: one newest-first query capped at 3000 papers dropped the oldest days of a busy week.
  The fetch now queries one submission day at a time (`submittedDate`), and a day that still hits
  `max_per_day` is reported in the fetch log and as a monitor alert.
- Several "more" batches on one day reused the same random stream.

Added
- Active-learning arm: `digest.n_uncertain` papers closest to the decision boundary per batch.
- Impressions log propensity, candidate-pool size and position (for off-policy evaluation later).
- Labels log whether a link was opened and a hash of the judged text. Nothing about time spent is
  recorded, on purpose: judging should never feel timed. Existing
  databases are migrated in place.
- `gte-base` and `gte-large` embedders.
- A one-line labelling guideline in the review card.

## 0.3.0 (2026-10-13)

Added
- "More papers" (key `m`): the next ranked batch from papers already downloaded, without fetching,
  so you can label as many papers as you like in one sitting. The model is refitted for every batch.
- The header shows how many unseen papers are left and how far the model is from leaving cold start.

Changed
- `digest.lookback_days` default 3 -> 7, so a weekly rhythm does not skip papers.

## 0.2.0 (2026-10-13)

Fixed
- Evaluation leak: the random-negative pool excluded every paper that is ever labelled, so past
  windows knew which papers the user would pick up later. The pool now excludes only training labels
  and the window's test items, as in the live system. A regression test adds late labels on old
  papers and checks that earlier windows do not change (it failed before the fix).
- The learning-curve pool had the same problem; fixed the same way.
- `python -m paperscreen.cli` started before the last commands were registered.
- CLI commands left SQLite connections open.

Added
- Model registry (`registry.py`, `paperscreen models`): every digest model is saved with a model card;
  impressions record the model version; saved models reproduce their scores; old versions are pruned.
- Monitoring (`monitor.py`, `paperscreen monitor`, Monitor tab): model picks vs random picks with
  Wilson/Newcombe intervals, per-source data quality and freshness, label backlog, alerts, non-zero
  exit on alerts.
- Config validation with all problems listed at once.
- Label snapshot hash and runtime in every evaluation report.
- SQLite WAL mode and a busy timeout (web app and background refresh write at the same time).
- Docker: non-root user, health check, models kept in the data volume; CI builds and boots the image.
- CI: mypy, coverage floor (85%), stricter ruff rules (bugbear, imports, pyupgrade); `zip(strict=True)`
  everywhere; pre-commit hooks that refuse to commit personal data; `make check`.

## 0.1.0 (2026-10-13)

First version: arXiv/PubMed/OpenAlex fetching with cross-source deduplication, embedding and TF-IDF
rankers, blind daily digest with random picks, leakage-aware prequential evaluation with cluster
bootstrap and Holm correction, local web app, Docker.
