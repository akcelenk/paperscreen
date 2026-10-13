# paperscreen

[![ci](https://github.com/akcelenk/paperscreen/actions/workflows/ci.yml/badge.svg)](https://github.com/akcelenk/paperscreen/actions/workflows/ci.yml)

A personal screening and ranking system for new research papers. It pulls fresh papers from
arXiv, PubMed and OpenAlex, shows you a short daily digest, learns from your yes/no judgements,
and measures honestly how much the learning helps.

The ranking is the easy part. The point of the project is the **evaluation discipline** around it:
time-split evaluation that cannot leak the future, uniformly random picks mixed into every digest
so the model is not only graded on its own taste, paired bootstrap confidence intervals, a fixed
analysis plan, and an ablation roadmap that compares simple baselines (TF-IDF) with embedding
models before anything heavier is tried.

> Status: v1. Runs on 2 CPU cores, no GPU. Everything is tested offline; the demo uses synthetic
> data. Claims about which method works best are deliberately absent: that is what your own labels
> and `paperscreen report` are for. Results are about one user and one period, not about methods in general.

## Quick start

Install once (this is the only terminal step), then use the browser app:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[embed,plots]"        # drop `embed` to use only the dependency-free hashing embedder
paperscreen init --write-config        # creates config.yaml: edit interests, contact_email, proxy link
export OPENALEX_API_KEY=...            # free key; OpenAlex is skipped without it
paperscreen refresh --days 60          # first run only: two months of papers to learn from (~20-30 min)
paperscreen app                        # opens http://127.0.0.1:8765 in your browser
```

In the app: **Fetch new papers and build digest**, then judge papers one at a time with the buttons
or the keyboard (`y` interesting, `n` not, `s` not sure (it will not come back), `z` undo, `1`-`5`
toggle interest tags, `m` next batch from papers already downloaded, no network). There is no schedule
and no quota: open it whenever you like, judge as many as you like (the header shows how many unseen
papers are left and when the model starts learning). After a break, **Fetch** catches up on everything
since the last run (up to `sources.max_catchup_days`, 60 by default). The
**Report** tab runs the offline evaluation; the **Monitor** tab shows how the live system is doing. `scripts/start.bat` (Windows) and
`scripts/start.sh` (macOS/Linux) start the app with a double click once it is installed.
The app listens on 127.0.0.1 only, rejects foreign Host headers and cross-site posts, and your data
never leaves the computer.

The same things are available from the command line:

```bash
paperscreen demo                       # whole pipeline on synthetic data: no network, no personal data
paperscreen refresh                    # fetch from all sources + build today's digest (the daily routine)
paperscreen label                      # judge shown papers in the terminal
paperscreen report                     # time-split evaluation -> reports/<date>_<hash>/report.md
paperscreen monitor                    # online quality, data quality, freshness, backlog; exit 1 on alerts
paperscreen models                     # saved model versions with their training data and config
paperscreen stats
```

### Docker (optional)

```bash
docker compose up --build              # then open http://127.0.0.1:8765
```

Everything you want to keep (labels, database, config, digests, reports, model cache) lives in `./data`,
which the compose file mounts, so rebuilding the image loses nothing. The port is published to
`127.0.0.1` only; do not expose it to a network, the app has no login. Edit `data/config.yaml` after the
first start to change interests, `contact_email` or the university proxy link.

## How it works

```
 sources (arXiv, PubMed, OpenAlex)          your judgements
        │  dedupe by arXiv id / DOI / PMID / title        ▲
        ▼                                                 │
   SQLite ──► embeddings (cached) ──► ranker ──► digest ──┘
 papers, labels,                       ▲           │ quota per interest + unmarked random picks
 impressions, embeddings               │           ▼
                                       └──── report (leakage-safe, weekly windows, bootstrap CIs)
```

* **Cold start.** Until there are enough labels (`model.min_labels/min_positives/min_negatives`) the
  digest ranks by similarity to your interest descriptions (zero-shot). Afterwards it uses the
  configured model, refit on all labels at every digest (fits take seconds).
* **Interests.** Five starter interests live in `config.yaml`. Each is a name, a description and
  optional anchor abstracts. Papers are grouped under their closest interest, and you tag positives
  with the interests they satisfy. Per-interest metrics make it visible when a small interest is being
  swamped by a big one.
* **Blind digest.** Items carry no scores or markers and are shuffled inside each interest group.
  A few (default 5) are drawn uniformly at random from all candidates; the impression log records
  which. Those are what make later evaluation unbiased.
* **Active learning.** Once the model is trained, each batch also holds a few papers it is least sure
  about (`digest.n_uncertain`, logged as `uncertain`): a label near the decision boundary teaches the
  model the most. They are kept out of the unbiased random slice and shown as their own arm in the monitor.
* **What is logged, because it cannot be recovered later.** Every impression stores its arm, the
  probability that arm picked it (1 for deterministic arms, k/N for the uniform picks), the size of
  the candidate pool and its position in the batch, so new rankers can later be scored off-policy
  (inverse propensity weighting) on old data. Every label stores whether a link was opened first and a hash of the exact text that was judged.
  Nothing about time spent is recorded: judging should never feel timed.

* **Paywalled journals.** Ranking uses titles and abstracts only, which OpenAlex and PubMed provide
  for subscription journals without any subscription. Set `digest.proxy_prefix` to your university's
  proxy (for example an EZproxy login URL) and each digest entry gets a "via university" link for when
  you decide to read it. The project never downloads publisher PDFs.

## Models (all in `src/paperscreen/rankers.py`)

| name | what it is |
|---|---|
| `random`, `recency`, `category` | baselines that every learned model must beat |
| `keyword` | TF-IDF similarity to the interest descriptions (no labels) |
| `zeroshot` | embedding similarity to the interest descriptions (no labels) |
| `tfidf_svm` | TF-IDF + class-balanced linear SVM (the arxiv-sanity-lite recipe) |
| `emb_logreg` | frozen embedding + weighted L2 logistic regression; random unlabeled papers act as weak negatives (`C`, `S`, `V` knobs) |
| `prototype` | max cosine to per-interest centroids of your positives |
| `per_interest_logreg` | one logistic regression per interest, max-combined |
| `fusion_tfidf_emb` | rank average of sparse and dense scores |

Embedders: `hash-256` (offline, lexical), `bge-small`, `minilm`, `specter2-base`, `scincl`,
`gte-base`, `gte-large` (Scholar Inbox found GTE-large best on explicit ratings; slower on CPU)
(`embedding.model`). Check each model card's licence before redistributing anything derived from it.

## Evaluation design

The harness is in `src/paperscreen/evaluation/`; the rules it enforces are tested in
`tests/test_leakage.py`.

* Weekly expanding windows on **label time**. A model for window `[D, D+7d)` sees only labels made
  before day `D`. Rankers are rebuilt per window; the TF-IDF vocabulary and IDF are fitted on
  training-period text only (a test plants a sentinel word to prove it).
* The pool of random negatives holds papers published before `D` that are not training labels and not
  in the test window. Papers labelled later stay in it, as they would in the live system on day `D`
  (excluding them would leak which papers you pick up in the future; a test checks this).
* Metric: average precision per window (also P@10, nDCG@20), averaged over windows. Ties are broken
  randomly with a seed, never by input order.
* Uncertainty: cluster bootstrap over windows; paired differences; **Holm correction over three primary
  comparisons** fixed in `ANALYSIS_PLAN.md`. Everything else is labelled exploratory.
* Two views: all labelled items (more power, biased by earlier models) and the random slice only
  (unbiased, few positives, so expect wide intervals).
* Label-efficiency curve: train on the first N labels, test on a fixed later period.
* Labels made by an LLM are never used for evaluation (the database records each label's origin).

The ablation roadmap (which axes, in which order, with what label budget) is in
[`docs/ABLATIONS.md`](docs/ABLATIONS.md).

## Production concerns

The parts of an ML system that are not the model, at the scale of one user:

* **Model registry and lineage.** Every model that ranks a digest is saved under `models/<version>/`
  with a model card (`card.json`: ranker, parameters, embedder, number of labels, a hash of the
  exact training labels, config fingerprint, git commit, fit time). Each impression stores the
  version id, so any paper you were shown can be traced to the model, data and code that chose it,
  and `registry.load_model` reproduces its scores (tested). Old versions are pruned (`keep_versions`).
* **Online evaluation with a control group.** The random picks in each digest are shown exactly like
  model picks. `paperscreen monitor` compares how often each arm is judged interesting (Wilson
  intervals per week, Newcombe interval for the difference, lift). This is an A/B test that runs by
  itself; the offline report and the online numbers should tell the same story.
* **Data quality and freshness.** Per source: volume, share of records without abstract or authors,
  age of the newest paper. Alerts fire on a stale source, a parser that suddenly loses abstracts, a
  growing label backlog, a week without judgements, or model picks doing worse than random.
  `monitor` exits non-zero on alerts, so `scripts/daily.sh` under cron mails you only then.
* **Train/serve parity.** The evaluation harness and the digest build training data with the same
  rules (labels before the cutoff; random negatives from papers unlabelled at that time, excluding the
  items being ranked), so offline numbers describe the system you actually run.
* **Reproducibility.** Seeds everywhere, config fingerprint + git commit + label snapshot hash in every
  report and model card, an analysis plan written before the data (`ANALYSIS_PLAN.md`).
* **Config validation.** The merged config is checked at start-up and every problem is listed at once
  (misspelt ranker or embedder, out-of-range weights, a proxy link that is not https).
* **Quality gates.** CI runs ruff (incl. bugbear and import order), mypy, pytest with a coverage floor
  of 85% (currently above 90%), the synthetic end-to-end demo, and builds and boots the Docker image.
  `pre-commit` runs the linters locally and refuses to commit databases, labels or `config.yaml`.
* **Operations.** SQLite in WAL mode so the web app can save labels while a background refresh writes;
  a container that runs as a non-root user with a health check; one background job at a time.

## Data, licences and privacy

* arXiv metadata (title, abstract, authors, categories) is released under CC0 per the arXiv API
  terms of use; full texts are not fetched. The API allows one request per three seconds with a
  single connection, summed over all your machines; the client enforces this. For history use the
  Kaggle metadata snapshot (`paperscreen import-jsonl`) rather than the API.
* PubMed (NCBI E-utilities): set `NCBI_API_KEY` for a higher rate limit and `contact_email` in the config.
* OpenAlex requires a free API key (`OPENALEX_API_KEY`).
* Semantic Scholar is **not** used: its API licence limits use to non-commercial internal use and
  forbids sharing the data, which does not fit a public repository.
* `data/`, `digests/`, `reports/`, `models/` and `config.yaml` are git-ignored: your labels are
  personal data. Terms change; check the sources' current terms before you publish a dataset.

## Scheduling

```cron
30 6 * * * cd ~/paperscreen && scripts/daily.sh     # fetch and build the digest (or use the app's button)
0  7 * * 1 cd ~/paperscreen && scripts/weekly.sh    # evaluation report
```

Keep the fetch to one process: the arXiv rate limit is global, not per process.

## Development

```bash
make install     # editable install with all extras + pre-commit hook
make check       # ruff + mypy + tests with coverage (what CI runs)
make demo        # synthetic end-to-end run: digest, report, monitor.md
```

## Known limitations

* The OpenAlex parser is tested against a hand-written fixture that follows the documented schema,
  not a recorded live response (the shared free quota was exhausted during development). Run
  `paperscreen fetch --source openalex` once with your key and look at the output.
* arXiv is queried one submission day at a time, so a busy week is never cut off by a global cap; a day
  that still hits `max_per_day` raises a monitor alert.
* The arXiv API can lag by days around weekends; the fetch watermark follows the newest paper
  seen, with a 3-day overlap, so late-indexed papers are picked up.
* One labeller. Power is low with a few hundred labels: expect wide intervals and say so.
* Not implemented yet: calibration, coverage@k, embedding-drift monitoring, SetFit, LLM re-ranking
  and weak labels, systematic-review simulation (see the roadmap).
* The Docker image is built and smoke-tested in CI; it was not built on the development machine.

## Layout

```
src/paperscreen/  sources/ (arxiv, pubmed, openalex)  db.py  embeddings.py  rankers.py  digest.py
                  evaluation/ (runner, bootstrap, metrics, report)  registry.py  monitor.py
                  pipeline.py  webui.py + web/index.html  config.py  cli.py  simulate.py
tests/            offline; fixtures: one recorded arXiv record (CC0 metadata), one PubMed record with the
                  publisher abstract replaced by placeholder text, one hand-written OpenAlex response
docs/ABLATIONS.md  ANALYSIS_PLAN.md  CHANGELOG.md  scripts/  docker/  Dockerfile  docker-compose.yml
.github/workflows/ci.yml  .pre-commit-config.yaml  Makefile
```

MIT licensed.
