"""Leakage-aware evaluation: expanding-window prequential runs and label-efficiency curves.

Rules enforced here (and covered by tests/test_leakage.py):

* Time is the *label* time. A model trained for window ``[D, D+step)`` sees only labels made on
  days strictly before ``D``.
* The pool of random negatives contains papers published before ``D`` that are neither training
  labels nor items of the test window. Papers labelled *after* the window stay in the pool, exactly
  as they would in the live system on day ``D``; excluding every ever-labelled paper would tell past
  windows which papers the user picks up later (a leak, covered by a test).
* TF-IDF vocabulary/IDF, centroids, classifiers: all fitted inside ``Ranker.fit`` on that data only.
* Rankers are rebuilt from scratch for every window.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..paper import Paper
from ..rankers import Context, TrainData, build_rankers
from .bootstrap import holm, mean_ci, paired_difference
from .metrics import average_precision, window_metrics


class InsufficientData(RuntimeError):
    pass


@dataclass
class EvalSettings:
    step_days: int = 7
    min_train_labels: int = 50
    min_train_positives: int = 5
    k: int = 10
    ndcg_k: int = 20
    n_boot: int = 2000
    seed: int = 0
    min_interest_positives: int = 20

    @classmethod
    def from_config(cls, ev: dict, seed: int = 0) -> EvalSettings:
        return cls(
            step_days=ev["step_days"], min_train_labels=ev["min_train_labels"],
            min_train_positives=ev["min_train_positives"], k=ev["k"], ndcg_k=ev["ndcg_k"],
            n_boot=ev["n_boot"], seed=seed,
            min_interest_positives=ev.get("min_interest_positives", 20),
        )


@dataclass
class PrequentialResult:
    scores: pd.DataFrame   # window, ranker, paper_id, y, tags, score, first_reason
    windows: pd.DataFrame  # window, start, end, n_train, n_train_pos, n_test, n_test_pos, n_pool

    def restrict_random(self) -> PrequentialResult:
        """Keep only test items first shown as uniformly random picks (the unbiased slice).

        Training is unchanged (it used every label); only the evaluated items are restricted.
        """
        sc = self.scores[self.scores["first_reason"] == "random"]
        wins = self.windows.copy()
        if sc.empty:
            raise InsufficientData("no labelled items came from the random slice yet")
        first = sc[sc["ranker"] == sc["ranker"].iloc[0]]
        grp = first.groupby("window")["y"].agg(["size", "sum"])
        wins = wins[wins["window"].isin(grp.index)].copy()
        wins["n_test"] = wins["window"].map(grp["size"])
        wins["n_test_pos"] = wins["window"].map(grp["sum"]).astype(int)
        return PrequentialResult(sc.reset_index(drop=True), wins.reset_index(drop=True))


def add_day(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    out["day"] = pd.to_datetime(out["labeled_at"], utc=True).dt.strftime("%Y-%m-%d")
    return out


def first_cutoff(labels: pd.DataFrame, min_labels: int, min_pos: int) -> date | None:
    """Earliest day D such that labels made before D satisfy the minimum training size."""
    days = sorted(labels["day"].unique())
    for d in days:
        cut = date.fromisoformat(d) + timedelta(days=1)
        train = labels[labels["day"] < cut.isoformat()]
        if len(train) >= min_labels and int(train["label"].sum()) >= min_pos:
            return cut
    return None


def _train_data(train: pd.DataFrame, by_id: dict[str, Paper], pool: list[Paper]) -> TrainData:
    return TrainData(
        papers=[by_id[i] for i in train["paper_id"]],
        y=train["label"].to_numpy(dtype=int),
        tags=[tuple(t) for t in train["tags"]],
        pool=pool,
    )


def _pool(all_papers: list[Paper], dates: list[str], cutoff: str, exclude: set[str]) -> list[Paper]:
    """Unlabelled-at-cutoff papers published before ``cutoff`` (``all_papers`` sorted by date)."""
    return [p for p in all_papers[: bisect_left(dates, cutoff)] if p.paper_id not in exclude]


def _fit_all(rankers: dict, data: TrainData) -> None:
    for r in rankers.values():
        if r.needs_fit:
            r.fit(data)


def run_prequential(
    labels: pd.DataFrame, all_papers: list[Paper], specs: list[dict], ctx: Context,
    settings: EvalSettings,
) -> PrequentialResult:
    """Weekly (``step_days``) expanding-window evaluation. ``all_papers`` must be sorted by date."""
    if labels.empty:
        raise InsufficientData("no labels yet")
    labels = add_day(labels)
    by_id = {p.paper_id: p for p in all_papers}
    pool_dates = [p.published for p in all_papers]

    cut = first_cutoff(labels, settings.min_train_labels, settings.min_train_positives)
    if cut is None:
        raise InsufficientData(
            f"need >= {settings.min_train_labels} labels with >= {settings.min_train_positives} "
            f"positives before the first test window; have {len(labels)} labels"
        )
    last_day = labels["day"].max()
    rows: list[dict] = []
    wins: list[dict] = []
    w = 0
    while cut.isoformat() <= last_day:
        end = cut + timedelta(days=settings.step_days)
        train = labels[labels["day"] < cut.isoformat()]
        test = labels[(labels["day"] >= cut.isoformat()) & (labels["day"] < end.isoformat())]
        if not test.empty:
            assert train["day"].max() < cut.isoformat() <= test["day"].min(), "label-time leakage"
            pool = _pool(all_papers, pool_dates, cut.isoformat(), set(train["paper_id"]) | set(test["paper_id"]))
            assert all(p.published < cut.isoformat() for p in pool), "pool contains future papers"
            data = _train_data(train, by_id, pool)
            rankers = build_rankers(specs, ctx)
            _fit_all(rankers, data)
            test_papers = [by_id[i] for i in test["paper_id"]]
            tag_str = [" ".join(t) for t in test["tags"]]
            for name, r in rankers.items():
                s = r.score(test_papers)
                rows.append(pd.DataFrame({
                    "window": w, "ranker": name, "paper_id": test["paper_id"].to_numpy(),
                    "y": test["label"].to_numpy(), "tags": tag_str, "score": s,
                    "first_reason": test["first_reason"].fillna("").to_numpy(),
                }))
            wins.append({
                "window": w, "start": cut.isoformat(), "end": end.isoformat(),
                "n_train": len(train), "n_train_pos": int(train["label"].sum()),
                "n_test": len(test), "n_test_pos": int(test["label"].sum()), "n_pool": len(pool),
            })
        w += 1
        cut = end
    if not rows:
        raise InsufficientData("no test windows contain labels")
    return PrequentialResult(pd.concat(rows, ignore_index=True), pd.DataFrame(wins))


def window_table(res: PrequentialResult, s: EvalSettings) -> pd.DataFrame:
    recs = []
    for (w, name), g in res.scores.groupby(["window", "ranker"], sort=False):
        m = window_metrics(g["y"].to_numpy(), g["score"].to_numpy(), s.k, s.ndcg_k, seed=s.seed + int(w))
        recs.append({"window": w, "ranker": name, **m})
    return pd.DataFrame(recs)


def _matrix(wt: pd.DataFrame, col: str) -> tuple[pd.DataFrame, list[int]]:
    """Ranker x window matrix of ``col`` over windows that contain both classes."""
    ref = wt.drop_duplicates("window").set_index("window")
    valid = [w for w in ref.index if ref.loc[w, "n_pos"] >= 1 and ref.loc[w, "n_pos"] < ref.loc[w, "n"]]
    piv = wt[wt["window"].isin(valid)].pivot(index="ranker", columns="window", values=col)
    order = list(dict.fromkeys(wt["ranker"]))
    return piv.loc[order], valid


def per_interest_ap(res: PrequentialResult, interest_ids: list[str], s: EvalSettings):
    """Mean per-window AP of each ranker on each interest's positives, and positives per interest."""
    out: dict[str, dict[str, float]] = {}
    npos: dict[str, int] = {}
    first = res.scores["ranker"].iloc[0]
    one = res.scores[res.scores["ranker"] == first]
    for iid in interest_ids:
        vals: dict[str, list[float]] = {}
        npos[iid] = int(((one["y"] == 1) & one["tags"].map(lambda t, i=iid: i in t.split())).sum())
        for (w, name), g in res.scores.groupby(["window", "ranker"], sort=False):
            has = g["tags"].map(lambda t, i=iid: i in t.split()).to_numpy()
            keep = (g["y"].to_numpy() == 0) | has
            yt = (g["y"].to_numpy() == 1) & has
            yt, sc = yt[keep].astype(int), g["score"].to_numpy()[keep]
            if yt.sum() == 0:
                continue
            vals.setdefault(name, []).append(average_precision(yt, sc, seed=s.seed + int(w)))
        for name, v in vals.items():
            out.setdefault(name, {})[iid] = float(np.mean(v))
    df = pd.DataFrame(out).T.reindex(columns=interest_ids)
    df.index.name = "ranker"
    return df, npos


def summarize(res: PrequentialResult, s: EvalSettings, primary: list[list[str]],
              interest_ids: list[str]) -> dict:
    wt = window_table(res, s)
    ap, valid = _matrix(wt, "ap")
    pk, _ = _matrix(wt, "p_at_k")
    nd, _ = _matrix(wt, "ndcg")
    table = []
    for name in ap.index:
        m, lo, hi = mean_ci(ap.loc[name].to_numpy(), s.n_boot, s.seed)
        table.append({"ranker": name, "map": m, "map_lo": lo, "map_hi": hi,
                      "p_at_k": float(np.nanmean(pk.loc[name])), "ndcg": float(np.nanmean(nd.loc[name]))})
    prim: list[dict] = []
    for a, b in primary:
        if a in ap.index and b in ap.index:
            prim.append({"a": a, "b": b, **paired_difference(ap.loc[a].to_numpy(), ap.loc[b].to_numpy(),
                                                             s.n_boot, s.seed)})
    adj = holm({f"{r['a']}>{r['b']}": r["p"] for r in prim})
    for r in prim:
        r["p_holm"] = adj[f"{r['a']}>{r['b']}"]
    expl = []
    for ref in ("tfidf_svm", "recency"):
        if ref in ap.index:
            for name in ap.index:
                if name != ref:
                    expl.append({"a": name, "b": ref, **paired_difference(
                        ap.loc[name].to_numpy(), ap.loc[ref].to_numpy(), s.n_boot, s.seed)})
    first = res.scores["ranker"].iloc[0]
    one = res.scores[res.scores["ranker"] == first]
    pi, pi_n = per_interest_ap(res, interest_ids, s)
    return {
        "map_table": pd.DataFrame(table),
        "primary": pd.DataFrame(prim),
        "exploratory": pd.DataFrame(expl),
        "per_interest": pi,
        "per_interest_n": pi_n,
        "window_metrics": wt,
        "windows": res.windows,
        "n_windows_used": len(valid),
        "n_windows_total": int(res.windows.shape[0]),
        "n_test_labels": int(len(one)),
        "n_test_pos": int(one["y"].sum()),
    }


def run_learning_curve(
    labels: pd.DataFrame, all_papers: list[Paper], specs: list[dict], ctx: Context,
    settings: EvalSettings, sizes: list[int], test_fraction: float = 0.3,
) -> pd.DataFrame:
    """Train on the first N labels (in time), test on a fixed final stretch, by weekly units."""
    labels = add_day(labels)
    n_test = max(1, int(len(labels) * test_fraction))
    test_start = labels.iloc[len(labels) - n_test]["day"]
    test = labels[labels["day"] >= test_start]
    cands = labels[labels["day"] < test_start]
    by_id = {p.paper_id: p for p in all_papers}
    pool_dates = [p.published for p in all_papers]
    test_ids = set(test["paper_id"])
    t0 = date.fromisoformat(test_start)
    unit = test["day"].map(lambda d: (date.fromisoformat(d) - t0).days // settings.step_days).to_numpy()
    test_papers = [by_id[i] for i in test["paper_id"]]
    y_test = test["label"].to_numpy(dtype=int)
    recs = []
    for n in sizes:
        if n > len(cands):
            continue
        train = cands.iloc[:n]
        cutoff = (date.fromisoformat(train["day"].iloc[-1]) + timedelta(days=1)).isoformat()
        assert train["day"].max() < test_start
        pool = _pool(all_papers, pool_dates, cutoff, set(train["paper_id"]) | test_ids)
        data = _train_data(train, by_id, pool)
        rankers = build_rankers(specs, ctx)
        _fit_all(rankers, data)
        for name, r in rankers.items():
            sc = r.score(test_papers)
            aps = []
            for u in np.unique(unit):
                m = unit == u
                if 0 < y_test[m].sum() < m.sum():
                    aps.append(average_precision(y_test[m], sc[m], seed=settings.seed + int(u)))
            mm, lo, hi = mean_ci(np.array(aps), settings.n_boot, settings.seed)
            recs.append({"n_labels": n, "n_train_pos": int(train["label"].sum()), "ranker": name,
                         "map": mm, "lo": lo, "hi": hi, "n_units": len(aps)})
    return pd.DataFrame(recs)
