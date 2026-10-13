"""The properties that keep the evaluation honest. If these fail, no reported number can be trusted."""

import numpy as np
import pandas as pd
import pytest
from conftest import make_ctx

from paperscreen.evaluation.runner import EvalSettings, InsufficientData, add_day, first_cutoff, run_prequential
from paperscreen.paper import Paper
from paperscreen.rankers import RANKER_TYPES, Ranker, TfidfSvmRanker

SETTINGS = EvalSettings(step_days=7, min_train_labels=50, min_train_positives=5, n_boot=100)


class Spy(Ranker):
    seen: list = []

    def fit(self, data):
        Spy.seen.append(data)
        return self

    def score(self, papers):
        return np.zeros(len(papers))


@pytest.fixture
def spy(monkeypatch):
    Spy.seen = []
    monkeypatch.setitem(RANKER_TYPES, "spy", Spy)
    return Spy


def test_training_data_is_strictly_in_the_past(demo_db, cfg, spy):
    db, _ = demo_db
    labels, papers = db.labels_frame(), db.query_papers()
    res = run_prequential(labels, papers, [{"name": "spy", "type": "spy"}], make_ctx(db, cfg), SETTINGS)
    assert len(spy.seen) == len(res.windows) >= 4
    day = add_day(labels).set_index("paper_id")["day"]
    for data, (_, w) in zip(spy.seen, res.windows.iterrows(), strict=True):
        assert max(day[p.paper_id] for p in data.papers) < w["start"]
        assert len(data.papers) == w["n_train"]
        test_ids = set(res.scores[res.scores["window"] == w["window"]]["paper_id"])
        pool_ids = {p.paper_id for p in data.pool}
        assert not (pool_ids & {p.paper_id for p in data.papers}), "training label in the negative pool"
        assert not (pool_ids & test_ids), "test paper in the negative pool"
        assert all(p.published < w["start"] for p in data.pool), "future paper in the negative pool"
        assert not (test_ids & {p.paper_id for p in data.papers}), "test paper was in training"


def test_future_labels_cannot_change_past_scores(demo_db, cfg):
    db, _ = demo_db
    labels, papers, ctx = db.labels_frame(), db.query_papers(), make_ctx(db, cfg)
    specs = [{"name": n, "type": n} for n in ("emb_logreg", "tfidf_svm", "per_interest_logreg", "prototype")]
    T = "2026-07-03"
    flipped = add_day(labels)
    late = flipped["day"] >= T
    flipped.loc[late, "label"] = 1 - flipped.loc[late, "label"]
    flipped.loc[late, "tags"] = pd.Series([("topic1",)] * int(late.sum()), index=flipped.index[late])
    a = run_prequential(labels, papers, specs, ctx, SETTINGS)
    b = run_prequential(flipped.drop(columns="day"), papers, specs, ctx, SETTINGS)
    key = ["window", "ranker", "paper_id"]
    m = a.scores.merge(b.scores, on=key, suffixes=("_a", "_b"))
    starts = a.windows.set_index("window")["start"]
    m["start"] = m["window"].map(starts)
    past = m[m["start"] < T]
    assert len(past) > 100
    assert np.allclose(past["score_a"], past["score_b"]), "scores before T depend on labels made after T"
    future = m[(m["start"] >= T) & (m["ranker"] == "emb_logreg")]
    assert len(future) and not np.allclose(future["score_a"], future["score_b"]), "test is vacuous"


def test_tfidf_vocabulary_excludes_test_period_text(demo_db, cfg, monkeypatch):
    db, _ = demo_db
    token = "zzsentinelqq"
    for i in range(3):
        pid, _ = db.upsert_paper(Paper(title=f"{token} paper {i} about nothing in particular here",
                                       abstract=f"{token} filler text", published="2026-07-01",
                                       source="synthetic", arxiv_id=f"2607.9900{i}"))
        db.set_label(pid, 1, ["topic1"], labeled_at="2026-07-02T18:00:00Z")
    orig, vocabs = TfidfSvmRanker.fit, []

    def wrapped(self, data):
        r = orig(self, data)
        vocabs.append(set(self.vec.vocabulary_))
        return r

    monkeypatch.setattr(TfidfSvmRanker, "fit", wrapped)
    res = run_prequential(db.labels_frame(), db.query_papers(), [{"name": "tfidf_svm", "type": "tfidf_svm"}],
                          make_ctx(db, cfg), SETTINGS)
    w = res.windows
    idx = int(w.index[(w["start"] <= "2026-07-02") & ("2026-07-02" < w["end"])][0])
    assert token not in vocabs[idx], "test-period text leaked into the TF-IDF vocabulary"
    assert token in vocabs[idx + 1], "sentinel should be learnable once it is in the past"


def test_insufficient_data_is_reported():
    labels = pd.DataFrame({"paper_id": ["a", "b"], "label": [1, 0], "tags": [(), ()],
                           "labeled_at": ["2026-01-01T10:00:00Z"] * 2, "first_reason": ["exploit"] * 2})
    assert first_cutoff(add_day(labels), 50, 5) is None
    with pytest.raises(InsufficientData):
        run_prequential(labels, [], [], None, SETTINGS)
    with pytest.raises(InsufficientData):
        run_prequential(labels.iloc[:0], [], [], None, SETTINGS)


def test_random_slice_restriction(demo_db, cfg):
    db, _ = demo_db
    res = run_prequential(db.labels_frame(), db.query_papers(), [{"name": "random", "type": "random"}],
                          make_ctx(db, cfg), SETTINGS)
    rs = res.restrict_random()
    assert set(rs.scores["first_reason"]) == {"random"} and 0 < len(rs.scores) < len(res.scores)
    assert rs.windows["n_test"].sum() == len(rs.scores)


def test_past_windows_do_not_depend_on_which_papers_get_labelled_later(demo_db, cfg):
    """Labels made after T, even on old papers, must leave every window that ends by T unchanged.

    This catches a subtle leak: building the negative pool by excluding *every* paper that is ever
    labelled tells past windows which papers the user will pick up in the future. It only shows up
    when old papers are labelled late (the demo labels papers right after publication), so the test
    adds late labels on old, never-labelled papers.
    """
    db, _ = demo_db
    labels, papers, ctx = db.labels_frame(), db.query_papers(), make_ctx(db, cfg)
    T = "2026-07-03"
    labeled = set(labels["paper_id"])
    old = [p for p in papers if p.paper_id not in labeled and p.published < "2026-06-15"][:300]
    assert len(old) == 300
    late = pd.DataFrame({"paper_id": [p.paper_id for p in old], "label": [1, 0, 0] * 100,
                         "tags": [("topic1",)] * 300, "labeled_at": ["2026-07-20T12:00:00Z"] * 300,
                         "first_reason": ["exploit"] * 300})
    specs = [{"name": n, "type": n} for n in ("emb_logreg", "tfidf_svm", "per_interest_logreg")]
    a = run_prequential(labels, papers, specs, ctx, SETTINGS)
    b = run_prequential(pd.concat([labels, late], ignore_index=True), papers, specs, ctx, SETTINGS)
    ends = a.windows.set_index("window")["end"]
    past = [w for w in ends.index if ends[w] <= T]
    assert past, "test is vacuous"
    key = ["window", "ranker", "paper_id"]
    m = a.scores[a.scores["window"].isin(past)].merge(b.scores, on=key, suffixes=("_a", "_b"))
    assert len(m) == int(a.scores["window"].isin(past).sum())
    assert np.allclose(m["score_a"], m["score_b"]), "past scores depend on which papers are labelled later"
