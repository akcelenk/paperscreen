import numpy as np
import pytest
from conftest import make_ctx

from paperscreen.digest import training_data
from paperscreen.evaluation.metrics import average_precision
from paperscreen.paper import Paper
from paperscreen.rankers import RANKER_TYPES, TrainData, build_active_ranker, build_rankers, default_specs, negative_weights


def test_negative_weights_mass_and_share():
    we, wr = negative_weights(n_pos=10, n_exp=20, n_rand=100, S=2.0, V=0.25)
    assert we * 20 + wr * 100 == pytest.approx(20.0)
    assert wr * 100 == pytest.approx(20.0 * 0.25)
    assert negative_weights(5, 0, 50, 1.0, 0.5) == (0.0, pytest.approx(5 / 50))
    assert negative_weights(5, 10, 0, 1.0, 0.5) == (pytest.approx(5 / 10), 0.0)


def _split(demo_db, cfg, day_cut):
    db, _ = demo_db
    data = training_data(db)
    labels = db.labels_frame()
    keep = (labels["labeled_at"] < day_cut).to_numpy()
    by = {p.paper_id: p for p in data.papers}
    train = TrainData(
        papers=[p for p, k in zip(data.papers, keep, strict=True) if k], y=data.y[keep],
        tags=[t for t, k in zip(data.tags, keep, strict=True) if k],
        pool=[p for p in data.pool if p.published < day_cut[:10]],
    )
    test_ids = labels.loc[~keep, "paper_id"].tolist()
    return train, [by[i] for i in test_ids], data.y[~keep]


def test_learned_rankers_beat_random_on_synthetic_stream(demo_db, cfg):
    db, _ = demo_db
    ctx = make_ctx(db, cfg)
    train, test, y = _split(demo_db, cfg, "2026-06-25")
    specs = default_specs({"C": 1.0, "S": 1.0, "V": 0.5})
    rankers = build_rankers(specs, ctx)
    for r in rankers.values():
        if r.needs_fit:
            r.fit(train)
    ap = {n: average_precision(y, r.score(test), seed=0) for n, r in rankers.items()}
    for name in ("tfidf_svm", "emb_logreg", "prototype", "per_interest_logreg", "keyword", "zeroshot"):
        assert ap[name] > ap["random"] + 0.2, (name, ap)
    assert set(ap) == {s["name"] for s in specs}


def test_rankers_survive_degenerate_training_data(db, cfg):
    ctx = make_ctx(db, cfg)
    papers = []
    for i in range(3):
        pid, _ = db.upsert_paper(Paper(title=f"title {i}", abstract="alpha beta", published="2026-01-01",
                                       source="synthetic", arxiv_id=f"2610.1000{i}"))
        papers += db.get_papers([pid])
    for y in (np.zeros(3, dtype=int), np.ones(3, dtype=int)):
        data = TrainData(papers, y, [()] * 3, pool=[])
        for typ in ("tfidf_svm", "emb_logreg", "prototype", "per_interest_logreg"):
            r = RANKER_TYPES[typ](ctx).fit(data)
            assert r.score(papers).shape == (3,)


def test_active_ranker_uses_zeroshot_until_enough_labels(demo_db, cfg):
    db, _ = demo_db
    ctx = make_ctx(db, cfg)
    train, _, _ = _split(demo_db, cfg, "2026-06-25")
    small = TrainData(train.papers[:20], train.y[:20], train.tags[:20], train.pool)
    assert build_active_ranker(cfg["model"] | {"ranker": "emb_logreg"}, ctx, small)[1] == "zeroshot"
    assert build_active_ranker(cfg["model"] | {"ranker": "emb_logreg"}, ctx, train)[1] == "emb_logreg"


def test_fusion_refs_and_duplicate_names(db, cfg):
    ctx = make_ctx(db, cfg)
    with pytest.raises(ValueError):
        build_rankers([{"name": "a", "type": "random"}, {"name": "a", "type": "recency"}], ctx)
    with pytest.raises(ValueError):
        build_rankers([{"name": "a", "type": "nope"}], ctx)
    r = build_rankers([{"name": "r", "type": "recency"}, {"name": "f", "type": "fusion", "of": ["r"]}], ctx)
    ps = [Paper(paper_id=str(i), title="t", published=f"2026-01-0{i + 1}") for i in range(3)]
    assert list(np.argsort(r["f"].score(ps))) == [0, 1, 2]


def test_prototype_kmeans_option(demo_db, cfg):
    db, _ = demo_db
    ctx = make_ctx(db, cfg)
    train, test, _ = _split(demo_db, cfg, "2026-06-25")
    r = RANKER_TYPES["prototype"](ctx, k=2).fit(train)
    assert r.score(test).shape == (len(test),)
