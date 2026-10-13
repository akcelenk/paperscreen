import json

import numpy as np
import pytest
from conftest import make_ctx

from paperscreen import registry
from paperscreen.digest import build_digest, training_data
from paperscreen.rankers import RANKER_TYPES


@pytest.mark.parametrize("typ", ["emb_logreg", "tfidf_svm", "per_interest_logreg", "prototype", "keyword"])
def test_saved_model_reproduces_scores(demo_db, cfg, tmp_path, typ):
    db, _ = demo_db
    ctx = make_ctx(db, cfg)
    data = training_data(db)
    r = RANKER_TYPES[typ](ctx).fit(data)
    snap = registry.label_snapshot(data)
    card = registry.ModelCard(version=registry.make_version(typ, snap), created="2026-10-13T00:00:00Z", ranker=typ,
                              params={}, embedder="hash-256", input_format=None, n_labels=len(data.papers),
                              n_pos=data.n_pos, n_neg=data.n_neg, n_pool=len(data.pool), label_snapshot=snap,
                              config="x", git=None, fit_seconds=0.1)
    path = registry.save_model(tmp_path, r, card)
    r2, card2 = registry.load_model(path, make_ctx(db, cfg))
    papers = db.query_papers()[:200]
    assert registry.same_scores(r.score(papers), r2.score(papers))
    assert card2 == card


def test_label_snapshot_is_order_independent_and_sensitive(demo_db):
    db, _ = demo_db
    data = training_data(db)
    a = registry.label_snapshot(data)
    order = np.arange(len(data.papers))[::-1]
    data_rev = type(data)([data.papers[i] for i in order], data.y[order], [data.tags[i] for i in order], data.pool)
    assert registry.label_snapshot(data_rev) == a
    data.y = data.y.copy()
    data.y[0] = 1 - data.y[0]
    assert registry.label_snapshot(data) != a


def test_digest_registers_model_and_links_impressions(demo_db, cfg, tmp_path):
    db, last = demo_db
    root = tmp_path / "models"
    c = cfg.with_overrides(model={"registry_dir": str(root), "keep_versions": 2},
                           digest={"out_dir": str(tmp_path / "digests")})
    ctx = make_ctx(db, c)
    versions = []
    for _ in range(3):  # three digests on the same day: three versions, the oldest pruned
        dg = build_digest(db, c, ctx, today=last)
        versions.append(dg.model_id)
    cards = registry.list_models(root)
    assert [x.version for x in cards] == versions[1:]
    card = json.loads((root / versions[-1] / "card.json").read_text())
    assert card["ranker"] == "emb_logreg" and card["n_labels"] > 0 and card["config"] == c.fingerprint()
    ids = {r[0] for r in db.conn.execute("SELECT DISTINCT model_id FROM impressions WHERE model_id LIKE 'emb_logreg-%'")}
    assert set(versions) <= ids
