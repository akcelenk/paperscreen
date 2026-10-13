from collections import Counter
from datetime import date

from conftest import make_ctx

from paperscreen.db import Database
from paperscreen.digest import build_digest
from paperscreen.paper import Paper


def test_digest_structure_and_logging(demo_db, cfg):
    db, last = demo_db
    ctx = make_ctx(db, cfg)
    dg = build_digest(db, cfg, ctx, today=last, write=False)
    dc = cfg["digest"]
    assert dg.model_id == "emb_logreg"
    reasons = Counter(e.reason for e in dg.entries)
    assert reasons["random"] == dc["n_random"]
    assert reasons["exploit"] <= dc["per_interest"] * len(cfg.interests)
    ids = {e.paper.paper_id for e in dg.entries}
    assert len(ids) == len(dg.entries)
    logged = db.conn.execute("SELECT reason, COUNT(*) n FROM impressions WHERE shown_at LIKE ? GROUP BY reason",
                             (f"{last.isoformat()}T%",)).fetchall()
    assert {r["reason"]: r["n"] for r in logged}.get("random", 0) >= dc["n_random"]
    md = dg.markdown.lower()
    assert "random" not in md.replace("randomly", "") and "score" not in md  # blind: no markers, no scores
    assert all(f"`{i}`" in dg.markdown for i in ids)


def test_second_digest_never_repeats_papers(demo_db, cfg):
    db, last = demo_db
    ctx = make_ctx(db, cfg)
    a = build_digest(db, cfg, ctx, today=last, write=False)
    b = build_digest(db, cfg, ctx, today=last, write=False)
    assert not ({e.paper.paper_id for e in a.entries} & {e.paper.paper_id for e in b.entries})


def test_cold_start_uses_zero_shot(db, cfg):
    ctx = make_ctx(db, cfg)
    for i in range(30):
        db.upsert_paper(Paper(title=f"Cold start paper number {i} with words", abstract="alpha beta gamma",
                              published="2026-10-13", source="synthetic", arxiv_id=f"2610.{i:05d}"),
                        now="2026-10-13T06:00:00Z")
    dg = build_digest(db, cfg, ctx, today=date(2026, 10, 13), write=False)
    assert dg.model_id == "zeroshot" and {e.reason for e in dg.entries} <= {"coldstart", "random"}


def test_no_candidates(db, cfg):
    dg = build_digest(db, cfg, make_ctx(db, cfg), today=date(2026, 10, 13), write=False)
    assert dg.entries == [] and "No new candidates" in dg.markdown


def test_proxy_link_and_no_overwrite(tmp_path, cfg):
    db = Database(":memory:")
    ctx = make_ctx(db, cfg)
    for i in range(12):
        db.upsert_paper(Paper(title=f"Paywalled paper number {i} on something", abstract="alpha beta",
                              published="2026-10-13", source="openalex", doi=f"10.1/abc{i}",
                              url=f"https://x.test/{i}"), now="2026-10-13T06:00:00Z")
    c2 = cfg.with_overrides(digest={"proxy_prefix": "https://ezproxy.example/login?url=",
                                    "out_dir": str(tmp_path), "per_interest": 1, "n_random": 1})
    first = build_digest(db, c2, ctx, today=date(2026, 10, 13))
    assert "https://ezproxy.example/login?url=https://doi.org/10.1/abc" in first.markdown
    second = build_digest(db, c2, ctx, today=date(2026, 10, 13))
    assert first.path != second.path and len(list(tmp_path.glob("*.md"))) == 2
    db.close()


def test_impressions_log_propensity_position_and_uncertain_arm(demo_db, cfg, tmp_path):
    db, last = demo_db
    c = cfg.with_overrides(digest={"out_dir": str(tmp_path), "n_uncertain": 3})
    ctx = make_ctx(db, c)
    dg = build_digest(db, c, ctx, today=last)
    rows = db.conn.execute("SELECT reason, propensity, n_candidates, position FROM impressions "
                           "WHERE model_id=? ORDER BY position", (dg.model_id,)).fetchall()
    reasons = [r[0] for r in rows]
    assert reasons.count("uncertain") == 3 and reasons.count("random") == c["digest"]["n_random"]
    assert [r[3] for r in rows] == list(range(len(rows)))
    n = rows[0][2]
    assert n == dg.n_candidates and all(r[2] == n for r in rows)
    for reason, prop, *_ in rows:
        assert (0 < prop < 1) if reason == "random" else prop == 1.0
    # the uncertain picks sit closest to the boundary among papers the model did not pick
    unc = {e.paper.paper_id: e.score for e in dg.entries if e.reason == "uncertain"}
    assert max(abs(s) for s in unc.values()) <= min(abs(e.score) for e in dg.entries if e.reason == "random")


def test_more_batches_on_one_day_draw_fresh_random_picks(demo_db, cfg, tmp_path):
    db, last = demo_db
    c = cfg.with_overrides(digest={"out_dir": str(tmp_path)})
    ctx = make_ctx(db, c)
    a = build_digest(db, c, ctx, today=last)
    b = build_digest(db, c, ctx, today=last)
    pa = {e.paper.paper_id for e in a.entries}
    assert pa.isdisjoint(e.paper.paper_id for e in b.entries)


def test_no_uncertain_arm_during_cold_start(cfg, tmp_path):
    db = Database(":memory:")
    ctx = make_ctx(db, cfg)
    for i in range(30):
        db.upsert_paper(Paper(title=f"Cold start paper {i} about something", abstract="alpha beta gamma",
                              published="2026-10-13", source="arxiv", arxiv_id=f"2610.0400{i:02d}"))
    dg = build_digest(db, cfg.with_overrides(digest={"out_dir": str(tmp_path)}), ctx, today=date(2026, 10, 13))
    assert dg.model_id.startswith("zeroshot") and {e.reason for e in dg.entries} <= {"coldstart", "random"}
    db.close()


def test_random_negative_pool_survives_a_wide_lookback(demo_db, cfg, tmp_path, monkeypatch):
    """Right after a big first fetch every paper is a candidate; the pool must not go empty."""
    import paperscreen.digest as digest_mod

    db, last = demo_db
    seen = []
    orig = digest_mod.build_active_ranker

    def spy(cfg_model, ctx, data):
        seen.append(len(data.pool))
        return orig(cfg_model, ctx, data)

    monkeypatch.setattr(digest_mod, "build_active_ranker", spy)
    c = cfg.with_overrides(digest={"out_dir": str(tmp_path), "lookback_days": 3650})
    build_digest(db, c, make_ctx(db, c), today=last)
    n_unlabeled = db.count_papers() - len(db.labels_frame())
    assert seen[0] > 0.9 * n_unlabeled
    waiting = {p.paper_id for p in db.unlabeled_impressions()}
    build_digest(db, c, make_ctx(db, c), today=last)
    assert seen[1] == n_unlabeled - len(waiting)  # only papers waiting for a judgement are held out


def test_candidate_count_matches_candidate_list(demo_db, cfg):
    from paperscreen.digest import candidate_since, candidates

    db, last = demo_db
    c = cfg.with_overrides(digest={"lookback_days": 20})
    assert db.count_candidates(candidate_since(c, last)) == len(candidates(db, c, last)) > 0


def test_skipped_papers_leave_the_waiting_list(demo_db, cfg, tmp_path):
    db, last = demo_db
    c = cfg.with_overrides(digest={"out_dir": str(tmp_path)})
    build_digest(db, c, make_ctx(db, c), today=last)
    waiting = db.unlabeled_impressions()
    first = waiting[0].paper_id
    db.skip(first)
    assert first not in {p.paper_id for p in db.unlabeled_impressions()}
    db.unskip(first)
    assert first in {p.paper_id for p in db.unlabeled_impressions()}
    db.skip(first)
    db.set_label(first, 1)  # judging a skipped paper later clears the skip
    assert db.conn.execute("SELECT COUNT(*) FROM skips").fetchone()[0] == 0
