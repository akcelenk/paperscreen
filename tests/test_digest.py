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
