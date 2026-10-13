"""The daily routine as one function (used by `paperscreen refresh` and the web app)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from .config import Config
from .db import Database
from .digest import Digest, build_digest
from .rankers import Context
from .sources import build_sources, fetch_all


def refresh(cfg: Config, db: Database, ctx: Context, days: int | None = None,
            log: Callable[[str], None] = print, today: date | None = None) -> Digest:
    """Fetch new papers from every enabled source, then build today's digest."""
    sources, skipped = build_sources(cfg)
    for s in skipped:
        log(f"{s.source}: skipped ({s.skipped})")
    for r in fetch_all(db, sources, days=days, today=today):
        log(f"{r.source}: ERROR {r.error}" if r.error else f"{r.source}: {r.fetched} fetched, {r.new} new")
        if r.warning:
            log(f"{r.source}: WARNING {r.warning}")
    log("ranking and building the digest (the first run downloads the embedding model)...")
    dg = build_digest(db, cfg, ctx, today)
    log(f"digest ready: {len(dg.entries)} papers ({dg.model_id})" if dg.entries else "no new candidates")
    return dg
