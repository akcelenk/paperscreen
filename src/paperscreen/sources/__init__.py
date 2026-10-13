"""Paper sources and the fetch pipeline that stores them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from ..config import Config
from ..db import Database
from .arxiv import ArxivSource, import_kaggle_jsonl
from .base import Source, SourceError
from .openalex import OpenAlexSource
from .pubmed import PubMedSource

__all__ = [
    "ArxivSource", "OpenAlexSource", "PubMedSource", "Source", "SourceError",
    "FetchResult", "build_sources", "fetch_all", "import_kaggle_jsonl",
]


@dataclass
class FetchResult:
    source: str
    fetched: int = 0
    new: int = 0
    error: str | None = None
    skipped: str | None = None


def build_sources(cfg: Config, only: list[str] | None = None) -> tuple[list[Source], list[FetchResult]]:
    """Instantiate enabled sources; ones that cannot start (e.g. no key) come back as skipped."""
    sources: list[Source] = []
    skipped: list[FetchResult] = []
    email = cfg.get("contact_email")
    sc = cfg["sources"]

    def wanted(name: str) -> bool:
        return sc.get(name, {}).get("enabled", False) and (not only or name in only)

    if wanted("arxiv"):
        a = sc["arxiv"]
        sources.append(ArxivSource(a["categories"], page_size=a.get("page_size", 100),
                                   max_papers=a.get("max_papers", 3000), contact=email))
    if wanted("pubmed"):
        p = sc["pubmed"]
        sources.append(PubMedSource(p["queries"], api_key=cfg.secret("NCBI_API_KEY"), email=email,
                                    max_per_query=p.get("max_per_query", 200),
                                    require_abstract=p.get("require_abstract", True)))
    if wanted("openalex"):
        o = sc["openalex"]
        try:
            sources.append(OpenAlexSource(o["queries"], api_key=cfg.secret("OPENALEX_API_KEY"), email=email,
                                          max_per_query=o.get("max_per_query", 200)))
        except SourceError as exc:
            skipped.append(FetchResult("openalex", skipped=str(exc)))
    return sources, skipped


def fetch_all(
    db: Database, sources: list[Source], days: int | None = None, today: date | None = None,
) -> list[FetchResult]:
    """Fetch every source since its watermark (or ``days`` back) and upsert with deduplication."""
    today = today or datetime.now(timezone.utc).date()
    results: list[FetchResult] = []
    for src in sources:
        res = FetchResult(src.name)
        mark = db.kv_get(f"watermark:{src.name}")
        if days is not None:
            since = today - timedelta(days=days)
        elif mark:
            # Overlap by 3 days (indexing lags, weekends) but never look back more than 14 days.
            since = max(date.fromisoformat(mark) - timedelta(days=3), today - timedelta(days=14))
        else:
            since = today - timedelta(days=7)
        newest = ""
        try:
            for paper in src.fetch(since, today):
                if not paper.published:
                    paper.published = today.isoformat()
                _, created = db.upsert_paper(paper)
                res.fetched += 1
                res.new += int(created)
                newest = max(newest, min(paper.published, today.isoformat()))
            # The watermark follows the newest paper actually seen, not the wall clock: if an API
            # lags (arXiv over weekends), the next run still covers the days that are missing.
            if newest:
                db.kv_set(f"watermark:{src.name}", newest)
        except Exception as exc:  # one broken source must not stop the others
            res.error = f"{type(exc).__name__}: {exc}"
        results.append(res)
    return results
