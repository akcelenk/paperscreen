"""arXiv source (Atom API) and an importer for the Kaggle metadata snapshot.

Metadata (title, abstract, authors, categories) is CC0 per the arXiv API terms. Full texts are
not fetched or stored. The API allows one request per three seconds, one connection at a time,
summed over all your machines; the client below enforces the interval.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from datetime import date, timedelta
from pathlib import Path

from ..http import HttpClient
from ..paper import Paper
from ..text import collapse_ws, normalize_arxiv_id, normalize_doi, to_iso_date
from .base import Source

API_URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    out: list[Paper] = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        aid = normalize_arxiv_id(raw_id)
        title = collapse_ws(e.findtext("a:title", default="", namespaces=NS))
        if not aid or not title or title.lower() == "error":
            continue  # the API returns an "Error" entry for malformed queries
        cats = [c.get("term", "") for c in e.findall("a:category", NS)]
        prim = e.find("x:primary_category", NS)
        if prim is not None and prim.get("term"):
            cats = [prim.get("term", "")] + [c for c in cats if c != prim.get("term")]
        out.append(
            Paper(
                title=title,
                abstract=collapse_ws(e.findtext("a:summary", default="", namespaces=NS)),
                published=to_iso_date(e.findtext("a:published", default="", namespaces=NS)),
                source="arxiv",
                arxiv_id=aid,
                doi=normalize_doi(e.findtext("x:doi", default="", namespaces=NS)),
                authors=", ".join(
                    collapse_ws(a.findtext("a:name", default="", namespaces=NS))
                    for a in e.findall("a:author", NS)
                ),
                categories=" ".join(c for c in cats if c),
                url=f"https://arxiv.org/abs/{aid}",
            )
        )
    return out


class ArxivSource(Source):
    """arXiv API, queried one submission day at a time.

    A single "newest first" query with a global cap silently drops the oldest days once the
    selected categories produce more papers than the cap (cs.LG and cs.AI alone had ~44k
    submissions each in 2025). Slicing by ``submittedDate`` keeps every day complete; a day that
    still hits ``max_per_day`` is recorded in ``truncated_days`` so the gap is visible.
    """

    name = "arxiv"

    def __init__(
        self,
        categories: Iterable[str],
        http: HttpClient | None = None,
        page_size: int = 500,
        max_per_day: int = 3000,
        contact: str | None = None,
    ):
        self.categories = list(categories)
        ua = "paperscreen/0.3" + (f" (mailto:{contact})" if contact else "")
        self.http = http or HttpClient(min_interval=3.1, user_agent=ua)
        self.page_size = page_size
        self.max_per_day = max_per_day
        self.truncated_days: list[str] = []

    def day_query(self, day: date) -> str:
        cats = " OR ".join(f"cat:{c}" for c in self.categories)
        d = day.strftime("%Y%m%d")
        return f"({cats}) AND submittedDate:[{d}0000 TO {d}2359]"  # GMT, per the API manual

    def fetch(self, since: date, until: date | None = None) -> Iterator[Paper]:
        until = until or date.today()
        self.truncated_days = []
        day = until
        while day >= since:
            start = 0
            while True:
                r = self.http.get(
                    API_URL,
                    params={
                        "search_query": self.day_query(day), "start": start, "max_results": self.page_size,
                        "sortBy": "submittedDate", "sortOrder": "descending",
                    },
                )
                page = parse_atom(r.text)
                yield from page
                start += len(page)
                if len(page) < self.page_size:
                    break
                if start >= self.max_per_day:
                    self.truncated_days.append(day.isoformat())
                    break
            day -= timedelta(days=1)


def import_kaggle_jsonl(
    path: str | Path, since: date | None = None, categories: Iterable[str] | None = None,
    limit: int | None = None,
) -> Iterator[Paper]:
    """Read the Kaggle ``arxiv-metadata-oversnapshot.json`` (one JSON object per line).

    Use this for the historical backfill instead of hammering the API. A paper is kept if it has
    any category in ``categories`` (when given) and was first submitted on/after ``since``.
    """
    wanted = set(categories or [])
    since_s = since.isoformat() if since else ""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cats = (d.get("categories") or "").split()
            if wanted and not wanted.intersection(cats):
                continue
            versions = d.get("versions") or []
            published = to_iso_date(d.get("update_date"))
            if versions and versions[0].get("created"):
                # "Mon, 2 Apr 2007 19:18:42 GMT": parse day/month/year loosely.
                published = _rfc2822_to_iso(versions[0]["created"]) or published
            if since_s and published < since_s:
                continue
            aid = normalize_arxiv_id(d.get("id"))
            title = collapse_ws(d.get("title", ""))
            if not aid or not title:
                continue
            yield Paper(
                title=title, abstract=collapse_ws(d.get("abstract", "")), published=published,
                source="arxiv", arxiv_id=aid, doi=normalize_doi(d.get("doi")),
                authors=collapse_ws(d.get("authors", "")), categories=" ".join(cats),
                url=f"https://arxiv.org/abs/{aid}",
            )
            n += 1
            if limit and n >= limit:
                return


def _rfc2822_to_iso(s: str) -> str:
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(s).date().isoformat()
    except (TypeError, ValueError):
        return ""
