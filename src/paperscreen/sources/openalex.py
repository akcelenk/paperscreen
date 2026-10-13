"""OpenAlex source (covers paywalled publishers' metadata and abstracts, no subscription needed).

OpenAlex requires a free API key (set ``OPENALEX_API_KEY``). Abstracts arrive as an inverted index
and are reassembled here. NOTE: the parser is tested against a hand-written fixture that follows
the documented schema, not a recorded live response (the shared quota was exhausted while this was
developed); run ``paperscreen fetch --source openalex`` once with your key and check the output.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date

from ..http import HttpClient, HttpError
from ..paper import Paper
from ..text import arxiv_id_from_doi, collapse_ws, normalize_arxiv_id, normalize_doi, to_iso_date
from .base import Source, SourceError

API_URL = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_date,abstract_inverted_index,authorships,primary_location,ids"


def rebuild_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inv.items():
        for pos in positions:
            slots[pos] = word
    return collapse_ws(" ".join(slots[i] for i in sorted(slots)))


def parse_work(w: dict) -> Paper | None:
    title = collapse_ws(w.get("title") or w.get("display_name") or "")
    abstract = rebuild_abstract(w.get("abstract_inverted_index"))
    if not title:
        return None
    oa_id = (w.get("id") or "").rsplit("/", 1)[-1] or None
    doi = normalize_doi(w.get("doi"))
    ids = w.get("ids") or {}
    pmid = (ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
    loc = w.get("primary_location") or {}
    landing = loc.get("landing_page_url") or ""
    aid = arxiv_id_from_doi(doi) or (normalize_arxiv_id(landing) if "arxiv.org" in landing else None)
    venue = ((loc.get("source") or {}).get("display_name")) or ""
    authors = ", ".join(
        (a.get("author") or {}).get("display_name", "") for a in (w.get("authorships") or [])[:20]
    )
    return Paper(
        title=title, abstract=abstract, published=to_iso_date(w.get("publication_date")),
        source="openalex", arxiv_id=aid, doi=doi, pmid=pmid if (pmid or "").isdigit() else None,
        openalex_id=oa_id, authors=authors, venue=venue,
        url=landing or (f"https://doi.org/{doi}" if doi else (w.get("id") or "")),
    )


class OpenAlexSource(Source):
    name = "openalex"

    def __init__(
        self,
        queries: Iterable[str],
        api_key: str | None,
        http: HttpClient | None = None,
        email: str | None = None,
        max_per_query: int = 200,
        require_abstract: bool = True,
    ):
        if not api_key:
            raise SourceError("OpenAlex needs a free API key: set OPENALEX_API_KEY (see openalex.org).")
        self.queries = list(queries)
        self.api_key = api_key
        self.http = http or HttpClient(min_interval=0.2, user_agent="paperscreen/0.1")
        self.email = email
        self.max_per_query = max_per_query
        self.require_abstract = require_abstract

    def fetch(self, since: date, until: date | None = None) -> Iterator[Paper]:
        until = until or date.today()
        for q in self.queries:
            cursor: str | None = "*"
            got = 0
            while cursor and got < self.max_per_query:
                params = {
                    "search": q,
                    "filter": f"from_publication_date:{since.isoformat()},"
                    f"to_publication_date:{until.isoformat()},has_abstract:true",
                    "per-page": min(100, self.max_per_query - got) or 100,
                    "cursor": cursor,
                    "select": SELECT,
                    "api_key": self.api_key,
                }
                if self.email:
                    params["mailto"] = self.email
                try:
                    data = self.http.get(API_URL, params=params).json()
                except HttpError as exc:
                    raise SourceError(f"OpenAlex request failed: {exc}") from exc
                for w in data.get("results", []):
                    p = parse_work(w)
                    if p and (p.abstract or not self.require_abstract):
                        got += 1
                        yield p
                cursor = (data.get("meta") or {}).get("next_cursor")
                if not data.get("results"):
                    break
