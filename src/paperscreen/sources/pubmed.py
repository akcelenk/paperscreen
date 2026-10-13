"""PubMed source via NCBI E-utilities (esearch + efetch). Titles and abstracts only."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from datetime import date

from ..http import HttpClient
from ..paper import Paper
from ..text import collapse_ws, normalize_doi
from .base import Source

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _text(el: ET.Element | None) -> str:
    return collapse_ws("".join(el.itertext())) if el is not None else ""


def _date_from(el: ET.Element | None) -> str:
    if el is None:
        return ""
    y = el.findtext("Year")
    if not y or not y.isdigit():
        medline = el.findtext("MedlineDate") or ""  # e.g. "2020 Jan-Feb"
        y = medline[:4] if medline[:4].isdigit() else ""
        return f"{y}-01-01" if y else ""
    m_raw = (el.findtext("Month") or "1").strip().lower()
    m = int(m_raw) if m_raw.isdigit() else _MONTHS.get(m_raw[:3], 1)
    d_raw = (el.findtext("Day") or "1").strip()
    d = int(d_raw) if d_raw.isdigit() else 1
    return f"{int(y):04d}-{min(max(m, 1), 12):02d}-{min(max(d, 1), 28 if m == 2 else 31):02d}"


def parse_pubmed_xml(xml_text: str, require_abstract: bool = True) -> list[Paper]:
    root = ET.fromstring(xml_text)
    out: list[Paper] = []
    for art in root.findall("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid = (cit.findtext("PMID") or "").strip()
        article = cit.find("Article")
        if not pmid or article is None:
            continue
        title = _text(article.find("ArticleTitle")).rstrip(".")
        parts = []
        for at in article.findall("Abstract/AbstractText"):
            label = at.get("Label")
            txt = _text(at)
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(parts)
        if not title or (require_abstract and not abstract):
            continue
        doi = None
        for aid in art.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = normalize_doi(aid.text)
        if doi is None:
            for el in article.findall("ELocationID"):
                if el.get("EIdType") == "doi":
                    doi = normalize_doi(el.text)
        published = _date_from(article.find("ArticleDate")) or _date_from(
            article.find("Journal/JournalIssue/PubDate")
        )
        authors = ", ".join(
            collapse_ws(f"{a.findtext('ForeName') or a.findtext('Initials') or ''} {a.findtext('LastName') or ''}")
            for a in article.findall("AuthorList/Author")
            if a.findtext("LastName")
        )
        out.append(
            Paper(
                title=title, abstract=abstract, published=published, source="pubmed", pmid=pmid,
                doi=doi, authors=authors, venue=_text(article.find("Journal/Title")),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return out


class PubMedSource(Source):
    name = "pubmed"

    def __init__(
        self,
        queries: Iterable[str],
        http: HttpClient | None = None,
        api_key: str | None = None,
        email: str | None = None,
        max_per_query: int = 200,
        require_abstract: bool = True,
    ):
        self.queries = list(queries)
        # NCBI allows 3 requests/s without a key and 10/s with one.
        self.http = http or HttpClient(
            min_interval=0.11 if api_key else 0.4, user_agent="paperscreen/0.1"
        )
        self.api_key = api_key
        self.email = email
        self.max_per_query = max_per_query
        self.require_abstract = require_abstract

    def _common(self) -> dict:
        p = {"tool": "paperscreen"}
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def fetch(self, since: date, until: date | None = None) -> Iterator[Paper]:
        until = until or date.today()
        seen: set[str] = set()
        for q in self.queries:
            r = self.http.get(
                f"{BASE}/esearch.fcgi",
                params={
                    **self._common(), "db": "pubmed", "term": q, "retmode": "json",
                    "datetype": "edat", "mindate": since.strftime("%Y/%m/%d"),
                    "maxdate": until.strftime("%Y/%m/%d"), "retmax": self.max_per_query,
                    "sort": "date",
                },
            )
            ids = [i for i in r.json().get("esearchresult", {}).get("idlist", []) if i not in seen]
            seen.update(ids)
            for i in range(0, len(ids), 100):
                batch = ids[i : i + 100]
                x = self.http.get(
                    f"{BASE}/efetch.fcgi",
                    params={**self._common(), "db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                )
                yield from parse_pubmed_xml(x.text, self.require_abstract)
