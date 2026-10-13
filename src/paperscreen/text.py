"""Identifier normalisation and the text that is fed to embedders."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime

from .paper import Paper

_ARXIV_NEW = re.compile(r"\d{4}\.\d{4,5}(v\d+)?")
_ARXIV_OLD = re.compile(r"[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?")
_VERSION = re.compile(r"v\d+$")

INPUT_FORMATS = ("title", "title_abstract", "title_sep_abstract")


def normalize_arxiv_id(raw: str | None) -> str | None:
    """Version-less arXiv id (``2401.01234`` or ``hep-th/9901001``), or None."""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^(https?://)?(www\.)?arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s)
    if _ARXIV_NEW.fullmatch(s) or _ARXIV_OLD.fullmatch(s):
        return _VERSION.sub("", s)
    return None


def normalize_doi(raw: str | None) -> str | None:
    """Lower-cased bare DOI (``10.xxxx/...``), or None."""
    if not raw:
        return None
    s = raw.strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s if s.startswith("10.") else None


def arxiv_id_from_doi(doi: str | None) -> str | None:
    """arXiv-issued DOIs look like ``10.48550/arxiv.2401.01234``."""
    d = normalize_doi(doi)
    if d and d.startswith("10.48550/arxiv."):
        return normalize_arxiv_id(d[len("10.48550/arxiv.") :])
    return None


def title_key(title: str) -> str | None:
    """Exact-match key for a title (lower-case alphanumerics); None if too short to be safe."""
    k = re.sub(r"[^a-z0-9]", "", title.lower())
    return k if len(k) >= 25 else None


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def to_iso_date(value: str | date | datetime | None) -> str:
    """``YYYY-MM-DD`` from an ISO date/datetime string or date object ('' if unparseable)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(value))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def paper_text(paper: Paper, fmt: str = "title_abstract") -> str:
    """Text fed to an embedder or TF-IDF.

    ``title_sep_abstract`` is the SPECTER/SciNCL convention (title ``[SEP]`` abstract).
    """
    title = collapse_ws(paper.title)
    abstract = collapse_ws(paper.abstract)
    if fmt == "title":
        return title
    if fmt == "title_abstract":
        return f"{title}. {abstract}".strip() if abstract else title
    if fmt == "title_sep_abstract":
        return f"{title} [SEP] {abstract}" if abstract else title
    raise ValueError(f"unknown input format {fmt!r}; choose from {INPUT_FORMATS}")


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
