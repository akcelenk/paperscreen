"""The ``Paper`` record shared by sources, storage, models and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """One paper (title + abstract + identifiers).

    Sources yield ``Paper`` objects with ``paper_id`` empty; the database assigns the
    canonical id when the record is upserted (see ``Database.upsert_paper``).
    ``published`` is an ISO date (``YYYY-MM-DD``).
    """

    title: str
    abstract: str = ""
    published: str = ""
    source: str = ""
    paper_id: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    pmid: str | None = None
    openalex_id: str | None = None
    authors: str = ""
    categories: str = ""  # space separated (arXiv categories, or venue tags)
    venue: str = ""
    url: str = ""
    first_seen: str = field(default="")

    @property
    def category_set(self) -> set[str]:
        return set(self.categories.split())
