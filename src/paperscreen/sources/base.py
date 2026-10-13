"""Source interface. A source yields ``Paper`` records for a date window."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date

from ..paper import Paper


class SourceError(RuntimeError):
    """A source cannot be used right now (missing key, quota exhausted, ...)."""


class Source(ABC):
    name: str = "source"

    @abstractmethod
    def fetch(self, since: date, until: date | None = None) -> Iterator[Paper]:
        """Yield papers published in ``[since, until]`` (``until`` defaults to today)."""
