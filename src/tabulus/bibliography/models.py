from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BibliographyEntry:
    """One bibliography entry extracted from a scientific PDF.

    ``raw`` preserves the source citation text. The remaining bibliographic
    fields are structured metadata parsed by GROBID when available. They are
    optional because historical and poorly formatted references may contain
    only partial metadata.
    """

    index: int
    raw: str
    doi: str
    source: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "raw": self.raw,
            "doi": self.doi,
            "source": self.source,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
        }


@dataclass(frozen=True)
class Bibliography:
    """Normalized bibliography artifact for one scientific PDF."""

    source: str
    entries: tuple[BibliographyEntry, ...]

    @property
    def bibliography_count(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bibliography_count": self.bibliography_count,
            "bibliography_source": self.source,
            "entries": [
                entry.to_dict()
                for entry in self.entries
            ],
        }
