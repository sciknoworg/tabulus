from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from tabulus.reference_resolution.models import ResolutionCandidate
from tabulus.reference_resolution.scoring import normalize_doi


DEFAULT_CROSSREF_BASE_URL = "https://api.crossref.org"
DEFAULT_CROSSREF_TIMEOUT_SECONDS = 30.0
DEFAULT_CROSSREF_ROWS = 5


class CrossrefError(RuntimeError):
    """Raised when Crossref cannot return a usable API response."""


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _extract_year(item: dict[str, Any]) -> int | None:
    for field in (
        "published-print",
        "published-online",
        "issued",
        "created",
    ):
        value = item.get(field)

        if not isinstance(value, dict):
            continue

        date_parts = value.get("date-parts")

        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            year = date_parts[0][0]

            if isinstance(year, int):
                return year

        # Crossref's `created` object normally uses date-time rather than
        # date-parts, so support that representation as a final fallback.
        date_time = value.get("date-time")

        if isinstance(date_time, str) and len(date_time) >= 4:
            try:
                return int(date_time[:4])
            except ValueError:
                pass

    return None


def _extract_authors(item: dict[str, Any]) -> tuple[str, ...]:
    authors = item.get("author")

    if not isinstance(authors, list):
        return ()

    result: list[str] = []

    for author in authors:
        if not isinstance(author, dict):
            continue

        given = str(author.get("given") or "").strip()
        family = str(author.get("family") or "").strip()

        name = " ".join(part for part in (given, family) if part)

        if name:
            result.append(name)

    return tuple(result)


def candidate_from_crossref_item(
    item: dict[str, Any],
) -> ResolutionCandidate:
    """Normalize one Crossref work record into the Stage 6 candidate model."""

    doi = normalize_doi(str(item.get("DOI") or ""))

    return ResolutionCandidate(
        source="crossref",
        source_id=doi or str(item.get("URL") or ""),
        doi=doi,
        title=_first_text(item.get("title")),
        authors=_extract_authors(item),
        year=_extract_year(item),
        venue=_first_text(item.get("container-title")),
        volume=str(item.get("volume") or "").strip(),
        issue=str(item.get("issue") or "").strip(),
        pages=str(item.get("page") or "").strip(),
        url=str(item.get("URL") or "").strip(),
    )


def _request_json(
    url: str,
    *,
    user_agent: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            payload = response.read()

    except urllib.error.HTTPError:
        raise

    except urllib.error.URLError as error:
        raise CrossrefError(
            f"Could not reach Crossref: {error.reason}"
        ) from error

    if not payload:
        raise CrossrefError("Crossref returned an empty response.")

    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossrefError(
            "Crossref returned invalid JSON."
        ) from error

    if not isinstance(value, dict):
        raise CrossrefError(
            "Crossref response must be a JSON object."
        )

    return value


@dataclass(frozen=True)
class CrossrefClient:
    """Small dependency-free client for Stage 6 Crossref retrieval."""

    mailto: str
    base_url: str = DEFAULT_CROSSREF_BASE_URL
    timeout_seconds: float = DEFAULT_CROSSREF_TIMEOUT_SECONDS
    rows: int = DEFAULT_CROSSREF_ROWS

    def __post_init__(self) -> None:
        if not self.mailto.strip():
            raise ValueError(
                "Crossref mailto must be provided for identifiable API requests."
            )

        if self.rows <= 0:
            raise ValueError("Crossref rows must be greater than zero.")

        if self.timeout_seconds <= 0:
            raise ValueError(
                "Crossref timeout_seconds must be greater than zero."
            )

    @property
    def user_agent(self) -> str:
        return f"tabulus/0.1 (mailto:{self.mailto.strip()})"

    def get_by_doi(
        self,
        doi: str,
    ) -> ResolutionCandidate | None:
        """Fetch one Crossref work by DOI.

        A Crossref 404 means the DOI could not be validated in Crossref and is
        represented as ``None``. Other HTTP failures remain explicit errors.
        """

        normalized = normalize_doi(doi)

        if not normalized:
            return None

        encoded_doi = urllib.parse.quote(normalized, safe="")
        url = (
            f"{self.base_url.rstrip('/')}/works/"
            f"{encoded_doi}"
        )

        try:
            payload = _request_json(
                url,
                user_agent=self.user_agent,
                timeout_seconds=self.timeout_seconds,
            )

        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None

            raise CrossrefError(
                f"Crossref DOI lookup returned HTTP {error.code}."
            ) from error

        message = payload.get("message")

        if not isinstance(message, dict):
            raise CrossrefError(
                "Crossref DOI lookup response has no work object."
            )

        return candidate_from_crossref_item(message)

    def search_bibliographic(
        self,
        reference_text: str,
    ) -> tuple[ResolutionCandidate, ...]:
        """Retrieve candidate works for one raw bibliography string."""

        reference_text = str(reference_text or "").strip()

        if not reference_text:
            return ()

        query = urllib.parse.urlencode(
            {
                "query.bibliographic": reference_text,
                "rows": self.rows,
                "mailto": self.mailto.strip(),
            }
        )

        url = (
            f"{self.base_url.rstrip('/')}/works?"
            f"{query}"
        )

        try:
            payload = _request_json(
                url,
                user_agent=self.user_agent,
                timeout_seconds=self.timeout_seconds,
            )

        except urllib.error.HTTPError as error:
            raise CrossrefError(
                f"Crossref bibliographic search returned HTTP {error.code}."
            ) from error

        message = payload.get("message")

        if not isinstance(message, dict):
            raise CrossrefError(
                "Crossref search response has no message object."
            )

        items = message.get("items")

        if not isinstance(items, list):
            raise CrossrefError(
                "Crossref search response has no items list."
            )

        candidates: list[ResolutionCandidate] = []

        for item in items:
            if isinstance(item, dict):
                candidates.append(
                    candidate_from_crossref_item(item)
                )

        return tuple(candidates)
