from __future__ import annotations

import json
import logging
import time
import re
from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import unicodedata

from tabulus.reference_resolution.models import (
    ResolutionCandidate,
)
from tabulus.reference_resolution.scoring import (
    normalize_doi,
)


DEFAULT_CORE_BASE_URL = "https://api.core.ac.uk/v3"
DEFAULT_CORE_TIMEOUT_SECONDS = 30.0
DEFAULT_CORE_LIMIT = 5

DEFAULT_CORE_MAX_ATTEMPTS = 4
DEFAULT_CORE_BACKOFF_SECONDS = 1.0
DEFAULT_CORE_MAX_RETRY_DELAY_SECONDS = 30.0

CORE_RETRYABLE_HTTP_STATUSES = frozenset(
    {
        429,
        500,
        502,
        503,
        504,
    }
)


class CoreError(RuntimeError):
    """Raised when CORE cannot return a usable API response."""


@dataclass(frozen=True)
class CoreRateLimit:
    """Rate-limit state reported by CORE for one API response."""

    limit: int | None = None
    remaining: int | None = None
    retry_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "retry_after": self.retry_after,
        }


@dataclass(frozen=True)
class CoreSearchResponse:
    """Normalized result of one CORE ``search/works`` request."""

    candidates: tuple[ResolutionCandidate, ...]
    total_hits: int | None
    search_id: str
    limit: int | None
    offset: int | None
    rate_limit: CoreRateLimit

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "total_hits": self.total_hits,
            "search_id": self.search_id,
            "limit": self.limit,
            "offset": self.offset,
            "rate_limit": self.rate_limit.to_dict(),
        }


def _text(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    text = _text(value)

    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def _year_from_value(
    value: Any,
) -> int | None:
    if isinstance(value, int):
        if 1800 <= value <= 2199:
            return value
        return None

    text = _text(value)

    match = re.search(
        r"\b((?:18|19|20|21)\d{2})\b",
        text,
    )

    if match is None:
        return None

    return int(match.group(1))


def _extract_year(
    item: dict[str, Any],
) -> int | None:
    """Extract the best available publication year.

    ``publishedDate`` is preferred when CORE supplies an explicit publication
    date. ``yearPublished`` is used as the fallback. The value is evidence, not
    an independently authoritative validation signal.
    """

    published = _year_from_value(
        item.get("publishedDate")
    )

    if published is not None:
        return published

    return _year_from_value(
        item.get("yearPublished")
    )


def _extract_authors(
    value: Any,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()

    result: list[str] = []

    for author in value:
        if isinstance(author, dict):
            name = _text(
                author.get("name")
            )
        else:
            name = _text(author)

        if name:
            result.append(name)

    return tuple(result)


def _journal_metadata(
    item: dict[str, Any],
) -> tuple[str, str, str]:
    """Return venue, volume, issue from CORE journal metadata when present."""

    journals = item.get("journals")

    if not isinstance(journals, list):
        return "", "", ""

    for journal in journals:
        if isinstance(journal, str):
            value = _text(journal)

            if value:
                return value, "", ""

            continue

        if not isinstance(journal, dict):
            continue

        venue = ""

        for field in (
            "title",
            "name",
        ):
            venue = _text(
                journal.get(field)
            )

            if venue:
                break

        volume = _text(
            journal.get("volume")
        )
        issue = _text(
            journal.get("issue")
            or journal.get("number")
        )

        if venue or volume or issue:
            return venue, volume, issue

    return "", "", ""


def _best_url(
    item: dict[str, Any],
) -> str:
    links = item.get("links")

    if isinstance(links, list):
        # Prefer CORE's display page over an arbitrary auxiliary link.
        for link in links:
            if (
                isinstance(link, dict)
                and _text(link.get("type")).casefold()
                == "display"
            ):
                url = _text(
                    link.get("url")
                )

                if url:
                    return url

        for link in links:
            if isinstance(link, dict):
                url = _text(
                    link.get("url")
                )
            else:
                url = _text(link)

            if url:
                return url

    return _text(
        item.get("downloadUrl")
    )


def candidate_from_core_work(
    item: dict[str, Any],
) -> ResolutionCandidate:
    """Normalize one CORE v3 work record into the shared Stage 6 model."""

    venue, volume, issue = (
        _journal_metadata(item)
    )

    doi = normalize_doi(
        _text(item.get("doi"))
    )

    return ResolutionCandidate(
        source="core",
        source_id=_text(
            item.get("id")
        ),
        doi=doi,
        title=_text(
            item.get("title")
        ),
        authors=_extract_authors(
            item.get("authors")
        ),
        year=_extract_year(item),
        venue=venue,
        volume=volume,
        issue=issue,
        pages="",
        url=_best_url(item),
    )


def _rate_limit_from_headers(
    headers: Any,
) -> CoreRateLimit:
    return CoreRateLimit(
        limit=_integer(
            headers.get(
                "x-ratelimit-limit"
            )
        ),
        remaining=_integer(
            headers.get(
                "x-ratelimit-remaining"
            )
        ),
        retry_after=_text(
            headers.get(
                "x-ratelimit-retry-after"
            )
        ),
    )


def _retry_delay_seconds(
    error: urllib.error.HTTPError,
    *,
    attempt: int,
) -> float:
    """Return a bounded delay for one retryable CORE response.

    Prefer explicit service retry metadata when available. Otherwise use
    exponential backoff. Delays are capped so one failing reference cannot
    stall a complete Stage 6 batch indefinitely.
    """

    headers = error.headers

    if headers is not None:
        retry_after = _text(
            headers.get("Retry-After")
        )

        if retry_after:
            try:
                value = float(
                    retry_after
                )
            except ValueError:
                pass
            else:
                if value >= 0:
                    return min(
                        value,
                        DEFAULT_CORE_MAX_RETRY_DELAY_SECONDS,
                    )

    fallback = (
        DEFAULT_CORE_BACKOFF_SECONDS
        * (2 ** max(attempt - 1, 0))
    )

    return min(
        fallback,
        DEFAULT_CORE_MAX_RETRY_DELAY_SECONDS,
    )



# CORE's work-search endpoint interprets query-language metacharacters
# rather than treating arbitrary bibliography text entirely literally.
# Scientific citations frequently contain these characters, for example
# journal volume "82/83". Sending such text verbatim can trigger server-side
# errors. The original citation remains unchanged; this transformation is
# used only for the external CORE search query.
LOGGER = logging.getLogger(__name__)


_CORE_QUERY_METACHARACTERS = re.compile(
    r'[+\-=><!(){}\[\]^"~*?:\\/|&]+'
)


def sanitize_core_query(
    value: str,
) -> str:
    """Return a search-safe CORE query without altering source evidence."""

    value = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    # Replace parser-sensitive syntax with whitespace rather than deleting
    # it, keeping neighboring bibliographic tokens distinct.
    value = _CORE_QUERY_METACHARACTERS.sub(
        " ",
        value,
    )

    # These punctuation marks carry no useful identity signal in this
    # free-text retrieval setting.
    value = re.sub(
        r"[,;]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _request_json(
    url: str,
    *,
    api_key: str,
    timeout_seconds: float,
) -> tuple[
    dict[str, Any],
    CoreRateLimit,
]:
    """Request JSON from CORE with bounded retries for transient HTTP errors."""

    last_error: urllib.error.HTTPError | None = None

    for attempt in range(
        1,
        DEFAULT_CORE_MAX_ATTEMPTS + 1,
    ):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": (
                    f"Bearer {api_key}"
                ),
                "User-Agent": "tabulus/0.1",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                payload = response.read()
                rate_limit = (
                    _rate_limit_from_headers(
                        response.headers
                    )
                )

        except urllib.error.HTTPError as error:
            last_error = error

            retryable = (
                error.code
                in CORE_RETRYABLE_HTTP_STATUSES
            )

            if (
                retryable
                and attempt
                < DEFAULT_CORE_MAX_ATTEMPTS
            ):
                time.sleep(
                    _retry_delay_seconds(
                        error,
                        attempt=attempt,
                    )
                )
                continue

            if retryable:
                raise CoreError(
                    f"CORE request returned HTTP "
                    f"{error.code} after "
                    f"{attempt} attempts."
                ) from error

            raise CoreError(
                f"CORE request returned HTTP "
                f"{error.code}."
            ) from error

        except TimeoutError as error:
            if attempt < DEFAULT_CORE_MAX_ATTEMPTS:
                delay = min(
                    DEFAULT_CORE_BACKOFF_SECONDS
                    * (2 ** (attempt - 1)),
                    DEFAULT_CORE_MAX_RETRY_DELAY_SECONDS,
                )

                LOGGER.warning(
                    "[CORE] transient timeout; retrying in %.1fs "
                    "(attempt %d/%d)",
                    delay,
                    attempt + 1,
                    DEFAULT_CORE_MAX_ATTEMPTS,
                )

                time.sleep(delay)
                continue

            raise CoreError(
                "CORE request timed out after "
                f"{attempt} attempts."
            ) from error

        except urllib.error.URLError as error:
            raise CoreError(
                f"Could not reach CORE: "
                f"{error.reason}"
            ) from error

        if not payload:
            raise CoreError(
                "CORE returned an empty response."
            )

        try:
            value = json.loads(
                payload.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise CoreError(
                "CORE returned invalid JSON."
            ) from error

        if not isinstance(value, dict):
            raise CoreError(
                "CORE response must be a JSON object."
            )

        return value, rate_limit

    # Defensive invariant: every loop path either returns, retries, or raises.
    if last_error is not None:
        raise CoreError(
            "CORE request failed after the retry budget "
            "was exhausted."
        ) from last_error

    raise CoreError(
        "CORE request failed unexpectedly."
    )

@dataclass(frozen=True)
class CoreClient:
    """Dependency-free client for the CORE API v3 works search."""

    api_key: str
    base_url: str = DEFAULT_CORE_BASE_URL
    timeout_seconds: float = (
        DEFAULT_CORE_TIMEOUT_SECONDS
    )
    limit: int = DEFAULT_CORE_LIMIT

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "CORE api_key must be provided."
            )

        if self.limit <= 0:
            raise ValueError(
                "CORE limit must be greater than zero."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "CORE timeout_seconds must be "
                "greater than zero."
            )

    def search_works(
        self,
        reference_text: str,
    ) -> CoreSearchResponse:
        """Search CORE for scholarly-work candidates."""

        reference_text = _text(
            reference_text
        )

        if not reference_text:
            return CoreSearchResponse(
                candidates=(),
                total_hits=0,
                search_id="",
                limit=self.limit,
                offset=0,
                rate_limit=CoreRateLimit(),
            )

        search_query = sanitize_core_query(
            reference_text
        )

        if not search_query:
            return CoreSearchResponse(
                candidates=(),
                total_hits=0,
                search_id="",
                limit=self.limit,
                offset=0,
                rate_limit=CoreRateLimit(),
            )

        query = urllib.parse.urlencode(
            {
                "q": search_query,
                "limit": self.limit,
            }
        )

        # The live CORE API canonicalizes this endpoint with a trailing slash.
        url = (
            f"{self.base_url.rstrip('/')}"
            f"/search/works/?{query}"
        )

        payload, rate_limit = (
            _request_json(
                url,
                api_key=self.api_key.strip(),
                timeout_seconds=(
                    self.timeout_seconds
                ),
            )
        )

        results = payload.get(
            "results"
        )

        if not isinstance(results, list):
            raise CoreError(
                "CORE search response has no "
                "results list."
            )

        candidates = tuple(
            candidate_from_core_work(
                result
            )
            for result in results
            if isinstance(result, dict)
        )

        return CoreSearchResponse(
            candidates=candidates,
            total_hits=_integer(
                payload.get("totalHits")
            ),
            search_id=_text(
                payload.get("searchId")
            ),
            limit=_integer(
                payload.get("limit")
            ),
            offset=_integer(
                payload.get("offset")
            ),
            rate_limit=rate_limit,
        )
