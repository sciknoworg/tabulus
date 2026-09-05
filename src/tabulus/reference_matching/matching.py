from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

from tabulus.bibliography.grobid import extract_doi
from tabulus.bibliography.models import BibliographyEntry


NUMERIC_POSITION_METHOD = "numeric_position"
DOI_EXACT_METHOD = "doi_exact"
AUTHOR_YEAR_METHOD = "author_year"
AUTHOR_ONLY_METHOD = "author_only"
TEXT_CONTAINMENT_METHOD = "text_containment"

REFERENCE_HEADER_PATTERN = re.compile(
    r"\b("
    r"refs?\.?|references?|citations?|authors?|sources?|papers?|"
    r"literatures?|publications?|works?|stud(?:y|ies)"
    r")\b",
    re.IGNORECASE,
)
AUTHOR_YEAR_PATTERN = re.compile(
    r"\b(?P<author>[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)"
    r".*?(?P<year>(?:19|20)\d{2}[a-z]?)\b",
    re.IGNORECASE,
)
AUTHOR_ONLY_PATTERN = re.compile(
    r"^[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+"
    r"(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)?$"
)
SQUARE_BRACKET_NUMERIC_GROUP_PATTERN = re.compile(
    r"\[\s*([0-9,;\s&\-–]+(?:\band\b[0-9,;\s&\-–]*)?)\s*\]",
    re.IGNORECASE,
)
PARENTHESIZED_NUMERIC_LIST_PATTERN = re.compile(
    r"^\(\s*(\d{1,5}(?:\s*[-–]\s*\d{1,5})?"
    r"(?:\s*(?:,|;|&|\band\b)\s*(?:\band\b\s*)?"
    r"\d{1,5}(?:\s*[-–]\s*\d{1,5})?)*)\s*\)$",
    re.IGNORECASE,
)
BARE_NUMERIC_LIST_PATTERN = re.compile(
    r"^\s*\d{1,5}(?:\s*[-–]\s*\d{1,5})?"
    r"(?:\s*(?:,|;|&|\band\b)\s*(?:\band\b\s*)?"
    r"\d{1,5}(?:\s*[-–]\s*\d{1,5})?)*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReferenceCandidate:
    reference: BibliographyEntry
    method: str
    token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_index": self.reference.index,
            "method": self.method,
            "token": self.token,
        }


@dataclass(frozen=True)
class ReferenceValueMatch:
    value: str
    candidates: tuple[ReferenceCandidate, ...]
    tokens: tuple[str, ...]
    unmatched_tokens: tuple[str, ...]

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    @property
    def matched_reference_indices(self) -> tuple[int, ...]:
        return tuple(candidate.reference.index for candidate in self.candidates)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_for_match(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def _normalize_author_for_match(value: Any) -> str:
    text = unicodedata.normalize("NFKD", normalize_text(value)).casefold()
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("–", "-")
        .replace("—", "-")
    )


def _contains_author_token(raw_reference: str, author: str) -> bool:
    raw = _normalize_author_for_match(raw_reference)
    return bool(
        re.search(
            rf"(?<![\w'-]){re.escape(author)}(?![\w'-])",
            raw,
        )
    )


def _matches_author_only_position(
    raw_reference: str,
    token: str,
) -> bool:
    raw = _normalize_author_for_match(raw_reference).strip()
    author = _normalize_author_for_match(token).strip()

    if not raw or not author:
        return False

    # Some raw citations may retain a leading numeric label.
    raw = re.sub(
        r"^\s*(?:\[\d+\]|\d+\s*[.)])\s*",
        "",
        raw,
    )

    surname = re.escape(author)

    # Surname-first bibliography styles:
    #   Smith J. ...
    #   Smith, J. ...
    if re.match(
        rf"^{surname}(?![\w'-])",
        raw,
    ):
        return True

    # Initials-first bibliography styles:
    #   J. Smith ...
    #   J. A. Smith ...
    return bool(
        re.match(
            rf"^(?:[a-z]\.?\s+){{1,3}}"
            rf"{surname}(?![\w'-])",
            raw,
        )
    )


def looks_like_reference_header(value: Any) -> bool:
    return bool(REFERENCE_HEADER_PATTERN.search(normalize_text(value)))


def _expand_numeric_part(part: str) -> list[str]:
    part = part.strip()
    range_match = re.fullmatch(r"(\d{1,5})\s*[-–]\s*(\d{1,5})", part)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start <= end and end - start <= 10000:
            return [str(value) for value in range(start, end + 1)]
        return []
    if part.isdigit():
        return [part]
    return []


def _parse_numeric_group(group: str) -> list[str]:
    cleaned = re.sub(r"\band\b|&", ",", group, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"[,;]", cleaned) if part.strip()]
    tokens: list[str] = []
    for part in parts:
        tokens.extend(_expand_numeric_part(part))
    return tokens


def _looks_like_numeric_only_sequence(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return False

    # Remove citation-list conjunctions even when OCR has lost spacing,
    # e.g. "1989 and1990" or "and1883-1974".
    residual = re.sub(r"and", "", text, flags=re.IGNORECASE)

    # Any remaining alphabetic content means this is not numeric-only
    # reference syntax (e.g. "Smith 2020").
    if re.search(r"[A-Za-z]", residual):
        return False

    return bool(
        re.search(r"\d", residual)
        and re.fullmatch(
            r"[0-9\s,;&\-–\[\]\(\)]+",
            residual,
        )
    )


def _parse_loose_numeric_sequence(value: str) -> list[str]:
    parts = re.findall(
        r"\d{1,5}(?:\s*[-–]\s*\d{1,5})?",
        value,
    )

    tokens: list[str] = []
    for part in parts:
        tokens.extend(_expand_numeric_part(part))

    return tokens


def extract_numeric_reference_tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    if not text:
        return []

    tokens: list[str] = []
    groups = SQUARE_BRACKET_NUMERIC_GROUP_PATTERN.findall(text)
    for group in groups:
        tokens.extend(_parse_numeric_group(group))

    if not groups:
        parenthesized = PARENTHESIZED_NUMERIC_LIST_PATTERN.fullmatch(text)
        if parenthesized:
            tokens.extend(_parse_numeric_group(parenthesized.group(1)))
        elif BARE_NUMERIC_LIST_PATTERN.fullmatch(text):
            tokens.extend(_parse_numeric_group(text))
        elif _looks_like_numeric_only_sequence(text):
            tokens.extend(_parse_loose_numeric_sequence(text))

    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def split_reference_cell(value: Any) -> list[str]:
    raw_text = str(value or "")
    if not raw_text.strip():
        return []

    parts = re.split(r"\s*;\s*|[\r\n]+", raw_text)
    final_parts: list[str] = []
    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        comma_parts = re.split(
            r",\s+(?=[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+"
            r"(?:\s+et\s+al\.?|\s+and|\s+&|\s*,)?"
            r"\s*.*?(?:19|20)\d{2})",
            part,
        )
        final_parts.extend(item.strip() for item in comma_parts if item.strip())
    return final_parts


def _author_year(value: str) -> tuple[str, str] | None:
    match = AUTHOR_YEAR_PATTERN.search(value)
    if match is None:
        return None
    return (
        _normalize_author_for_match(match.group("author")),
        match.group("year").lower(),
    )


def _deduplicate_candidates(
    candidates: Sequence[ReferenceCandidate],
) -> tuple[ReferenceCandidate, ...]:
    seen: set[int] = set()
    result: list[ReferenceCandidate] = []
    for candidate in candidates:
        index = candidate.reference.index
        if index not in seen:
            seen.add(index)
            result.append(candidate)
    return tuple(result)


def _match_text_token(
    token: str,
    bibliography: Sequence[BibliographyEntry],
) -> list[ReferenceCandidate]:
    doi = extract_doi(token)
    if doi:
        matches = [
            ReferenceCandidate(ref, DOI_EXACT_METHOD, token)
            for ref in bibliography
            if ref.doi and ref.doi.lower() == doi.lower()
        ]
        if matches:
            return matches

    author_year = _author_year(token)
    if author_year is not None:
        author, year = author_year
        matches = []
        for ref in bibliography:
            raw_norm = normalize_for_match(ref.raw)
            author_ok = _contains_author_token(ref.raw, author)
            year_ok = year[:4] in raw_norm
            if author_ok and year_ok:
                matches.append(ReferenceCandidate(ref, AUTHOR_YEAR_METHOD, token))
        if matches:
            return matches

    token_norm = normalize_for_match(token)
    if (
        AUTHOR_ONLY_PATTERN.fullmatch(normalize_text(token))
        and len(token_norm) >= 4
        and not re.search(r"\d", token_norm)
        and not looks_like_reference_header(token)
        and token_norm not in {"study", "studies", "no", "info", "no info"}
    ):
        matches = []
        for ref in bibliography:
            if _matches_author_only_position(ref.raw, token):
                matches.append(
                    ReferenceCandidate(
                        ref,
                        AUTHOR_ONLY_METHOD,
                        token,
                    )
                )
        if matches:
            return matches

    if len(token_norm) >= 20 and len(token_norm.split()) >= 4:
        matches = []
        for ref in bibliography:
            raw_norm = normalize_for_match(ref.raw)
            if token_norm in raw_norm or raw_norm in token_norm:
                matches.append(ReferenceCandidate(ref, TEXT_CONTAINMENT_METHOD, token))
        if matches:
            return matches

    return []


def match_reference_value(
    value: Any,
    bibliography: Sequence[BibliographyEntry],
) -> ReferenceValueMatch:
    raw_value = str(value or "")
    text = normalize_text(raw_value)
    if not text or text.lower() == "no info":
        return ReferenceValueMatch(text, (), (), ())

    by_index = {entry.index: entry for entry in bibliography}
    numeric_tokens = extract_numeric_reference_tokens(text)
    if numeric_tokens:
        candidates: list[ReferenceCandidate] = []
        unmatched: list[str] = []
        for token in numeric_tokens:
            ref = by_index.get(int(token))
            if ref is None:
                unmatched.append(token)
            else:
                candidates.append(
                    ReferenceCandidate(ref, NUMERIC_POSITION_METHOD, token)
                )
        return ReferenceValueMatch(
            value=text,
            candidates=_deduplicate_candidates(candidates),
            tokens=tuple(numeric_tokens),
            unmatched_tokens=tuple(unmatched),
        )

    tokens = split_reference_cell(raw_value)
    if not tokens:
        tokens = [text]

    candidates = []
    unmatched = []
    for token in tokens:
        matches = _match_text_token(token, bibliography)
        if matches:
            candidates.extend(matches)
        else:
            unmatched.append(token)

    return ReferenceValueMatch(
        value=text,
        candidates=_deduplicate_candidates(candidates),
        tokens=tuple(tokens),
        unmatched_tokens=tuple(unmatched),
    )


def _looks_like_author_year(value: str) -> bool:
    return _author_year(value) is not None


def detect_reference_column(rows: Sequence[Sequence[Any]]) -> int | None:
    if not rows:
        return None

    max_cols = max((len(row) for row in rows), default=0)
    best_column: int | None = None
    best_score = 0

    for column in range(max_cols):
        score = 0
        first_non_empty_seen = False

        for row_index, row in enumerate(rows):
            cell = normalize_text(row[column] if column < len(row) else "")
            if not cell:
                continue

            if not first_non_empty_seen:
                first_non_empty_seen = True
                if looks_like_reference_header(cell):
                    score += 12

            numeric = extract_numeric_reference_tokens(cell)
            if numeric:
                score += 6
            elif extract_doi(cell):
                score += 7
            elif _looks_like_author_year(cell):
                score += 6
            elif row_index > 0 and not re.search(r"\d", cell) and len(cell.split()) <= 4:
                score += 1

        if score > best_score:
            best_score = score
            best_column = column

    return best_column if best_score > 0 else None
