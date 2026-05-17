from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from lxml import etree

from app.reference_matching.kreuzberg_reference_fallback import (
    extract_bibliography_with_kreuzberg,
)


# =========================
# GROBID / XML
# =========================

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XP_BIBLS = etree.XPath("//tei:listBibl//tei:biblStruct", namespaces=TEI_NS)

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


# =========================
# BASIC HELPERS
# =========================

def norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def extract_text(el) -> str:
    return norm_text("".join(el.itertext()))


def normalize_for_match(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def extract_doi(text: str) -> str:
    if not text:
        return ""

    t = text.strip()
    t = t.replace("https://doi.org/", "")
    t = t.replace("http://doi.org/", "")
    t = t.replace("doi.org/", "")
    t = t.replace("DOI:", "doi:")
    t = t.replace("Doi:", "doi:")
    t = t.replace("doi:", "").strip()

    match = DOI_RE.search(t)
    return match.group(1).rstrip(").,;]") if match else ""


def looks_like_ref_header(text: str) -> bool:
    return bool(
        re.search(
            r"\b(ref|refs|reference|references|citation|citations|author|authors|source|sources|paper|papers)\b",
            str(text),
            re.I,
        )
    )


# =========================
# TOKEN EXTRACTION
# =========================

def expand_numeric_range(part: str) -> List[str]:
    part = part.strip()

    range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", part)

    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))

        if start <= end:
            return [str(i) for i in range(start, end + 1)]

    if part.isdigit():
        return [part]

    return []


def extract_numeric_tokens(text: str) -> List[str]:
    tokens = []

    bracket_groups = re.findall(r"\[\s*([0-9,\s\-–]+)\s*\]", text)

    for group in bracket_groups:
        parts = re.split(r"\s*,\s*", group)

        for part in parts:
            tokens.extend(expand_numeric_range(part))

    paren_groups = re.findall(r"\(\s*([0-9,\s\-–]+)\s*\)", text)

    for group in paren_groups:
        parts = re.split(r"\s*,\s*", group)

        for part in parts:
            tokens.extend(expand_numeric_range(part))

    if text.strip().isdigit():
        tokens.append(text.strip())

    return tokens


def extract_author_year_ref(s: str) -> Optional[Dict[str, str]]:
    """
    Handles:
    - Smith et al. 2020
    - Smith et al., 2020
    - Smith and Jones 2019
    - Smith & Jones (2018)
    - Smith, 2020
    """
    if not s:
        return None

    text = norm_text(s)

    match = re.search(
        r"\b(?P<author>[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)"
        r"(?:\s*(?:,|and|&|et\s+al\.?)?\s*[A-ZÀ-ÖØ-Ý]?[A-Za-zÀ-ÖØ-öø-ÿ'`\-]*)?"
        r".*?"
        r"(?P<year>(?:19|20)\d{2}[a-z]?)\b",
        text,
        re.I,
    )

    if not match:
        return None

    return {
        "author": match.group("author").lower(),
        "year": match.group("year").lower(),
    }


def split_reference_cell(text: str) -> List[str]:
    text = norm_text(text)

    if not text:
        return []

    parts = re.split(r"\s*;\s*|\s*\n\s*", text)

    final_parts = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        comma_parts = re.split(
            r",\s+(?=[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+et\s+al\.?|\s+and|\s+&|\s*,)?\s*.*?(?:19|20)\d{2})",
            part,
        )

        final_parts.extend([p.strip() for p in comma_parts if p.strip()])

    return final_parts


def extract_reference_tokens(s: str) -> List[str]:
    if not s:
        return []

    text = norm_text(s)

    tokens = []

    numeric_tokens = extract_numeric_tokens(text)
    tokens.extend(numeric_tokens)

    for part in split_reference_cell(text):
        if re.fullmatch(r"\[\s*[0-9,\s\-–]+\s*\]", part):
            continue

        if re.fullmatch(r"\(\s*[0-9,\s\-–]+\s*\)", part):
            continue

        if part.strip().isdigit() and part.strip() in tokens:
            continue

        tokens.append(part)

    seen = set()
    unique_tokens = []

    for token in tokens:
        key = token.lower().strip()

        if key and key not in seen:
            seen.add(key)
            unique_tokens.append(token)

    return unique_tokens


# =========================
# GROBID
# =========================

def call_grobid(pdf_path: Path, grobid_url: str, timeout: int = 300) -> str:
    endpoint = f"{grobid_url.rstrip('/')}/api/processReferences"

    with requests.Session() as session:
        with pdf_path.open("rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            response = session.post(endpoint, files=files, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(f"GROBID error: {response.status_code}: {response.text[:500]}")

    return response.text


def extract_bibliography(tei_xml: str) -> List[Dict[str, Any]]:
    root = etree.fromstring(tei_xml.encode("utf-8", errors="ignore"))
    bibls = XP_BIBLS(root)

    refs = []

    for i, bibl in enumerate(bibls, start=1):
        raw = extract_text(bibl)

        refs.append(
            {
                "index": i,
                "raw": raw,
                "doi": extract_doi(raw),
                "source": "grobid",
            }
        )

    return refs


# =========================
# MATCHING
# =========================

def is_text_match(a: str, b: str) -> bool:
    a = normalize_for_match(a)
    b = normalize_for_match(b)

    if not a or not b:
        return False

    return a in b or b in a


def match_numeric_reference(
    token: str,
    bibliography: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not token.isdigit():
        return []

    num = int(token)

    for ref in bibliography:
        if ref.get("index") == num:
            return [ref]

    return []


def match_doi_reference(
    token: str,
    bibliography: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    doi = extract_doi(token)

    if not doi:
        return []

    for ref in bibliography:
        if doi.lower() == str(ref.get("doi", "")).lower():
            return [ref]

    return []


def match_author_year_reference(
    token: str,
    bibliography: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ay = extract_author_year_ref(token)

    if not ay:
        return []

    matches = []

    for ref in bibliography:
        raw_norm = normalize_for_match(ref.get("raw", ""))

        author_ok = ay["author"] in raw_norm
        year_ok = ay["year"][:4] in raw_norm

        if author_ok and year_ok:
            matches.append(ref)

    return matches


def match_author_only_reference(
    token: str,
    bibliography: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    token_norm = normalize_for_match(token)

    if not token_norm:
        return []

    if len(token_norm) < 4:
        return []

    if token_norm in {
        "ref",
        "refs",
        "reference",
        "references",
        "author",
        "authors",
        "source",
        "sources",
        "paper",
        "papers",
        "study",
        "studies",
        "no",
        "info",
        "no info",
    }:
        return []

    matches = []

    for ref in bibliography:
        raw_norm = normalize_for_match(ref.get("raw", ""))

        if re.search(rf"\b{re.escape(token_norm)}\b", raw_norm):
            matches.append(ref)

    return matches


def find_matches(
    table_ref: str,
    bibliography: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tokens = extract_reference_tokens(table_ref)

    found = []

    for token in tokens:
        token = norm_text(token)

        if not token:
            continue

        if token.lower() == "no info":
            continue

        numeric_matches = match_numeric_reference(token, bibliography)

        if numeric_matches:
            found.extend(numeric_matches)
            continue

        doi_matches = match_doi_reference(token, bibliography)

        if doi_matches:
            found.extend(doi_matches)
            continue

        author_year_matches = match_author_year_reference(token, bibliography)

        if author_year_matches:
            found.extend(author_year_matches)
            continue

        author_only_matches = match_author_only_reference(token, bibliography)

        if author_only_matches:
            found.extend(author_only_matches)
            continue

        for ref in bibliography:
            if is_text_match(token, ref.get("raw", "")):
                found.append(ref)

    unique = {}

    for ref in found:
        unique[ref["index"]] = ref

    return list(unique.values())


# =========================
# CROSSREF
# =========================

def parse_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": (item.get("title") or [None])[0],
        "authors": [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (item.get("author") or [])
        ],
        "year": (item.get("issued", {}).get("date-parts", [[None]])[0][0]),
        "doi": item.get("DOI"),
        "url": item.get("URL"),
    }


def query_crossref(reference_text: str, mailto: str) -> Optional[Dict[str, Any]]:
    if not reference_text:
        return None

    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": reference_text,
                "rows": 1,
                "mailto": mailto,
            },
            headers={"User-Agent": f"ref-resolver (mailto:{mailto})"},
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        items = data.get("message", {}).get("items", [])

        if not items:
            return None

        return parse_crossref_item(items[0])

    except Exception:
        return None


# =========================
# COLUMN DETECTION
# =========================

def looks_like_author_year(text: str) -> bool:
    return bool(
        re.search(
            r"\b[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+"
            r"(?:\s+(?:et\s+al\.?|and|&)\s+[A-ZÀ-ÖØ-Ý]?[A-Za-zÀ-ÖØ-öø-ÿ'`\-]*)?"
            r".*?"
            r"(19|20)\d{2}[a-z]?\b",
            text,
            re.I,
        )
    )


def match_author_only_cell(cell: str) -> bool:
    cell = norm_text(cell)

    if not cell:
        return False

    if len(cell) < 4:
        return False

    if len(cell.split()) > 4:
        return False

    if re.search(r"\d", cell):
        return False

    if looks_like_ref_header(cell):
        return False

    return bool(
        re.fullmatch(
            r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÖØ-öø-ÿ'`\-]+)?",
            cell,
        )
    )


def detect_reference_column(rows: List[List[str]]) -> Optional[int]:
    if not rows:
        return None

    max_cols = max(len(r) for r in rows)

    best_col = None
    best_score = 0

    for col in range(max_cols):
        score = 0

        for row_idx, row in enumerate(rows):
            cell = row[col] if col < len(row) else ""
            cell = norm_text(cell)

            if not cell:
                continue

            if row_idx == 0 and looks_like_ref_header(cell):
                score += 10

            if re.fullmatch(r"\[\s*\d+\s*\]", cell):
                score += 6
            elif re.search(
                r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]",
                cell,
            ):
                score += 6
            elif re.fullmatch(r"\d+", cell):
                score += 3
            elif looks_like_author_year(cell):
                score += 6
            elif re.search(r"(19|20)\d{2}", cell):
                score += 2
            elif row_idx > 0 and match_author_only_cell(cell):
                score += 2

        if score > best_score:
            best_score = score
            best_col = col

    return best_col if best_score > 0 else None


# =========================
# BIBLIOGRAPHY SOURCE
# =========================

def load_bibliography_with_grobid_or_kreuzberg(
    pdf_path: Path,
    grobid_url: str,
    kreuzberg_api_url: Optional[str],
    refs_start_page: Optional[int],
) -> Dict[str, Any]:
    grobid_error = None
    kreuzberg_used = False
    kreuzberg_reason = "GROBID succeeded; kreuzberg_service fallback not needed."
    kreuzberg_pattern_used = None
    kreuzberg_pattern_counts = None
    kreuzberg_fallback_error = None

    try:
        tei_xml = call_grobid(pdf_path, grobid_url)
        bibliography = extract_bibliography(tei_xml)

        if not bibliography:
            raise RuntimeError("GROBID returned no bibliography entries.")

        return {
            "bibliography": bibliography,
            "bibliography_source": "grobid",
            "grobid_error": None,
            "kreuzberg_used": False,
            "kreuzberg_reason": kreuzberg_reason,
            "kreuzberg_pattern_used": None,
            "kreuzberg_pattern_counts": None,
            "kreuzberg_fallback_error": None,
        }

    except Exception as e:
        grobid_error = str(e)

    if not kreuzberg_api_url:
        raise RuntimeError(
            f"GROBID failed and kreuzberg_service fallback is disabled. GROBID error: {grobid_error}"
        )

    try:
        fallback_data = extract_bibliography_with_kreuzberg(
            pdf_path=pdf_path,
            kreuzberg_api_url=kreuzberg_api_url,
            refs_start_page=refs_start_page,
        )

        bibliography = fallback_data["bibliography"]
        kreuzberg_used = True
        kreuzberg_reason = (
            "GROBID failed. kreuzberg_service fallback was used because the raw "
            "reference section matched the numbered bibliography pattern."
        )
        kreuzberg_pattern_used = fallback_data.get("pattern_used")
        kreuzberg_pattern_counts = fallback_data.get("pattern_counts")
        kreuzberg_fallback_error = None

        return {
            "bibliography": bibliography,
            "bibliography_source": "kreuzberg_fallback",
            "grobid_error": grobid_error,
            "kreuzberg_used": kreuzberg_used,
            "kreuzberg_reason": kreuzberg_reason,
            "kreuzberg_pattern_used": kreuzberg_pattern_used,
            "kreuzberg_pattern_counts": kreuzberg_pattern_counts,
            "kreuzberg_fallback_error": kreuzberg_fallback_error,
        }

    except Exception as fallback_error:
        kreuzberg_fallback_error = str(fallback_error)

        raise RuntimeError(
            f"GROBID failed and kreuzberg_service fallback failed/skipped. "
            f"GROBID error: {grobid_error}. "
            f"kreuzberg_service error: {kreuzberg_fallback_error}"
        )


# =========================
# MAIN PIPELINE STEP
# =========================

def match_reference_tables_with_grobid(
    pdf_path: Path,
    ocr_tables: List[Dict[str, Any]],
    grobid_url: str,
    crossref_mailto: str,
    use_crossref: bool = False,
    kreuzberg_api_url: Optional[str] = None,
    refs_start_page: Optional[int] = None,
) -> Dict[str, Any]:

    bibliography_data = load_bibliography_with_grobid_or_kreuzberg(
        pdf_path=pdf_path,
        grobid_url=grobid_url,
        kreuzberg_api_url=kreuzberg_api_url,
        refs_start_page=refs_start_page,
    )

    bibliography = bibliography_data["bibliography"]

    matched_tables = []

    for table in ocr_tables:
        if not table.get("is_reference_table"):
            continue

        rows = table.get("rows", [])
        col_idx = detect_reference_column(rows)

        if col_idx is None:
            continue

        matches = []

        for i, row in enumerate(rows):
            if col_idx >= len(row):
                continue

            value = norm_text(row[col_idx])

            if not value:
                continue

            if i == 0 and looks_like_ref_header(value):
                matches.append(
                    {
                        "row_index": i,
                        "value": value,
                        "table_reference": value,
                        "found": False,
                        "matched_reference_indices": [],
                        "matched_references": [],
                        "doi": [],
                        "is_header": True,
                    }
                )
                continue

            matches_for_cell = find_matches(value, bibliography)

            if not matches_for_cell:
                matches.append(
                    {
                        "row_index": i,
                        "value": value,
                        "table_reference": value,
                        "found": False,
                        "matched_reference_indices": [],
                        "matched_references": [],
                        "doi": [],
                        "is_header": False,
                    }
                )
                continue

            dois = []

            for match in matches_for_cell:
                doi = match.get("doi")

                if not doi and use_crossref:
                    meta = query_crossref(match.get("raw", ""), crossref_mailto)

                    if meta:
                        doi = meta.get("doi") or meta.get("url")

                if doi:
                    dois.append(doi)

            matches.append(
                {
                    "row_index": i,
                    "value": value,
                    "table_reference": value,
                    "found": True,
                    "matched_reference_indices": [m.get("index") for m in matches_for_cell],
                    "matched_references": [m.get("raw") for m in matches_for_cell],
                    "doi": dois,
                    "is_header": False,
                }
            )

        matched_tables.append(
            {
                "table_id": table.get("table_id"),
                "source_file": table.get("source_file"),
                "reference_column_index": col_idx,
                "matches_found": len([m for m in matches if m.get("found")]),
                "matches_total": len([m for m in matches if not m.get("is_header")]),
                "matches": matches,
            }
        )

    return {
        "bibliography_count": len(bibliography),
        "bibliography_source": bibliography_data.get("bibliography_source"),
        "grobid_error": bibliography_data.get("grobid_error"),
        "kreuzberg_used": bibliography_data.get("kreuzberg_used"),
        "kreuzberg_reason": bibliography_data.get("kreuzberg_reason"),
        "kreuzberg_pattern_used": bibliography_data.get("kreuzberg_pattern_used"),
        "kreuzberg_pattern_counts": bibliography_data.get("kreuzberg_pattern_counts"),
        "kreuzberg_fallback_error": bibliography_data.get("kreuzberg_fallback_error"),
        "reference_tables_checked": len(matched_tables),
        "matched_tables": matched_tables,
    }


# =========================
# CSV EXPORT
# =========================

def write_resolved_reference_table_csvs(
    job_out_dir: Path,
    ocr_tables: List[Dict[str, Any]],
    match_data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    out_dir = job_out_dir / "resolved_reference_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_map = {
        t.get("source_file"): t
        for t in ocr_tables
        if t.get("source_file")
    }

    written = []

    for table in match_data.get("matched_tables", []):
        src = table.get("source_file")
        col = table.get("reference_column_index")

        if src is None or col is None:
            continue

        ocr_table = ocr_map.get(src)

        if not ocr_table:
            continue

        rows = [list(r) for r in ocr_table.get("rows", [])]

        if not rows:
            continue

        while len(rows[0]) <= col:
            rows[0].append("")

        rows[0][col] = "DOI"

        replacements = 0

        for m in table.get("matches", []):
            r = m.get("row_index")

            if r is None or r == 0 or r >= len(rows):
                continue

            while len(rows[r]) <= col:
                rows[r].append("")

            doi_values = m.get("doi") or []

            # successful match
            if m.get("found") and doi_values:
                rows[r][col] = "; ".join(doi_values)
                replacements += 1

            # nothing found -> clear cell
            else:
                rows[r][col] = ""

        csv_name = f"{Path(src).stem}_resolved.csv"
        csv_path = out_dir / csv_name

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        written.append(
            {
                "source_file": src,
                "csv_name": csv_name,
                "csv_path": str(csv_path),
                "reference_column_index": col,
                "replacements": replacements,
            }
        )

    return written