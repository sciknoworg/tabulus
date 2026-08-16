from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


def clean_ref(s: str) -> str:
    s = str(s or "")
    s = s.replace("\x02", "")
    s = s.replace("\u00ad", "")
    s = s.replace("", "")
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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


def remove_noise(text: str) -> str:
    text = str(text or "")
    text = text.replace("\x02", "")
    text = text.replace("\u00ad", "")
    text = text.replace("", "")

    patterns = [
        r"Rubiales et al\.\s+Legume Breeding Needs: European Perspective",
        r"Frontiers in Plant Science\s+\|\s+www\.frontiersin\.org.*?Article\s+\d+",
        r"Agron\. Sustain\. Dev\.\s+\(\d{4}\).*?Page\s+\d+\s+of\s+\d+\s+\d*",
        r"\d+\s+Page\s+\d+\s+of\s+\d+\s+Agron\. Sustain\. Dev\.\s+\(\d{4}\).*",
        r"ECOLOGICAL APPLICATIONS\s+\d+\s+of\s+\d+",
        r"\d+\s+of\s+\d+\s+ROILO ET AL\.",
        r"19395582,\s*\d{4},\s*\d+,?",
        r"Downloaded from https://.*?Creative Commons License",
    ]

    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I | re.S)

    return text


def call_kreuzberg_raw_text(
    pdf_path: Path,
    kreuzberg_api_url: str,
    timeout: int = 3600,
) -> str:
    with pdf_path.open("rb") as f:
        response = requests.post(
            kreuzberg_api_url,
            files={"files": (pdf_path.name, f, "application/pdf")},
            timeout=(10, timeout),
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"kreuzberg_service error: {response.status_code}: {response.text[:500]}"
        )

    payload = response.json()

    if isinstance(payload, list) and payload:
        payload = payload[0]

    return payload.get("content", "") or ""


def get_references_section(text: str) -> str:
    text = remove_noise(text)

    match = re.search(r"\bREFERENCES\b|\bReferences\b", text)

    if not match:
        return ""

    refs = text[match.end():]

    stop = re.search(
        r"\bSUPPORTING INFORMATION\b|"
        r"\bHow to cite this article\b|"
        r"\bConflict of Interest\b|"
        r"\bPublisher[’']?s note\b|"
        r"\bFuller\. 2012\. Predicted\b|"
        r"^\s*Copyright ©|"
        r"^\s*Copyright ",
        refs,
        flags=re.I | re.M,
    )

    if stop:
        refs = refs[:stop.start()]

    return refs.strip()


def extract_numbered_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = remove_noise(refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n")

    refs_text = re.split(
        r"\bSummary of systematic\b|"
        r"\bRQS score\b|"
        r"\bModality\b|"
        r"\bBiological correlate\b|"
        r"\bDisclaimer/Publisher[’']?s Note\b",
        refs_text,
        flags=re.I,
    )[0]

    refs_text = re.sub(
        r"^\s*.*J Epidemiol Community Health.*$",
        " ",
        refs_text,
        flags=re.I | re.M,
    )
    refs_text = re.sub(
        r"^\s*Protected by copyright.*$",
        " ",
        refs_text,
        flags=re.I | re.M,
    )
    refs_text = re.sub(
        r"^\s*.*Downloaded from.*$",
        " ",
        refs_text,
        flags=re.I | re.M,
    )

    pattern_brackets = re.compile(
        r"(?ms)"
        r"\[(\d{1,3})\]\s+"
        r"(.*?)"
        r"(?=\s*\[\d{1,3}\]\s+|\Z)"
    )

    refs = []

    for nr, ref in pattern_brackets.findall(refs_text):
        ref = clean_ref(ref)

        if len(ref) < 25:
            continue

        refs.append({"nr": str(nr), "ref": ref})

    if len(refs) >= 5:
        return refs

    pattern_dot = re.compile(
        r"(?ms)"
        r"(?:^|\n)\s*(\d{1,3})\.\s+"
        r"(.*?)"
        r"(?=(?:\n\s*\d{1,3}\.\s+)|\Z)"
    )

    refs = []

    for nr, ref in pattern_dot.findall(refs_text):
        ref = clean_ref(ref)

        if len(ref) < 25:
            continue

        refs.append({"nr": str(nr), "ref": ref})

    if len(refs) >= 5:
        return refs

    pattern_bmj = re.compile(
        r"(?ms)"
        r"(?:^|\n)\s*(\d{1,3})\s+"
        r"(.*?)"
        r"(?=(?:\n\s*\d{1,3}\s+[A-ZÀ-ÖØ-Þ])|\Z)"
    )

    refs = []

    for nr, ref in pattern_bmj.findall(refs_text):
        ref = clean_ref(ref)

        if len(ref) < 30:
            continue

        if re.search(
            r"Protected by copyright|Downloaded from|"
            r"J Epidemiol Community Health|doi:",
            ref,
            re.I,
        ):
            continue

        if not re.search(r"\b(?:19|20)\d{2}\b|https?://|www\.", ref):
            continue

        refs.append({"nr": str(nr), "ref": ref})

    if len(refs) >= 5:
        return refs

    return []


def extract_apa_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = clean_ref(remove_noise(refs_text))

    author = (
        r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+"
        r"(?:\s+[A-ZÀ-ÖØ-Þa-zà-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+){0,3}"
    )

    pattern = re.compile(
        rf"""
        (
            {author},\s+.*?\(\d{{4}}[a-z]?\)\..*?
        )
        (?=
            \s+{author},\s+.*?\(\d{{4}}[a-z]?\)\.
            |
            \Z
        )
        """,
        re.X | re.S,
    )

    refs = []

    for match in pattern.finditer(refs_text):
        ref = clean_ref(match.group(1))

        if len(ref) >= 40:
            refs.append(
                {
                    "nr": str(len(refs) + 1),
                    "ref": ref,
                }
            )

    return refs


def extract_springer_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = remove_noise(refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n")

    start_re = re.compile(r"^[A-Z].*\(\d{4}", re.M)

    refs = []
    current = []

    for line in refs_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if start_re.match(line):
            if current:
                refs.append(clean_ref(" ".join(current)))

            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        refs.append(clean_ref(" ".join(current)))

    final = []

    for ref in refs:
        if len(ref) >= 35 and re.search(r"\(\d{4}", ref):
            final.append(
                {
                    "nr": str(len(final) + 1),
                    "ref": ref,
                }
            )

    return final


def extract_wiley_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = remove_noise(refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n")

    refs_text = re.split(
        r"\bSUPPORTING INFORMATION\b|"
        r"\bHow to cite this article\b|"
        r"\bPublisher[’']?s note\b|"
        r"\bConflict of Interest\b|"
        r"\bFuller\. 2012\. Predicted\b|"
        r"Copyright",
        refs_text,
        flags=re.I,
    )[0]

    raw_lines = refs_text.split("\n")

    lines = []

    for line in raw_lines:
        line = line.strip()

        if not line:
            continue

        if re.search(r"ECOLOGICAL APPLICATIONS|ROILO ET AL\.", line, re.I):
            continue

        if re.search(r"Downloaded from|Wiley Online Library|Creative Commons License", line, re.I):
            continue

        lines.append(line)

    def is_reference_start(line: str, lookahead: str) -> bool:
        line = line.strip()
        joined = clean_ref(lookahead)

        if len(line) < 3:
            return False

        if not re.search(r"\b(?:19|20)\d{2}\.", joined[:350]):
            return False

        special = (
            r"BMEL\.",
            r"SMEKUL\.",
            r"R Core Team\.",
            r"Staatsbetrieb Geobasisinformation und Vermessung Sachsen\.",
        )

        if any(re.match(p, line) for p in special):
            return True

        return bool(
            re.match(
                r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s+"
                r"(?:[A-Z]\.\s*){1,4}",
                line,
            )
        )

    def looks_complete_ref(ref: str) -> bool:
        ref = clean_ref(ref)

        if len(ref) < 35:
            return False

        if not re.search(r"\b(?:19|20)\d{2}\.", ref):
            return False

        if re.search(
            r"Downloaded from|Wiley Online Library|Creative Commons License|Terms and Conditions",
            ref,
            flags=re.I,
        ):
            return False

        return True

    refs = []
    current = []

    for i, line in enumerate(lines):
        lookahead = " ".join(lines[i:i + 6])

        if is_reference_start(line, lookahead):
            if current:
                ref = clean_ref(" ".join(current))

                if looks_complete_ref(ref):
                    refs.append(ref)

            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        ref = clean_ref(" ".join(current))

        if looks_complete_ref(ref):
            refs.append(ref)

    return [{"nr": str(i + 1), "ref": ref} for i, ref in enumerate(refs)]


def extract_frontiers_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = remove_noise(refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n")

    refs_text = re.split(
        r"Frontiers in .*?frontiersin\.org|"
        r"Kumar et al\.\s+10\.3389|"
        r"Downloaded from",
        refs_text,
        flags=re.I | re.S,
    )[0]

    lines = []

    for line in refs_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if re.search(r"Frontiers in|frontiersin\.org|Kumar et al\.", line, re.I):
            continue

        lines.append(line)

    def is_frontiers_start(line: str, lookahead: str) -> bool:
        joined = clean_ref(lookahead)

        if not re.search(r"\((?:19|20)\d{2}[a-z]?\)\.", joined[:450]):
            return False

        if re.match(
            r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s+"
            r"(?:[A-Z]\.\s*){1,4}",
            line,
        ):
            return True

        if re.match(
            r"^(?:de|dos|da|van|von)\s+"
            r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s+",
            line,
        ):
            return True

        return False

    refs = []
    current = []

    for i, line in enumerate(lines):
        lookahead = " ".join(lines[i:i + 8])

        if is_frontiers_start(line, lookahead):
            if current:
                ref = clean_ref(" ".join(current))

                if len(ref) >= 35 and re.search(r"\((?:19|20)\d{2}[a-z]?\)\.", ref):
                    refs.append(ref)

            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        ref = clean_ref(" ".join(current))

        if len(ref) >= 35 and re.search(r"\((?:19|20)\d{2}[a-z]?\)\.", ref):
            refs.append(ref)

    return [{"nr": str(i + 1), "ref": ref} for i, ref in enumerate(refs)]


def extract_bes_refs(refs_text: str) -> List[Dict[str, str]]:
    refs_text = remove_noise(refs_text)
    refs_text = refs_text.replace("\r\n", "\n").replace("\r", "\n")

    refs_text = re.sub(
        r"\bFig\.\s*\d+\..*?(?=^\s*[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s*[A-Z]|"
        r"^\s*[A-Z]{2,10}\s+\((?:19|20)\d{2}\)|"
        r"^\s*(?:van|von|de|del|da|dos)\s+[A-ZÀ-ÖØ-Þ]|"
        r"\Z)",
        " ",
        refs_text,
        flags=re.I | re.S | re.M,
    )

    refs_text = re.split(
        r"\bReceived\s+\d{1,2}\s+\w+\s+\d{4}\b|"
        r"\bSupporting Information\b|"
        r"Downloaded from|"
        r"Wiley Online Library",
        refs_text,
        flags=re.I,
    )[0]

    lines = []

    for line in refs_text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if re.search(
            r"Journal of Ecology|British Ecological Society|Benefits of diversifying|"
            r"F\.\s*Isbell et al\.|^\d+\s*$|Downloaded from|Wiley Online Library|"
            r"Creative Commons License|Terms and Conditions",
            line,
            re.I,
        ):
            continue

        lines.append(line)

    def is_bes_start(line: str, lookahead: str) -> bool:
        joined = clean_ref(lookahead)

        if not re.search(r"\((?:19|20)\d{2}[a-z]?\)", joined[:450]):
            return False

        if re.match(
            r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s*"
            r"[A-Z](?:\.|-)",
            line,
        ):
            return True

        if re.match(
            r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s*"
            r"[A-Z](?:\.[A-Z]\.)?\s*&",
            line,
        ):
            return True

        if re.match(r"^[A-Z]{2,10}\s+\((?:19|20)\d{2}\)", line):
            return True

        if re.match(
            r"^(?:van|von|de|del|da|dos)\s+"
            r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+,\s*",
            line,
        ):
            return True

        return False

    refs = []
    current = []

    for i, line in enumerate(lines):
        lookahead = " ".join(lines[i:i + 8])

        if is_bes_start(line, lookahead):
            if current:
                ref = clean_ref(" ".join(current))

                if len(ref) >= 35 and re.search(r"\((?:19|20)\d{2}[a-z]?\)", ref):
                    refs.append(ref)

            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        ref = clean_ref(" ".join(current))

        if len(ref) >= 35 and re.search(r"\((?:19|20)\d{2}[a-z]?\)", ref):
            refs.append(ref)

    return [{"nr": str(i + 1), "ref": ref} for i, ref in enumerate(refs)]


def extract_references_with_pattern(text: str):
    refs_section = get_references_section(text)

    if not refs_section:
        return [], "no_references_heading", {}

    numbered = extract_numbered_refs(refs_section)
    apa = extract_apa_refs(refs_section)
    springer = extract_springer_refs(refs_section)
    wiley = extract_wiley_refs(refs_section)
    frontiers = extract_frontiers_refs(refs_section)
    bes = extract_bes_refs(refs_section)

    counts = {
        "numbered": len(numbered),
        "apa": len(apa),
        "springer": len(springer),
        "wiley": len(wiley),
        "frontiers": len(frontiers),
        "bes": len(bes),
    }

    if len(numbered) >= 7:
        return numbered, "numbered", counts

    if len(frontiers) >= 10:
        return frontiers, "frontiers", counts

    if len(bes) >= 10:
        return bes, "bes", counts

    if len(wiley) >= 10:
        return wiley, "wiley", counts

    if len(springer) >= 10:
        return springer, "springer", counts

    return apa, "apa", counts


def extract_bibliography_with_kreuzberg(
    pdf_path: Path,
    kreuzberg_api_url: str,
    refs_start_page: Optional[int] = None,
) -> Dict[str, Any]:
    raw_text = call_kreuzberg_raw_text(
        pdf_path=pdf_path,
        kreuzberg_api_url=kreuzberg_api_url,
    )

    if not raw_text:
        raise RuntimeError("kreuzberg_service returned no raw text.")

    refs, pattern_used, pattern_counts = extract_references_with_pattern(raw_text)

    if pattern_used != "numbered":
        raise RuntimeError(
            f"kreuzberg_service fallback skipped. Expected numbered bibliography pattern, "
            f"but detected pattern was '{pattern_used}'. Counts: {pattern_counts}"
        )

    bibliography = []

    for item in refs:
        nr = item.get("nr")
        raw = item.get("ref", "")

        try:
            index = int(nr)
        except Exception:
            index = len(bibliography) + 1

        bibliography.append(
            {
                "index": index,
                "raw": raw,
                "doi": extract_doi(raw),
                "source": "kreuzberg_fallback",
            }
        )

    return {
        "bibliography": bibliography,
        "pattern_used": pattern_used,
        "pattern_counts": pattern_counts,
        "raw_text_length": len(raw_text),
        "refs_start_page": refs_start_page,
    }