"""Step 6 scholarly-identity resolution quality evaluation for Tabulus."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import json
import re
import unicodedata


VALIDATED = {"validated_with_doi", "validated_without_doi"}
STOP = {"of", "the", "and", "for", "in", "on", "a", "an"}


@dataclass(frozen=True)
class ReferenceResolutionEvaluation:
    gold_path: Path
    prediction_path: Path
    gold_targets: int
    matched_predictions: int
    missing_predictions: int
    resolved_targets: int
    rejected_targets: int
    resolution_yield: float
    doi_evaluable: int
    doi_predictions: int
    doi_exact_matches: int
    doi_precision: float
    doi_recall: float
    doi_f1: float
    title_evaluable: int
    title_correct: int
    title_accuracy: float
    authors_evaluable: int
    authors_correct: int
    authors_accuracy: float
    year_evaluable: int
    year_correct: int
    year_accuracy: float
    venue_evaluable: int
    venue_correct: int
    venue_accuracy: float
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_path"] = str(self.gold_path)
        payload["prediction_path"] = str(self.prediction_path)
        return payload

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _load(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return path, value


def _ascii(value: Any) -> str:
    s = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(c for c in s if not unicodedata.combining(c))


def _normalize_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _ascii(value)).split())


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii(value))


def _doi(value: Any) -> str:
    s = _ascii(value).strip()
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:?\s*)", "", s)
    return s.rstrip(".,;)")


def _available(value: Any) -> bool:
    if value is None: return False
    if isinstance(value, str): return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)): return bool(value)
    return True


def _gold_field(record: dict[str, Any], name: str) -> Any:
    """Read the canonical Step 6 gold field, with legacy aliases."""
    canonical = {
        "doi": "gold_doi",
        "title": "title",
        "authors": "authors",
        "year": "year",
        "venue": "venue",
    }[name]

    if canonical in record:
        return record.get(canonical)

    return record.get(f"gold_{name}")


def _title_ok(gold: Any, pred: Any) -> bool:
    g, p = _normalize_text(gold), _normalize_text(pred)
    return bool(
        g
        and p
        and (
            g == p
            or SequenceMatcher(None, g, p, autojunk=False).ratio() >= 0.95
        )
    )


def _surname(value: Any) -> str:
    text = _ascii(value).strip()
    if "," in text:
        text = text.split(",", 1)[0]
    parts = re.findall(r"[a-z0-9]+", text)
    return parts[-1] if parts else ""


def _authors(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [
            part.strip()
            for part in re.split(r"\s+and\s+", value, flags=re.IGNORECASE)
            if part.strip()
        ]
    elif isinstance(value, list):
        values = value
    else:
        return ()

    result = []
    for item in values:
        surname = _surname(item)
        if surname and surname != "others":
            result.append(surname)

    return tuple(result)


def _venue_tokens(value: Any) -> list[str]:
    return [x for x in re.findall(r"[a-z0-9]+", _ascii(value)) if x not in STOP]


def _venue_ok(gold: Any, pred: Any) -> bool:
    g, p = _venue_tokens(gold), _venue_tokens(pred)
    if not g or not p:
        return False

    if g == p:
        return True

    short, long = (g, p) if len(g) <= len(p) else (p, g)

    def compatible(a: str, b: str) -> bool:
        if a == b:
            return True

        shorter, longer = (
            (a, b) if len(a) <= len(b) else (b, a)
        )

        if len(shorter) == 1:
            return longer.startswith(shorter)

        k = min(len(a), len(b), 4)
        return k >= 2 and a[:k] == b[:k]

    matched = sum(
        any(compatible(token, candidate) for candidate in long)
        for token in short
    )

    return matched / len(short) >= 0.80


def evaluate_reference_resolution(
    gold_json: Path,
    prediction_json: Path,
) -> ReferenceResolutionEvaluation:
    """Evaluate final Step 6 identities against independent source gold."""

    gold_path, gobj = _load(gold_json, "Step 6 gold")
    pred_path, pobj = _load(prediction_json, "Step 6 prediction")
    records, entries = gobj.get("records"), pobj.get("entries")
    if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
        raise ValueError("Step 6 gold has no valid 'records' list.")
    if not isinstance(entries, list) or not all(isinstance(x, dict) for x in entries):
        raise ValueError("Step 6 prediction has no valid 'entries' list.")

    pby = {}
    for entry in entries:
        res = entry.get("resolution")
        if not isinstance(res, dict) or not isinstance(res.get("reference_index"), int):
            continue
        idx = res["reference_index"]
        if idx in pby: raise ValueError(f"Duplicate Step 6 reference_index: {idx}")
        pby[idx] = res

    status = {"validated_with_doi": 0, "validated_without_doi": 0, "rejected": 0}
    matched = resolved = rejected = 0
    de = dp = dx = te = tc = ae = ac = ye = yc = ve = vc = 0

    for g in records:
        # Frozen controlled Step 6 predictions are keyed to the independent
        # scholarly-identity gold index. step4_index remains provenance for
        # mapping back to the complete bibliography.
        idx = g.get("gold_index")
        if not isinstance(idx, int):
            idx = g.get("step4_index")
        if not isinstance(idx, int):
            raise ValueError("Gold record lacks integer gold_index.")
        p = pby.get(idx)
        if p is not None:
            matched += 1
            s = p.get("status")
            if s in status: status[s] += 1
            if s in VALIDATED: resolved += 1
            elif s == "rejected": rejected += 1

        if _available(_gold_field(g, "doi")):
            de += 1
            pd = p.get("canonical_doi") if p else None
            if _available(pd):
                dp += 1
                dx += _doi(pd) == _doi(_gold_field(g, "doi"))

        if _available(_gold_field(g, "title")):
            te += 1
            tc += _title_ok(_gold_field(g, "title"), p.get("canonical_title") if p else None)

        if _available(_gold_field(g, "authors")):
            ae += 1
            ac += bool(_authors(_gold_field(g, "authors"))) and _authors(_gold_field(g, "authors")) == _authors(
                p.get("canonical_authors") if p else None
            )

        if _available(_gold_field(g, "year")):
            ye += 1
            yc += str(p.get("canonical_year") if p else "").strip() == str(_gold_field(g, "year")).strip()

        if _available(_gold_field(g, "venue")):
            ve += 1
            vc += _venue_ok(_gold_field(g, "venue"), p.get("canonical_venue") if p else None)

    dprec = dx / dp if dp else 0.0
    drec = dx / de if de else 0.0
    df1 = 2 * dprec * drec / (dprec + drec) if dprec + drec else 0.0
    total = len(records)

    return ReferenceResolutionEvaluation(
        gold_path, pred_path, total, matched, total - matched, resolved, rejected,
        resolved / total if total else 0.0,
        de, dp, dx, dprec, drec, df1,
        te, tc, tc / te if te else 0.0,
        ae, ac, ac / ae if ae else 0.0,
        ye, yc, yc / ye if ye else 0.0,
        ve, vc, vc / ve if ve else 0.0,
        status,
    )
