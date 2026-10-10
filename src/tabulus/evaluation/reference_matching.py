"""Step 5 citation-to-bibliography matching evaluation for Tabulus."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import json
import re


@dataclass(frozen=True)
class ReferenceMatchingEvaluation:
    gold_path: Path
    prediction_path: Path
    gold_tables: int
    aligned_tables: int
    table_coverage: float
    reference_columns_correct: int
    reference_column_accuracy: float
    gold_citation_cells: int
    surface_matched_cells: int
    citation_surface_recall: float
    exact_link_set_cells: int
    exact_cell_accuracy: float
    gold_links: int
    predicted_links: int
    link_true_positives: int
    link_false_positives: int
    link_false_negatives: int
    link_precision: float
    link_recall: float
    link_f1: float

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


def _table_keys(item: dict[str, Any]) -> set[str]:
    out = {str(item["table_id"])} if item.get("table_id") is not None else set()
    for name in ("source_file", "source_prediction", "source_parsed"):
        value = item.get(name)
        if isinstance(value, str) and value:
            out.add(Path(value).stem)
    return out


def _signature(value: Any) -> tuple[int, ...]:
    text = str(value or "").replace("–", "-").replace("—", "-")
    vals: list[int] = []
    spans = []
    for m in re.finditer(r"\b(\d{1,4})\s*-{1,2}\s*(\d{1,4})\b", text):
        a, b = int(m.group(1)), int(m.group(2))
        spans.append(m.span())
        vals.extend(range(a, b + 1) if a <= b and b - a <= 5000 else (a, b))
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            chars[i] = " "
    vals.extend(int(x) for x in re.findall(r"\b\d{1,4}\b", "".join(chars)))
    return tuple(dict.fromkeys(vals))


def _lcs(gold: list[tuple[int, ...]], pred: list[tuple[int, ...]]) -> list[tuple[int, int]]:
    m, n = len(gold), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if gold[i] and gold[i] == pred[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < m and j < n:
        if gold[i] and gold[i] == pred[j] and dp[i][j] == 1 + dp[i + 1][j + 1]:
            pairs.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _indices(values: Any, mapping: Mapping[int, int] | None) -> set[int]:
    if not isinstance(values, list):
        return set()
    out = set()
    for value in values:
        if not isinstance(value, int):
            continue
        if mapping is None:
            out.add(value)
        elif value in mapping:
            out.add(int(mapping[value]))
    return out


def evaluate_reference_matching(
    gold_json: Path,
    prediction_json: Path,
    *,
    bibliography_index_map: Mapping[int, int] | None = None,
) -> ReferenceMatchingEvaluation:
    """Evaluate Step 5; optionally map production Step 4 indices to gold."""

    gold_path, gobj = _load(gold_json, "Step 5 gold")
    pred_path, pobj = _load(prediction_json, "Step 5 prediction")
    gtables, ptables = gobj.get("tables"), pobj.get("matched_tables")
    if not isinstance(gtables, list) or not all(isinstance(x, dict) for x in gtables):
        raise ValueError("Step 5 gold has no valid 'tables' list.")
    if not isinstance(ptables, list) or not all(isinstance(x, dict) for x in ptables):
        raise ValueError("Step 5 prediction has no valid 'matched_tables' list.")

    pindex: dict[str, list[int]] = {}
    for j, table in enumerate(ptables):
        for key in _table_keys(table):
            pindex.setdefault(key, []).append(j)

    used = set()
    aligned = ref_ok = gcells_n = surface = exact = glinks_n = plinks_n = 0
    tp = fp = fn = 0

    for gtable in gtables:
        gid = str(gtable.get("table_id"))
        candidates = [j for j in pindex.get(gid, []) if j not in used]
        gcs = [x for x in gtable.get("citation_cells", []) if isinstance(x, dict)]
        glinks = [_indices(x.get("bibliography_indices"), None) for x in gcs]
        gcells_n += len(gcs); glinks_n += sum(map(len, glinks))

        if len(candidates) != 1:
            fn += sum(map(len, glinks))
            continue

        j = candidates[0]; used.add(j); aligned += 1
        ptable = ptables[j]
        ref_ok += ptable.get("reference_column_index") == gtable.get("reference_column_index")
        pcs = [
            x for x in ptable.get("matches", [])
            if isinstance(x, dict) and not x.get("is_header", False)
        ]
        plinks = [
            _indices(x.get("matched_reference_indices"), bibliography_index_map)
            for x in pcs
        ]
        plinks_n += sum(map(len, plinks))
        gold_signatures = [_signature(x.get("value")) for x in gcs]
        pred_has_surface = all(
            isinstance(x.get("value"), str) and bool(x.get("value").strip())
            for x in pcs
        )

        if pred_has_surface:
            pairs = _lcs(
                gold_signatures,
                [_signature(x.get("value")) for x in pcs],
            )
        else:
            # Historical controlled Step 5 artifacts preserve the ordered
            # per-cell link sets but omit the original cell surface. In that
            # contract, table-local match order is the canonical alignment.
            pairs = [(i, i) for i in range(min(len(gcs), len(pcs)))]

        surface += len(pairs)
        pg, pp = {a for a, _ in pairs}, {b for _, b in pairs}

        for a, b in pairs:
            gs, ps = glinks[a], plinks[b]
            tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
            exact += gs == ps
        fn += sum(len(glinks[a]) for a in range(len(glinks)) if a not in pg)
        fp += sum(len(plinks[b]) for b in range(len(plinks)) if b not in pp)

    for j, table in enumerate(ptables):
        if j in used:
            continue
        for cell in table.get("matches", []):
            if isinstance(cell, dict) and not cell.get("is_header", False):
                links = _indices(cell.get("matched_reference_indices"), bibliography_index_map)
                plinks_n += len(links); fp += len(links)

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ntable, ncell = len(gtables), gcells_n
    return ReferenceMatchingEvaluation(
        gold_path, pred_path, ntable, aligned, aligned / ntable if ntable else 0.0,
        ref_ok, ref_ok / ntable if ntable else 0.0,
        ncell, surface, surface / ncell if ncell else 0.0,
        exact, exact / ncell if ncell else 0.0,
        glinks_n, plinks_n, tp, fp, fn, p, r, f1,
    )
