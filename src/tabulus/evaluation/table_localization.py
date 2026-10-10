"""Step 1 physical-table localization evaluation for Tabulus."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class TableLocalizationEvaluation:
    gold_path: Path
    prediction_path: Path
    gold_fragments: int
    predicted_fragments: int
    matched_fragments: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    matched_by_identity: int
    matched_by_page_order: int
    matches: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gold_path"] = str(self.gold_path)
        payload["prediction_path"] = str(self.prediction_path)
        payload["geometry_metric"] = None
        payload["geometry_note"] = (
            "No IoU is reported because canonical Step 1 gold has no "
            "authoritative bounding boxes."
        )
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


def _stem(value: Any) -> str | None:
    return Path(value).stem if isinstance(value, str) and value else None


def _gold_keys(x: dict[str, Any]) -> set[str]:
    values = [x.get("table_id"), x.get("crop")]
    ann = x.get("annotation")
    if isinstance(ann, dict):
        values += [
            ann.get("benchmark_table_key"),
            ann.get("image"),
            ann.get("image_name"),
        ]
    out: set[str] = set()
    for value in values:
        if isinstance(value, str) and value:
            out.add(value)
            out.add(Path(value).stem)
    return out


def _pred_keys(x: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    if x.get("table_id") is not None:
        out.add(str(x["table_id"]))
    for name in ("image", "image_name"):
        value = x.get(name)
        if isinstance(value, str) and value:
            out.add(value)
            out.add(Path(value).stem)
    return out


def _gold_page(x: dict[str, Any]) -> int | None:
    if isinstance(x.get("page"), int):
        return x["page"]
    ann = x.get("annotation")
    if isinstance(ann, dict) and isinstance(ann.get("page_nr"), int):
        return ann["page_nr"]
    return None


def _pred_page(x: dict[str, Any]) -> int | None:
    return x.get("page_nr") if isinstance(x.get("page_nr"), int) else None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate_table_localization(
    gold_json: Path,
    prediction_json: Path,
) -> TableLocalizationEvaluation:
    """Score curated physical-fragment recovery, not bounding-box IoU."""

    gold_path, gold_obj = _load(gold_json, "Step 1 gold")
    pred_path, pred_obj = _load(prediction_json, "Step 1 prediction")
    gold = gold_obj.get("tables")
    pred = pred_obj.get("tables")
    if not isinstance(gold, list) or not all(isinstance(x, dict) for x in gold):
        raise ValueError("Step 1 gold has no valid 'tables' list.")
    if not isinstance(pred, list) or not all(isinstance(x, dict) for x in pred):
        raise ValueError("Step 1 prediction has no valid 'tables' list.")

    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[dict[str, Any]] = []

    pred_index: dict[str, list[int]] = {}
    for j, item in enumerate(pred):
        for key in _pred_keys(item):
            pred_index.setdefault(key, []).append(j)

    for i, item in enumerate(gold):
        candidates = {
            j
            for key in _gold_keys(item)
            for j in pred_index.get(key, [])
            if j not in used_p
        }
        if len(candidates) != 1:
            continue
        j = next(iter(candidates))
        gp, pp = _gold_page(item), _pred_page(pred[j])
        if gp is not None and pp is not None and gp != pp:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append({
            "gold_table_id": item.get("table_id"),
            "prediction_table_id": pred[j].get("table_id"),
            "page": gp,
            "method": "identity",
        })

    gpages: dict[int, list[int]] = {}
    ppages: dict[int, list[int]] = {}
    for i, item in enumerate(gold):
        page = _gold_page(item)
        if i not in used_g and page is not None:
            gpages.setdefault(page, []).append(i)
    for j, item in enumerate(pred):
        page = _pred_page(item)
        if j not in used_p and page is not None:
            ppages.setdefault(page, []).append(j)

    for page in sorted(set(gpages) & set(ppages)):
        for i, j in zip(gpages[page], ppages[page]):
            used_g.add(i)
            used_p.add(j)
            matches.append({
                "gold_table_id": gold[i].get("table_id"),
                "prediction_table_id": pred[j].get("table_id"),
                "page": page,
                "method": "page_order",
            })

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gold) - tp
    p, r, f1 = _prf(tp, fp, fn)
    return TableLocalizationEvaluation(
        gold_path, pred_path, len(gold), len(pred), tp, fp, fn, p, r, f1,
        sum(x["method"] == "identity" for x in matches),
        sum(x["method"] == "page_order" for x in matches),
        tuple(matches),
    )
