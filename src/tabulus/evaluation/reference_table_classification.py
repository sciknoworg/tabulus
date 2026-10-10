"""Step 3 reference-table classification evaluation for Tabulus."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ReferenceTableClassificationEvaluation:
    gold_path: Path
    prediction_path: Path
    gold_tables: int
    positive_gold: int
    negative_gold: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    specificity: float | None
    accuracy: float
    balanced_accuracy: float | None

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


def _pred_keys(item: dict[str, Any]) -> set[str]:
    out = {str(item["table_id"])} if item.get("table_id") is not None else set()
    for name in ("source_prediction", "source_parsed"):
        value = item.get(name)
        if isinstance(value, str) and value:
            out.add(Path(value).stem)
    return out


def _div(a: int, b: int) -> float:
    return a / b if b else 0.0


def evaluate_reference_table_classification(
    gold_json: Path,
    prediction_json: Path,
) -> ReferenceTableClassificationEvaluation:
    """Evaluate one complete controlled Step 3 classification manifest."""

    gold_path, gobj = _load(gold_json, "Step 3 gold")
    pred_path, pobj = _load(prediction_json, "Step 3 prediction")
    gold = gobj.get("tables")
    pred = pobj.get("tables")
    if not isinstance(gold, list) or not all(isinstance(x, dict) for x in gold):
        raise ValueError("Step 3 gold has no valid 'tables' list.")
    if not isinstance(pred, list) or not all(isinstance(x, dict) for x in pred):
        raise ValueError("Step 3 prediction has no valid 'tables' list.")

    index: dict[str, list[dict[str, Any]]] = {}
    for item in pred:
        for key in _pred_keys(item):
            index.setdefault(key, []).append(item)

    pairs: list[tuple[bool, bool]] = []
    failed = False
    for item in gold:
        key = str(item.get("table_id"))
        candidates = index.get(key, [])
        if len(candidates) != 1:
            failed = True
            break
        gv, pv = item.get("is_reference_table"), candidates[0].get("is_reference_table")
        if not isinstance(gv, bool) or not isinstance(pv, bool):
            raise ValueError(f"Invalid Step 3 boolean label for {key}.")
        pairs.append((gv, pv))

    # Controlled manifests preserve order even when source path provenance is
    # absent. Production/end-to-end alignment belongs in corpus orchestration.
    if failed:
        if len(gold) != len(pred):
            raise ValueError(
                "Step 3 prediction does not cover the complete gold set. "
                "Align production tables upstream before pair-level scoring."
            )
        pairs = []
        for g, p in zip(gold, pred):
            gv, pv = g.get("is_reference_table"), p.get("is_reference_table")
            if not isinstance(gv, bool) or not isinstance(pv, bool):
                raise ValueError("Invalid Step 3 boolean label.")
            pairs.append((gv, pv))

    tp = sum(g and p for g, p in pairs)
    fp = sum((not g) and p for g, p in pairs)
    tn = sum((not g) and (not p) for g, p in pairs)
    fn = sum(g and (not p) for g, p in pairs)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    has_positive = (tp + fn) > 0
    has_negative = (tn + fp) > 0
    specificity = _div(tn, tn + fp) if has_negative else None
    accuracy = _div(tp + tn, len(pairs))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (
        (recall + specificity) / 2
        if has_positive and has_negative and specificity is not None
        else None
    )

    return ReferenceTableClassificationEvaluation(
        gold_path, pred_path, len(pairs), sum(g for g, _ in pairs),
        sum(not g for g, _ in pairs), tp, fp, tn, fn, precision, recall, f1,
        specificity, accuracy, balanced_accuracy,
    )
