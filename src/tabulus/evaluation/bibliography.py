"""Component-aware bibliography extraction evaluation for Tabulus.

This module implements bibliography/reference extraction evaluation for
Step 4.  The matching core is the frozen component-aware v5 algorithm used
for the controlled TabulusBench evaluation.

The public API is deliberately dataset-agnostic.  It compares one curated
gold bibliography with one Tabulus ``references/bibliography.json`` artifact.
TabulusBench corpus discovery, domain aggregation, and reporting remain
outside the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict
from statistics import mean, median
import json
import re
import unicodedata


MAX_SPLIT = 12

MAX_MERGE = 6

LOOKAHEAD = 12

MATCH_THRESHOLD = 0.62

BLOCK_F1_THRESHOLD = 0.55

COMPONENT_WEIGHTS = {
    "doi": 4.0,
    "authors": 3.0,
    "year": 1.5,
    "title": 2.5,
    "venue": 1.0,
    "locator": 1.5,
}

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from",
    "in", "of", "on", "or", "the", "to", "via", "with",
}

AUTHOR_SUFFIXES = {
    "jr", "sr", "ii", "iii", "iv",
}

def ascii_text(value):
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    return text

def tokens(value):
    return re.findall(r"[a-z0-9]+", ascii_text(value))

def informative_tokens(value, *, min_len=3):
    return [
        t for t in tokens(value)
        if len(t) >= min_len and t not in STOPWORDS
    ]

def compact(value):
    return "".join(tokens(value))

def normalize_doi(value):
    text = ascii_text(value).strip()
    text = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:?\s*)",
        "",
        text,
    )
    return text.strip().rstrip(".,;)")

def surname_from_author(author):
    parts = [
        p for p in tokens(author)
        if p not in AUTHOR_SUFFIXES
    ]
    if not parts:
        return None
    return parts[-1]

def years_in_text(value):
    return set(
        re.findall(
            r"\b(?:17|18|19|20)\d{2}[a-z]?\b",
            ascii_text(value),
        )
    )

def numeric_tokens(value):
    return re.findall(r"\d+", ascii_text(value))

def token_containment(component_text, gold_text, *, min_len=3):
    """
    Fraction of informative component tokens represented in the
    gold raw citation. This is token-based, not character-fuzzy.
    """
    wanted = informative_tokens(
        component_text,
        min_len=min_len,
    )

    if not wanted:
        return None

    gold = Counter(
        informative_tokens(
            gold_text,
            min_len=min_len,
        )
    )
    wanted_counter = Counter(wanted)

    overlap = sum((wanted_counter & gold).values())
    total = sum(wanted_counter.values())

    return overlap / total if total else None

def collect_predicted_components(entries):
    surnames = []
    years = []
    dois = []
    titles = []
    venues = []
    volumes = []
    pages = []

    for entry in entries:
        for author in entry.get("authors") or []:
            surname = surname_from_author(author)
            if surname:
                surnames.append(surname)

        year = entry.get("year")
        if year not in (None, ""):
            years.append(str(year).casefold())

        doi = normalize_doi(entry.get("doi"))
        if doi:
            dois.append(doi)

        title = str(entry.get("title") or "").strip()
        if title:
            titles.append(title)

        venue = str(entry.get("venue") or "").strip()
        if venue:
            venues.append(venue)

        volume = str(entry.get("volume") or "").strip()
        if volume:
            volumes.append(volume)

        page = str(entry.get("pages") or "").strip()
        if page:
            pages.append(page)

    return {
        "surnames": surnames,
        "years": years,
        "dois": dois,
        "titles": titles,
        "venues": venues,
        "volumes": volumes,
        "pages": pages,
    }

def author_tokens(value):
    """
    Produce exact normalized name tokens while preserving boundaries
    recoverable from capitalization.

    Examples:
      GuoQ       -> guo, q
      Sadeghi-TehranP -> sadeghi, tehran, p

    No fuzzy character matching is used for author names.
    """
    raw = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )

    raw = "".join(
        c for c in raw
        if not unicodedata.combining(c)
    )

    # Recover boundaries lost during PDF text extraction.
    raw = re.sub(
        r"(?<=[a-z])(?=[A-Z])",
        " ",
        raw,
    )

    return re.findall(
        r"[A-Za-z]+",
        raw.casefold(),
    )

def author_score(predicted_surnames, gold_text):
    if not predicted_surnames:
        return None

    gold_name_tokens = set(
        author_tokens(gold_text)
    )

    matches = sum(
        surname.casefold() in gold_name_tokens
        for surname in predicted_surnames
    )

    return matches / len(predicted_surnames)

def year_score(predicted_years, gold_text):
    if not predicted_years:
        return None

    pred_years = set(predicted_years)
    gold_years = years_in_text(gold_text)

    if not pred_years or not gold_years:
        return None

    overlap = len(pred_years & gold_years)

    precision = overlap / len(pred_years)
    recall = overlap / len(gold_years)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )

def doi_score(predicted_dois, gold_text):
    if not predicted_dois:
        return None

    gold_compact = ascii_text(gold_text)
    matches = sum(
        doi in gold_compact
        for doi in predicted_dois
    )

    return matches / len(predicted_dois)

def compact_component(value):
    """
    Remove spacing and punctuation while retaining alphanumeric content.
    This makes comparison robust to PDF artifacts such as:

      "im ages"       vs "images"
      "high-throughputcropphenotyping"
      "high-throughput crop phenotyping"
    """
    return re.sub(
        r"[^a-z0-9]+",
        "",
        ascii_text(value),
    )

def character_ngrams(value, n=3):
    value = compact_component(value)

    if not value:
        return Counter()

    if len(value) < n:
        return Counter([value])

    return Counter(
        value[i:i+n]
        for i in range(len(value) - n + 1)
    )

def component_containment(component_text, gold_text):
    """
    Character-n-gram recall of one predicted bibliographic component
    against the raw gold citation.

    This is suitable for title/venue text, where spaces and line-break
    hyphenation are unreliable in PDF-derived gold strings.
    """
    component = compact_component(component_text)
    gold = compact_component(gold_text)

    if not component:
        return None

    # Exact compact containment is the strongest possible result.
    if component in gold:
        return 1.0

    wanted = character_ngrams(component)
    available = character_ngrams(gold)

    total = sum(wanted.values())

    if not total:
        return None

    overlap = sum(
        (wanted & available).values()
    )

    return overlap / total

def averaged_containment(values, gold_text, *, min_len=3):
    if not values:
        return None

    scores = []

    for value in values:
        if len(compact_component(value)) < min_len:
            continue

        score = component_containment(
            value,
            gold_text,
        )

        if score is not None:
            scores.append(score)

    if not scores:
        return None

    return mean(scores)

def locator_score(volumes, pages, gold_text):
    scores = []

    gold_numeric = set(numeric_tokens(gold_text))
    gold_tokens = set(tokens(gold_text))

    for volume in volumes:
        vt = tokens(volume)
        if not vt:
            continue

        matches = sum(
            token in gold_tokens
            for token in vt
        )
        scores.append(matches / len(vt))

    for page in pages:
        pt = numeric_tokens(page)

        if pt:
            matches = sum(
                token in gold_numeric
                for token in pt
            )
            scores.append(matches / len(pt))
        else:
            pc = compact(page)
            if pc:
                scores.append(
                    1.0 if pc in compact(gold_text) else 0.0
                )

    if not scores:
        return None

    return mean(scores)

def block_coverage(gold_text, pred_entries):
    """
    Bidirectional character-n-gram coverage between the complete gold
    citation block and the complete raw GROBID block.

    Whitespace and punctuation are removed first. This makes the
    completeness check robust to PDF extraction artifacts without
    fuzzy-matching individual author names.
    """
    pred_text = " ".join(
        str(entry.get("raw") or "")
        for entry in pred_entries
    )

    gold_ngrams = character_ngrams(
        gold_text,
        n=3,
    )
    pred_ngrams = character_ngrams(
        pred_text,
        n=3,
    )

    gold_n = sum(
        gold_ngrams.values()
    )
    pred_n = sum(
        pred_ngrams.values()
    )

    if gold_n == 0 or pred_n == 0:
        return 0.0, 0.0, 0.0

    overlap = sum(
        (gold_ngrams & pred_ngrams).values()
    )

    precision = overlap / pred_n
    recall = overlap / gold_n

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1

def component_match(gold_text, pred_entries):
    """
    Bibliographic alignment evidence.

    Author names are never character-fuzzy matched. Predicted surnames
    are normalized and checked as exact tokens in the gold citation.

    Year, title, venue, volume/pages and DOI are considered separately.

    Whole-reference block coverage is then used to ensure that a
    prediction represents the complete gold citation rather than merely
    one constituent subreference.
    """
    comp = collect_predicted_components(pred_entries)

    scores = {
        "doi": doi_score(
            comp["dois"],
            gold_text,
        ),
        "authors": author_score(
            comp["surnames"],
            gold_text,
        ),
        "year": year_score(
            comp["years"],
            gold_text,
        ),
        "title": averaged_containment(
            comp["titles"],
            gold_text,
            min_len=3,
        ),
        "venue": averaged_containment(
            comp["venues"],
            gold_text,
            min_len=2,
        ),
        "locator": locator_score(
            comp["volumes"],
            comp["pages"],
            gold_text,
        ),
    }

    block_precision, block_recall, block_f1 = (
        block_coverage(
            gold_text,
            pred_entries,
        )
    )

    # Core components can contribute positive or negative evidence.
    core_names = (
        "authors",
        "year",
        "venue",
        "locator",
    )

    numerator = 0.0
    denominator = 0.0

    for name in core_names:
        score = scores[name]

        if score is None:
            continue

        weight = COMPONENT_WEIGHTS[name]
        numerator += weight * score
        denominator += weight

    # Title and DOI are positive corroborating evidence only.
    # Many bibliography styles omit titles and/or DOIs altogether,
    # so their absence from the raw gold citation must not count
    # against an otherwise correct reference.
    for name in ("title", "doi"):
        score = scores[name]

        if score is None or score <= 0:
            continue

        weight = COMPONENT_WEIGHTS[name]
        numerator += weight * score
        denominator += weight

    component_score = (
        numerator / denominator
        if denominator
        else 0.0
    )

    # Complete raw-block agreement receives most of the weight.
    # Component evidence provides bibliographic corroboration.
    cumulative = (
        0.70 * block_f1
        + 0.30 * component_score
    )

    positive_families = sum(
        score is not None and score >= 0.5
        for score in scores.values()
    )

    strong_doi = (
        scores["doi"] == 1.0
        if scores["doi"] is not None
        else False
    )

    # Very strong whole-reference overlap is sufficient evidence even
    # if GROBID did not populate some structured metadata correctly.
    strong_raw_match = block_f1 >= 0.78

    corroborated_match = (
        block_f1 >= BLOCK_F1_THRESHOLD
        and cumulative >= MATCH_THRESHOLD
        and positive_families >= 2
    )

    doi_rescue = (
        strong_doi
        and max(block_precision, block_recall) >= 0.35
    )

    identity_anchor = (
        (
            scores["authors"] is not None
            and scores["authors"] >= 0.5
        )
        or (
            scores["title"] is not None
            and scores["title"] >= 0.75
        )
        or strong_doi
    )

    # If one complete citation is contained in a noisier block,
    # symmetric F1 can be low despite correct bibliographic identity.
    # Strong component evidence plus one-way block coverage rescues
    # these cases without fuzzy matching author names.
    asymmetric_identity_match = (
        component_score >= 0.72
        and positive_families >= 2
        and identity_anchor
        and max(block_precision, block_recall) >= 0.60
    )

    accepted = (
        strong_raw_match
        or corroborated_match
        or asymmetric_identity_match
        or doi_rescue
    )

    return {
        "score": cumulative,
        "component_score": component_score,
        "block_precision": block_precision,
        "block_recall": block_recall,
        "block_f1": block_f1,
        "accepted": accepted,
        "positive_families": positive_families,
        **{
            f"{name}_score": score
            for name, score in scores.items()
        },
    }

def join_gold(gold, start, count):
    return " ".join(
        gold[k]["ref"]
        for k in range(start, start + count)
    )

def pred_block(pred, start, count):
    return pred[start:start + count]

def candidate_key(candidate):
    """
    Prefer the candidate that best explains the complete citation
    block. Component evidence then resolves close cases.
    """
    return (
        candidate["block_f1"],
        candidate["score"],
        candidate["positive_families"],
        -(candidate["gold_count"] + candidate["pred_count"]),
    )

def best_current_block(gold, pred, i, j):
    candidates = []

    # 1 gold -> k GROBID entries
    for pcount in range(
        1,
        min(MAX_SPLIT, len(pred) - j) + 1,
    ):
        result = component_match(
            gold[i]["ref"],
            pred_block(pred, j, pcount),
        )

        if result["accepted"]:
            candidates.append({
                "gold_offset": 0,
                "pred_offset": 0,
                "gold_count": 1,
                "pred_count": pcount,
                **result,
            })

    # k gold -> 1 GROBID entry
    for gcount in range(
        2,
        min(MAX_MERGE, len(gold) - i) + 1,
    ):
        result = component_match(
            join_gold(gold, i, gcount),
            pred_block(pred, j, 1),
        )

        # A genuine many-gold -> one-prediction merge must
        # represent a substantial fraction of the combined gold block.
        # This prevents one unrelated citation from being accepted merely
        # because it happens to match one constituent of a larger block.
        if (
            result["accepted"]
            and result["block_recall"] >= 0.50
        ):
            candidates.append({
                "gold_offset": 0,
                "pred_offset": 0,
                "gold_count": gcount,
                "pred_count": 1,
                **result,
            })

    if not candidates:
        return None

    return max(candidates, key=candidate_key)

def best_resync_anchor(gold, pred, i, j):
    candidates = []

    max_g = min(
        LOOKAHEAD,
        len(gold) - i - 1,
    )
    max_p = min(
        LOOKAHEAD,
        len(pred) - j - 1,
    )

    for goff in range(max_g + 1):
        for poff in range(max_p + 1):
            if goff == 0 and poff == 0:
                continue

            result = component_match(
                gold[i + goff]["ref"],
                pred_block(pred, j + poff, 1),
            )

            if result["accepted"]:
                candidates.append({
                    "gold_offset": goff,
                    "pred_offset": poff,
                    "gold_count": 1,
                    "pred_count": 1,
                    **result,
                })

    if not candidates:
        return None

    # Prefer strongest evidence, then nearest resynchronization.
    return min(
        candidates,
        key=lambda x: (
            x["gold_offset"] + x["pred_offset"],
            max(x["gold_offset"], x["pred_offset"]),
            -x["block_f1"],
            -x["score"],
            -x["positive_families"],
        ),
    )

def choose_gap_direction(gold, pred, i, j):
    """
    If no accepted local alignment exists, look ahead using the same
    component-aware matcher and move whichever side most plausibly
    restores alignment.
    """
    best_if_gold_skipped = -1.0

    for goff in range(
        1,
        min(LOOKAHEAD, len(gold) - i - 1) + 1,
    ):
        result = component_match(
            gold[i + goff]["ref"],
            pred_block(pred, j, 1),
        )
        best_if_gold_skipped = max(
            best_if_gold_skipped,
            result["score"],
        )

    best_if_pred_skipped = -1.0

    for poff in range(
        1,
        min(LOOKAHEAD, len(pred) - j - 1) + 1,
    ):
        result = component_match(
            gold[i]["ref"],
            pred_block(pred, j + poff, 1),
        )
        best_if_pred_skipped = max(
            best_if_pred_skipped,
            result["score"],
        )

    if best_if_gold_skipped >= best_if_pred_skipped:
        return "gold"

    return "pred"

def align_paper(gold, pred):
    i = 0
    j = 0

    operations = []
    unmatched_gold = []
    unmatched_pred = []

    while i < len(gold) and j < len(pred):
        current = best_current_block(
            gold,
            pred,
            i,
            j,
        )

        if current is not None:
            gcount = current["gold_count"]
            pcount = current["pred_count"]

            if gcount == 1 and pcount == 1:
                kind = "one_to_one"
            elif gcount == 1:
                kind = "split"
            else:
                kind = "merge"

            operations.append({
                "kind": kind,
                "gold_start": i + 1,
                "gold_count": gcount,
                "pred_start": j + 1,
                "pred_count": pcount,
                **{
                    key: value
                    for key, value in current.items()
                    if key not in {
                        "gold_offset",
                        "pred_offset",
                        "gold_count",
                        "pred_count",
                        "accepted",
                    }
                },
            })

            i += gcount
            j += pcount
            continue

        anchor = best_resync_anchor(
            gold,
            pred,
            i,
            j,
        )

        if anchor is not None:
            goff = anchor["gold_offset"]
            poff = anchor["pred_offset"]

            for k in range(i, i + goff):
                unmatched_gold.append(k + 1)

            for k in range(j, j + poff):
                unmatched_pred.append(k + 1)

            gi = i + goff
            pj = j + poff

            operations.append({
                "kind": "one_to_one",
                "gold_start": gi + 1,
                "gold_count": 1,
                "pred_start": pj + 1,
                "pred_count": 1,
                **{
                    key: value
                    for key, value in anchor.items()
                    if key not in {
                        "gold_offset",
                        "pred_offset",
                        "gold_count",
                        "pred_count",
                        "accepted",
                    }
                },
            })

            i = gi + 1
            j = pj + 1
            continue

        direction = choose_gap_direction(
            gold,
            pred,
            i,
            j,
        )

        if direction == "gold":
            unmatched_gold.append(i + 1)
            i += 1
        else:
            unmatched_pred.append(j + 1)
            j += 1

    while i < len(gold):
        unmatched_gold.append(i + 1)
        i += 1

    while j < len(pred):
        unmatched_pred.append(j + 1)
        j += 1

    return (
        operations,
        unmatched_gold,
        unmatched_pred,
    )

def coverage_prf(
    gold_used,
    pred_used,
    gold_n,
    pred_n,
):
    precision = (
        pred_used / pred_n
        if pred_n else 0.0
    )
    recall = (
        gold_used / gold_n
        if gold_n else 0.0
    )
    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1

# ----------------------------------------------------------------------
# Public Tabulus evaluation API
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BibliographyEvaluation:
    """Evaluation result for one gold/prediction bibliography pair."""

    gold_path: Path
    prediction_path: Path

    gold_entries: int
    predicted_entries: int

    content_recovered_gold: int
    content_supported_predictions: int

    content_precision: float
    content_recall: float
    content_f1: float

    one_to_one_gold: int

    split_gold_groups: int
    split_pred_entries: int

    merge_groups: int
    merged_gold_entries: int

    unmatched_gold: int
    unmatched_pred: int

    position_preserved_gold: int
    position_preservation_rate: float

    mean_abs_position_error_1to1: float
    median_abs_position_error_1to1: float
    mean_alignment_score: float

    predicted_component_availability: dict[str, int]

    operations: tuple[dict[str, Any], ...]
    unmatched_gold_positions: tuple[int, ...]
    unmatched_pred_positions: tuple[int, ...]

    @property
    def strict_entry_true_positives(self) -> int:
        """Number of accepted gold-to-prediction 1:1 bibliography matches."""
        return self.one_to_one_gold

    @property
    def strict_entry_false_positives(self) -> int:
        """Predicted entries not participating in an accepted 1:1 match."""
        return (
            self.predicted_entries
            - self.strict_entry_true_positives
        )

    @property
    def strict_entry_false_negatives(self) -> int:
        """Gold entries not participating in an accepted 1:1 match."""
        return (
            self.gold_entries
            - self.strict_entry_true_positives
        )

    @property
    def strict_entry_precision(self) -> float:
        """Strict bibliography-entry precision using only 1:1 matches."""
        if not self.predicted_entries:
            return 0.0

        return (
            self.strict_entry_true_positives
            / self.predicted_entries
        )

    @property
    def strict_entry_recall(self) -> float:
        """Strict bibliography-entry recall using only 1:1 matches."""
        if not self.gold_entries:
            return 0.0

        return (
            self.strict_entry_true_positives
            / self.gold_entries
        )

    @property
    def strict_entry_f1(self) -> float:
        """Strict bibliography-entry F1 using only 1:1 matches."""
        precision = self.strict_entry_precision
        recall = self.strict_entry_recall

        if not (precision + recall):
            return 0.0

        return (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    def to_dict(self):
        """Serialize evaluation results including strict entry metrics."""
        payload = self._to_dict_without_strict()

        payload.update(
            {
                "strict_entry_true_positives": (
                    self.strict_entry_true_positives
                ),
                "strict_entry_false_positives": (
                    self.strict_entry_false_positives
                ),
                "strict_entry_false_negatives": (
                    self.strict_entry_false_negatives
                ),
                "strict_entry_precision": (
                    self.strict_entry_precision
                ),
                "strict_entry_recall": (
                    self.strict_entry_recall
                ),
                "strict_entry_f1": (
                    self.strict_entry_f1
                ),
            }
        )

        return payload

    def _to_dict_without_strict(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation payload."""

        return {
            "gold_path": str(self.gold_path),
            "prediction_path": str(self.prediction_path),
            "gold_entries": self.gold_entries,
            "predicted_entries": self.predicted_entries,
            "content_recovered_gold": self.content_recovered_gold,
            "content_supported_predictions": (
                self.content_supported_predictions
            ),
            "content_precision": self.content_precision,
            "content_recall": self.content_recall,
            "content_f1": self.content_f1,
            "one_to_one_gold": self.one_to_one_gold,
            "split_gold_groups": self.split_gold_groups,
            "split_pred_entries": self.split_pred_entries,
            "merge_groups": self.merge_groups,
            "merged_gold_entries": self.merged_gold_entries,
            "unmatched_gold": self.unmatched_gold,
            "unmatched_pred": self.unmatched_pred,
            "position_preserved_gold": self.position_preserved_gold,
            "position_preservation_rate": (
                self.position_preservation_rate
            ),
            "mean_abs_position_error_1to1": (
                self.mean_abs_position_error_1to1
            ),
            "median_abs_position_error_1to1": (
                self.median_abs_position_error_1to1
            ),
            "mean_alignment_score": self.mean_alignment_score,
            "match_threshold": MATCH_THRESHOLD,
            "block_f1_threshold": BLOCK_F1_THRESHOLD,
            "max_split": MAX_SPLIT,
            "max_merge": MAX_MERGE,
            "lookahead": LOOKAHEAD,
            "component_weights": dict(COMPONENT_WEIGHTS),
            "predicted_component_availability": dict(
                self.predicted_component_availability
            ),
            "operations": [
                dict(operation)
                for operation in self.operations
            ],
            "unmatched_gold_positions": list(
                self.unmatched_gold_positions
            ),
            "unmatched_pred_positions": list(
                self.unmatched_pred_positions
            ),
        }

    def write_json(self, path: Path) -> Path:
        """Write this pair-level evaluation to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return output_path


def _require_json(
    path: Path,
    *,
    label: str,
) -> Path:
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} JSON not found: {resolved}"
        )

    if resolved.suffix.lower() != ".json":
        raise ValueError(
            f"{label} must be a JSON file: {resolved}"
        )

    return resolved


def _load_gold_bibliography(
    path: Path,
) -> list[dict[str, Any]]:
    """Load the curated ``bibliography/gold.json`` contract."""

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "Gold bibliography must be a JSON list."
        )

    for position, entry in enumerate(
        data,
        start=1,
    ):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Gold entry {position} is not an object."
            )

        if not isinstance(entry.get("ref"), str):
            raise ValueError(
                f"Gold entry {position} has no string 'ref'."
            )

    return data


def _load_prediction_bibliography(
    path: Path,
) -> list[dict[str, Any]]:
    """Load a Step 4 ``references/bibliography.json`` artifact."""

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Prediction bibliography must be a JSON object."
        )

    entries = data.get("entries")

    if not isinstance(entries, list):
        raise ValueError(
            "Prediction bibliography has no valid 'entries' list."
        )

    count = data.get("bibliography_count")

    if count is not None and count != len(entries):
        raise ValueError(
            "Prediction bibliography_count does not match "
            f"len(entries): {count} != {len(entries)}"
        )

    for position, entry in enumerate(
        entries,
        start=1,
    ):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Prediction entry {position} is not an object."
            )

    return entries


def _component_availability(
    entries: list[dict[str, Any]],
) -> dict[str, int]:
    """Count structured components available in predictions."""

    availability = Counter()

    for entry in entries:
        availability["entries"] += 1

        if entry.get("authors"):
            availability["authors"] += 1

        if entry.get("year") not in (
            None,
            "",
        ):
            availability["year"] += 1

        if str(
            entry.get("title") or ""
        ).strip():
            availability["title"] += 1

        if str(
            entry.get("venue") or ""
        ).strip():
            availability["venue"] += 1

        if str(
            entry.get("volume") or ""
        ).strip():
            availability["volume"] += 1

        if str(
            entry.get("pages") or ""
        ).strip():
            availability["pages"] += 1

        if normalize_doi(
            entry.get("doi")
        ):
            availability["doi"] += 1

    return {
        key: availability[key]
        for key in (
            "entries",
            "authors",
            "year",
            "title",
            "venue",
            "volume",
            "pages",
            "doi",
        )
    }


def evaluate_bibliography(
    gold_json: Path,
    prediction_json: Path,
) -> BibliographyEvaluation:
    """Evaluate one extracted bibliography against curated gold.

    This function implements the pair-level portion of the frozen
    component-aware v5 TabulusBench Step 4 evaluator.

    It is intentionally independent of TabulusBench paths, paper identifiers,
    domains, and result directories.
    """

    gold_path = _require_json(
        gold_json,
        label="Gold bibliography",
    )

    prediction_path = _require_json(
        prediction_json,
        label="Prediction bibliography",
    )

    gold = _load_gold_bibliography(
        gold_path
    )

    pred = _load_prediction_bibliography(
        prediction_path
    )

    (
        operations,
        unmatched_gold,
        unmatched_pred,
    ) = align_paper(
        gold,
        pred,
    )

    gold_used = sum(
        operation["gold_count"]
        for operation in operations
    )

    pred_used = sum(
        operation["pred_count"]
        for operation in operations
    )

    (
        precision,
        recall,
        f1,
    ) = coverage_prf(
        gold_used,
        pred_used,
        len(gold),
        len(pred),
    )

    one_to_one = [
        operation
        for operation in operations
        if operation["kind"] == "one_to_one"
    ]

    splits = [
        operation
        for operation in operations
        if operation["kind"] == "split"
    ]

    merges = [
        operation
        for operation in operations
        if operation["kind"] == "merge"
    ]

    exact_position = [
        operation
        for operation in one_to_one
        if (
            operation["gold_start"]
            == operation["pred_start"]
        )
    ]

    displacements = [
        abs(
            operation["pred_start"]
            - operation["gold_start"]
        )
        for operation in one_to_one
    ]

    return BibliographyEvaluation(
        gold_path=gold_path,
        prediction_path=prediction_path,
        gold_entries=len(gold),
        predicted_entries=len(pred),
        content_recovered_gold=gold_used,
        content_supported_predictions=pred_used,
        content_precision=precision,
        content_recall=recall,
        content_f1=f1,
        one_to_one_gold=len(one_to_one),
        split_gold_groups=len(splits),
        split_pred_entries=sum(
            operation["pred_count"]
            for operation in splits
        ),
        merge_groups=len(merges),
        merged_gold_entries=sum(
            operation["gold_count"]
            for operation in merges
        ),
        unmatched_gold=len(
            unmatched_gold
        ),
        unmatched_pred=len(
            unmatched_pred
        ),
        position_preserved_gold=len(
            exact_position
        ),
        position_preservation_rate=(
            len(exact_position) / len(gold)
            if gold else 0.0
        ),
        mean_abs_position_error_1to1=(
            mean(displacements)
            if displacements
            else 0.0
        ),
        median_abs_position_error_1to1=(
            median(displacements)
            if displacements
            else 0.0
        ),
        mean_alignment_score=(
            mean(
                operation["score"]
                for operation in operations
            )
            if operations
            else 0.0
        ),
        predicted_component_availability=(
            _component_availability(pred)
        ),
        operations=tuple(
            dict(operation)
            for operation in operations
        ),
        unmatched_gold_positions=tuple(
            unmatched_gold
        ),
        unmatched_pred_positions=tuple(
            unmatched_pred
        ),
    )
