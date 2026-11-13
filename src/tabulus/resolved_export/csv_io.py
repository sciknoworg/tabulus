
"""Shared CSV reading for Step 7 and its evaluation artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock


# Resolved JSON arrays can exceed Python's default limit even for valid tables.
MAX_FIELD_CHARS = 1_000_000
_FIELD_LIMIT_LOCK = Lock()


def read_csv(path: Path) -> list[list[str]]:
    with _FIELD_LIMIT_LOCK:
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(max(previous_limit, MAX_FIELD_CHARS))
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                return list(csv.reader(handle, strict=True))
        finally:
            csv.field_size_limit(previous_limit)
