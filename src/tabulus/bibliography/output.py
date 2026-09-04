from __future__ import annotations

import json
from pathlib import Path

from tabulus.bibliography.models import Bibliography


BIBLIOGRAPHY_NAME = "bibliography.json"


def write_bibliography_json(
    bibliography: Bibliography,
    output_path: Path,
) -> Path:
    """Persist a normalized bibliography artifact without mutating inputs."""

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            bibliography.to_dict(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
