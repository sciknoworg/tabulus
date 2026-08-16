# Python Library Setup

The new Tabulus code is organized as an installable Python library with a standard `src/tabulus` package layout.

Use this setup for library development and unit tests that do not require GPU execution.

## Development Install

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

After installation, verify the MinerU output reader can be imported:

```bash
python - <<'PY'
from tabulus.mineru import discover_tables

print(discover_tables)
PY
```

## Unit Tests

The first implemented module is designed to be testable without GPU access because it reads existing MinerU output files from disk.

Run the tests with:

```bash
python -m pytest
```

The current tests cover:

- recursive `*_content_list.json` discovery
- table-region extraction
- page and provenance handling
- reference-section detection
- missing-output error handling

## Current Import Boundary

The current public entry point is:

```python
from pathlib import Path
from tabulus.mineru import discover_tables

tables, refs_start_page = discover_tables(Path("work/mineru/puurunen_2005"))
```

This call expects MinerU to have already produced its document output directory.
