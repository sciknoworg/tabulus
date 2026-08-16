# Python Library Setup

The new Tabulus code is organized as an installable Python library with a standard `src/tabulus` package layout.

Use this setup for core library development and unit tests that do not require GPU execution. For a validated Windows CPU-only MinerU installation, use `installation/windows-cpu`. For GPU-accelerated MinerU profiling, use `installation/gpu-server`.

## Development Install

From the repository root:

```bash
python -m pip install -e .
```

If you intend to run the test suite, install the development extra:

```bash
python -m pip install -e ".[dev]"
```

After installation, verify the MinerU output reader can be imported:

```bash
python - <<'PY'
from tabulus.mineru import discover_tables

print(discover_tables)
PY
```

## Unit Tests

The implemented library is designed so most unit tests do not require GPU access. Tests cover MinerU output discovery, backend selection, command construction, mocked MinerU execution, and table-crop export.

Run the tests with:

```bash
python -m pytest
```

The current Windows validation used Python 3.12.10 and pytest 9.1.1:

```text
21 passed
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

This call inspects a MinerU document output directory. To create that output through Tabulus, use `tabulus profile` in an environment where MinerU is installed.

## Current CLI Commands

After installation, these commands should be available:

```bash
tabulus --version
tabulus profile --help
tabulus export-table-crops --help
```

`tabulus profile` can launch MinerU when MinerU is installed in the active environment. The default `pipeline` backend is CPU-compatible; `hybrid-engine` is selected only when requested and a suitable CUDA GPU is visible.

If `hybrid-engine` is requested but the GPU requirements are not satisfied, Tabulus reports the reason and falls back to `pipeline`.

The profiling CLI separates the profiler from its backend. `mineru` is currently the only profiler; `pipeline` and `hybrid-engine` are MinerU backends.

When `--out` is omitted, Tabulus writes profiling output beside the PDF:

```text
<PDF directory>/tabulus-output/<PDF stem>/profiling/<profiler>/<backend>/
```

`--out` remains available as an explicit override. If `hybrid-engine` falls back to `pipeline`, the automatic directory uses the resolved backend, `pipeline`.

`tabulus export-table-crops` consumes an existing MinerU output directory and writes:

```text
work/table_crops/
  tables_index.json
  images/
```
