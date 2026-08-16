# MinerU GPU Execution

This page records the tested MinerU command sequence for producing document outputs that the new Tabulus library can inspect.

It does not describe the full Tabulus pipeline. At this stage, Tabulus consumes existing MinerU outputs; it does not launch MinerU itself.

## Environment

MinerU was tested in its own Conda environment with Python 3.12.

```bash
conda activate tabulus-mineru
cd ~/tabulus
```

Confirm the environment:

```bash
which python
mineru --version
```

The tested MinerU version was:

```text
MinerU 3.4.5
```

The installation command was:

```bash
python -m pip install "mineru[all]==3.4.5"
```

## Verify GPU Access

Before processing a document, verify PyTorch and CUDA visibility:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

In the tested environment this resolved to one visible NVIDIA L40S GPU.

GPU assignment is deliberately external:

```bash
CUDA_VISIBLE_DEVICES=0
```

Do not hard-code the GPU selection inside Tabulus. Keeping GPU assignment external makes the same workflow compatible with GPU servers and schedulers such as Slurm.

## Prepare Work Directories

The current headless workflow uses:

```text
work/
  input/
  mineru/
  table_crops/
  paddleocr/
```

Create the first required directories:

```bash
mkdir -p work/input
mkdir -p work/mineru
```

Place the PDF under `work/input/`.

## Run MinerU

The configuration successfully tested for scientific PDF table processing was:

```bash
CUDA_VISIBLE_DEVICES=0 mineru \
  -p "$HOME/tabulus/work/input/Puurunen - February 2005.pdf" \
  -o "$HOME/tabulus/work/mineru/puurunen_2005" \
  -b hybrid-engine \
  --effort high \
  -m auto \
  -t true \
  -f false \
  --image-analysis false
```

Option meaning:

| Option | Meaning |
| --- | --- |
| `-b hybrid-engine` | Use MinerU's hybrid document-processing backend. |
| `--effort high` | Use the high-effort processing configuration. |
| `-m auto` | Automatically determine text and OCR handling. |
| `-t true` | Enable table processing. |
| `-f false` | Disable formula processing for this workflow. |
| `--image-analysis false` | Disable additional image and chart analysis. |

## First Run Versus Subsequent Runs

The first MinerU invocation can be substantially slower because it may:

- download the MinerU VLM checkpoint
- download supporting OCR and layout models
- initialize the inference engine
- compile kernels
- populate caches

For the tested 53-page PDF, MinerU downloaded its VLM and supporting OCR/layout models before processing. Treat that first run separately from steady-state performance measurements.

## Verify MinerU Output

After completion, inspect the output tree:

```bash
find "$HOME/tabulus/work/mineru/puurunen_2005" \
  -maxdepth 5 -type f | sort
```

The document-specific MinerU directory should contain structured JSON outputs, layout visualization, Markdown reconstruction, an original PDF copy, and an `images/` directory. See `data-contracts/mineru-output-files` for the file-level reference.

## Validate With The Tabulus Library

The current library-level validation is:

```bash
python - <<'PY'
from pathlib import Path
from tabulus.mineru import discover_tables

root = Path.home() / "tabulus/work/mineru/puurunen_2005"

tables, refs_start_page = discover_tables(root)

print("Tables:", len(tables))
print("References start:", refs_start_page)

for table in tables:
    print(
        table.table_id,
        table.page_nr,
        table.image_path.name,
        table.caption,
    )
PY
```

For the tested document, the validated result was:

```text
Tables: 23
```

The detected table regions began on page 6 and ended on page 22.

## Implementation Boundary

```text
                    ALREADY VALIDATED
                           |
PDF
 |
 v
MinerU 3.4.5 on GPU -------+
 |
 v
MinerU structured output
 |
 v
new tabulus library --------+
 |
 +-- locate content_list.json
 +-- parse document elements
 +-- select table regions
 +-- resolve image crops
 +-- retain provenance
 +-- expose typed TableRegion objects


                    NOT YET IMPLEMENTED
                           |
                           v
                  crop export / PNG
                           |
                           v
                     PaddleOCR-VL
                           |
                           v
                  reference processing
                           |
                           v
                    final pipeline
```

The new library currently provides typed access to existing MinerU outputs. These stages are not yet implemented in the new library:

- MinerU process launching
- table JPG to PNG export
- `tables_index.json` generation
- PaddleOCR-VL execution
- GROBID, Kreuzberg, or Crossref integration
- full Tabulus process command
