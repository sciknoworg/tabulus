# GPU Server Run

The GPU server workflow should run the same modules as the local workflow, but with GPU-backed adapters for layout detection and table OCR.

## Assumptions

- Python environment is managed directly on the server.
- GPU libraries are installed outside Docker.
- Input PDFs are already available in a papers folder.
- Output runs are written to a persistent runs folder.

## Recommended Shape

Use a manifest when processing many papers:

```text
paper_id,pdf_path
P51,/data/papers/P51.pdf
P52,/data/papers/P52.pdf
```

Then run:

```bash
python -m tabulus_pipeline.profile_manifest --manifest /data/papers.csv --runs-root /data/runs --adapter mineru
```

Each later module should process either one run or all runs with a selected status.
