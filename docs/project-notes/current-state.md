# Current State

The repository currently mixes three concerns:

- `src/Tabulus`: the production-oriented pipeline.
- `src/ocr_models`: OCR services, model experiments, and benchmark runners.
- `evaluation`: evaluation scripts, plots, and result summaries.

The documentation should keep these concerns separate so the main tutorial remains a clear processing flow.

## Known Documentation Drift

- Some READMEs mention `tabulus/pipeline`, but the current production folder is `src/Tabulus`.
- Some READMEs mention `paddle_service`, but the actual folder is `paddleocr_service`.
- Docker instructions are stale for Windows machines without NVIDIA GPUs.
- The root `requirements.txt` captures a broad research environment and should not be treated as the production install contract.
