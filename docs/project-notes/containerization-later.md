# Containerization Later

Docker should be treated as a packaging step after the modules work independently.

## Recommended Order

1. Stabilize file-based module contracts.
2. Make each module runnable from the command line.
3. Add tests or fixture runs for each module.
4. Add an end-to-end orchestrator.
5. Containerize stable modules.

This avoids hiding module-level failures inside stale container images.
