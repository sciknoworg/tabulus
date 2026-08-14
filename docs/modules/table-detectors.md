# Table Detectors

Table detectors identify table regions and table metadata.

## Responsibility

- Assign stable table ids.
- Record page number and bounding box or source image path.
- Capture captions and footnotes if available.

## Current Adapter

The current MinerU runner filters layout items with `type == "table"`.
