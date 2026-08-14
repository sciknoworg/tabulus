# Layout Detectors

Layout detectors find document regions such as tables, figures, captions, headings, and paragraphs.

## Responsibility

- Produce normalized layout items.
- Preserve source-specific raw metadata when useful.
- Avoid forcing downstream components to parse library-specific formats.

## Current Adapter

MinerU currently provides layout output through `content_list.json`.
