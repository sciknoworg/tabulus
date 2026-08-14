# Page Renderers

Page renderers convert PDF pages into images.

## Responsibility

- Render pages at a fixed DPI.
- Write deterministic image names.
- Record image dimensions and rendering settings.

## Candidate Tools

- `PyMuPDF`
- Poppler utilities

The renderer output becomes `metadata/page_rendering.json` plus `pages/page_XXX.png`.
