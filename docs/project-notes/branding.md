# Branding

Use the Tabulus logo from the repository as the canonical logo asset:

```text
assets/logo.png
```

The Sphinx documentation serves a copy from:

```text
docs/_static/tabulus-logo.png
```

That copy should only be refreshed from `assets/logo.png`.

## Usage Rule

Do not create alternate logos for documentation pages, generated diagrams, package metadata, or future container dashboards. If a derived image size is needed, generate it from `assets/logo.png` and record the derived file as a build artifact or explicitly named variant.

## Current Logo

```{image} ../_static/tabulus-logo.png
:alt: Tabulus logo
:width: 240px
```
