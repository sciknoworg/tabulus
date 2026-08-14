# Branding

Use the Tabulus logo from the repository as the canonical logo asset:

```text
assets/logo.png
```

The Sphinx documentation configuration points to this file directly through `html_logo`.

## Usage Rule

Do not create alternate logos for documentation pages, generated diagrams, package metadata, or future container dashboards. If a derived image size is needed, generate it from `assets/logo.png` and record the derived file as a build artifact or explicitly named variant.

## Current Logo

```{image} ../../assets/logo.png
:alt: Tabulus logo
:width: 240px
```
