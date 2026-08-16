# Kreuzberg OCR Service

## Overview

This folder contains the Docker configuration for the Kreuzberg OCR service used in the pipeline.

Kreuzberg is used as a fallback OCR component for bibliography extraction when GROBID fails to extract usable bibliography entries from scientific PDFs.

The service exposes an HTTP API which receives PDF files and returns extracted raw text.

Within the pipeline, the extracted raw text is further processed with regex-based bibliography extraction methods.

---

# Purpose in the Pipeline

The Kreuzberg service is not used as the primary bibliography extraction system.

The default bibliography extraction workflow uses:

1. GROBID
2. TEI XML parsing
3. Structured bibliography extraction

However, some PDFs contain:

* OCR artifacts,
* unsupported bibliography layouts,
* malformed references,
* or bibliography structures that GROBID cannot parse reliably.

In these cases, Kreuzberg is used as a fallback OCR system.

The backend then applies custom regex-based bibliography extraction patterns on the returned raw text.

---

# Pipeline Workflow

```text
Scientific PDF
        ↓
GROBID bibliography extraction
        ↓
GROBID failed or returned no usable references
        ↓
Kreuzberg OCR extraction
        ↓
Raw text extraction
        ↓
Regex-based bibliography extraction
        ↓
Reference matching
```

---

# Dockerfile

```dockerfile
FROM ghcr.io/kreuzberg-dev/kreuzberg:latest

EXPOSE 8010

CMD ["serve", "--host", "0.0.0.0", "--port", "8010"]
```

---

# Docker Configuration

The service runs on port:

```text
8010
```

Inside Docker Compose, the service is reachable through:

```text
http://kreuzberg:8010
```

The backend uses the endpoint:

```text
http://kreuzberg:8010/extract
```

---

# Service Responsibilities

The Kreuzberg service is responsible for:

* receiving PDF files,
* performing OCR extraction,
* returning extracted raw text,
* enabling bibliography fallback extraction.

The service itself does not perform bibliography matching or DOI resolution.

These tasks are handled inside the backend.

---

# Backend Integration

The backend calls the service using:

```python
KREUZBERG_API_URL = os.getenv(
    "KREUZBERG_API_URL",
    "http://kreuzberg:8010/extract",
)
```

The request is sent as multipart PDF upload.

The response contains extracted OCR text which is processed by:

```text
kreuzberg_reference_fallback.py
```

---

# Regex-based Bibliography Extraction

After OCR extraction, the backend applies several bibliography extraction heuristics.

Supported bibliography styles include:

| Pattern   | Description                           |
| --------- | ------------------------------------- |
| Numbered  | `[1]`, `1.` reference styles          |
| APA       | Author-year references                |
| Springer  | Springer-like bibliography formatting |
| Wiley     | Wiley publisher formatting            |
| Frontiers | Frontiers journal formatting          |
| BES       | British Ecological Society formatting |

Currently, the pipeline primarily accepts numbered bibliography patterns as fallback extraction output.

---

# Noise Removal

Before bibliography extraction, the backend removes common OCR and publisher noise such as:

* page headers,
* page footers,
* copyright notices,
* download notices,
* publisher watermarks,
* duplicated OCR lines.

This improves bibliography extraction quality.

---

# Typical Use Cases

The Kreuzberg fallback is especially useful for:

* scanned PDFs,
* noisy OCR documents,
* malformed bibliography layouts,
* PDFs where GROBID returns no bibliography entries,
* reference sections with inconsistent formatting.

---

# Output

The Kreuzberg service itself returns extracted OCR text.

The backend transforms this into structured bibliography entries:

```text
[
  {
    "index": 1,
    "raw": "Example reference...",
    "doi": "10.xxxx/xxxxx",
    "source": "kreuzberg_fallback"
  }
]
```

---

# Notes

The Kreuzberg service is used as a lightweight OCR fallback component within the scientific PDF processing pipeline.

The integration focuses on:

* robustness,
* fallback bibliography extraction,
* OCR-based recovery,
* and reference resolution support.
