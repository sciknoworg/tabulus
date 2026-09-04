from pathlib import Path

import tabulus.bibliography.pipeline as bibliography_pipeline


def test_extract_bibliography_artifact_uses_original_pdf_and_contract_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifact_root = tmp_path / "paper-output"

    bibliography = object()
    calls: dict[str, object] = {}

    class FakeExtractor:
        def __init__(self, *, grobid_url: str, timeout_seconds: float) -> None:
            calls["grobid_url"] = grobid_url
            calls["timeout_seconds"] = timeout_seconds

        def extract(self, received_pdf: Path):
            calls["pdf_path"] = received_pdf
            return bibliography

    def fake_write(received_bibliography, output_path: Path) -> Path:
        calls["bibliography"] = received_bibliography
        calls["output_path"] = output_path
        return output_path

    monkeypatch.setattr(
        bibliography_pipeline,
        "GrobidBibliographyExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        bibliography_pipeline,
        "write_bibliography_json",
        fake_write,
    )

    result = bibliography_pipeline.extract_bibliography_artifact(
        pdf_path,
        artifact_root,
        grobid_url="http://localhost:8070",
        timeout_seconds=42.0,
    )

    expected_output = artifact_root / "references" / "bibliography.json"

    assert calls == {
        "grobid_url": "http://localhost:8070",
        "timeout_seconds": 42.0,
        "pdf_path": pdf_path,
        "bibliography": bibliography,
        "output_path": expected_output,
    }
    assert result == expected_output
