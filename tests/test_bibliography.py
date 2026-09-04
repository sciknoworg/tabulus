from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabulus.bibliography import (
    BIBLIOGRAPHY_NAME,
    extract_doi,
    parse_grobid_tei,
    write_bibliography_json,
)


GROBID_FIXTURE = """\
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct xml:id="b0">
            <analytic>
              <author><persName><surname>Smith</surname></persName></author>
              <title>Structured title</title>
            </analytic>
            <note type="raw_reference">
              Smith J. Example paper. 2020. https://doi.org/10.1234/Example.1.
            </note>
          </biblStruct>
          <biblStruct xml:id="b1">
            <analytic>
              <author><persName><surname>Müller</surname></persName></author>
              <title>Second paper</title>
            </analytic>
            <monogr><imprint><date when="2021">2021</date></imprint></monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>
"""


def test_extract_doi_normalizes_common_forms() -> None:
    assert (
        extract_doi("https://doi.org/10.1016/j.example.2026.01.001.")
        == "10.1016/j.example.2026.01.001"
    )
    assert extract_doi("DOI: 10.1000/ABC-123)") == "10.1000/ABC-123"
    assert extract_doi("No persistent identifier") == ""


def test_parse_grobid_tei_prefers_raw_reference_and_preserves_order() -> None:
    bibliography = parse_grobid_tei(GROBID_FIXTURE)

    assert bibliography.source == "grobid"
    assert bibliography.bibliography_count == 2

    first, second = bibliography.entries
    assert first.index == 1
    assert first.raw == (
        "Smith J. Example paper. 2020. "
        "https://doi.org/10.1234/Example.1."
    )
    assert first.doi == "10.1234/Example.1"

    assert second.index == 2
    assert "Müller" in second.raw
    assert "Second paper" in second.raw
    assert second.doi == ""


def test_parse_grobid_tei_keeps_empty_bibliography_positions() -> None:
    tei = """\
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text><back><listBibl>
        <biblStruct xml:id="b0" />
        <biblStruct xml:id="b1"><note type="raw_reference">Second</note></biblStruct>
      </listBibl></back></text>
    </TEI>
    """

    bibliography = parse_grobid_tei(tei)

    assert [entry.index for entry in bibliography.entries] == [1, 2]
    assert bibliography.entries[0].raw == ""
    assert bibliography.entries[1].raw == "Second"


def test_parse_grobid_tei_rejects_invalid_xml() -> None:
    with pytest.raises(ValueError, match="valid TEI XML"):
        parse_grobid_tei("<TEI>")


def test_write_bibliography_json_matches_data_contract(tmp_path: Path) -> None:
    bibliography = parse_grobid_tei(GROBID_FIXTURE)
    output_path = tmp_path / "references" / BIBLIOGRAPHY_NAME

    written = write_bibliography_json(bibliography, output_path)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == output_path
    assert payload["bibliography_count"] == 2
    assert payload["bibliography_source"] == "grobid"
    assert payload["entries"][0] == {
        "index": 1,
        "raw": (
            "Smith J. Example paper. 2020. "
            "https://doi.org/10.1234/Example.1."
        ),
        "doi": "10.1234/Example.1",
        "source": "grobid",
    }
