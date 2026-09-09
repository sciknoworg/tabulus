from tabulus.table_continuations import (
    annotate_continuations,
    caption_text,
    continuation_links_from_records,
    is_explicit_continuation_caption,
    table_label,
)


def test_common_continuation_caption_forms() -> None:
    cases = (
        ("Table 1 (continued)", "1"),
        ("Table 1. Continued", "1"),
        ("TABLE A. CONTINUED", "A"),
        ("Table a — continued", "A"),
        ("Table I. Continued", "I"),
        ("Table IV (cont.)", "IV"),
        ("Table S1, contd.", "S1"),
        ("Supplementary Table S 1 — cont'd", "S1"),
        ("Table A-1 continued", "A1"),
        ("Table 1-A continued", "1A"),
        ("Table 2.1 (continued)", "2.1"),
        ("TABLE III. \x01Continued.-", "III"),
    )

    for raw_caption, expected_label in cases:
        caption = caption_text(raw_caption)

        assert table_label(caption) == expected_label
        assert is_explicit_continuation_caption(caption) is True


def test_descriptive_continued_text_is_not_a_continuation() -> None:
    captions = (
        "TABLE V. Typical processes (references in Table III).",
        "Table 5. Results discussed and continued in the text.",
        "Table A. Comparison with Table I.",
    )

    for raw_caption in captions:
        assert (
            is_explicit_continuation_caption(
                caption_text(raw_caption)
            )
            is False
        )


def test_continuation_chain_records_parent_and_root() -> None:
    records = [
        {
            "table_id": 8,
            "page_nr": 4,
            "table_caption": ["Table 4. Comparison"],
        },
        {
            "table_id": 9,
            "page_nr": 5,
            "table_caption": ["Table 4 (continued)"],
        },
        {
            "table_id": 10,
            "page_nr": 6,
            "table_caption": ["Table 4 — continued"],
        },
        {
            "table_id": 11,
            "page_nr": 7,
            "table_caption": ["Table 5. Other results"],
        },
    ]

    annotated = annotate_continuations(records)

    root = annotated[0]["continuation"]
    first_cont = annotated[1]["continuation"]
    second_cont = annotated[2]["continuation"]
    standalone = annotated[3]["continuation"]

    assert root == {
        "is_continuation": False,
        "continued_from_table_id": None,
        "continuation_root_table_id": 8,
        "printed_table_label": "4",
        "evidence": None,
        "link_status": "not_continuation",
    }

    assert first_cont["continued_from_table_id"] == 8
    assert first_cont["continuation_root_table_id"] == 8
    assert first_cont["link_status"] == "linked"

    assert second_cont["continued_from_table_id"] == 9
    assert second_cont["continuation_root_table_id"] == 8
    assert second_cont["link_status"] == "linked"

    assert standalone["continued_from_table_id"] is None
    assert standalone["continuation_root_table_id"] == 11


def test_structured_metadata_is_preferred_over_caption_reparsing() -> None:
    records = [
        {
            "table_id": 1,
            "table_caption": ["Table 7. Results"],
            "continuation": {
                "is_continuation": False,
                "continued_from_table_id": None,
                "continuation_root_table_id": 1,
                "printed_table_label": "7",
                "evidence": None,
                "link_status": "not_continuation",
            },
        },
        {
            "table_id": 2,
            # Deliberately not an explicit continuation caption:
            # structured Step 1 metadata is authoritative.
            "table_caption": ["fragment two"],
            "continuation": {
                "is_continuation": True,
                "continued_from_table_id": 1,
                "continuation_root_table_id": 1,
                "printed_table_label": "7",
                "evidence": "explicit_caption",
                "link_status": "linked",
            },
        },
    ]

    links = continuation_links_from_records(records)

    assert links[2].parent_table_id == 1
    assert links[2].root_table_id == 1


def test_legacy_caption_only_index_remains_supported() -> None:
    records = [
        {
            "table_id": 1,
            "table_caption": ["Table 3. Results"],
        },
        {
            "table_id": 2,
            "table_caption": ["Table 3 (continued)"],
        },
    ]

    links = continuation_links_from_records(records)

    assert links[2].parent_table_id == 1
    assert links[2].root_table_id == 1
