from pathlib import Path

from tabulus import cli


def test_profile_parser_accepts_folder_input() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "profile",
            "--folder",
            "papers",
            "--backend",
            "pipeline",
        ]
    )

    assert args.pdf is None
    assert args.folder == Path("papers")
    assert args.pdf_list is None


def test_profile_parser_accepts_pdf_list_input() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "profile",
            "--pdf-list",
            "papers.txt",
            "--backend",
            "pipeline",
        ]
    )

    assert args.pdf is None
    assert args.folder is None
    assert args.pdf_list == Path("papers.txt")


def test_profile_main_processes_folder_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    alpha = papers / "Alpha.pdf"
    zeta = papers / "zeta.pdf"
    alpha.write_bytes(b"%PDF")
    zeta.write_bytes(b"%PDF")

    mineru_out = tmp_path / "mineru-output"
    crops_out = tmp_path / "table-crops"
    calls = {"runs": [], "exports": []}

    monkeypatch.setattr(
        "sys.argv",
        [
            "tabulus",
            "profile",
            "--folder",
            str(papers),
            "--out",
            str(mineru_out),
            "--backend",
            "pipeline",
            "--table-crops-out",
            str(crops_out),
        ],
    )
    monkeypatch.setattr(
        cli,
        "select_backend",
        lambda requested: "pipeline",
    )

    def fake_run_mineru(
        pdf_path,
        output_dir,
        *,
        requested_backend,
        effort,
        method,
    ):
        calls["runs"].append(
            (
                pdf_path,
                output_dir,
                requested_backend,
                effort,
                method,
            )
        )
        return output_dir / pdf_path.stem / "auto"

    def fake_export_mineru_table_crops(
        *,
        mineru_output_dir,
        output_dir,
    ):
        calls["exports"].append(
            (mineru_output_dir, output_dir)
        )

        class FakeResult:
            tables_found = 1
            crops_saved = 1
            index_path = output_dir / "tables_index.json"

        return FakeResult()

    monkeypatch.setattr(cli, "run_mineru", fake_run_mineru)
    monkeypatch.setattr(
        cli,
        "export_mineru_table_crops",
        fake_export_mineru_table_crops,
    )

    cli.main()

    assert [call[0] for call in calls["runs"]] == [
        alpha,
        zeta,
    ]
    assert all(
        call[1] == mineru_out
        for call in calls["runs"]
    )
    assert [call[1] for call in calls["exports"]] == [
        crops_out / "Alpha",
        crops_out / "zeta",
    ]
