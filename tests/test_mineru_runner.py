from pathlib import Path

from types import SimpleNamespace

import pytest

import tabulus.mineru.runner as runner


def test_pipeline_command_does_not_include_effort(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "find_mineru_executable",
        lambda: "mineru",
    )

    command = runner.build_mineru_command(
        tmp_path / "paper.pdf",
        tmp_path / "output",
        backend="pipeline",
    )

    assert "-b" in command
    assert command[command.index("-b") + 1] == "pipeline"
    assert "-m" in command
    assert command[command.index("-m") + 1] == "auto"
    assert "-t" in command
    assert command[command.index("-t") + 1] == "true"
    assert "-f" in command
    assert command[command.index("-f") + 1] == "false"
    assert "--image-analysis" in command
    assert command[
        command.index("--image-analysis") + 1
    ] == "false"
    assert "--effort" not in command


def test_hybrid_command_includes_effort(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "find_mineru_executable",
        lambda: "mineru",
    )

    command = runner.build_mineru_command(
        tmp_path / "paper.pdf",
        tmp_path / "output",
        backend="hybrid-engine",
        effort="high",
    )

    assert command[command.index("-b") + 1] == "hybrid-engine"
    assert "--effort" in command
    assert command[command.index("--effort") + 1] == "high"


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.run_mineru(
            tmp_path / "missing.pdf",
            tmp_path / "output",
        )

def test_run_mineru_returns_native_hybrid_run_dir(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    output_dir = tmp_path / "output"
    native_run_dir = (
        output_dir
        / "paper"
        / "hybrid_auto"
    )

    monkeypatch.setattr(
        runner,
        "find_mineru_executable",
        lambda: "mineru",
    )

    monkeypatch.setattr(
        runner,
        "resolve_backend",
        lambda requested_backend: (
            requested_backend,
            None,
        ),
    )

    def fake_subprocess_run(*args, **kwargs):
        native_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            native_run_dir
            / "paper_content_list.json"
        ).write_text(
            "[]",
            encoding="utf-8",
        )

        return SimpleNamespace(
            returncode=0,
            stdout="MinerU completed",
            stderr="",
        )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        fake_subprocess_run,
    )

    result = runner.run_mineru(
        pdf_path,
        output_dir,
        requested_backend="hybrid-engine",
        effort="high",
        method="auto",
    )

    assert result == native_run_dir

    assert (
        native_run_dir / "mineru_stdout.log"
    ).is_file()

    assert (
        native_run_dir / "mineru_stderr.log"
    ).is_file()

    assert (
        native_run_dir / "tabulus_run.txt"
    ).is_file()

    assert not (
        output_dir
        / "paper"
        / "auto"
    ).exists()