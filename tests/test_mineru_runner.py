from pathlib import Path

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
