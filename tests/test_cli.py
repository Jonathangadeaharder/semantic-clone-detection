"""Tests for clone_detection.cli.main (Click CLI commands).

The CLI instantiates ``GraphCodeBERTEmbedder`` directly, which would require
torch + a network download. These tests monkeypatch the embedder class and the
index builder's ``train``/``add`` path so the full ingest/search/info flow can
be exercised deterministically and offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from clone_detection.cli import main as cli_main
from clone_detection.query.metadata import MetadataStore

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import FakeEmbedder


def test_cli_group_help() -> None:
    """The CLI exposes a help message listing its subcommands."""
    result = CliRunner().invoke(cli_main.cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "search" in result.output
    assert "info" in result.output


def test_cli_version() -> None:
    """--version prints the package version."""
    result = CliRunner().invoke(cli_main.cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_ingest_builds_index_and_metadata(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Ingest parses, embeds, builds a FLAT index, and writes metadata + index."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0, result.output
    assert index_out.exists()
    assert db_out.exists()

    store = MetadataStore(str(db_out))
    assert store.count() == 2
    assert "python" in store.get_languages()
    store.close()


def test_ingest_no_snippets_exits_nonzero(
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """An empty source directory yields exit code 1."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(empty_dir),
            "--index-output",
            str(tmp_path / "x.index"),
            "--metadata-db",
            str(tmp_path / "x.db"),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 1
    assert "No code snippets" in result.output


def test_search_by_query_code(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Search --query-code returns matching clones from a built index."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"

    ingest = CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )
    assert ingest.exit_code == 0, ingest.output

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "search",
            "--index",
            str(index_out),
            "--metadata-db",
            str(db_out),
            # Match the exact code tree-sitter extracts (no trailing newline),
            # so the FakeEmbedder hashes the query to the same vector as the
            # indexed snippet.
            "--query-code",
            "def add(a, b):\n    return a + b",
            "--similarity",
            "0.9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Found" in result.output


def test_search_by_query_file(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Search --query-file reads the file and searches for its clones."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"
    query_file = tmp_path / "query.py"
    # No trailing newline: must match the exact code tree-sitter extracts so
    # the FakeEmbedder hashes the query to the indexed vector.
    query_file.write_text("def add(a, b):\n    return a + b")

    ingest = CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )
    assert ingest.exit_code == 0, ingest.output

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "search",
            "--index",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--query-file",
            str(query_file),
            "--similarity",
            "0.9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Found" in result.output


def test_search_by_location(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Search --query-file with --line-number finds clones by location."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"
    query_file = source_dir / "a.py"

    ingest = CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )
    assert ingest.exit_code == 0, ingest.output

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "search",
            "--index",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--query-file",
            str(query_file),
            "--line-number",
            "1",
            "--similarity",
            "0.9",
        ],
    )
    assert result.exit_code == 0, result.output


def test_search_requires_query_or_file(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Search without --query-code/--query-file exits with code 1."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"

    CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )

    result = CliRunner().invoke(
        cli_main.cli,
        ["search", "--index", str(index_out), "--metadata-db", str(db_out)],
    )
    assert result.exit_code == 1
    assert "must be provided" in result.output


def test_info_prints_index_stats(
    source_dir: Path,
    tmp_path: Path,
    patched_embedder: FakeEmbedder,
    flat_index_builder: None,
) -> None:
    """Info loads an index and metadata and prints a stats table."""
    index_out = tmp_path / "clones.index"
    db_out = tmp_path / "clones.db"

    CliRunner().invoke(
        cli_main.cli,
        [
            "ingest",
            "--source-dir",
            str(source_dir),
            "--index-output",
            str(index_out),
            "--metadata-db",
            str(db_out),
            "--languages",
            "python",
            "--seed",
            "42",
        ],
    )

    result = CliRunner().invoke(
        cli_main.cli,
        ["info", "--index", str(index_out), "--metadata-db", str(db_out)],
    )
    assert result.exit_code == 0, result.output
    assert "Index Type" in result.output
    assert "Metadata Count" in result.output


def test_info_missing_index_exits_nonzero(tmp_path: Path) -> None:
    """Info on a nonexistent index exits nonzero."""
    result = CliRunner().invoke(
        cli_main.cli,
        [
            "info",
            "--index",
            str(tmp_path / "nope.index"),
            "--metadata-db",
            str(tmp_path / "nope.db"),
        ],
    )
    # click.Path(exists=True) rejects nonexistent paths -> exit code 2.
    assert result.exit_code != 0
