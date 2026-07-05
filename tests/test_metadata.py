"""Tests for clone_detection.query.metadata.MetadataStore."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from clone_detection.parsers.tree_sitter_parser import CodeSnippet
from clone_detection.query.metadata import MetadataStore

if TYPE_CHECKING:
    from pathlib import Path


def _snippet(
    i: int,
    *,
    file_path: str = "x.py",
    start_line: int = 1,
    end_line: int = 2,
    function_name: str | None = None,
) -> CodeSnippet:
    """Build a small CodeSnippet for use in metadata tests."""
    return CodeSnippet(
        code=f"# snippet {i}\n",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language="python",
        function_name=function_name,
    )


def test_add_and_get_snippet(tmp_db_path: Path) -> None:
    """add_snippet then get_snippet round-trips the metadata."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(1, _snippet(1, function_name="foo"))
    got = store.get_snippet(1)
    assert got is not None
    assert got["id"] == 1
    assert got["function_name"] == "foo"
    assert got["language"] == "python"
    store.close()


def test_get_snippet_missing_returns_none(tmp_db_path: Path) -> None:
    """get_snippet returns None for an unknown ID."""
    store = MetadataStore(str(tmp_db_path))
    assert store.get_snippet(999) is None
    store.close()


def test_add_snippet_replaces_existing(tmp_db_path: Path) -> None:
    """add_snippet with an existing ID replaces the row."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(1, _snippet(1, function_name="foo"))
    store.add_snippet(1, _snippet(2, function_name="bar"))
    got = store.get_snippet(1)
    assert got is not None
    assert got["function_name"] == "bar"
    store.close()


def test_add_snippets_batch(tmp_db_path: Path) -> None:
    """add_snippets_batch inserts multiple rows."""
    store = MetadataStore(str(tmp_db_path))
    snippets = [(i, _snippet(i, function_name=f"f{i}")) for i in range(5)]
    store.add_snippets_batch(snippets)
    assert store.count() == 5
    store.close()


def test_add_snippets_batch_empty_is_noop(tmp_db_path: Path) -> None:
    """add_snippets_batch with an empty list is a no-op."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippets_batch([])
    assert store.count() == 0
    store.close()


def test_get_snippets_multiple(tmp_db_path: Path) -> None:
    """get_snippets returns rows for the requested IDs."""
    store = MetadataStore(str(tmp_db_path))
    for i in range(3):
        store.add_snippet(i, _snippet(i))
    result = store.get_snippets([0, 2])
    ids = {r["id"] for r in result}
    assert ids == {0, 2}
    store.close()


def test_get_snippets_empty_list_returns_empty(tmp_db_path: Path) -> None:
    """get_snippets with an empty list returns an empty list."""
    store = MetadataStore(str(tmp_db_path))
    assert store.get_snippets([]) == []
    store.close()


def test_get_snippet_by_location(tmp_db_path: Path) -> None:
    """get_snippet_by_location finds the row containing the line."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(
        1, _snippet(1, file_path="x.py", start_line=10, end_line=20, function_name="f")
    )
    found = store.get_snippet_by_location("x.py", 15)
    assert found is not None
    assert found["id"] == 1


def test_get_snippet_by_location_outside_range_returns_none(
    tmp_db_path: Path,
) -> None:
    """get_snippet_by_location returns None when the line is outside the range."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(1, _snippet(1, file_path="x.py", start_line=10, end_line=20))
    assert store.get_snippet_by_location("x.py", 5) is None
    assert store.get_snippet_by_location("x.py", 25) is None


def test_get_snippet_by_location_nested_returns_first_match(
    tmp_db_path: Path,
) -> None:
    """For nested functions, get_snippet_by_location returns the first matching row.

    Both the outer (1-10) and inner (3-5) ranges contain line 4; the query
    returns the first row by insertion order. This documents the
    outermost-match behavior relied on by find_clones_by_location.
    """
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(
        1,
        CodeSnippet(
            code="outer",
            file_path="f.py",
            start_line=1,
            end_line=10,
            language="python",
            function_name="outer",
        ),
    )
    store.add_snippet(
        2,
        CodeSnippet(
            code="inner",
            file_path="f.py",
            start_line=3,
            end_line=5,
            language="python",
            function_name="inner",
        ),
    )
    found = store.get_snippet_by_location("f.py", 4)
    assert found is not None
    assert found["function_name"] == "outer"
    # Line 7 is only in the outer function.
    outer_only = store.get_snippet_by_location("f.py", 7)
    assert outer_only is not None
    assert outer_only["function_name"] == "outer"
    store.close()


def test_get_snippet_by_location_missing_file_returns_none(
    tmp_db_path: Path,
) -> None:
    """get_snippet_by_location returns None for an unknown file."""
    store = MetadataStore(str(tmp_db_path))
    assert store.get_snippet_by_location("missing.py", 1) is None


def test_count_empty_returns_zero(tmp_db_path: Path) -> None:
    """count() on an empty store returns 0."""
    store = MetadataStore(str(tmp_db_path))
    assert store.count() == 0
    store.close()


def test_get_languages(tmp_db_path: Path) -> None:
    """get_languages returns the distinct languages in the store."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(1, _snippet(1))
    store.add_snippet(
        2,
        CodeSnippet(
            code="x",
            file_path="x.js",
            start_line=1,
            end_line=1,
            language="javascript",
        ),
    )
    langs = set(store.get_languages())
    assert langs == {"python", "javascript"}
    store.close()


def test_get_languages_empty(tmp_db_path: Path) -> None:
    """get_languages on an empty store returns an empty list."""
    store = MetadataStore(str(tmp_db_path))
    assert store.get_languages() == []
    store.close()


def test_context_manager_closes(tmp_db_path: Path) -> None:
    """The context manager closes the connection on exit."""
    with MetadataStore(str(tmp_db_path)) as store:
        store.add_snippet(1, _snippet(1))
        assert store.count() == 1
    assert store.conn is None


def test_concurrent_writes_are_thread_safe(tmp_db_path: Path) -> None:
    """Concurrent add_snippet calls must not corrupt the database.

    SQLite with check_same_thread=False plus the internal write lock must
    serialize writers. After the threads join, every ID must be present.
    """
    store = MetadataStore(str(tmp_db_path))
    n_threads = 8
    per_thread = 25

    def writer(thread_id: int) -> None:
        for j in range(per_thread):
            sid = thread_id * per_thread + j
            store.add_snippet(sid, _snippet(sid, function_name=f"t{thread_id}_{j}"))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.count() == n_threads * per_thread
    for sid in [0, per_thread, per_thread * 4 + 3, n_threads * per_thread - 1]:
        assert store.get_snippet(sid) is not None
    store.close()


def test_persistence_across_reopen(tmp_db_path: Path) -> None:
    """Snippets persist across close/reopen of the store."""
    store = MetadataStore(str(tmp_db_path))
    store.add_snippet(1, _snippet(1, function_name="keep"))
    store.close()

    reopened = MetadataStore(str(tmp_db_path))
    got = reopened.get_snippet(1)
    assert got is not None
    assert got["function_name"] == "keep"
    reopened.close()
