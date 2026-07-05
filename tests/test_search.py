"""Tests for clone_detection.query.search.CloneSearcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from clone_detection.indexing.faiss_index import FAISSIndexBuilder, IndexType
from clone_detection.parsers.tree_sitter_parser import CodeSnippet
from clone_detection.query.metadata import MetadataStore
from clone_detection.query.search import CloneSearcher

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import FakeEmbedder

DIM = 32


def _seed_index(
    embedder: FakeEmbedder,
    store: MetadataStore,
    codes: list[str],
) -> FAISSIndexBuilder:
    """Build a FLAT index + metadata store from a list of code strings."""
    vecs = embedder.embed_batch(codes).astype(np.float32)
    builder = FAISSIndexBuilder(dimension=embedder.dimension, index_type=IndexType.FLAT)
    ids = np.arange(len(codes), dtype=np.int64)
    builder.build(vecs, ids)
    for i, code in enumerate(codes):
        store.add_snippet(
            i,
            CodeSnippet(
                code=code,
                file_path=f"f{i}.py",
                start_line=i + 1,
                end_line=i + 2,
                language="python",
                function_name=f"fn{i}",
            ),
        )
    return builder


def _make_searcher(
    fake_embedder: FakeEmbedder, tmp_db_path: Path, codes: list[str]
) -> tuple[CloneSearcher, MetadataStore]:
    """Build a searcher over a seeded index and metadata store."""
    store = MetadataStore(str(tmp_db_path))
    builder = _seed_index(fake_embedder, store, codes)
    searcher = CloneSearcher(builder.index, fake_embedder, store)  # type: ignore[arg-type]
    return searcher, store


def test_find_clones_finds_exact_match_excluding_self(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """find_clones finds a duplicate but the self is excluded by ID."""
    codes = [
        "def add(a, b): return a + b",
        "def add(a, b): return a + b",
        "def totally_different(): return None",
    ]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)

    clones = searcher.find_clones(codes[0], similarity_threshold=0.99, max_results=10)
    ids = {c.snippet_id for c in clones}
    assert 1 in ids
    store.close()


def test_find_clones_returns_sorted_descending(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """find_clones returns matches sorted by descending similarity."""
    codes = ["a", "a", "b"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)

    clones = searcher.find_clones(codes[0], similarity_threshold=0.0, max_results=10)
    sims = [c.similarity for c in clones]
    assert sims == sorted(sims, reverse=True)
    store.close()


def test_find_clones_max_results_zero_returns_empty(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """max_results=0 short-circuits to an empty result list."""
    codes = ["a", "a"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    assert searcher.find_clones(codes[0], max_results=0) == []
    store.close()


def test_find_clones_max_results_limits_count(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """max_results caps the number of returned matches."""
    codes = ["x"] * 5
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(codes[0], similarity_threshold=0.0, max_results=2)
    assert len(clones) == 2
    store.close()


def test_find_clones_max_results_none_returns_all(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """max_results=None returns every matching snippet (including self)."""
    codes = ["x"] * 4
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(codes[0], similarity_threshold=0.0, max_results=None)
    assert len(clones) == 4
    store.close()


def test_find_clones_exclude_self_false_includes_self(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """exclude_self=False without an ID keeps the self match."""
    codes = ["dup", "dup"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(
        codes[0], similarity_threshold=0.99, max_results=10, exclude_self=False
    )
    ids = {c.snippet_id for c in clones}
    assert 0 in ids
    assert 1 in ids
    store.close()


def test_find_clones_high_threshold_returns_empty(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """A high threshold with an excluded self returns no clones."""
    codes = ["different_one", "totally_different_two"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(
        codes[0], similarity_threshold=0.99, max_results=10, exclude_snippet_id=0
    )
    assert clones == []
    store.close()


def test_find_clones_exclude_snippet_id_drops_only_that_id(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """exclude_snippet_id drops only that ID, keeping genuine clones."""
    codes = ["dup", "dup", "dup"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(
        codes[0], similarity_threshold=0.99, max_results=10, exclude_snippet_id=1
    )
    ids = {c.snippet_id for c in clones}
    assert 1 not in ids
    assert 2 in ids
    store.close()


def test_find_clones_batch_consistent_with_single(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """find_clones_batch returns the same IDs as one-by-one find_clones."""
    codes = ["a", "a", "b", "c"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)

    batch = searcher.find_clones_batch(codes, similarity_threshold=0.0)
    single = [searcher.find_clones(c, similarity_threshold=0.0, max_results=None) for c in codes]
    for i in range(len(codes)):
        batch_ids = {c.snippet_id for c in batch[i]}
        single_ids = {c.snippet_id for c in single[i]}
        assert batch_ids == single_ids
    store.close()


def test_find_clones_batch_empty_returns_empty(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """find_clones_batch on an empty query list returns an empty list."""
    store = MetadataStore(str(tmp_db_path))
    builder = FAISSIndexBuilder(dimension=fake_embedder.dimension, index_type=IndexType.FLAT)
    builder.build()
    searcher = CloneSearcher(builder.index, fake_embedder, store)  # type: ignore[arg-type]
    assert searcher.find_clones_batch([], similarity_threshold=0.5) == []
    store.close()


def test_find_clones_batch_sorts_each_result(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """Each find_clones_batch result list is sorted by descending similarity."""
    codes = ["x", "x", "x"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    batch = searcher.find_clones_batch(codes, similarity_threshold=0.0)
    for result in batch:
        sims = [c.similarity for c in result]
        assert sims == sorted(sims, reverse=True)
    store.close()


def test_find_clones_by_location(fake_embedder: FakeEmbedder, tmp_db_path: Path) -> None:
    """find_clones_by_location finds the duplicate and excludes self by ID."""
    codes = ["def add(a, b): return a + b", "def add(a, b): return a + b"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)

    clones = searcher.find_clones_by_location("f0.py", 1, similarity_threshold=0.99, max_results=10)
    ids = {c.snippet_id for c in clones}
    assert 1 in ids
    assert 0 not in ids
    store.close()


def test_find_clones_by_location_missing_returns_empty(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """find_clones_by_location returns [] for an unknown location."""
    codes = ["x"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    assert searcher.find_clones_by_location("missing.py", 1) == []
    store.close()


def test_get_statistics(fake_embedder: FakeEmbedder, tmp_db_path: Path) -> None:
    """get_statistics reports index size, metadata count, and dimension."""
    codes = ["a", "b", "c"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    stats = searcher.get_statistics()
    assert stats["index_size"] == 3
    assert stats["metadata_count"] == 3
    assert stats["embedding_dimension"] == fake_embedder.dimension
    assert "python" in stats["languages"]
    store.close()


def test_clone_match_to_dict_and_repr(fake_embedder: FakeEmbedder, tmp_db_path: Path) -> None:
    """CloneMatch.to_dict and __repr__ expose the match's fields."""
    codes = ["dup", "dup"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    clones = searcher.find_clones(codes[0], similarity_threshold=0.99, max_results=1)
    assert clones
    c = clones[0]
    d = c.to_dict()
    assert d["snippet_id"] == c.snippet_id
    assert "file_path" in d
    r = repr(c)
    assert "CloneMatch" in r
    store.close()


def test_hydrate_results_skips_missing_metadata(
    fake_embedder: FakeEmbedder, tmp_db_path: Path
) -> None:
    """_hydrate_results drops IDs without metadata rather than crashing."""
    codes = ["a"]

    codes = ["a"]
    searcher, store = _make_searcher(fake_embedder, tmp_db_path, codes)
    # ID 999 has no metadata row.
    clones = searcher._hydrate_results(np.array([0, 999], dtype=np.int64), [0.5, 0.4])
    ids = {c.snippet_id for c in clones}
    assert ids == {0}
    store.close()
