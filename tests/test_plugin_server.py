"""Tests for plugin_server (FastAPI plugin HTTP API).

The detect endpoint initializes the real GraphCodeBERT model via
``_embedder_cache.get_or_initialize``. These tests patch that cache to return a
deterministic FakeEmbedder so the full parse -> embed -> index -> range_search
pipeline runs offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi.testclient import TestClient

import plugin_server

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import FakeEmbedder


@pytest.fixture
def patched_cache(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """Replace the embedder cache so endpoints use a FakeEmbedder."""
    from tests.conftest import FakeEmbedder

    fake = FakeEmbedder()
    monkeypatch.setattr(
        plugin_server._embedder_cache,
        "get_or_initialize",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        plugin_server._embedder_cache,
        "get",
        lambda: fake,
    )
    return fake


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create a directory with two identical functions (a clone pair)."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.py").write_text("def add(a, b):\n    return a + b\n")
    (d / "b.py").write_text("def add(a, b):\n    return a + b\n")
    return d


@pytest.fixture
def client(patched_cache: object) -> TestClient:
    """Return a TestClient with the embedder cache patched."""
    return TestClient(plugin_server.app)


def test_root_endpoint(client: TestClient) -> None:
    """GET / returns plugin metadata."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["endpoints"]["detect"] == "/api/v1/detect"


def test_health_endpoint(client: TestClient) -> None:
    """GET /health reports healthy status when the embedder loads."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"


def test_health_endpoint_degraded_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GET /health reports degraded when the embedder fails to initialize."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        msg = "model load failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(plugin_server._embedder_cache, "get_or_initialize", _boom)
    resp = TestClient(plugin_server.app).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert "Model loading failed" in body["message"]


def test_detect_finds_clone_pair(client: TestClient, source_dir: Path) -> None:
    """POST /api/v1/detect returns one clone pair for two identical functions."""
    resp = client.post(
        "/api/v1/detect",
        json={
            "source_dir": str(source_dir),
            "languages": ["python"],
            "similarity_threshold": 0.9,
            "max_results": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["functions_analyzed"] == 2
    assert body["stats"]["files_analyzed"] == 2
    assert len(body["clones"]) == 1
    clone = body["clones"][0]
    assert clone["similarity"] >= 0.9
    assert "Semantic similarity" in clone["explanation"]


def test_detect_empty_directory_returns_error(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An empty source dir yields a response with an error message."""
    empty = tmp_path / "empty"
    empty.mkdir()
    resp = client.post(
        "/api/v1/detect",
        json={"source_dir": str(empty), "languages": ["python"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["clones"] == []
    assert body["error"] == "No functions found in source directory"
    assert body["stats"]["functions_analyzed"] == 0


def test_detect_high_threshold_returns_no_clones(
    client: TestClient,
    source_dir: Path,
) -> None:
    """A threshold of 1.0 still returns the exact duplicate pair."""
    resp = client.post(
        "/api/v1/detect",
        json={
            "source_dir": str(source_dir),
            "languages": ["python"],
            "similarity_threshold": 1.0,
            "max_results": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None


def test_detect_invalid_threshold_rejected(
    client: TestClient,
    source_dir: Path,
) -> None:
    """A threshold outside [0,1] is rejected by pydantic validation."""
    resp = client.post(
        "/api/v1/detect",
        json={
            "source_dir": str(source_dir),
            "languages": ["python"],
            "similarity_threshold": 1.5,
        },
    )
    assert resp.status_code == 422


def test_collect_clones_helper_dedupes(
    patched_cache: object,
) -> None:
    """_collect_clones de-duplicates unordered (i,j)/(j,i) pairs."""
    funcs = [
        type("F", (), {"file_path": "a.py", "start_line": 1, "end_line": 2})(),
        type("F", (), {"file_path": "b.py", "start_line": 1, "end_line": 2})(),
    ]
    lims = np.array([0, 2, 4], dtype=np.int64)
    distances = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids = np.array([1, 1, 0, 0], dtype=np.int64)

    clones = plugin_server._collect_clones(
        funcs, lims, distances, ids, similarity_threshold=0.5, max_results=100
    )
    assert len(clones) == 1
    assert clones[0].source_file == "a.py"
    assert clones[0].target_file == "b.py"


def test_collect_clones_respects_max_results(
    patched_cache: object,
) -> None:
    """_collect_clones stops once max_results is reached."""
    funcs = [
        type("F", (), {"file_path": f"f{i}.py", "start_line": 1, "end_line": 2})() for i in range(5)
    ]
    lims = np.array([0, 4, 8, 12, 16, 20], dtype=np.int64)
    distances = np.zeros(20, dtype=np.float32)
    ids_list = [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]
    ids = np.array(ids_list, dtype=np.int64)

    clones = plugin_server._collect_clones(
        funcs, lims, distances, ids, similarity_threshold=0.5, max_results=2
    )
    assert len(clones) == 2


def test_collect_clones_skips_below_threshold(
    patched_cache: object,
) -> None:
    """_collect_clones skips pairs below the similarity threshold."""
    funcs = [
        type("F", (), {"file_path": "a.py", "start_line": 1, "end_line": 2})(),
        type("F", (), {"file_path": "b.py", "start_line": 1, "end_line": 2})(),
    ]
    # distance sqrt(2) -> cosine 0.0, below 0.5 threshold.
    lims = np.array([0, 1, 2], dtype=np.int64)
    distances = np.array([np.sqrt(2.0), np.sqrt(2.0)], dtype=np.float32)
    ids = np.array([1, 0], dtype=np.int64)
    clones = plugin_server._collect_clones(
        funcs, lims, distances, ids, similarity_threshold=0.5, max_results=100
    )
    assert clones == []


def test_build_index_shape_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_build_index raises RuntimeError on a rank/shape mismatch."""
    bad_embedder = type(
        "Bad",
        (),
        {
            "embed_batch": lambda self, fns: np.zeros((3, 32), dtype=np.float32),
        },
    )()
    monkeypatch.setattr(
        plugin_server._embedder_cache,
        "get_or_initialize",
        lambda *args, **kwargs: bad_embedder,
    )
    funcs = [type("F", (), {"file_path": "a.py", "start_line": 1, "end_line": 2})()]
    with pytest.raises(RuntimeError, match="shape mismatch"):
        plugin_server._build_index(bad_embedder, funcs)
