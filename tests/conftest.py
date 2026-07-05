"""Shared test fixtures for the semantic-clone-detection test suite."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pytest

# faiss-cpu and torch both bundle an OpenMP runtime. When both shared
# libraries are loaded into the same process, OpenMP aborts with
# "Error #15: Initializing libomp.dylib, but found libomp.dylib already
# initialized." KMP_DUPLICATE_LIB_OK=TRUE lets both runtimes coexist in one
# process. It MUST be set before importing either faiss or torch, hence it
# lives at the top of conftest (which pytest imports first).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

from clone_detection.parsers.tree_sitter_parser import CodeSnippet


class FakeEmbedder:
    """Deterministic, dependency-free stand-in for ``GraphCodeBERTEmbedder``.

    Produces a unit-norm embedding for each code snippet by hashing the source
    text. Two identical snippets yield identical vectors (similarity == 1.0),
    and two unrelated snippets are nearly orthogonal. This is exactly the
    contract ``CloneSearcher`` relies on, so the full query pipeline can be
    exercised without downloading the real model.

    The dimension is intentionally small (32) to keep tests fast.
    """

    DIMENSION = 32

    def __init__(self, dimension: int = DIMENSION) -> None:
        """Store the embedding dimension used by ``_vector_for``."""
        self.dimension = dimension

    def _vector_for(self, code: str) -> NDArray[np.float32]:
        """Hash ``code`` into a deterministic unit-norm float32 vector."""
        h = hashlib.sha256(code.encode("utf-8")).digest()
        # Repeat the digest to fill ``dimension`` uint8 values, then cast to
        # float32. The array length equals ``dimension`` exactly.
        material = (h * (self.dimension // len(h) + 1))[: self.dimension]
        vec = np.frombuffer(material, dtype=np.uint8).astype(np.float32)
        vec -= vec.mean()
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            vec[0] = 1.0
            norm = 1.0
        return vec / norm

    def embed_batch(
        self, code_snippets: Sequence[str] | Sequence[CodeSnippet]
    ) -> NDArray[np.float32]:
        """Embed a batch of code strings or CodeSnippet objects."""
        if code_snippets and isinstance(code_snippets[0], CodeSnippet):
            codes = [s.code for s in code_snippets]
        else:
            codes = list(code_snippets)
        if not codes:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector_for(c) for c in codes])

    def embed_single(self, code: str) -> NDArray[np.float32]:
        """Embed a single code string into a (dimension,) vector."""
        return self._vector_for(code)

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension used by this fake embedder."""
        return self.dimension


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Return a deterministic FakeEmbedder for use as a GraphCodeBERT stand-in."""
    return FakeEmbedder()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a path for a temporary SQLite metadata database."""
    return tmp_path / "metadata.db"


@pytest.fixture
def tmp_index_path(tmp_path: Path) -> Path:
    """Return a path for a temporary FAISS index file."""
    return tmp_path / "test.index"


# Callable type alias for the python_snippet_factory fixture.
SnippetFactory = Callable[[str, str], "Path"]


@pytest.fixture
def python_snippet_factory(tmp_path: Path) -> SnippetFactory:
    """Provide a callable that writes Python source to a temp file and returns its path."""

    def _make(name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content)
        return p

    return _make


@pytest.fixture
def sample_python_file(python_snippet_factory: SnippetFactory) -> Path:
    """Provide a small Python file with multiple functions and a method."""
    return python_snippet_factory(
        "sample.py",
        """\
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


class Calculator:
    def mul(self, a, b):
        return a * b
""",
    )


def make_snippet(
    code: str,
    file_path: str = "sample.py",
    start_line: int = 1,
    end_line: int = 1,
    language: str = "python",
    function_name: str | None = None,
) -> CodeSnippet:
    """Build a CodeSnippet without a parser."""
    return CodeSnippet(
        code=code,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=language,
        function_name=function_name,
    )


# Callable type alias for the make_snippets fixture.
MakeSnippets = Callable[..., CodeSnippet]


@pytest.fixture
def make_snippets() -> MakeSnippets:
    """Expose the ``make_snippet`` helper as a fixture."""
    return make_snippet


# --- CLI fixtures ---


@pytest.fixture
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """Replace the CLI's embedder factory with one returning a FakeEmbedder."""
    from clone_detection.cli import main as cli_main

    fake = FakeEmbedder()

    def _factory(*_args: object, **_kwargs: object) -> FakeEmbedder:
        return fake

    monkeypatch.setattr(cli_main, "_get_embedder_class", lambda: _factory)
    return fake


@pytest.fixture
def flat_index_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CLI's ingest to build a FLAT index (no PQ training needed)."""
    from clone_detection.indexing.faiss_index import FAISSIndexBuilder, IndexType

    real_init = FAISSIndexBuilder.__init__

    def _patched_init(self: FAISSIndexBuilder, *args: object, **kwargs: object) -> None:
        kwargs["index_type"] = IndexType.FLAT
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FAISSIndexBuilder, "__init__", _patched_init)


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """Create a small source tree with two duplicate functions."""
    d = tmp_path / "src"
    d.mkdir()
    (d / "a.py").write_text("def add(a, b):\n    return a + b\n")
    (d / "b.py").write_text("def add(a, b):\n    return a + b\n")
    return d
