"""Semantic Code Clone Detection using GraphCodeBERT and FAISS.

This package provides a production-grade system for detecting semantic code clones
across multiple programming languages using deep learning and approximate nearest
neighbor search.

The public classes (``GraphCodeBERTEmbedder``, ``FAISSIndexBuilder``, etc.) are
imported lazily via ``__getattr__`` so that importing a single submodule (e.g.
``clone_detection.indexing.faiss_index``) does not force the heavy optional
dependencies (``torch``, ``transformers``) to load. This keeps lightweight
operations — parsing, indexing, querying against a pre-built index — usable
in environments where the ML stack is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__author__ = "structurelint"

if TYPE_CHECKING:
    from clone_detection.embeddings.graphcodebert import GraphCodeBERTEmbedder
    from clone_detection.indexing.faiss_index import FAISSIndexBuilder
    from clone_detection.parsers.tree_sitter_parser import CodeSnippet, TreeSitterParser
    from clone_detection.query.search import CloneMatch, CloneSearcher

__all__ = [
    "CloneMatch",
    "CloneSearcher",
    "CodeSnippet",
    "FAISSIndexBuilder",
    "GraphCodeBERTEmbedder",
    "TreeSitterParser",
]


def __getattr__(name: str) -> Any:
    """Lazily import public symbols to avoid eager heavy-dependency loading."""
    # Each import below is localized to the branch that needs it, gated
    # behind ``name in __all__``, so torch/transformers stay off the import
    # path of lightweight submodules.
    if name in __all__:
        if name == "GraphCodeBERTEmbedder":
            from clone_detection.embeddings.graphcodebert import (
                GraphCodeBERTEmbedder,
            )

            return GraphCodeBERTEmbedder
        if name == "FAISSIndexBuilder":
            from clone_detection.indexing.faiss_index import FAISSIndexBuilder

            return FAISSIndexBuilder
        if name in {"CodeSnippet", "TreeSitterParser"}:
            from clone_detection.parsers.tree_sitter_parser import (
                CodeSnippet,
                TreeSitterParser,
            )

            if name == "CodeSnippet":
                return CodeSnippet
            return TreeSitterParser
        if name in {"CloneMatch", "CloneSearcher"}:
            from clone_detection.query.search import CloneMatch, CloneSearcher

            if name == "CloneMatch":
                return CloneMatch
            return CloneSearcher
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
