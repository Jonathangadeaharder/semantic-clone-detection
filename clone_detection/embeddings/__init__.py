"""Part II: Vectorization & Fine-Tuning.

GraphCodeBERT-based semantic code embedding for transforming code snippets
into 768-dimensional vectors.

``GraphCodeBERTEmbedder`` is exposed lazily via ``__getattr__`` so that
``import clone_detection.embeddings`` does not eagerly load ``torch`` or
``transformers``. Only attribute access (e.g.
``from clone_detection.embeddings import GraphCodeBERTEmbedder``) triggers the
heavy import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clone_detection.embeddings.graphcodebert import GraphCodeBERTEmbedder

__all__ = ["GraphCodeBERTEmbedder"]


def __getattr__(name: str) -> Any:
    """Lazily import public symbols to avoid eager heavy-dependency loading."""
    if name in __all__:
        from clone_detection.embeddings.graphcodebert import GraphCodeBERTEmbedder

        return GraphCodeBERTEmbedder
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
