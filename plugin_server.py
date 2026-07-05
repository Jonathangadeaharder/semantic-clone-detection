"""HTTP Plugin Server for Semantic Clone Detection.

This server provides an HTTP API for the semantic clone detection functionality,
allowing it to be used as an optional plugin for structurelint.

The plugin architecture keeps the core binary small while providing advanced
ML-based clone detection as an opt-in feature.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from clone_detection.indexing.faiss_index import (
    FAISSIndexBuilder,
    IndexType,
    cosine_to_l2_threshold,
    l2_to_cosine_similarity,
)
from clone_detection.parsers.tree_sitter_parser import TreeSitterParser

if TYPE_CHECKING:
    from clone_detection.embeddings.graphcodebert import GraphCodeBERTEmbedder

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants.
DEFAULT_MODEL_NAME = "microsoft/graphcodebert-base"

# Create FastAPI app.
app = FastAPI(
    title="Structurelint Semantic Clone Detection Plugin",
    version="0.1.0",
    description="HTTP API for semantic code clone detection using GraphCodeBERT",
)


class _EmbedderCache:
    """Thread-safe lazy-initialized singleton for the embedder.

    Using a small wrapper class (instead of a bare ``global`` statement)
    localizes the mutable state and makes the double-checked locking
    pattern explicit and testable.
    """

    def __init__(self) -> None:
        self._embedder: GraphCodeBERTEmbedder | None = None
        self._lock = Lock()

    def get(self) -> GraphCodeBERTEmbedder | None:
        """Return the cached embedder, or None if it has not been set."""
        return self._embedder

    def get_or_initialize(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> GraphCodeBERTEmbedder:
        """Return the cached embedder, initializing it lazily if needed.

        Uses double-checked locking so the heavy model load happens at most
        once across threads. The ``GraphCodeBERTEmbedder`` import is localized
        to this point so importing ``plugin_server`` does not pull in
        ``torch``/``transformers``; the optional ``plugin`` extra stays
        importable on its own.
        """
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is None:
                from clone_detection.embeddings.graphcodebert import (
                    GraphCodeBERTEmbedder,
                )

                logger.info("Initializing GraphCodeBERT embedder...")
                self._embedder = GraphCodeBERTEmbedder(
                    model_name=model_name,
                    device=device,
                )
            return self._embedder


_embedder_cache = _EmbedderCache()


# --- API Models ---


class SemanticCloneRequest(BaseModel):
    """Request model for semantic clone detection."""

    source_dir: str = Field(..., description="Root directory to analyze")
    languages: list[str] | None = Field(
        default=["python", "go", "javascript"],
        description="Languages to analyze",
    )
    exclude_patterns: list[str] | None = Field(
        default=["**/*_test.*", "**/node_modules/**", "**/vendor/**"],
        description="Glob patterns to exclude",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (0.0-1.0)",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        description="Maximum number of clone pairs to return",
    )


class SemanticClone(BaseModel):
    """Model for a detected semantic clone pair."""

    source_file: str
    source_start_line: int
    source_end_line: int
    target_file: str
    target_start_line: int
    target_end_line: int
    similarity: float
    explanation: str | None = None


class SemanticCloneStats(BaseModel):
    """Statistics about the clone detection analysis."""

    files_analyzed: int
    functions_analyzed: int
    duration_ms: int
    model_used: str = DEFAULT_MODEL_NAME


class SemanticCloneResponse(BaseModel):
    """Response model for semantic clone detection."""

    clones: list[SemanticClone]
    stats: SemanticCloneStats
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    capabilities: list[str] = [
        "semantic-clone-detection",
        "graphcodebert-embeddings",
        "faiss-indexing",
    ]
    message: str | None = None


# --- API Endpoints ---


@app.get("/health")
def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the plugin's health status and capabilities.
    """
    # Check if model can be loaded (thread-safe). Any failure means the
    # plugin is in a degraded state but the endpoint must still respond.
    try:
        _embedder_cache.get_or_initialize(device="cpu")
    except Exception as e:
        return HealthResponse(
            status="degraded",
            message=f"Model loading failed: {e}",
            capabilities=["limited"],
        )
    return HealthResponse(status="healthy", message="Semantic clone detection ready")


def _parse_functions(request: SemanticCloneRequest) -> list:
    """Parse the requested source directory into function snippets."""
    safe_path = str(Path(request.source_dir).resolve())
    logger.info("Parsing source directory: %s", safe_path)
    parser = TreeSitterParser(
        languages=list(request.languages) if request.languages else [],
    )
    functions = parser.parse_directory(
        str(request.source_dir),
        exclude_patterns=list(request.exclude_patterns) if request.exclude_patterns else None,
    )
    logger.info(
        "Found %d functions across %d files",
        len(functions),
        len({f.file_path for f in functions}),
    )
    return functions


# Expected rank of an embedding matrix: (n_functions, dimension).
_EXPECTED_EMBEDDING_RANK = 2


def _build_index(
    embedder: GraphCodeBERTEmbedder,
    functions: list,
) -> tuple[faiss.Index, np.ndarray]:
    """Generate embeddings and build a flat FAISS index over them.

    Args:
        embedder: The GraphCodeBERT embedder (already initialized).
        functions: Parsed function snippets.

    Returns:
        A tuple ``(index, query_vectors)`` where ``query_vectors`` are the
        L2-normalized embeddings used as range-search queries.

    Raises:
        RuntimeError: If the embedding shape does not match the input.

    """
    logger.info("Generating embeddings...")
    embeddings_array = embedder.embed_batch(functions)
    n_functions = len(functions)
    shape_ok = (
        embeddings_array.ndim == _EXPECTED_EMBEDDING_RANK
        and embeddings_array.shape[0] == n_functions
    )
    if not shape_ok:
        msg = f"Embedding shape mismatch: got {embeddings_array.shape} for {n_functions} functions"
        raise RuntimeError(msg)

    logger.info("Building FAISS index...")
    index_builder = FAISSIndexBuilder(
        dimension=int(embeddings_array.shape[1]),
        index_type=IndexType.FLAT,  # Use flat index for accuracy.
    )
    index = index_builder.build(embeddings_array)

    # L2-normalize query embeddings once so L2 distance reflects cosine
    # similarity (Section 4.1 of the blueprint). The index already stores
    # normalized vectors (FAISSIndexBuilder.add normalizes on insert).
    query_vectors = embeddings_array.astype(np.float32).copy()
    faiss.normalize_L2(query_vectors)
    return index, query_vectors


def _collect_clones(
    functions: list,
    lims: np.ndarray,
    distances: np.ndarray,
    ids: np.ndarray,
    similarity_threshold: float,
    max_results: int,
) -> list[SemanticClone]:
    """Collect unordered clone pairs from a batched range_search result.

    Args:
        functions: Parsed function snippets (indexed by row).
        lims: range_search lims array (len = n_queries + 1).
        distances: range_search distances array.
        ids: range_search ids array.
        similarity_threshold: Minimum cosine similarity to keep.
        max_results: Maximum number of clone pairs to return.

    Returns:
        List of :class:`SemanticClone` pairs (de-duplicated, unordered).

    """
    clones: list[SemanticClone] = []
    seen_pairs: set[tuple[int, int]] = set()
    num_functions = len(functions)

    for i in range(num_functions):
        start_idx = lims[i]
        end_idx = lims[i + 1]
        for pos in range(start_idx, end_idx):
            idx = int(ids[pos])
            # Skip self-matches.
            if idx == i:
                continue

            # Convert FAISS L2 distance to cosine similarity using the
            # shared helper (cos = 1 - D^2/2 for normalized vectors). This
            # matches the canonical CloneSearcher pipeline in
            # clone_detection/query/search.py.
            similarity = l2_to_cosine_similarity(float(distances[pos]))

            if similarity < similarity_threshold:
                continue

            # Skip if we've already seen this unordered pair.
            pair_key = tuple(sorted((i, idx)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            source_func = functions[i]
            target_func = functions[idx]

            clones.append(
                SemanticClone(
                    source_file=str(source_func.file_path),
                    source_start_line=source_func.start_line,
                    source_end_line=source_func.end_line,
                    target_file=str(target_func.file_path),
                    target_start_line=target_func.start_line,
                    target_end_line=target_func.end_line,
                    similarity=float(similarity),
                    explanation=f"Semantic similarity: {similarity:.3f}",
                ),
            )

            if len(clones) >= max_results:
                return clones

    return clones


@app.post("/api/v1/detect")
def detect_clones(request: SemanticCloneRequest) -> SemanticCloneResponse:
    """Detect semantic clones in the specified source directory.

    This endpoint analyzes code using GraphCodeBERT embeddings and FAISS
    similarity search to find semantically similar code fragments.
    """
    start_time = time.time()

    try:
        # Initialize components (thread-safe).
        embedder = _embedder_cache.get_or_initialize(device="cpu")

        functions = _parse_functions(request)

        if len(functions) == 0:
            return SemanticCloneResponse(
                clones=[],
                stats=SemanticCloneStats(
                    files_analyzed=0,
                    functions_analyzed=0,
                    duration_ms=int((time.time() - start_time) * 1000),
                ),
                error="No functions found in source directory",
            )

        index, query_vectors = _build_index(embedder, functions)

        # Search for similar code via a single batched range_search (was: O(N^2)
        # loop of one-query-at-a-time top-k searches). Range search returns all
        # pairs within the cosine similarity threshold in one call.
        logger.info("Searching for semantic clones...")
        l2_threshold = cosine_to_l2_threshold(request.similarity_threshold)
        lims, distances, ids = index.range_search(query_vectors, l2_threshold)

        clones = _collect_clones(
            functions,
            lims,
            distances,
            ids,
            request.similarity_threshold,
            request.max_results,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info("Found %d semantic clones in %dms", len(clones), duration_ms)

        return SemanticCloneResponse(
            clones=clones,
            stats=SemanticCloneStats(
                files_analyzed=len({f.file_path for f in functions}),
                functions_analyzed=len(functions),
                duration_ms=duration_ms,
            ),
        )

    except Exception as e:
        logger.exception("Clone detection failed")
        duration_ms = int((time.time() - start_time) * 1000)

        return SemanticCloneResponse(
            clones=[],
            stats=SemanticCloneStats(
                files_analyzed=0,
                functions_analyzed=0,
                duration_ms=duration_ms,
            ),
            error=str(e),
        )


@app.get("/")
def root() -> dict[str, object]:
    """Root endpoint with plugin information."""
    return {
        "name": "Structurelint Semantic Clone Detection Plugin",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {"health": "/health", "detect": "/api/v1/detect"},
    }


def _run_server(host: str, port: int, *, reload: bool) -> None:
    """Run the plugin server with uvicorn.

    ``uvicorn`` is imported lazily so importing this module does not require
    the optional ``plugin`` extra to be installed.
    """
    import uvicorn

    uvicorn.run(
        "plugin_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    """Run the plugin server."""
    parser = argparse.ArgumentParser(
        description="Structurelint Semantic Clone Detection Plugin Server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    logger.info("Starting plugin server on %s:%s", args.host, args.port)

    _run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
