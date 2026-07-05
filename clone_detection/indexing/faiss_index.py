"""FAISS index builder for high-performance vector similarity search.

This module implements Part III of the blueprint: High-Scale Vector Indexing.
It provides IndexIVFPQ with proper L2 normalization to ensure mathematically
correct cosine similarity search using L2 distance metrics.

Critical Implementation Notes:
1. All vectors MUST be L2-normalized before adding to the index.
2. This ensures L2 distance is equivalent to cosine similarity.
3. Mathematical proof: D_L2^2 = 2 - 2*cos_sim (for normalized vectors).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """FAISS index types as defined in Table 3.1 of the blueprint."""

    FLAT = "Flat"
    IVF_FLAT = "IVF,Flat"
    IVF_PQ = "IVF,PQ"


class FAISSIndexBuilder:
    """Builder for FAISS indexes with proper configuration for code clone detection.

    This class implements the index architecture from Section 3.3, including:
    - IndexIVFPQ for production scale.
    - L2 normalization for cosine similarity equivalence.
    - Training and population phases.
    - Query-time parameter tuning (nprobe).

    Example:
        >>> builder = FAISSIndexBuilder(dimension=768, nlist=4096)
        >>> builder.train(training_vectors)
        >>> builder.add(all_vectors, all_ids)
        >>> builder.save("clones.index")

    """

    def __init__(
        self,
        dimension: int = 768,
        index_type: IndexType = IndexType.IVF_PQ,
        nlist: int = 4096,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 16,
        *,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the index builder.

        Args:
            dimension: Vector dimension (768 for GraphCodeBERT).
            index_type: Type of FAISS index to build.
            nlist: Number of IVF clusters (Table 3.2: 4*sqrt(N) to 16*sqrt(N)).
            m: Number of PQ sub-vectors (must divide dimension evenly).
            nbits: Bits per PQ code (typically 8 for 256 centroids).
            nprobe: Number of clusters to probe at query time.
            use_gpu: Whether to use GPU acceleration.

        Raises:
            ValueError: If ``m`` does not evenly divide ``dimension``.

        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.use_gpu = use_gpu

        self.index: faiss.Index | None = None
        self.is_trained = False

        # ``m`` (the number of PQ sub-vectors) is only meaningful for IVF_PQ;
        # only validate the divisor constraint for that index type.
        if index_type == IndexType.IVF_PQ and dimension % m != 0:
            msg = (
                f"m ({m}) must evenly divide dimension ({dimension}). "
                f"Valid values for d=768: 32, 64, 96, 128, 192, 256, 384, 768"
            )
            raise ValueError(
                msg,
            )

        logger.info("Initialized FAISS index builder: %s", self._get_index_description())

    def _get_index_description(self) -> str:
        """Get a human-readable description of the index configuration."""
        if self.index_type == IndexType.IVF_PQ:
            return (
                f"IndexIVFPQ(d={self.dimension}, nlist={self.nlist}, "
                f"m={self.m}, nbits={self.nbits}, nprobe={self.nprobe})"
            )
        if self.index_type == IndexType.IVF_FLAT:
            return f"IndexIVFFlat(d={self.dimension}, nlist={self.nlist}, nprobe={self.nprobe})"
        return f"IndexFlatL2(d={self.dimension})"

    def build(
        self,
        vectors: NDArray[Any] | None = None,
        ids: NDArray[Any] | None = None,
        train_vectors: NDArray[Any] | None = None,
    ) -> faiss.Index:
        """Build the FAISS index (train + add data).

        This is a convenience method that combines training and population.
        For large-scale applications, call train() and add() separately.

        Args:
            vectors: All vectors to index (N, 768).
            ids: Corresponding IDs for each vector (N,).
            train_vectors: Optional separate training set. If None, uses vectors.

        Returns:
            Trained and populated FAISS index.

        Example:
            >>> builder = FAISSIndexBuilder()
            >>> index = builder.build(embeddings, snippet_ids)

        """
        self._create_index()

        train_data = train_vectors if train_vectors is not None else vectors
        if train_data is not None:
            self.train(train_data)

        if vectors is not None:
            self.add(vectors, ids)

        return self.index  # type: ignore[return-value]

    def _create_index(self) -> None:
        """Create the FAISS index structure.

        Implements Step 1 from Section 3.3: Instantiate the Index.
        """
        if self.index_type == IndexType.FLAT:
            # Brute-force exact search.
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True  # Flat index doesn't require training.
        elif self.index_type == IndexType.IVF_FLAT:
            # IVF with full-precision vectors.
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif self.index_type == IndexType.IVF_PQ:
            # IVF with Product Quantization (production).
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.m,
                self.nbits,
            )
        else:
            msg = f"Unknown index type: {self.index_type}"
            raise ValueError(msg)

        # Wrap with IDMap to support custom IDs. This MUST be done before
        # adding any data.
        self.index = faiss.IndexIDMap(self.index)

        # Apply GPU acceleration if requested.
        if self.use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                logger.warning(
                    "GPU requested but faiss-gpu not available. "
                    "Install with: conda install -c conda-forge faiss-gpu",
                )
                # Reconcile use_gpu with the actual index location so save()
                # does not later call index_gpu_to_cpu on a CPU index.
                self.use_gpu = False
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Moved index to GPU")

        logger.info("Created index: %s", self._get_index_description())

    def train(self, train_vectors: NDArray[Any]) -> None:
        """Train the index on a representative sample.

        Implements Step 2 from Section 3.3: Train the Index.

        The training performs k-means clustering for:
        1. IVF: nlist cluster centroids.
        2. PQ: m * 2^nbits sub-vector centroids.

        Args:
            train_vectors: Training vectors (typically 1M-2M samples, shape: (N, 768)).

        Note:
            - Training is expensive but done only once (offline).
            - Use a representative sample, not necessarily all data.
            - For billion-scale indexes, 1-2M samples are sufficient.

        """
        if self.is_trained:
            logger.warning("Index is already trained, skipping")
            return

        if self.index is None:
            self._create_index()

        # CRITICAL STEP: L2-normalize training vectors.
        # This ensures L2 distance = cosine similarity (Section 4.1).
        logger.info("L2-normalizing training vectors...")
        train_vectors = train_vectors.astype(np.float32)
        faiss.normalize_L2(train_vectors)

        logger.info("Training index on %d vectors...", len(train_vectors))
        self.index.train(train_vectors)  # type: ignore[union-attr]
        self.is_trained = True
        logger.info("Index training complete")

    def add(
        self,
        vectors: NDArray[Any],
        ids: NDArray[Any] | None = None,
    ) -> None:
        """Add vectors to the trained index.

        Implements Step 3 from Section 3.3: Populate the Index.

        Args:
            vectors: Vectors to add (N, 768).
            ids: Custom IDs for each vector. If None, uses sequential IDs.

        Raises:
            RuntimeError: If the index is not trained or not created.

        Note:
            - Vectors are automatically L2-normalized (critical!).
            - Can be called multiple times to add in batches.
            - All vectors must be added AFTER training.

        """
        if not self.is_trained:
            msg = "Index must be trained before adding vectors. Call train() first."
            raise RuntimeError(
                msg,
            )
        if self.index is None:
            msg = "Index not created. Call build() or _create_index() first."
            raise RuntimeError(
                msg,
            )

        # Generate sequential IDs if not provided.
        if ids is None:
            current_size = self.index.ntotal
            ids = np.arange(current_size, current_size + len(vectors), dtype=np.int64)
        else:
            ids = np.asarray(ids, dtype=np.int64)

        # CRITICAL STEP: L2-normalize vectors in-place.
        # This is the linchpin that enables cosine similarity via L2 distance
        # (Section 4.1).
        vectors = vectors.astype(np.float32).copy()  # Copy to avoid modifying original.
        faiss.normalize_L2(vectors)

        logger.info("Adding %d vectors to index...", len(vectors))
        self.index.add_with_ids(vectors, ids)
        logger.info("Index now contains %d vectors", self.index.ntotal)

    def set_nprobe(self, nprobe: int) -> None:
        """Set the number of clusters to probe at query time.

        This is the most important runtime parameter for tuning speed/accuracy.

        Args:
            nprobe: Number of clusters to search (1 = fastest, nlist = exact).

        Guidelines (Table 3.2):
            - nprobe=1: Very fast, lower accuracy.
            - nprobe=16: Good balance (default).
            - nprobe=32: Higher accuracy, slower.
            - nprobe=nlist: Exact search (defeats IVF purpose).

        """
        if self.index_type == IndexType.FLAT:
            logger.warning("nprobe has no effect on Flat index (already exact search)")
            return

        # Recursively unwrap IDMap / GPU wrappers to reach the underlying IVF
        # index, which is the only level that actually owns the nprobe setting.
        index = self._unwrap_ivf_index(self.index)

        if hasattr(index, "nprobe"):
            index.nprobe = nprobe  # type: ignore[attr-defined]
            self.nprobe = nprobe
            logger.info("Set nprobe = %d", nprobe)
        else:
            logger.warning("Index type %s does not support nprobe", type(index))

    @staticmethod
    def _unwrap_ivf_index(index: faiss.Index) -> faiss.Index:
        """Recursively unwrap IDMap and GPU wrappers to reach the IVF index.

        FAISS wraps IVF indexes in ``IndexIDMap`` (for custom IDs) and, on GPU,
        in ``GpuIndexIDMap`` / ``GpuIndexIVFPQ``-style wrappers. ``nprobe``
        lives on the innermost IVF index, so we peel layers until we find an
        index that either supports ``nprobe`` directly or exposes a nested
        index we can descend into.
        """
        seen: set[int] = set()
        current: faiss.Index | None = index
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "nprobe"):
                return current
            inner = getattr(current, "index", None)
            if inner is None:
                break
            try:
                current = faiss.downcast_index(inner)
            except Exception:
                logger.exception("Failed to downcast index, falling back to inner")
                current = inner
        return current  # type: ignore[return-value]

    def save(self, file_path: str) -> None:
        """Save the index to disk.

        The index is written to ``file_path`` and a sidecar JSON metadata file
        is written to ``<file_path>.meta.json`` recording the builder parameters
        (index_type, nlist, m, nbits, nprobe) so that :meth:`load` can restore a
        fully-populated instance whose :meth:`get_stats` works correctly.

        Args:
            file_path: Path to save the index file.

        Raises:
            RuntimeError: If no index has been built yet.

        Example:
            >>> builder.save("clones.index")

        """
        if self.index is None:
            msg = "No index to save. Build the index first."
            raise RuntimeError(msg)

        resolved_path = str(Path(file_path).resolve())
        logger.info("Saving index to %s", resolved_path)

        # If using GPU, move to CPU before saving.
        index_to_save: faiss.Index = self.index
        if self.use_gpu:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(index_to_save, resolved_path)

        # Persist builder parameters as a sidecar metadata file so load() can
        # restore them. Without this, get_stats() crashes on loaded instances
        # because index_type / nlist / m / nbits / nprobe are unset.
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type.value,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "use_gpu": self.use_gpu,
        }
        meta_path = self._meta_path(resolved_path)
        with Path(meta_path).open("w") as f:
            json.dump(meta, f)

        logger.info("Index saved (%d vectors)", self.index.ntotal)

    @staticmethod
    def _meta_path(file_path: str) -> str:
        """Return the sidecar metadata path for a given index file path."""
        return str(file_path) + ".meta.json"

    @classmethod
    def load(cls, file_path: str, *, use_gpu: bool = False) -> FAISSIndexBuilder:
        """Load a saved index from disk.

        Restores both the FAISS index and the builder parameters (index_type,
        nlist, m, nbits, nprobe) from the sidecar ``<file_path>.meta.json``
        file written by :meth:`save`. If the sidecar is missing (e.g. for
        legacy index files), parameters fall back to safe defaults so that
        :meth:`get_stats` still works without crashing.

        Args:
            file_path: Path to the index file.
            use_gpu: Whether to move index to GPU after loading.

        Returns:
            FAISSIndexBuilder instance with loaded index.

        Example:
            >>> builder = FAISSIndexBuilder.load("clones.index")

        """
        resolved_path = str(Path(file_path).resolve())
        logger.info("Loading index from %s", resolved_path)

        # Load the index.
        index = faiss.read_index(resolved_path)

        # Read sidecar metadata if present.
        meta_path = cls._meta_path(resolved_path)
        meta: dict[str, Any] = {}
        if Path(meta_path).exists():
            with Path(meta_path).open() as f:
                meta = json.load(f)

        # Resolve index_type (default FLAT for legacy files / IndexFlatL2).
        index_type_value = meta.get("index_type")
        if index_type_value is not None:
            try:
                index_type = IndexType(index_type_value)
            except ValueError:
                logger.warning(
                    "Unknown index_type %r in metadata; defaulting to FLAT",
                    index_type_value,
                )
                index_type = IndexType.FLAT
        else:
            index_type = IndexType.FLAT

        # Construct a fully-initialized instance via __new__ so all attributes
        # are set, bypassing validation on m/dimension for legacy files where
        # the stored values may not satisfy the divisor check.
        instance = cls.__new__(cls)
        instance.dimension = meta.get("dimension", index.d)
        instance.index_type = index_type
        instance.nlist = meta.get("nlist", 0)
        instance.m = meta.get("m", 1)
        instance.nbits = meta.get("nbits", 8)
        instance.nprobe = meta.get("nprobe", 0)
        instance.use_gpu = use_gpu
        instance.index = index
        instance.is_trained = True

        # Move to GPU if requested.
        if use_gpu:
            if not hasattr(faiss, "StandardGpuResources"):
                logger.warning("GPU requested but faiss-gpu not available")
                # Keep use_gpu consistent with the actual index location so a
                # subsequent save() does not try to move a CPU index to CPU.
                instance.use_gpu = False
            else:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                instance.index = index
                logger.info("Moved index to GPU")

        logger.info(
            "Loaded index with %d vectors (dimension: %d)",
            index.ntotal,
            index.d,
        )
        return instance

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "not_created"}

        return {
            "index_type": self.index_type.value,
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
            "is_trained": self.is_trained,
            "nlist": self.nlist if self.index_type != IndexType.FLAT else None,
            "nprobe": self.nprobe if self.index_type != IndexType.FLAT else None,
            "m": self.m if self.index_type == IndexType.IVF_PQ else None,
            "nbits": self.nbits if self.index_type == IndexType.IVF_PQ else None,
            "use_gpu": self.use_gpu,
        }


def cosine_to_l2_threshold(cosine_similarity: float) -> float:
    """Convert cosine similarity threshold to L2 distance threshold.

    Implements the conversion formula from Section 4.3:
        D_L2 = sqrt(2 - 2 * cos_sim)

    This is valid ONLY for L2-normalized vectors.

    Args:
        cosine_similarity: Desired minimum cosine similarity (0 to 1).

    Returns:
        Corresponding L2 distance threshold.

    Raises:
        ValueError: If ``cosine_similarity`` is outside [0, 1] (with tolerance
            for float drift).

    Example:
        >>> threshold = cosine_to_l2_threshold(0.95)  # Returns 0.316
        >>> # Use with: index.range_search(query, threshold)

    """
    # Tolerance for float drift on round-trip (e.g. cos=1.0 may reappear as
    # 0.9999999999). 1e-9 is below any meaningful clone-detection resolution.
    _epsilon = 1e-9
    if not -_epsilon <= cosine_similarity <= 1 + _epsilon:
        msg = f"cosine_similarity must be in [0, 1], got {cosine_similarity}"
        raise ValueError(
            msg,
        )

    # Clamp tiny float drift (e.g. -2e-16 from a round-trip) to the valid
    # [0, 1] range before sqrt to avoid NaNs from 2 - 2*cos going negative.
    clamped = min(1.0, max(0.0, cosine_similarity))
    l2_distance = np.sqrt(2 - 2 * clamped)
    return float(l2_distance)


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity.

    Inverse of cosine_to_l2_threshold:
        cos_sim = 1 - (D_L2^2 / 2)

    Args:
        l2_distance: L2 distance between normalized vectors.

    Returns:
        Corresponding cosine similarity.

    Example:
        >>> sim = l2_to_cosine_similarity(0.316)  # Returns ~0.95

    """
    cosine_sim = 1 - (l2_distance**2) / 2
    return float(cosine_sim)
