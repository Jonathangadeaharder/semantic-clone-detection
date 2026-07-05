"""Tests for clone_detection.indexing.faiss_index."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import faiss
import numpy as np
import pytest

from clone_detection.indexing.faiss_index import (
    FAISSIndexBuilder,
    IndexType,
    cosine_to_l2_threshold,
    l2_to_cosine_similarity,
)

if TYPE_CHECKING:
    from pathlib import Path

DIM = 32


def _make_unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Generate ``n`` deterministic unit-norm row vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def test_invalid_m_does_not_divide_dimension() -> None:
    """Constructing an IVF_PQ index with a bad ``m`` raises ValueError."""
    with pytest.raises(ValueError, match="must evenly divide"):
        FAISSIndexBuilder(dimension=768, m=50)


def test_build_flat_index_returns_index() -> None:
    """build() on a FLAT index returns a populated, trained index."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(5)
    ids = np.arange(5, dtype=np.int64)
    index = builder.build(vecs, ids)
    assert index.ntotal == 5
    assert builder.is_trained is True


def test_build_ivf_flat_index() -> None:
    """build() on an IVF_FLAT index trains and populates the index."""
    builder = FAISSIndexBuilder(
        dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2, nprobe=1, m=8
    )
    vecs = _make_unit_vectors(20)
    ids = np.arange(20, dtype=np.int64)
    index = builder.build(vecs, ids)
    assert index.ntotal == 20


def test_build_ivf_pq_index() -> None:
    """build() on an IVF_PQ index trains and populates the index."""
    builder = FAISSIndexBuilder(
        dimension=DIM, index_type=IndexType.IVF_PQ, nlist=2, m=8, nbits=4, nprobe=1
    )
    vecs = _make_unit_vectors(40)
    ids = np.arange(40, dtype=np.int64)
    index = builder.build(vecs, ids)
    assert index.ntotal == 40


def test_add_requires_training_for_ivf() -> None:
    """add() on an untrained IVF index raises RuntimeError."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2, m=8)
    builder._create_index()
    vecs = _make_unit_vectors(3)
    with pytest.raises(RuntimeError, match="trained"):
        builder.add(vecs)


def test_add_without_index_raises() -> None:
    """add() before build/_create_index raises RuntimeError."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2, m=8)
    with pytest.raises(RuntimeError, match=r"not created|trained"):
        builder.add(_make_unit_vectors(3))


def test_sequential_ids_when_omitted() -> None:
    """Omitting ``ids`` assigns sequential IDs starting at 0."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(4)
    index = builder.build(vecs)
    distances, ids = index.search(vecs[:1], k=1)
    assert int(ids[0][0]) == 0
    assert distances[0][0] == pytest.approx(0.0, abs=1e-6)


def test_save_and_load_round_trip(tmp_index_path: Path) -> None:
    """save() then load() preserves the vectors and dimension."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(5)
    ids = np.arange(5, dtype=np.int64)
    builder.build(vecs, ids)
    builder.save(str(tmp_index_path))

    loaded = FAISSIndexBuilder.load(str(tmp_index_path))
    assert loaded.index is not None
    assert loaded.index.ntotal == 5
    assert loaded.dimension == DIM
    assert loaded.is_trained is True


def test_load_restores_sidecar_metadata(tmp_index_path: Path) -> None:
    """load() restores index_type/m/nbits/nlist/nprobe from the sidecar."""
    builder = FAISSIndexBuilder(
        dimension=DIM, index_type=IndexType.IVF_PQ, nlist=2, m=8, nbits=4, nprobe=1
    )
    vecs = _make_unit_vectors(40)
    ids = np.arange(40, dtype=np.int64)
    builder.build(vecs, ids)
    builder.save(str(tmp_index_path))

    loaded = FAISSIndexBuilder.load(str(tmp_index_path))
    stats = loaded.get_stats()
    assert stats["index_type"] == "IVF,PQ"
    assert stats["m"] == 8
    assert stats["nbits"] == 4
    assert stats["nlist"] == 2
    assert stats["nprobe"] == 1
    assert stats["num_vectors"] == 40
    assert stats["is_trained"] is True


def test_get_stats_not_created() -> None:
    """get_stats() on a fresh builder reports not_created."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    stats = builder.get_stats()
    assert stats["status"] == "not_created"


def test_get_stats_flat_hides_ivf_fields() -> None:
    """get_stats() on a FLAT index nulls IVF/PQ-only fields."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs)
    stats = builder.get_stats()
    assert stats["index_type"] == "Flat"
    assert stats["nlist"] is None
    assert stats["nprobe"] is None
    assert stats["m"] is None
    assert stats["nbits"] is None
    assert stats["num_vectors"] == 3


def test_set_nprobe_on_ivf_index() -> None:
    """set_nprobe() updates nprobe on an IVF index."""
    builder = FAISSIndexBuilder(
        dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2, nprobe=1, m=8
    )
    vecs = _make_unit_vectors(20)
    builder.build(vecs)
    builder.set_nprobe(2)
    assert builder.nprobe == 2


def test_set_nprobe_on_flat_index_warns() -> None:
    """set_nprobe() on a FLAT index is a no-op and leaves nprobe unchanged."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs)
    builder.set_nprobe(4)
    assert builder.nprobe == 16


def test_save_without_index_raises(tmp_index_path: Path) -> None:
    """save() before build raises RuntimeError."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    with pytest.raises(RuntimeError, match="No index to save"):
        builder.save(str(tmp_index_path))


def test_load_legacy_without_sidecar_defaults_flat(tmp_index_path: Path) -> None:
    """load() on an index without a sidecar falls back to FLAT defaults."""
    vecs = _make_unit_vectors(3)
    raw = faiss.IndexFlatL2(DIM)
    raw_with_ids = faiss.IndexIDMap(raw)
    v = vecs.astype(np.float32).copy()
    faiss.normalize_L2(v)
    raw_with_ids.add_with_ids(v, np.arange(3, dtype=np.int64))
    faiss.write_index(raw_with_ids, str(tmp_index_path))

    loaded = FAISSIndexBuilder.load(str(tmp_index_path))
    stats = loaded.get_stats()
    assert stats["index_type"] == "Flat"
    assert stats["num_vectors"] == 3
    assert stats["dimension"] == DIM


def test_unwrap_ivf_index_descends_through_idmap() -> None:
    """_unwrap_ivf_index descends through IndexIDMap to the IVF index."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2, m=8)
    vecs = _make_unit_vectors(20)
    builder.build(vecs)
    inner = FAISSIndexBuilder._unwrap_ivf_index(builder.index)
    assert hasattr(inner, "nprobe")


def test_range_search_finds_self_at_zero_distance() -> None:
    """range_search finds the query vector itself at ~0 distance."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    ids = np.arange(3, dtype=np.int64)
    builder.build(vecs, ids)

    query = vecs[0:1].copy()
    _lims, distances, found_ids = builder.index.range_search(query, 1e-5)
    assert int(found_ids[0]) == 0
    assert float(distances[0]) == pytest.approx(0.0, abs=1e-5)


def test_cosine_to_l2_threshold_clamps_float_drift() -> None:
    """cosine_to_l2_threshold clamps tiny float drift around [0, 1]."""
    assert cosine_to_l2_threshold(1.0 + 1e-12) == pytest.approx(0.0, abs=1e-9)
    assert cosine_to_l2_threshold(-1e-12) == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_cosine_to_l2_threshold_rejects_far_out_of_range() -> None:
    """cosine_to_l2_threshold rejects values clearly outside [0, 1]."""
    with pytest.raises(ValueError, match="must be in"):
        cosine_to_l2_threshold(2.0)
    with pytest.raises(ValueError, match="must be in"):
        cosine_to_l2_threshold(-1.0)


def test_l2_to_cosine_similarity_known_anchors() -> None:
    """l2_to_cosine_similarity maps known L2 distances to cosine values."""
    assert l2_to_cosine_similarity(0.0) == pytest.approx(1.0)
    assert l2_to_cosine_similarity(np.sqrt(2.0)) == pytest.approx(0.0)


def test_get_index_description_variants() -> None:
    """_get_index_description describes each index type."""
    assert (
        "IndexFlatL2"
        in FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)._get_index_description()
    )
    assert (
        "IndexIVFFlat"
        in FAISSIndexBuilder(
            dimension=DIM, index_type=IndexType.IVF_FLAT, nlist=2
        )._get_index_description()
    )
    assert (
        "IndexIVFPQ"
        in FAISSIndexBuilder(
            dimension=DIM, index_type=IndexType.IVF_PQ, nlist=2, m=8
        )._get_index_description()
    )


def test_unknown_index_type_in_create_raises() -> None:
    """_create_index raises ValueError for an unknown index_type."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    builder.index_type = "nope"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown index type"):
        builder._create_index()


def test_add_generates_sequential_ids_from_current_size() -> None:
    """add() assigns IDs continuing from the index's current size."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs, np.array([10, 20, 30], dtype=np.int64))
    extra = _make_unit_vectors(2, seed=1)
    builder.add(extra)
    _distances, ids = builder.index.search(extra[:1], k=1)
    # Sequential IDs continue from ntotal==3 -> ids 3, 4.
    assert int(ids[0][0]) == 3


def test_train_skips_when_already_trained(caplog: pytest.LogCaptureFixture) -> None:
    """train() on an already-trained index is a no-op."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs)
    with caplog.at_level("WARNING"):
        builder.train(vecs)
    assert any("already trained" in r.message for r in caplog.records)


def test_load_unknown_index_type_value_defaults_flat(tmp_index_path: Path) -> None:
    """load() falls back to FLAT for an unknown index_type in the sidecar."""
    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs, np.arange(3, dtype=np.int64))
    builder.save(str(tmp_index_path))
    # Corrupt the sidecar index_type so IndexType(value) raises ValueError.
    meta_path = tmp_index_path.parent / (tmp_index_path.name + ".meta.json")
    with meta_path.open() as f:
        meta = json.load(f)
    meta["index_type"] = "nope"
    with meta_path.open("w") as f:
        json.dump(meta, f)

    loaded = FAISSIndexBuilder.load(str(tmp_index_path))
    assert loaded.index_type == IndexType.FLAT


def test_use_gpu_reconciled_when_gpu_unavailable(tmp_index_path: Path) -> None:
    """Requesting GPU without faiss-gpu reconciles use_gpu to False.

    Previously ``use_gpu`` stayed True while the index remained on CPU, so a
    subsequent save() would call ``index_gpu_to_cpu`` on a CPU index and crash.
    The fix reconciles ``use_gpu`` with the actual index location.
    """
    import faiss

    if hasattr(faiss, "StandardGpuResources"):
        pytest.skip("faiss-gpu is installed; cannot test the unavailable path")

    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT, use_gpu=True)
    vecs = _make_unit_vectors(3)
    builder.build(vecs, np.arange(3, dtype=np.int64))
    assert builder.use_gpu is False
    # save() must not crash on the CPU index despite the initial use_gpu=True.
    builder.save(str(tmp_index_path))
    assert tmp_index_path.exists()


def test_load_use_gpu_reconciled_when_gpu_unavailable(tmp_index_path: Path) -> None:
    """load(use_gpu=True) without faiss-gpu reconciles use_gpu to False."""
    import faiss

    if hasattr(faiss, "StandardGpuResources"):
        pytest.skip("faiss-gpu is installed; cannot test the unavailable path")

    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT)
    vecs = _make_unit_vectors(3)
    builder.build(vecs, np.arange(3, dtype=np.int64))
    builder.save(str(tmp_index_path))

    loaded = FAISSIndexBuilder.load(str(tmp_index_path), use_gpu=True)
    assert loaded.use_gpu is False
    assert loaded.index is not None


def test_gpu_unavailable_fallback_takes_cpu_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_index_path: Path,
) -> None:
    """Force the GPU-unavailable branch and assert the CPU fallback runs.

    Removing ``faiss.StandardGpuResources`` makes ``hasattr(faiss,
    "StandardGpuResources")`` return False regardless of whether faiss-gpu is
    installed, so this test exercises the fallback path in every environment.
    Both ``_create_index`` (via build) and ``load`` must reconcile ``use_gpu``
    to False and leave a usable CPU index.
    """
    if hasattr(faiss, "StandardGpuResources"):
        monkeypatch.delattr(faiss, "StandardGpuResources")

    builder = FAISSIndexBuilder(dimension=DIM, index_type=IndexType.FLAT, use_gpu=True)
    vecs = _make_unit_vectors(3)
    builder.build(vecs, np.arange(3, dtype=np.int64))
    assert builder.use_gpu is False
    assert builder.index is not None
    assert builder.index.ntotal == 3

    builder.save(str(tmp_index_path))
    loaded = FAISSIndexBuilder.load(str(tmp_index_path), use_gpu=True)
    assert loaded.use_gpu is False
    assert loaded.index is not None
    assert loaded.index.ntotal == 3
