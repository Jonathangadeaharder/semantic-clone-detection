"""Tests for the cosine/L2 threshold math in clone_detection.indexing.faiss_index.

These conversions back the clone-detection threshold pipeline. The tests pin
the round-trip identity between cosine similarity and L2 distance and a set of
known anchor values across the meaningful similarity range.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from clone_detection.indexing.faiss_index import (
    cosine_to_l2_threshold,
    l2_to_cosine_similarity,
)

# Anchors spanning the meaningful similarity range for clone detection.
ANCHOR_SIMILARITIES = [0.0, 0.5, 0.85, 0.95, 1.0]


@pytest.mark.parametrize("cos", ANCHOR_SIMILARITIES)
def test_l2_to_cosine_uses_squared_distance(cos: float) -> None:
    """Verify the cosine↔L2 conversion uses the squared L2 distance.

    FAISS IndexFlatL2 returns SQUARED L2 distances, so the conversion is
    cos = 1 - (D^2 / 2). The test confirms ``l2_to_cosine_similarity`` recovers
    the input cosine value at each anchor.
    """
    l2 = cosine_to_l2_threshold(cos)
    # l2 here is the true L2 distance (sqrt(2 - 2*cos)), NOT squared.
    expected = 1.0 - (l2**2) / 2.0
    assert l2_to_cosine_similarity(l2) == pytest.approx(expected)
    assert l2_to_cosine_similarity(l2) == pytest.approx(cos)


@pytest.mark.parametrize("cos", ANCHOR_SIMILARITIES)
def test_round_trip_cosine_to_l2_and_back(cos: float) -> None:
    """l2_to_cosine_similarity must invert cosine_to_l2_threshold."""
    l2 = cosine_to_l2_threshold(cos)
    assert l2_to_cosine_similarity(l2) == pytest.approx(cos)


@pytest.mark.parametrize("cos", ANCHOR_SIMILARITIES)
def test_round_trip_l2_to_cosine_and_back(cos: float) -> None:
    """cosine_to_l2_threshold must invert l2_to_cosine_similarity for valid L2."""
    l2 = cosine_to_l2_threshold(cos)
    recovered_cos = l2_to_cosine_similarity(l2)
    recovered_l2 = cosine_to_l2_threshold(recovered_cos)
    assert recovered_l2 == pytest.approx(l2)


def test_known_anchor_values() -> None:
    """Pin exact known-good anchor values to catch formula drift."""
    # cos=1.0  -> D=0
    assert cosine_to_l2_threshold(1.0) == pytest.approx(0.0)
    assert l2_to_cosine_similarity(0.0) == pytest.approx(1.0)

    # cos=0.0  -> D=sqrt(2)
    assert cosine_to_l2_threshold(0.0) == pytest.approx(np.sqrt(2.0))
    assert l2_to_cosine_similarity(np.sqrt(2.0)) == pytest.approx(0.0)

    # cos=0.95 -> D=sqrt(0.1) ~= 0.3162277
    assert cosine_to_l2_threshold(0.95) == pytest.approx(np.sqrt(0.1))
    assert l2_to_cosine_similarity(np.sqrt(0.1)) == pytest.approx(0.95)

    # cos=0.5  -> D=sqrt(1.0) = 1.0
    assert cosine_to_l2_threshold(0.5) == pytest.approx(1.0)
    assert l2_to_cosine_similarity(1.0) == pytest.approx(0.5)


def test_cosine_to_l2_threshold_rejects_out_of_range() -> None:
    """Thresholds outside [0, 1] are invalid for cosine similarity."""
    with pytest.raises(ValueError, match="must be in"):
        cosine_to_l2_threshold(-0.01)
    with pytest.raises(ValueError, match="must be in"):
        cosine_to_l2_threshold(1.01)


def test_l2_to_cosine_similarity_range() -> None:
    """Similarity stays in [0, 1] and is monotonic for D in [0, sqrt(2)].

    For valid normalized-vector L2 distances in [0, sqrt(2)], cosine similarity
    stays within [-0.0, 1.0]. (At D=sqrt(2), cos=0; beyond is unreachable for
    normalized vectors but the function must still be monotonic.).
    """
    distances = np.linspace(0.0, np.sqrt(2.0), 50)
    sims = [l2_to_cosine_similarity(float(d)) for d in distances]
    assert sims[0] == pytest.approx(1.0)
    assert sims[-1] == pytest.approx(0.0)
    # Monotonically non-increasing as distance grows.
    for a, b in pairwise(sims):
        assert b <= a + 1e-12
