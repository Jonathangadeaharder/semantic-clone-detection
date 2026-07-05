"""Part IV: The Clone-Finding Query Pipeline.

Threshold-based range search for finding semantic code clones.
"""

from clone_detection.query.metadata import MetadataStore
from clone_detection.query.search import CloneMatch, CloneSearcher

__all__ = ["CloneMatch", "CloneSearcher", "MetadataStore"]
