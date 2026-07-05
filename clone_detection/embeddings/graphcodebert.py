"""GraphCodeBERT embedding generator for semantic code representation.

This module implements Part II of the blueprint: Vectorization.
It uses the GraphCodeBERT model to transform code snippets into 768-dimensional
semantic vectors, following the "Path A" implementation (no explicit DFG).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

from clone_detection.parsers.tree_sitter_parser import CodeSnippet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class GraphCodeBERTEmbedder:
    """Semantic code embedder using GraphCodeBERT.

    This class implements the embedding extraction strategy from Section 2.3:
    - Uses the <s> token's last hidden state as the code representation.
    - Supports batch inference for efficiency.
    - Can run on CPU or GPU.

    The model is pre-trained with Data Flow Graph awareness, providing superior
    semantic understanding compared to standard CodeBERT, even without explicit
    DFG input at inference time (Path A implementation).

    Example:
        >>> embedder = GraphCodeBERTEmbedder(device="cuda")
        >>> code = ["def add(a, b): return a + b"]
        >>> embeddings = embedder.embed_batch(code)
        >>> print(embeddings.shape)  # (1, 768)

    """

    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        """Initialize the GraphCodeBERT embedder.

        Args:
            model_name: HuggingFace model identifier. Can be:
                - "microsoft/graphcodebert-base" (default, pre-trained).
                - Path to a fine-tuned model checkpoint.
            device: Device to run on ("cuda", "cpu", or None for auto-detect).
            max_length: Maximum sequence length (GraphCodeBERT limit: 512).
            batch_size: Batch size for inference.

        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        # Auto-detect device if not specified.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing GraphCodeBERT on device: %s", self.device)

        # Load tokenizer and model.
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

        # Move model to device and set to evaluation mode.
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loaded model: %s", model_name)
        logger.info("Embedding dimension: 768")

    @torch.no_grad()
    def embed_batch(
        self,
        code_snippets: Sequence[str] | Sequence[CodeSnippet],
    ) -> NDArray[np.float32]:
        """Generate embeddings for a batch of code snippets.

        This implements the batch inference strategy from Section 2.3:
        1. Tokenize all snippets (truncate to max_length).
        2. Forward pass through GraphCodeBERT.
        3. Extract the <s> token's last hidden state.

        Args:
            code_snippets: List of code strings or CodeSnippet objects.

        Returns:
            NumPy array of shape (N, 768) containing the embeddings.

        Example:
            >>> embedder = GraphCodeBERTEmbedder()
            >>> code = ["def foo(): pass", "def bar(): return 1"]
            >>> embeddings = embedder.embed_batch(code)
            >>> print(embeddings.shape)  # (2, 768)

        """
        # Extract code strings if CodeSnippet objects were provided.
        if code_snippets and isinstance(code_snippets[0], CodeSnippet):
            code_strings = [snippet.code for snippet in code_snippets]
        else:
            code_strings = list(code_snippets)

        if not code_strings:
            return np.zeros((0, 768), dtype=np.float32)

        # Process in batches.
        all_embeddings: list[NDArray[np.float32]] = []

        for i in range(0, len(code_strings), self.batch_size):
            batch = code_strings[i : i + self.batch_size]
            batch_embeddings = self._embed_single_batch(batch)
            all_embeddings.append(batch_embeddings)

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                logger.debug(
                    "Processed %d/%d snippets",
                    i + len(batch),
                    len(code_strings),
                )

        # Concatenate all batch embeddings.
        embeddings = np.vstack(all_embeddings)

        logger.info("Generated %d embeddings", embeddings.shape[0])
        return embeddings

    def _embed_single_batch(self, code_batch: Sequence[str]) -> NDArray[np.float32]:
        """Embed a single batch of code snippets.

        This is the core inference logic from Section 2.3:
        - Tokenize with padding and truncation.
        - Forward pass through the model.
        - Extract <s> token representation.

        Args:
            code_batch: List of code strings (size <= batch_size).

        Returns:
            NumPy array of shape (batch_size, 768).

        """
        # Tokenize the batch.
        # - padding="max_length": Pad all sequences to max_length.
        # - truncation=True: Truncate sequences longer than max_length.
        # - return_tensors="pt": Return PyTorch tensors.
        inputs = self.tokenizer(
            list(code_batch),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass through GraphCodeBERT.
        outputs = self.model(**inputs)

        # Extract the <s> token's last hidden state.
        # outputs.last_hidden_state shape: (batch_size, seq_len, 768).
        # The <s> token is always at position 0.
        # Shape: (batch_size, 768).
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Move to CPU and convert to numpy.
        return embeddings.cpu().numpy()

    def embed_single(self, code: str) -> NDArray[np.float32]:
        """Generate embedding for a single code snippet.

        Args:
            code: Source code string.

        Returns:
            NumPy array of shape (768,).

        Example:
            >>> embedder = GraphCodeBERTEmbedder()
            >>> embedding = embedder.embed_single("def foo(): pass")
            >>> print(embedding.shape)  # (768,)

        """
        embeddings = self.embed_batch([code])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embeddings (always 768 for GraphCodeBERT)."""
        return 768
