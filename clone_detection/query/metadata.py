"""Metadata storage for code snippets.

This module provides a database layer for storing and retrieving metadata
associated with code snippets in the FAISS index.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clone_detection.parsers.tree_sitter_parser import CodeSnippet

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-based metadata storage for code snippets.

    Maps FAISS vector IDs to code snippet metadata (file path, line numbers, etc.).
    This is part of Blueprint A (Batch Ingestion Pipeline) and Blueprint B
    (Query Pipeline).

    Example:
        >>> store = MetadataStore("clones.db")
        >>> store.add_snippet(snippet_id=1, snippet=my_snippet)
        >>> metadata = store.get_snippet(snippet_id=1)

    """

    def __init__(self, db_path: str) -> None:
        """Initialize the metadata store.

        Args:
            db_path: Path to the SQLite database file.

        """
        self.db_path = str(Path(db_path).resolve())
        self.conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()
        self._connect()
        self._create_tables()

    def _connect(self) -> None:
        """Establish connection to the database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name.
        logger.info("Connected to metadata database: %s", self.db_path)

    def _create_tables(self) -> None:
        """Create the schema for storing code snippet metadata."""
        with self._write_lock, self.conn as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snippets (
                    id INTEGER PRIMARY KEY,
                    code TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    function_name TEXT
                )
                """,
            )

            # Create indexes for efficient queries.
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_path
                ON snippets(file_path)
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_language
                ON snippets(language)
                """,
            )

        logger.info("Metadata tables initialized")

    def add_snippet(self, snippet_id: int, snippet: CodeSnippet) -> None:
        """Add a single code snippet to the store.

        Args:
            snippet_id: Unique ID (must match FAISS index ID).
            snippet: CodeSnippet object with metadata.

        """
        with self._write_lock, self.conn as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snippets
                (id, code, file_path, start_line, end_line, language, function_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snippet_id,
                    snippet.code,
                    snippet.file_path,
                    snippet.start_line,
                    snippet.end_line,
                    snippet.language,
                    snippet.function_name,
                ),
            )

    def add_snippets_batch(
        self,
        snippets: Sequence[tuple[int, CodeSnippet]],
    ) -> None:
        """Add multiple snippets efficiently.

        Args:
            snippets: List of (snippet_id, CodeSnippet) tuples.

        """
        data = [
            (
                snippet_id,
                snippet.code,
                snippet.file_path,
                snippet.start_line,
                snippet.end_line,
                snippet.language,
                snippet.function_name,
            )
            for snippet_id, snippet in snippets
        ]

        with self._write_lock, self.conn as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO snippets
                (id, code, file_path, start_line, end_line, language, function_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )

        logger.info("Added %d snippets to metadata store", len(snippets))

    def get_snippet(self, snippet_id: int) -> dict[str, Any] | None:
        """Retrieve metadata for a single snippet.

        Args:
            snippet_id: The snippet ID.

        Returns:
            Dictionary with snippet metadata, or None if not found.

        """
        if self.conn is None:
            return None
        cursor = self.conn.execute(
            """
            SELECT id, code, file_path, start_line, end_line, language, function_name
            FROM snippets
            WHERE id = ?
            """,
            (snippet_id,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return dict(row)

    def get_snippets(self, snippet_ids: Sequence[int]) -> list[dict[str, Any]]:
        """Retrieve metadata for multiple snippets.

        Args:
            snippet_ids: List of snippet IDs.

        Returns:
            List of dictionaries with snippet metadata.

        """
        if not snippet_ids or self.conn is None:
            return []

        placeholders = ",".join("?" * len(snippet_ids))
        # Only `?` and `,` characters are interpolated; all user values are
        # bound via the parameterized second argument, so this is not injectable.
        cursor = self.conn.execute(
            f"""
            SELECT id, code, file_path, start_line, end_line, language, function_name
            FROM snippets
            WHERE id IN ({placeholders})
            """,
            list(snippet_ids),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_snippet_by_location(
        self,
        file_path: str,
        line_number: int,
    ) -> dict[str, Any] | None:
        """Find a snippet by file location.

        Args:
            file_path: Path to the source file.
            line_number: Line number within the file.

        Returns:
            Dictionary with snippet metadata, or None if not found.

        """
        if self.conn is None:
            return None
        cursor = self.conn.execute(
            """
            SELECT id, code, file_path, start_line, end_line, language, function_name
            FROM snippets
            WHERE file_path = ? AND start_line <= ? AND end_line >= ?
            """,
            (file_path, line_number, line_number),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return dict(row)

    def count(self) -> int:
        """Get the total number of snippets in the store."""
        if self.conn is None:
            return 0
        cursor = self.conn.execute("SELECT COUNT(*) FROM snippets")
        return int(cursor.fetchone()[0])

    def get_languages(self) -> list[str]:
        """Get list of all languages in the store."""
        if self.conn is None:
            return []
        cursor = self.conn.execute("SELECT DISTINCT language FROM snippets")
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Closed metadata database connection")

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()
