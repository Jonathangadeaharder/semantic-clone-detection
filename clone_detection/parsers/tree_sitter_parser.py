"""Tree-sitter-based code parser for extracting function-level code snippets.

This module implements Part I of the blueprint: Ingestion & Parsing.
It uses Tree-sitter to parse source code files and extract discrete,
semantically-coherent units (functions and methods) across multiple languages.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from clone_detection.parsers.language_configs import (
    LANGUAGE_CONFIGS,
    LanguageConfig,
    get_language_for_file,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """Represents a parsed code snippet with metadata.

    Attributes:
        code: The raw source code of the function.
        file_path: Path to the source file.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        language: Programming language.
        function_name: Optional name of the function.

    """

    code: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    function_name: str | None = None

    def __repr__(self) -> str:
        """Return a concise, human-readable representation of the snippet."""
        return (
            f"CodeSnippet(file={self.file_path}, "
            f"lines={self.start_line}-{self.end_line}, "
            f"lang={self.language}, "
            f"name={self.function_name})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "function_name": self.function_name,
        }


class TreeSitterParser:
    """Multi-language code parser using Tree-sitter.

    This parser implements the extraction strategy described in Section 1.3
    of the blueprint, using S-expression queries to find function definitions
    across multiple programming languages.

    Example:
        >>> parser = TreeSitterParser(languages=["python", "java"])
        >>> snippets = parser.parse_file("example.py")
        >>> print(f"Found {len(snippets)} functions")

    """

    def __init__(self, languages: Sequence[str] | None = None) -> None:
        """Initialize the parser with specified languages.

        Args:
            languages: List of language names to support. If None, all available
                languages are enabled.

        """
        self.enabled_languages: list[str] = (
            list(languages) if languages else list(LANGUAGE_CONFIGS.keys())
        )
        self.parsers: dict[str, Parser] = {}
        self.queries: dict[str, Query] = {}
        self._configs: dict[str, LanguageConfig] = {}

        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        """Load Tree-sitter grammars and compile queries for each enabled language.

        This implements the initialization described in Section 1.3.
        """
        for lang_name in self.enabled_languages:
            if lang_name not in LANGUAGE_CONFIGS:
                logger.warning("Unknown language: %s, skipping", lang_name)
                continue

            config = LANGUAGE_CONFIGS[lang_name]

            try:
                # Dynamically import the language grammar module.
                grammar_module = importlib.import_module(config.grammar_module)
                language = Language(grammar_module.language())

                # Create parser and compile the S-expression query.
                parser = Parser(language)
                query = Query(language, config.function_query)

                self.parsers[lang_name] = parser
                self.queries[lang_name] = query
                self._configs[lang_name] = config

                logger.info("Initialized parser for %s", lang_name)
            except ImportError:
                logger.exception(
                    "Failed to import grammar for %s. Install with: uv add %s",
                    lang_name,
                    config.grammar_module.replace("_", "-"),
                )
            except Exception:
                logger.exception("Failed to initialize parser for %s", lang_name)

    def parse_file(self, file_path: str) -> list[CodeSnippet]:
        """Parse a single source file and extract all function snippets.

        Args:
            file_path: Path to the source code file.

        Returns:
            List of extracted code snippets.

        Example:
            >>> parser = TreeSitterParser(languages=["python"])
            >>> snippets = parser.parse_file("my_module.py")

        """
        resolved_path = str(Path(file_path).resolve())

        # Determine language from file extension.
        lang_name = get_language_for_file(resolved_path)
        if lang_name is None:
            logger.debug("Unsupported file type: %s", resolved_path)
            return []

        if lang_name not in self.parsers:
            logger.debug(
                "Parser not initialized for %s: %s",
                lang_name,
                resolved_path,
            )
            return []

        # Read file content.
        try:
            with Path(resolved_path).open("rb") as f:
                source_bytes = f.read()
        except OSError:
            logger.exception("Failed to read file %s", resolved_path)
            return []

        return self._parse_source(source_bytes, resolved_path, lang_name)

    def _parse_source(
        self,
        source_bytes: bytes,
        file_path: str,
        lang_name: str,
    ) -> list[CodeSnippet]:
        """Parse source code bytes and extract function snippets.

        This implements the extraction logic from Section 1.3:
        1. Parse the file into an AST.
        2. Run the S-expression query.
        3. Extract function metadata and code.

        Args:
            source_bytes: Raw source code as bytes.
            file_path: Path to the source file.
            lang_name: Programming language name.

        Returns:
            List of extracted code snippets.

        """
        parser = self.parsers[lang_name]
        query = self.queries[lang_name]
        config = self._configs[lang_name]

        # Parse the source code.
        tree = parser.parse(source_bytes)

        # Run the query to find all function definitions.
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        # In tree-sitter 0.25+, captures() returns a dict mapping capture name
        # to a list of nodes. The nodes are NOT guaranteed to be in source
        # order (the cursor yields them in match order, which can differ for
        # nested/overlapping captures), so we sort by start byte to present
        # snippets in the order they appear in the file.
        definition_nodes = sorted(captures.get(config.capture_name, []), key=lambda n: n.start_byte)

        snippets: list[CodeSnippet] = []
        for node in definition_nodes:
            function_code = node.text.decode("utf8")

            # start_point/end_point are 0-indexed (row, column); convert to
            # 1-indexed line numbers.
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            function_name = self._extract_function_name(node)

            snippet = CodeSnippet(
                code=function_code,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=lang_name,
                function_name=function_name,
            )
            snippets.append(snippet)

        logger.debug("Extracted %d functions from %s", len(snippets), file_path)
        return snippets

    def _extract_function_name(self, node: Node) -> str | None:
        """Extract the function name from a function definition node.

        Args:
            node: Tree-sitter node representing the function.

        Returns:
            Function name if found, None otherwise.

        """
        # Look for a child node of type "identifier" or "name".
        for child in node.children:
            if child.type in ("identifier", "name"):
                return child.text.decode("utf8")

            # Recursive search in case the name is nested.
            for subchild in child.children:
                if subchild.type in ("identifier", "name"):
                    return subchild.text.decode("utf8")

        return None

    def parse_directory(
        self,
        directory: str,
        exclude_patterns: Sequence[str] | None = None,
        max_files: int | None = None,
    ) -> list[CodeSnippet]:
        """Recursively parse all source files in a directory.

        This implements the "CodebaseWalker" component from Blueprint A
        (Batch Ingestion Pipeline).

        Args:
            directory: Root directory to scan.
            exclude_patterns: List of glob patterns to exclude (e.g., "*.test.py").
            max_files: Maximum number of files to process (for testing).

        Returns:
            List of all extracted code snippets.

        Example:
            >>> parser = TreeSitterParser(languages=["python"])
            >>> snippets = parser.parse_directory(
            ...     "/path/to/codebase",
            ...     exclude_patterns=["**/test_*.py", "**/__pycache__/**"]
            ... )

        """
        # Build exclusion pathspec.
        exclude_spec: PathSpec | None = None
        if exclude_patterns:
            exclude_spec = PathSpec.from_lines(
                GitWildMatchPattern,
                list(exclude_patterns),
            )

        # Collect supported file extensions.
        supported_extensions: set[str] = set()
        for lang_name in self.enabled_languages:
            if lang_name in LANGUAGE_CONFIGS:
                supported_extensions.update(
                    ext.lower() for ext in LANGUAGE_CONFIGS[lang_name].extensions
                )

        # Walk directory and collect files.
        all_snippets: list[CodeSnippet] = []
        file_count = 0

        root_path = Path(directory).resolve()
        for file_path in root_path.rglob("*"):
            # Skip directories.
            if not file_path.is_file():
                continue

            # Check if file has a supported extension (case-insensitive).
            if file_path.suffix.lower() not in supported_extensions:
                continue

            # Check exclusion patterns.
            relative_path = file_path.relative_to(root_path)
            if exclude_spec and exclude_spec.match_file(str(relative_path)):
                logger.debug("Excluded: %s", relative_path)
                continue

            # Parse file.
            snippets = self.parse_file(str(file_path))
            all_snippets.extend(snippets)

            file_count += 1
            if max_files and file_count >= max_files:
                logger.info("Reached max_files limit: %d", max_files)
                break

        logger.info(
            "Parsed %d files, extracted %d function snippets",
            file_count,
            len(all_snippets),
        )
        return all_snippets
