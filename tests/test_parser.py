"""Tests for clone_detection.parsers.tree_sitter_parser.TreeSitterParser."""

from __future__ import annotations

from typing import TYPE_CHECKING

from clone_detection.parsers.language_configs import get_language_for_file
from clone_detection.parsers.tree_sitter_parser import CodeSnippet, TreeSitterParser

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_get_language_for_file_known_extensions() -> None:
    """get_language_for_file maps known extensions to language names."""
    assert get_language_for_file("foo.py") == "python"
    assert get_language_for_file("foo.js") == "javascript"
    assert get_language_for_file("foo.jsx") == "javascript"
    assert get_language_for_file("foo.mjs") == "javascript"
    assert get_language_for_file("Foo.java") == "java"
    assert get_language_for_file("foo.go") == "go"
    assert get_language_for_file("foo.cpp") == "cpp"
    assert get_language_for_file("foo.cs") == "csharp"


def test_get_language_for_file_unknown_returns_none() -> None:
    """get_language_for_file returns None for unrecognized extensions."""
    assert get_language_for_file("foo.txt") is None
    assert get_language_for_file("foo.rb") is None
    assert get_language_for_file("Makefile") is None


def test_get_language_for_file_case_insensitive() -> None:
    """get_language_for_file matches extensions case-insensitively."""
    assert get_language_for_file("FOO.PY") == "python"
    assert get_language_for_file("Foo.JS") == "javascript"


def test_parse_file_extracts_python_functions(sample_python_file: Path) -> None:
    """parse_file extracts all function names from a Python file."""
    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_file(str(sample_python_file))

    names = [s.function_name for s in snippets]
    assert names == ["add", "sub", "mul"]
    assert all(s.language == "python" for s in snippets)
    assert all(s.file_path == str(sample_python_file.resolve()) for s in snippets)


def test_parse_file_function_line_ranges(sample_python_file: Path) -> None:
    """parse_file reports correct 1-indexed start/end line ranges."""
    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_file(str(sample_python_file))

    add = next(s for s in snippets if s.function_name == "add")
    assert add.start_line == 1
    assert add.end_line == 2
    assert "return a + b" in add.code


def test_parse_file_includes_class_method(sample_python_file: Path) -> None:
    """parse_file extracts methods defined inside classes."""
    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_file(str(sample_python_file))
    mul = next(s for s in snippets if s.function_name == "mul")
    assert mul.start_line == 10
    assert "return a * b" in mul.code


def test_parse_file_nested_function(tmp_path: Path) -> None:
    """parse_file extracts both outer and nested inner functions."""
    src = tmp_path / "nested.py"
    src.write_text(
        """\
def outer():
    def inner():
        return 1
    return inner()
"""
    )
    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_file(str(src))
    names = {s.function_name for s in snippets}
    assert "outer" in names
    assert "inner" in names


def test_parse_file_javascript_arrow_and_function(tmp_path: Path) -> None:
    """parse_file extracts named functions, arrows, and methods from JS."""
    src = tmp_path / "mod.js"
    src.write_text(
        """\
function named() {
    return 1;
}
const arrow = () => 42;
class C {
    method() {
        return 3;
    }
}
"""
    )
    parser = TreeSitterParser(languages=["javascript"])
    snippets = parser.parse_file(str(src))
    assert len(snippets) == 3
    assert all(s.language == "javascript" for s in snippets)


def test_parse_file_typescript_unsupported_returns_empty(tmp_path: Path) -> None:
    """parse_file on an unsupported .ts extension returns an empty list."""
    src = tmp_path / "mod.ts"
    src.write_text("function f() { return 1; }\n")
    parser = TreeSitterParser(languages=["python", "javascript"])
    snippets = parser.parse_file(str(src))
    assert snippets == []


def test_parse_file_unsupported_extension_returns_empty(tmp_path: Path) -> None:
    """parse_file on an unsupported extension returns an empty list."""
    src = tmp_path / "mod.txt"
    src.write_text("def foo(): pass\n")
    parser = TreeSitterParser(languages=["python"])
    assert parser.parse_file(str(src)) == []


def test_parse_file_empty_file_returns_empty(tmp_path: Path) -> None:
    """parse_file on an empty file returns an empty list."""
    src = tmp_path / "empty.py"
    src.write_text("")
    parser = TreeSitterParser(languages=["python"])
    assert parser.parse_file(str(src)) == []


def test_parse_file_no_functions_returns_empty(tmp_path: Path) -> None:
    """parse_file on a file with no function definitions returns empty."""
    src = tmp_path / "nofunc.py"
    src.write_text("x = 1\ny = 2\nprint(x + y)\n")
    parser = TreeSitterParser(languages=["python"])
    assert parser.parse_file(str(src)) == []


def test_parse_file_syntax_error_still_parses_best_effort(tmp_path: Path) -> None:
    """parse_file does not crash on a syntax error and returns a list."""
    src = tmp_path / "broken.py"
    src.write_text("def foo(:\n    return 1\n")
    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_file(str(src))
    assert isinstance(snippets, list)


def test_parse_file_missing_file_returns_empty(tmp_path: Path) -> None:
    """parse_file on a nonexistent path returns an empty list."""
    parser = TreeSitterParser(languages=["python"])
    missing = str(tmp_path / "does_not_exist.py")
    assert parser.parse_file(missing) == []


def test_unknown_language_is_skipped() -> None:
    """An unknown language name is skipped without raising."""
    parser = TreeSitterParser(languages=["python", "cobol"])
    assert "python" in parser.parsers
    assert "cobol" not in parser.parsers


def test_default_languages_loads_all() -> None:
    """The default constructor enables all configured languages."""
    parser = TreeSitterParser()
    assert "python" in parser.parsers


def test_parse_directory_walks_tree(tmp_path: Path, sample_python_file: Path) -> None:
    """parse_directory recursively walks subdirectories for source files."""
    sub = tmp_path / "pkg"
    sub.mkdir()
    target = sub / "sample.py"
    target.write_text(sample_python_file.read_text())
    (sub / "README.md").write_text("# not code\n")

    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_directory(str(tmp_path))
    names = {s.function_name for s in snippets}
    assert {"add", "sub", "mul"} <= names


def test_parse_directory_exclude_patterns(tmp_path: Path) -> None:
    """parse_directory honors gitignore-style exclude patterns."""
    keep = tmp_path / "keep.py"
    keep.write_text("def keep(): return 1\n")
    skip_dir = tmp_path / "skip"
    skip_dir.mkdir()
    (skip_dir / "skip.py").write_text("def skip(): return 2\n")

    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_directory(str(tmp_path), exclude_patterns=["skip/**"])
    names = {s.function_name for s in snippets}
    assert "keep" in names
    assert "skip" not in names


def test_parse_directory_max_files_limit(tmp_path: Path) -> None:
    """parse_directory stops after processing ``max_files`` files."""
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text(f"def f{i}(): return {i}\n")

    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_directory(str(tmp_path), max_files=2)
    assert len(snippets) <= 2


def test_parse_directory_case_insensitive_extension(tmp_path: Path) -> None:
    """parse_directory matches file extensions case-insensitively."""
    (tmp_path / "Upper.PY").write_text("def up(): return 1\n")

    parser = TreeSitterParser(languages=["python"])
    snippets = parser.parse_directory(str(tmp_path))
    assert any(s.function_name == "up" for s in snippets)


def test_code_snippet_repr_and_to_dict() -> None:
    """CodeSnippet.__repr__ and to_dict expose the snippet's fields."""
    snippet = CodeSnippet(
        code="def foo(): pass\n",
        file_path="x.py",
        start_line=1,
        end_line=1,
        language="python",
        function_name="foo",
    )
    r = repr(snippet)
    assert "x.py" in r
    assert "foo" in r
    d = snippet.to_dict()
    assert d["function_name"] == "foo"
    assert d["code"] == "def foo(): pass\n"
    assert d["start_line"] == 1
    assert d["end_line"] == 1


def test_code_snippet_function_name_defaults_none() -> None:
    """CodeSnippet.function_name defaults to None."""
    snippet = CodeSnippet(code="x", file_path="x.py", start_line=1, end_line=1, language="python")
    assert snippet.function_name is None


def test_parse_directory_logs_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """parse_directory logs the number of files and snippets parsed."""
    (tmp_path / "a.py").write_text("def a(): return 1\n")
    parser = TreeSitterParser(languages=["python"])
    with caplog.at_level("INFO", logger="clone_detection.parsers.tree_sitter_parser"):
        parser.parse_directory(str(tmp_path))
    assert any("Parsed" in r.message for r in caplog.records)
