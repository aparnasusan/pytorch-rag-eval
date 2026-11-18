"""Parse sphinx-gallery tutorial scripts (pytorch/tutorials *.py) into clean text.

A gallery script is: an RST module docstring, then alternating code and text blocks.
A text block starts with a separator line (`#####...` of 20+ chars, or `# %%`) and
continues over `#`-prefixed lines. Code between `# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%`
markers is hidden on the rendered page, so it is dropped.

The whole script is converted to one RST document (code → `.. code-block:: python`)
and cleaned in a single pass, so heading levels stay consistent across blocks.
"""

from __future__ import annotations

import ast
import inspect
import re
import warnings

from ragbench.ingest.rst import clean_rst

_SEPARATOR = re.compile(r"^(#{20,}|# %%(?!%))\s*.*$")
_HIDDEN_TOGGLE = re.compile(r"^#\s*%%%%%%INVISIBLE_CODE_BLOCK%%%%%%\s*$")
# Breadcrumb nav lines at the top of a tutorial: "`Tensors <x.html>`_ ||" / "**Datasets** ||"
_NAV_LINE = re.compile(r"^\s*(`[^`]+`__?|\*\*[^*]+\*\*)\s*(\|\|)?\s*$")
_GALLERY_DIRECTIVE_COMMENT = re.compile(r"^#\s*sphinx_gallery_")


def _docstring_and_rest(source: str) -> tuple[str, list[str]]:
    lines = source.replace("\r\n", "\n").split("\n")
    try:
        with warnings.catch_warnings():  # tutorial code has e.g. "\s" in non-raw strings
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return "", lines
    first = tree.body[0] if tree.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return inspect.cleandoc(first.value.value), lines[first.end_lineno :]
    return "", lines


def _strip_comment(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    return line[1:] if line.startswith("#") else line


def gallery_to_rst(source: str) -> str:
    doc, rest = _docstring_and_rest(source)
    parts: list[str] = [
        "\n".join(ln for ln in doc.split("\n") if not (_NAV_LINE.match(ln) and "||" in ln))
    ]
    # Single nav items without "||" (the last one) only appear in the nav run at the top.
    parts[0] = re.sub(r"\A(\s*(`[^`]+`__?|\*\*[^*]+\*\*)\s*\n)+", "", parts[0])

    code: list[str] = []
    hidden = False

    def flush_code() -> None:
        body = "\n".join(code).strip("\n")
        if body.strip():
            parts.append(
                ".. code-block:: python\n\n"
                + "\n".join("    " + ln if ln.strip() else "" for ln in body.split("\n"))
            )
        code.clear()

    i = 0
    while i < len(rest):
        line = rest[i]
        if _HIDDEN_TOGGLE.match(line.strip()):
            hidden = not hidden
            i += 1
            continue
        if hidden:
            i += 1
            continue
        if _SEPARATOR.match(line):
            flush_code()
            i += 1
            text: list[str] = []
            while i < len(rest) and rest[i].startswith("#") and not _SEPARATOR.match(rest[i]):
                if _HIDDEN_TOGGLE.match(rest[i].strip()):
                    break
                text.append(_strip_comment(rest[i]))
                i += 1
            if any(t.strip() for t in text):
                parts.append("\n".join(text).strip("\n"))
            continue
        if not _GALLERY_DIRECTIVE_COMMENT.match(line):
            code.append(line)
        i += 1
    flush_code()
    return "\n\n".join(p for p in parts if p.strip())


def parse_gallery(source: str) -> str:
    """Clean text for a sphinx-gallery script (headings, prose, ```python code)."""
    return clean_rst(gallery_to_rst(source))
