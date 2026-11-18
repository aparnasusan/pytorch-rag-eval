"""Heading-based sections with stable anchors and character offsets.

Sections are what gold evidence spans and citations point at, so offsets are computed
on the final document text and headings inside code fences are ignored.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import TypedDict

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


class Section(TypedDict):
    title: str
    level: int
    anchor: str
    char_start: int
    char_end: int


def slugify(title: str) -> str:
    """Approximate docutils `make_id`, which Sphinx/MyST use for section ids."""
    text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    text = re.sub(r"[`*_]", "", text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"^[^a-z]+", "", text) or "section"


def iter_headings(text: str, max_level: int = 6):
    """Yield (char_offset, level, title) for markdown headings outside code fences."""
    offset = 0
    fence_char: str | None = None
    for line in text.split("\n"):
        f = _FENCE.match(line)
        if f:
            if fence_char is None:
                fence_char = f.group(1)
            elif f.group(1).startswith(fence_char[0]) and len(f.group(1)) >= len(fence_char):
                fence_char = None
        elif fence_char is None and (m := _HEADING.match(line)):
            level = len(m.group(1))
            if level <= max_level:
                yield offset, level, m.group(2)
        offset += len(line) + 1


def find_sections(
    text: str,
    anchor_fn: Callable[[str], str] = slugify,
    max_level: int = 6,
    default_title: str = "",
    default_anchor: str = "",
) -> list[Section]:
    """Split text into consecutive sections, one per heading (any level up to max_level).

    Text before the first heading becomes a leading section with the default title/anchor.
    """
    heads = list(iter_headings(text, max_level))
    sections: list[Section] = []
    if not heads or (heads[0][0] > 0 and text[: heads[0][0]].strip()):
        end = heads[0][0] if heads else len(text)
        sections.append(
            Section(title=default_title, level=0, anchor=default_anchor, char_start=0, char_end=end)
        )
    seen: dict[str, int] = {}
    for idx, (start, level, title) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(text)
        anchor = anchor_fn(title)
        if anchor in seen:  # Sphinx de-duplicates repeated ids with -1, -2, ...
            seen[anchor] += 1
            anchor = f"{anchor}-{seen[anchor]}"
        else:
            seen[anchor] = 0
        sections.append(
            Section(title=title, level=level, anchor=anchor, char_start=start, char_end=end)
        )
    return sections


def demote_headings(text: str, min_level: int) -> str:
    """Push every heading (outside code) down to at least `min_level`."""
    out, fence_char = [], None
    for line in text.split("\n"):
        f = _FENCE.match(line)
        if f:
            if fence_char is None:
                fence_char = f.group(1)
            elif f.group(1).startswith(fence_char[0]) and len(f.group(1)) >= len(fence_char):
                fence_char = None
        elif fence_char is None and (m := _HEADING.match(line)):
            level = min(max(len(m.group(1)), min_level), 6)
            line = "#" * level + " " + m.group(2)
        out.append(line)
    return "\n".join(out)
