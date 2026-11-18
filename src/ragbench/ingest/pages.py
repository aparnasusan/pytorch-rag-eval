"""Parse PyTorch docs pages (MyST Markdown with embedded eval-rst, or plain .rst).

Produces the page's readable text plus the list of API objects the page documents
(autosummary entries and inline auto* directives), tracking `currentmodule` the way
Sphinx does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragbench.ingest.rst import (
    ADMONITIONS,
    DIRECTIVE,
    clean_rst,
    collect_block,
    fence,
    indent_of,
    inline_outside_code,
    split_options,
    tidy,
)

_FENCE_OPEN = re.compile(r"^(\s*)(`{3,}|~{3,}|:{3,})\s*(\{[\w:.-]+\})?\s*(.*)$")
_MYST_TARGET = re.compile(r"^\(.+\)=\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

AUTO_OBJECT_DIRECTIVES = {
    "autoclass", "autofunction", "automethod", "autoattribute", "autodata",
    "autoexception", "autoproperty", "autodecorator",
}  # fmt: skip
MYST_DROP = {"toctree", "contents", "image", "figure", "raw", "include", "literalinclude"}


@dataclass
class ApiEntry:
    name: str  # as written in the directive (may be relative, may start with ~)
    module: str | None  # currentmodule at that point
    generated: bool  # True → has its own generated/<fullname>.html page
    directive: str


@dataclass
class PageParse:
    text: str
    entries: list[ApiEntry] = field(default_factory=list)
    module_docs: list[str] = field(default_factory=list)  # modules whose docstring was inlined
    warnings: list[str] = field(default_factory=list)


@dataclass
class _State:
    module: str | None = None
    entries: list[ApiEntry] = field(default_factory=list)
    module_docs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _module_docstring(module: str) -> str:
    import importlib
    import inspect

    try:
        return inspect.getdoc(importlib.import_module(module)) or ""
    except Exception:
        return ""


def _extract_api_directives(rst: str, state: _State) -> str:
    """Pull out currentmodule/automodule/autosummary/auto* directives; return remaining RST."""
    lines = rst.split("\n")
    kept: list[str] = []
    i = 0
    while i < len(lines):
        m = DIRECTIVE.match(lines[i].strip())
        name = m.group(1).lower() if m else ""
        if name in {"currentmodule", "py:currentmodule", "module", "py:module", "automodule"}:
            ind = indent_of(lines[i])
            block, end = collect_block(lines, i + 1, ind)
            opts, _ = split_options(block)
            arg = m.group(2).strip()
            if name in {"currentmodule", "py:currentmodule"}:
                state.module = None if arg == "None" else arg
            elif name == "automodule":
                state.module = arg
                if "members" in opts:
                    state.warnings.append(f"automodule {arg} has :members: (not expanded)")
                doc = _module_docstring(arg)
                if doc:
                    state.module_docs.append(arg)
                    kept.extend(" " * ind + ln for ln in doc.split("\n"))
            i = end
            continue
        if name == "autosummary":
            ind = indent_of(lines[i])
            block, end = collect_block(lines, i + 1, ind)
            opts, body = split_options(block)
            generated = "toctree" in opts
            for entry in body.split("\n"):
                entry = entry.strip()
                if entry and not entry.startswith(".."):
                    state.entries.append(ApiEntry(entry, state.module, generated, "autosummary"))
            i = end
            continue
        if name in AUTO_OBJECT_DIRECTIVES:
            ind = indent_of(lines[i])
            _, end = collect_block(lines, i + 1, ind)
            target = m.group(2).strip().split("(")[0].strip()
            state.entries.append(ApiEntry(target, state.module, False, name))
            i = end
            continue
        kept.append(lines[i])
        i += 1
    return "\n".join(kept)


def _split_myst_options(body: list[str]) -> tuple[dict[str, str], list[str]]:
    opts: dict[str, str] = {}
    i = 0
    if body and body[0].strip() == "---":  # YAML-style option block
        j = 1
        while j < len(body) and body[j].strip() != "---":
            key, _, val = body[j].partition(":")
            opts[key.strip()] = val.strip()
            j += 1
        return opts, body[j + 1 :]
    while i < len(body) and (m := re.match(r"^:([\w-]+):(?:\s+(.*))?$", body[i].strip())):
        opts[m.group(1)] = (m.group(2) or "").strip()
        i += 1
    return opts, body[i:]


def _parse_myst(text: str, state: _State) -> list[str]:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    if lines and lines[0].strip() == "---":  # front matter
        j = 1
        while j < len(lines) and lines[j].strip() != "---":
            j += 1
        i = j + 1
    while i < len(lines):
        line = lines[i]
        m = _FENCE_OPEN.match(line)
        if m:
            fence_tok, directive, info = m.group(2), m.group(3), m.group(4).strip()
            close = re.compile(rf"^\s*{re.escape(fence_tok[0])}{{{len(fence_tok)},}}\s*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            body = lines[i + 1 : j]
            i = j + 1
            if directive is None:
                if fence_tok[0] == ":":  # bare colon fence: just content
                    out.extend(_parse_myst("\n".join(body), state))
                else:
                    out.extend(fence("\n".join(body), info.split()[0] if info else ""))
                continue
            name = directive[1:-1].lower()
            opts, rest = _split_myst_options(body)
            if name == "eval-rst":
                remaining = _extract_api_directives("\n".join(body), state)
                out.extend(clean_rst(remaining).split("\n"))
            elif name in {"code-block", "code", "sourcecode", "testcode", "doctest"}:
                out.extend(fence("\n".join(rest), info or opts.get("language", "") or "python"))
            elif name == "math":
                out.extend(["$$", *("\n".join(filter(None, [info, *rest]))).split("\n"), "$$"])
            elif name in MYST_DROP:
                pass
            elif name in ADMONITIONS:
                label = ADMONITIONS[name] or info or "Note"
                content = rest if name == "admonition" else [info, *rest] if info else rest
                out.extend(["", f"{label}:", *_parse_myst("\n".join(content), state), ""])
            elif name.startswith("auto") or name in {"currentmodule", "py:currentmodule"}:
                rst = [f".. {name}:: {info}"]
                rst += [f"   :{k}: {v}".rstrip() for k, v in opts.items()]
                rst += [""] + ["   " + ln for ln in rest]
                remaining = _extract_api_directives("\n".join(rst), state)
                out.extend(clean_rst(remaining).split("\n"))
            else:
                out.extend(_parse_myst("\n".join(rest), state))
            continue
        if _MYST_TARGET.match(line.strip()) or line.startswith("%"):  # target / MyST comment
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def parse_page(text: str, is_rst: bool = False) -> PageParse:
    state = _State()
    text = text.replace("\r\n", "\n")
    if is_rst:
        body = clean_rst(_extract_api_directives(text, state))
    else:
        lines = _parse_myst(_HTML_COMMENT.sub("", text), state)
        body = tidy(inline_outside_code("\n".join(lines)))
    return PageParse(body, state.entries, state.module_docs, state.warnings)
