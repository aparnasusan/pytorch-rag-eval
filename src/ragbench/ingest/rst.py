"""Convert reStructuredText (docstrings, eval-rst blocks, .rst notes) into plain markdown-ish text.

Keeps headings, code blocks (as ``` fences) and admonitions; strips Sphinx markup.
This is deliberately a small line-based converter, not a full RST renderer: the goal
is clean retrievable text, not faithful HTML.
"""

from __future__ import annotations

import re
import textwrap

DIRECTIVE = re.compile(r"^\.\.\s+([\w:.-]+)::\s*(.*)$")
_OPTION = re.compile(r"^:([\w-]+):(?:\s+(.*))?$")
_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+.:_])\1{2,}$")

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_LITERAL = re.compile(r"``(.+?)``", re.S)
# Napoleon-style section headers that PyTorch writes as "Shape::" etc. but renders as prose.
PROSE_SECTIONS = {
    "shape", "shapes", "args", "arguments", "parameters", "returns", "return", "attributes",
    "inputs", "outputs", "variables", "keyword args", "keyword arguments", "raises",
    "note", "notes", "warning", "warnings", "see also",
}  # fmt: skip
_ROLE = re.compile(r"(?::([\w:+.-]+):|\{([\w:+.-]+)\})`([^`]+)`")
_LINK = re.compile(r"`([^`<]+?)\s*<([^>]+)>`__?")
# `target`_ reference: content can't start/end with a space and must be its own token,
# so "`Iterable` with `__len__`" isn't read as the reference "` with `_".
_REF = re.compile(r"(?<![`\w])`([^`\s](?:[^`]*[^`\s])?)`__?(?![\w`])")
_ESCAPED_SPACE = re.compile(r"\\ ")

CODE_ROLES = {
    "class", "func", "meth", "attr", "mod", "obj", "data", "exc", "const", "any",
    "py:class", "py:func", "py:meth", "py:attr", "py:mod", "py:obj", "py:data", "py:exc",
}  # fmt: skip

ADMONITIONS = {
    "note": "Note", "warning": "Warning", "tip": "Tip", "important": "Important",
    "caution": "Caution", "danger": "Danger", "attention": "Attention", "hint": "Hint",
    "seealso": "See also", "admonition": "",
}  # fmt: skip
VERSION_DIRECTIVES = {
    "versionadded": "New in version",
    "versionchanged": "Changed in version",
    "deprecated": "Deprecated since version",
}
CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "testcode", "doctest"}
PY_OBJECT_DIRECTIVES = {
    "class", "method", "function", "attribute", "data", "exception", "property",
    "py:class", "py:method", "py:function", "py:attribute", "py:data", "py:exception",
}  # fmt: skip
DROP_DIRECTIVES = {
    "toctree", "contents", "image", "figure", "raw", "role", "currentmodule", "module",
    "py:module", "py:currentmodule", "autosummary", "testsetup", "testcleanup", "include",
    "highlight", "default-role", "index", "tabularcolumns", "literalinclude", "graphviz",
    "automodule", "autoclass", "autofunction", "automethod", "autoattribute", "autodata",
    "autoexception", "autoproperty", "autodecorator",
}  # fmt: skip


# ---------------------------------------------------------------- helpers


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def collect_block(lines: list[str], start: int, base_indent: int) -> tuple[list[str], int]:
    """Lines from `start` that are blank or indented deeper than `base_indent`.

    Returns (block, end) where trailing blank lines are left out of the block.
    """
    j = start
    while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > base_indent):
        j += 1
    while j > start and not lines[j - 1].strip():
        j -= 1
    return lines[start:j], j


def split_options(block: list[str]) -> tuple[dict[str, str], str]:
    """Split a directive's block into its `:option: value` lines and dedented body."""
    body = textwrap.dedent("\n".join(block)).split("\n")
    opts: dict[str, str] = {}
    i = 0
    while i < len(body) and (m := _OPTION.match(body[i].strip())):
        opts[m.group(1)] = (m.group(2) or "").strip()
        i += 1
    return opts, "\n".join(body[i:]).strip("\n")


def fence(code: str, lang: str = "python", indent: int = 0) -> list[str]:
    pad = " " * indent
    code_lines = textwrap.dedent(code).strip("\n").split("\n")
    return [f"{pad}```{lang}", *[pad + ln if ln else "" for ln in code_lines], f"{pad}```"]


def _role_sub(m: re.Match[str]) -> str:
    role = m.group(1) or m.group(2)
    content = m.group(3)
    if role == "math":
        return content
    titled = re.match(r"^(.*?)\s*<([^>]+)>$", content, re.S)
    if titled and titled.group(1):
        return titled.group(1)
    target = titled.group(2) if titled else content
    target = target.lstrip("!")
    if target.startswith("~"):
        target = target[1:].split(".")[-1]
    return f"`{target}`" if role in CODE_ROLES else target


def inline(text: str) -> str:
    """Strip inline Sphinx/RST/MyST markup from one line of prose."""
    text = _INLINE_LITERAL.sub(r"`\1`", text)
    text = _ROLE.sub(_role_sub, text)
    text = _LINK.sub(r"\1", text)
    text = _REF.sub(r"\1", text)
    return _ESCAPED_SPACE.sub("", text)


def inline_outside_code(text: str) -> str:
    """Apply `inline` to each run of non-code lines, so roles that wrap lines are handled."""
    out: list[str] = []
    buf: list[str] = []
    fence_tok: str | None = None
    for line in text.split("\n"):
        f = _FENCE.match(line)
        if fence_tok is None and not f:
            buf.append(line)
            continue
        if buf:
            out.append(inline("\n".join(buf)))
            buf = []
        out.append(line)
        if f:
            if fence_tok is None:
                fence_tok = f.group(1)
            elif f.group(1)[0] == fence_tok[0] and len(f.group(1)) >= len(fence_tok):
                fence_tok = None
    if buf:
        out.append(inline("\n".join(buf)))
    return "\n".join(out)


def tidy(text: str) -> str:
    """Trim trailing spaces and collapse runs of blank lines."""
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------- directives


def render_directive(name: str, arg: str, opts: dict[str, str], body: str) -> list[str]:
    name = name.lower()
    if name in DROP_DIRECTIVES:
        return []
    if name in ADMONITIONS:
        label = ADMONITIONS[name] or arg or "Note"
        content = body if name == "admonition" else "\n".join(filter(None, [arg, body]))
        return ["", f"{label}:", *clean_rst(content).split("\n"), ""]
    if name in VERSION_DIRECTIVES:
        extra = " ".join(clean_rst(body).split()) if body else ""
        return [f"{VERSION_DIRECTIVES[name]} {arg}. {extra}".rstrip()]
    if name in CODE_DIRECTIVES:
        lang = arg or opts.get("language", "") or "python"
        return fence(body, lang) if body.strip() else []
    if name == "math":
        math = "\n".join(filter(None, [arg, body]))
        return ["$$", *textwrap.dedent(math).strip().split("\n"), "$$"]
    if name in PY_OBJECT_DIRECTIVES:
        return ["", f"`{arg}`", "", *clean_rst(body).split("\n")]
    # Unknown directive (tables, only, etc.): keep its text.
    content = "\n".join(filter(None, [arg, body]))
    return clean_rst(content).split("\n")


# ---------------------------------------------------------------- main converter


def clean_rst(text: str) -> str:
    lines = text.expandtabs(4).replace("\r\n", "\n").split("\n")
    out: list[str] = []
    heading_styles: list[str] = []
    i, n = 0, len(lines)

    def heading(style: str, title: str) -> str:
        if style not in heading_styles:
            heading_styles.append(style)
        level = min(heading_styles.index(style) + 1, 6)
        return "#" * level + " " + inline(title.strip())

    while i < n:
        line = lines[i]
        s = line.strip()
        ind = indent_of(line)

        # Over+underlined heading
        if (
            ind == 0
            and _UNDERLINE.match(s)
            and i + 2 < n
            and lines[i + 1].strip()
            and lines[i + 2].strip() == s
        ):
            out.append(heading("o" + s[0], lines[i + 1]))
            i += 3
            continue
        # Underlined heading
        if (
            ind == 0
            and s
            and i + 1 < n
            and _UNDERLINE.match(nxt := lines[i + 1].strip())
            and len(nxt) >= len(s)
            and not s.startswith("..")
        ):
            out.append(heading(nxt[0], s))
            i += 2
            continue
        # Transition line
        if ind == 0 and _UNDERLINE.match(s) and len(s) >= 4:
            i += 1
            continue

        if m := DIRECTIVE.match(s):
            block, end = collect_block(lines, i + 1, ind)
            opts, body = split_options(block)
            pad = " " * ind
            out.extend(pad + ln if ln else "" for ln in render_directive(m[1], m[2], opts, body))
            i = end
            continue
        if s == ".." or s.startswith(".. "):  # comment, link target, substitution
            _, i = collect_block(lines, i + 1, ind)  # always advances past this line
            continue

        if s.endswith("::"):
            prefix = line.rstrip()[:-2].rstrip()
            if prefix.strip():
                out.append(prefix + ":")
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and indent_of(lines[j]) > ind:
                block, end = collect_block(lines, j, ind)
                if prefix.strip().lower() in PROSE_SECTIONS:
                    pad = " " * (ind + 4)
                    body = clean_rst(textwrap.dedent("\n".join(block)))
                    out.extend(pad + ln if ln else "" for ln in body.split("\n"))
                else:
                    out.extend(fence("\n".join(block), "python", ind))
                i = end
            else:
                i += 1
            continue

        if s.startswith(">>>"):
            j = i
            while j < n and lines[j].strip():
                j += 1
            out.extend(fence("\n".join(lines[i:j]), "python", ind))
            i = j
            continue

        out.append(line)
        i += 1

    return tidy(inline_outside_code("\n".join(out)))
