"""Resolve documented API names to live objects and render their docstrings locally."""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any

from ragbench.ingest.rst import clean_rst
from ragbench.ingest.sections import demote_headings

MISSING = object()


def import_by_name(dotted: str) -> Any:
    """Import the longest importable module prefix, then getattr the rest."""
    parts = dotted.split(".")
    for k in range(len(parts), 0, -1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj: Any = importlib.import_module(".".join(parts[:k]))
        except Exception:
            continue
        try:
            for attr in parts[k:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return MISSING
        return obj
    return MISSING


def resolve(name: str, module: str | None) -> tuple[str, Any] | None:
    """Resolve a directive name the way Sphinx does: currentmodule first, then absolute."""
    name = name.lstrip("~").strip()
    candidates = [f"{module}.{name}"] if module else []
    # Already absolute (e.g. "torch.default_generator" under currentmodule torch): try it
    # first, so attribute chains like torch.torch.* don't win.
    if module and name.split(".")[0] == module.split(".")[0]:
        candidates.insert(0, name)
    else:
        candidates.append(name)
    for full in candidates:
        obj = import_by_name(full)
        if obj is not MISSING:
            return full, obj
    return None


def _signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _own_doc(member: Any) -> str:
    """The member's own docstring (no inheritance fallback, unlike inspect.getdoc)."""
    if isinstance(member, (classmethod, staticmethod)):
        member = member.__func__
    doc = getattr(member, "__doc__", None)
    return inspect.cleandoc(doc) if isinstance(doc, str) and doc.strip() else ""


def _is_documentable_member(member: Any) -> bool:
    if inspect.isclass(member):
        return False
    return (
        callable(member)
        or isinstance(member, (property, classmethod, staticmethod))
        or inspect.isdatadescriptor(member)
        or inspect.ismethoddescriptor(member)
    )


def _header(full: str, obj: Any, level: int) -> list[str]:
    lines = ["#" * level + " " + full, ""]
    target = obj.__func__ if isinstance(obj, (classmethod, staticmethod)) else obj
    sig = None if isinstance(obj, property) else _signature(target)
    if sig:
        prefix = "class " if inspect.isclass(obj) else ""
        lines += ["```python", f"{prefix}{full}{sig}", "```", ""]
    return lines


def render_api(full: str, obj: Any) -> tuple[str, list[str]]:
    """Render an object as text. Returns (text, member_names_included).

    Classes include their own public members (not inherited ones), so e.g.
    `Module.forward`'s doc isn't copied onto every layer.
    """
    lines = _header(full, obj, 1)
    doc = inspect.getdoc(obj) or ""
    if doc:
        lines.append(demote_headings(clean_rst(doc), 3))
    members: list[str] = []
    if inspect.isclass(obj):
        for mname, member in vars(obj).items():
            if mname.startswith("_") or not _is_documentable_member(member):
                continue
            mdoc = _own_doc(member)
            if not mdoc:
                continue
            members.append(mname)
            lines += ["", *_header(f"{full}.{mname}", member, 2)]
            lines.append(demote_headings(clean_rst(mdoc), 3))
    return "\n".join(lines).strip() + "\n", members
