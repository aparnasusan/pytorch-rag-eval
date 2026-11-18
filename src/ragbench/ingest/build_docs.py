"""Build data/corpus/docs.jsonl: cleaned source documents with sections and citation URLs.

    python -m ragbench.ingest.build_docs

Every record is one citable document:
  - kind="page":  a docs page's own prose (e.g. optim.html "How to use an optimizer")
  - kind="note":  a notes page (CUDA semantics, reproducibility, AMP, ...)
  - kind="api":   one documented object, rendered from the installed torch's docstrings

Sections carry char offsets into `text`; gold evidence spans and chunks both point here.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ragbench.config import PROJECT_ROOT, get_config
from ragbench.ingest.api import render_api, resolve
from ragbench.ingest.fetch import fetch_pytorch_docs, fetch_tutorials
from ragbench.ingest.gallery import parse_gallery
from ragbench.ingest.pages import ApiEntry, parse_page
from ragbench.ingest.sections import find_sections

MIN_PAGE_CHARS = 200  # skip index/toctree-only pages


def _page_url(base: str, rel: str) -> str:
    return base + rel.rsplit(".", 1)[0] + ".html"


def _with_section_urls(sections: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    root = url.split("#", 1)[0]
    return [{**s, "url": f"{root}#{s['anchor']}" if s["anchor"] else root} for s in sections]


def build_page_doc(path: Path, rel: str, kind: str, base: str, version: str):
    parsed = parse_page(path.read_text(encoding="utf-8"), is_rst=path.suffix == ".rst")
    url = _page_url(base, rel)
    sections = find_sections(parsed.text)
    title = next((s["title"] for s in sections if s["level"] == 1), rel)
    record = {
        "doc_id": f"{kind}:{rel.rsplit('.', 1)[0]}",
        "kind": kind,
        "title": title,
        "url": url,
        "source": f"pytorch/docs/source/{rel}",
        "object_name": None,
        "page": rel,
        "torch_version": version,
        "text": parsed.text,
        "sections": _with_section_urls(sections, url),
    }
    return record, parsed


def build_api_doc(entry: ApiEntry, page_rel: str, base: str, version: str):
    resolved = resolve(entry.name, entry.module)
    if resolved is None:
        return None
    full, obj = resolved
    if entry.generated:
        url = f"{base}generated/{full}.html"
    else:
        url = f"{_page_url(base, page_rel)}#{full}"
    text, members = render_api(full, obj)
    sections = find_sections(
        text, anchor_fn=lambda t: t.strip(), max_level=2, default_title=full, default_anchor=full
    )
    return {
        "doc_id": f"api:{full}",
        "kind": "api",
        "title": full,
        "url": url,
        "source": f"docstring:{full}",
        "object_name": full,
        "page": page_rel,
        "torch_version": version,
        "text": text,
        "sections": _with_section_urls(sections, url),
        "has_docstring": bool(inspect.getdoc(obj)) or bool(members),
    }


def build_tutorial_doc(path: Path, rel: str, base: str, commit: str, version: str):
    """rel is relative to beginner_source/, e.g. 'basics/data_tutorial.py'."""
    source = path.read_text(encoding="utf-8")
    text = parse_page(source, is_rst=True).text if path.suffix == ".rst" else parse_gallery(source)
    stem = "beginner/" + rel.rsplit(".", 1)[0]
    url = f"{base}{stem}.html"
    sections = find_sections(text)
    title = next((s["title"] for s in sections if s["level"] == 1), rel)
    return {
        "doc_id": f"tutorial:{stem}",
        "kind": "tutorial",
        "title": title,
        "url": url,
        "source": f"pytorch/tutorials/beginner_source/{rel}@{commit[:7]}",
        "object_name": None,
        "page": rel,
        "torch_version": version,
        "text": text,
        "sections": _with_section_urls(sections, url),
    }


def build(config: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch

    cfg = (config or get_config())["corpus"]
    version = cfg["torch_version"]
    if not torch.__version__.startswith(version):
        sys.exit(f"Installed torch {torch.__version__} != pinned {version}")
    base = cfg["docs_base_url"]
    docs_src = fetch_pytorch_docs(PROJECT_ROOT / cfg["raw_dir"], cfg["torch_git_commit"])

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    report: dict[str, Any] = {
        "unresolved": [],
        "empty_docstring": [],
        "skipped_pages": [],
        "missing_tutorials": [],
        "warnings": [],
    }

    def add(record: dict[str, Any]) -> None:
        if record["doc_id"] not in seen:
            seen.add(record["doc_id"])
            records.append(record)

    for rel in cfg["api_pages"]:
        page, parsed = build_page_doc(docs_src / rel, rel, "page", base, version)
        report["warnings"] += [f"{rel}: {w}" for w in parsed.warnings]
        if len(page["text"]) >= MIN_PAGE_CHARS:
            add(page)
        else:
            report["skipped_pages"].append(rel)
        for entry in parsed.entries:
            doc = build_api_doc(entry, rel, base, version)
            if doc is None:
                report["unresolved"].append(f"{rel}: {entry.module}.{entry.name}")
                continue
            if not doc.pop("has_docstring"):  # heading-only on the docs site; nothing to retrieve
                report["empty_docstring"].append(doc["object_name"])
                continue
            add(doc)

    for path in sorted(docs_src.glob(cfg["notes_glob"])):
        if path.suffix not in {".md", ".rst"}:
            continue
        rel = path.relative_to(docs_src).as_posix()
        note, parsed = build_page_doc(path, rel, "note", base, version)
        if len(note["text"]) >= MIN_PAGE_CHARS:
            add(note)
        else:
            report["skipped_pages"].append(rel)

    tut_commit = cfg["tutorials_git_commit"]
    tut_src = fetch_tutorials(PROJECT_ROOT / cfg["raw_dir"], tut_commit)
    for rel in cfg["tutorials"]:
        path = tut_src / rel
        if not path.exists():
            report["missing_tutorials"].append(rel)
            continue
        add(build_tutorial_doc(path, rel, cfg["tutorials_base_url"], tut_commit, version))

    out_path = PROJECT_ROOT / cfg["docs_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["counts"] = dict(Counter(r["kind"] for r in records))
    report["total_docs"] = len(records)
    report["total_chars"] = sum(len(r["text"]) for r in records)
    report_path = PROJECT_ROOT / cfg["report_path"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = build()
    print(json.dumps({k: report[k] for k in ("counts", "total_docs", "total_chars")}, indent=2))
    for key in ("unresolved", "empty_docstring", "skipped_pages", "missing_tutorials", "warnings"):
        print(f"{key}: {len(report[key])}")


if __name__ == "__main__":
    main()
