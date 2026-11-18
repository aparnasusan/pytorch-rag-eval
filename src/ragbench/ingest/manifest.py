"""Corpus manifest: what the corpus was built from, and checksums to detect drift."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragbench.config import PROJECT_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _chunk_file_stats(path: Path) -> dict[str, Any]:
    tokens: list[int] = []
    size = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tokens.append(rec["n_tokens"])
            size = rec["chunk_size"]
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "chunk_size": size,
        "chunks": len(tokens),
        "tokens_min": min(tokens, default=0),
        "tokens_median": statistics.median(tokens) if tokens else 0,
        "tokens_max": max(tokens, default=0),
        "sha256": sha256_file(path),
    }


def write_manifest(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    corpus, retrieval = config["corpus"], config["retrieval"]
    docs_path = PROJECT_ROOT / corpus["docs_path"]
    kinds: dict[str, int] = {}
    with docs_path.open(encoding="utf-8") as f:
        for line in f:
            kind = json.loads(line)["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1

    chunk_files = sorted(docs_path.parent.glob("chunks*.jsonl"))
    report_path = PROJECT_ROOT / corpus["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "torch_version": corpus["torch_version"],
            "torch_installed": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "pytorch_docs_commit": corpus["torch_git_commit"],
            "tutorials_commit": corpus["tutorials_git_commit"],
            "docs_base_url": corpus["docs_base_url"],
            "tutorials_base_url": corpus["tutorials_base_url"],
        },
        "build_env": {"ragbench_git": _git_state(), "python": platform.python_version()},
        "docs": {
            "path": corpus["docs_path"],
            "count": sum(kinds.values()),
            "by_kind": dict(sorted(kinds.items())),
            "sha256": sha256_file(docs_path),
        },
        "chunking": {
            "tokenizer": retrieval["embedding_model"],
            "embedding_max_tokens": retrieval["embedding_max_tokens"],
            "overlap_tokens": retrieval["chunk_overlap_tokens"],
            "min_chunk_tokens": retrieval["min_chunk_tokens"],
            "files": [_chunk_file_stats(p) for p in chunk_files],
        },
        "ingest_report": {k: len(v) for k, v in report.items() if isinstance(v, list)},
    }
    out = PROJECT_ROOT / corpus["manifest_path"]
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest
