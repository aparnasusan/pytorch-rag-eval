"""Fetch PyTorch docs sources with a sparse, shallow git checkout at a pinned commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

PYTORCH_REPO = "https://github.com/pytorch/pytorch"
TUTORIALS_REPO = "https://github.com/pytorch/tutorials"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def fetch_sparse(
    dest: Path, repo: str, commit: str, paths: list[str], *, force: bool = False
) -> Path:
    """Check out only `paths` of `repo` at `commit` into `dest` (no-op if already there)."""
    if (dest / ".git").exists() and not force:
        try:
            if _git(dest, "rev-parse", "HEAD") == commit and all(
                (dest / p.strip("/")).exists() for p in paths
            ):
                return dest
        except subprocess.CalledProcessError:
            pass
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(dest, "init", "-q")
        _git(dest, "remote", "add", "origin", repo)
    _git(dest, "sparse-checkout", "set", "--no-cone", *[f"/{p.strip('/')}/" for p in paths])
    _git(dest, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    _git(dest, "checkout", "-q", "FETCH_HEAD")
    return dest


def fetch_pytorch_docs(raw_dir: Path, commit: str) -> Path:
    """Return the path to docs/source of pytorch/pytorch at `commit`."""
    repo_dir = fetch_sparse(raw_dir / "pytorch", PYTORCH_REPO, commit, ["docs/source"])
    return repo_dir / "docs" / "source"


def fetch_tutorials(raw_dir: Path, commit: str) -> Path:
    """Return the path to beginner_source/ of pytorch/tutorials at `commit`."""
    repo_dir = fetch_sparse(raw_dir / "tutorials", TUTORIALS_REPO, commit, ["beginner_source"])
    return repo_dir / "beginner_source"
