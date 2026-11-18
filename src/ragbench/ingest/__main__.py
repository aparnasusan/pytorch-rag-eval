"""Build the corpus end to end: docs → chunks → manifest.

python -m ragbench.ingest                    # default chunk size from config
python -m ragbench.ingest --chunk-size 256   # extra size for the Phase 5 sweep
python -m ragbench.ingest --skip-docs        # re-chunk an existing docs.jsonl
"""

from __future__ import annotations

import argparse
import json

from ragbench.config import PROJECT_ROOT, get_config
from ragbench.ingest import build_docs
from ragbench.ingest.chunk import HFTokenCounter, chunk_docs
from ragbench.ingest.manifest import write_manifest


def main() -> None:
    cfg = get_config()
    corpus, retrieval = cfg["corpus"], cfg["retrieval"]
    parser = argparse.ArgumentParser(prog="python -m ragbench.ingest")
    parser.add_argument("--chunk-size", type=int, default=retrieval["chunk_size_tokens"])
    parser.add_argument("--skip-docs", action="store_true", help="reuse existing docs.jsonl")
    args = parser.parse_args()

    if not args.skip_docs:
        report = build_docs.build(cfg)
        print("docs:", json.dumps(report["counts"]))

    docs_path = PROJECT_ROOT / corpus["docs_path"]
    with docs_path.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    tok = HFTokenCounter(retrieval["embedding_model"])
    chunks = chunk_docs(
        docs,
        tok,
        chunk_size=args.chunk_size,
        overlap=retrieval["chunk_overlap_tokens"],
        min_tokens=retrieval["min_chunk_tokens"],
        max_window=retrieval["embedding_max_tokens"],
    )
    if args.chunk_size == retrieval["chunk_size_tokens"]:
        out = PROJECT_ROOT / corpus["chunks_path"]
    else:
        out = PROJECT_ROOT / corpus["chunks_path_sized"].format(size=args.chunk_size)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"chunks: {len(chunks)} -> {out.relative_to(PROJECT_ROOT).as_posix()}")

    manifest = write_manifest(cfg)
    for fs in manifest["chunking"]["files"]:
        print(
            f"  {fs['path']}: {fs['chunks']} chunks, tokens min/median/max "
            f"{fs['tokens_min']}/{fs['tokens_median']}/{fs['tokens_max']}"
        )
    print(f"manifest -> {corpus['manifest_path']}")


if __name__ == "__main__":
    main()
