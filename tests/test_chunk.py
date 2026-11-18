"""Chunker tests with a whitespace tokenizer (1 token per whitespace-separated word)."""

from __future__ import annotations

import re

import pytest

from ragbench.ingest.chunk import chunk_doc, chunk_docs
from ragbench.ingest.manifest import sha256_file
from ragbench.ingest.sections import find_sections


class WordTokenizer:
    def count(self, text: str) -> int:
        return len(text.split())

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]


TOK = WordTokenizer()


def make_doc(text: str, doc_id: str = "page:x") -> dict:
    url = "https://example/x.html"
    sections = [{**s, "url": f"{url}#{s['anchor']}"} for s in find_sections(text)]
    return {"doc_id": doc_id, "kind": "page", "title": "X", "url": url, "text": text,
            "sections": sections, "object_name": None, "torch_version": "2.15.0"}  # fmt: skip


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def run(doc, size=40, overlap=0, min_tokens=5):
    return chunk_doc(doc, TOK, chunk_size=size, overlap=overlap, min_tokens=min_tokens)


def test_chunks_are_exact_slices_and_within_budget():
    text = "# A\n\n" + "\n\n".join(words(12, f"p{i}_") for i in range(8)) + "\n\n# B\n\n" + words(9)
    doc = make_doc(text)
    chunks = run(doc)
    for c in chunks:
        assert c["text"] == text[c["char_start"] : c["char_end"]]
        assert c["chunk_id"] == f"page:x@{c['char_start']}-{c['char_end']}"
        assert c["n_tokens"] + TOK.count(c["context"]) + 2 <= 40
    # every non-space char is covered
    covered = set()
    for c in chunks:
        covered.update(range(c["char_start"], c["char_end"]))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_no_chunk_crosses_a_section():
    text = "# A\n\n" + words(20, "a") + "\n\n# B\n\n" + words(20, "b")
    chunks = run(make_doc(text), size=100)
    assert len(chunks) == 2
    assert all(("a0" in c["text"]) != ("b0" in c["text"]) for c in chunks)
    assert [c["section_anchor"] for c in chunks] == ["a", "b"]
    assert chunks[1]["url"].endswith("#b")


def test_heading_only_section_merges_into_next():
    text = "# Parent\n\n## Child\n\n" + words(20)
    chunks = run(make_doc(text), size=100)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("# Parent")
    assert chunks[0]["section_anchor"] == "child"  # the section holding the content


def test_code_fence_kept_whole():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(5)) + "\n\n\ny = 2\n```"
    text = "# A\n\n" + words(25) + "\n\n" + code + "\n\n" + words(5, "z")
    chunks = run(make_doc(text), size=40)
    fence_chunks = [c for c in chunks if "```python" in c["text"]]
    assert len(fence_chunks) == 1 and fence_chunks[0]["text"].count("```") == 2


def test_oversize_block_split_by_lines_then_tokens():
    text = "# A\n\n" + words(100)  # one 100-word line, budget ~36
    chunks = run(make_doc(text), size=40)
    assert len(chunks) >= 3
    assert all(c["n_tokens"] <= 40 for c in chunks)


def test_overlap_repeats_trailing_block():
    paras = [words(10, f"p{i}_") for i in range(6)]
    text = "# A\n\n" + "\n\n".join(paras)
    chunks = run(make_doc(text), size=40, overlap=12)
    assert len(chunks) >= 2
    assert chunks[0]["char_end"] > chunks[1]["char_start"]  # overlapping spans


def test_tiny_tail_merges():
    text = "# A\n\n" + words(30, "a") + "\n\n" + words(2, "t")
    chunks = run(make_doc(text), size=40, min_tokens=5)
    assert len(chunks) == 1 and "t1" in chunks[0]["text"]


def test_deterministic():
    text = "# A\n\n" + "\n\n".join(words(7, f"p{i}_") for i in range(10))
    assert run(make_doc(text), overlap=8) == run(make_doc(text), overlap=8)


def test_rejects_chunk_size_over_window():
    with pytest.raises(ValueError, match="exceeds the embedder window"):
        chunk_docs([], TOK, chunk_size=1024, overlap=0, min_tokens=5, max_window=512)


def test_sha256_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert sha256_file(a) == sha256_file(b)


def test_lead_in_merges_with_following_long_line():
    # Budget = 40 - 3 (context "X > A") - 2 = 35. Heading (2) + 33 words fill chunk 1 exactly,
    # so "Example:" can't join it and must merge forward into the first window of the long
    # line (60 words → two equal 30-word windows, leaving slack for the label).
    text = "# A\n\n" + words(33, "a") + "\n\nExample:\n\n" + words(60, "x")
    chunks = run(make_doc(text), size=40, min_tokens=5)
    assert all(c["n_tokens"] >= 5 for c in chunks)
    assert any(c["text"].startswith("Example:") and "x0" in c["text"] for c in chunks)
