"""Split docs.jsonl into retrieval chunks with exact char offsets.

Rules (deterministic):
  - Chunks follow the doc's heading sections. Sections smaller than `min_tokens`
    (e.g. a heading-only parent like "## Containers") merge into the next section;
    a tiny last section merges into the previous one.
  - Inside a segment, text is split into blocks: paragraphs and whole code fences.
    Blocks are packed greedily up to the token budget; a block over budget is split
    into lines, and a line over budget into token windows.
  - Consecutive chunks overlap by whole trailing blocks totalling <= `overlap` tokens.
  - chunk.text == doc.text[char_start:char_end], always (gold spans rely on this).

Token counts use the embedder's own tokenizer, and the budget leaves room for the
context header ("title > section") and [CLS]/[SEP], so header + body fits the window.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class HFTokenCounter:
    """Token counting with a Hugging Face `tokenizers` tokenizer (no special tokens)."""

    def __init__(self, model_name: str):
        from tokenizers import Tokenizer

        self.name = model_name
        self._tok = Tokenizer.from_pretrained(model_name)
        self._tok.no_truncation()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._tok.encode(text, add_special_tokens=False).offsets)


@dataclass(frozen=True)
class Unit:
    start: int  # absolute char offsets into doc text
    end: int
    tokens: int


# ---------------------------------------------------------------- splitting


def _blocks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraphs and whole code fences within text[start:end], as absolute spans."""
    spans: list[tuple[int, int]] = []
    pos = start
    cur: int | None = None  # start of the block in progress
    last = start  # end of the last non-blank line of that block
    fence_tok: str | None = None
    for line in text[start:end].split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1
        f = _FENCE.match(line)
        if fence_tok is not None:  # inside a fence: blank lines don't end it
            last = le
            if f and f.group(1)[0] == fence_tok[0] and len(f.group(1)) >= len(fence_tok):
                spans.append((cur, le))
                cur, fence_tok = None, None
            continue
        if f:  # an opening fence ends any paragraph in progress
            if cur is not None:
                spans.append((cur, last))
            cur, last, fence_tok = ls, le, f.group(1)
            continue
        if line.strip():
            if cur is None:
                cur = ls
            last = le
        elif cur is not None:
            spans.append((cur, last))
            cur = None
    if cur is not None:  # paragraph or unterminated fence at the end
        spans.append((cur, last))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans, pos = [], start
    for line in text[start:end].split("\n"):
        if line.strip():
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _units(text: str, start: int, end: int, budget: int, tok: TokenCounter) -> list[Unit]:
    """Blocks that fit the budget; oversize blocks broken into lines, then token windows."""
    units: list[Unit] = []
    for bs, be in _blocks(text, start, end):
        n = tok.count(text[bs:be])
        if n <= budget:
            units.append(Unit(bs, be, n))
            continue
        for ls, le in _lines(text, bs, be):
            n = tok.count(text[ls:le])
            if n <= budget:
                units.append(Unit(ls, le, n))
                continue
            # Equal-size windows (not full ones), leaving slack so a short lead-in
            # like "Example:" can still merge with the first window.
            offs = tok.offsets(text[ls:le])
            size = math.ceil(len(offs) / math.ceil(len(offs) / budget))
            for i in range(0, len(offs), size):
                window = offs[i : i + size]
                units.append(Unit(ls + window[0][0], ls + window[-1][1], len(window)))
    return units


def _pack(units: list[Unit], budget: int, overlap: int) -> list[tuple[int, int, int]]:
    """Greedy packing → (start_idx, end_idx_exclusive, tokens) over units."""
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and total + units[j].tokens <= budget:
            total += units[j].tokens
            j += 1
        if j == i:  # cannot happen after _units, but never loop forever
            j, total = i + 1, units[i].tokens
        out.append((i, j, total))
        if j >= len(units):
            break
        k, ov = j, 0
        while k - 1 > i and ov + units[k - 1].tokens <= overlap:
            k -= 1
            ov += units[k].tokens
        i = k
    return out


def _merge_tiny(
    packed: list[tuple[int, int, int]],
    units: list[Unit],
    text: str,
    tok: TokenCounter,
    budget: int,
    min_tokens: int,
) -> list[tuple[int, int, int]]:
    """Merge chunks under min_tokens into the next chunk (a lead-in like "Example:"
    belongs with what follows), else into the previous one, if the union fits."""
    out = list(packed)

    def union(x: int, y: int) -> tuple[int, int, int]:
        a, b = out[x][0], out[y][1]
        return a, b, tok.count(text[units[a].start : units[b - 1].end])

    i = 0
    while i < len(out):
        if out[i][2] < min_tokens and len(out) > 1:
            if i + 1 < len(out) and (m := union(i, i + 1))[2] <= budget:
                out[i : i + 2] = [m]
                continue
            if i > 0 and (m := union(i - 1, i))[2] <= budget:
                out[i - 1 : i + 1] = [m]
                continue
        i += 1
    return out


# ---------------------------------------------------------------- segments


def _segments(
    doc: dict[str, Any], tok: TokenCounter, min_tokens: int
) -> list[tuple[int, int, dict[str, Any]]]:
    """Merge tiny sections into neighbours. Returns (start, end, representative_section)."""
    text = doc["text"]
    sized = [(s, tok.count(text[s["char_start"] : s["char_end"]])) for s in doc["sections"]]
    sized = [(s, n) for s, n in sized if text[s["char_start"] : s["char_end"]].strip()]
    segs: list[list[Any]] = []  # [start, end, [(section, tokens), ...]]
    pending: list[Any] | None = None
    for sec, n in sized:
        if pending is None:
            pending = [sec["char_start"], sec["char_end"], [(sec, n)]]
        else:
            pending[1] = sec["char_end"]
            pending[2].append((sec, n))
        if sum(t for _, t in pending[2]) >= min_tokens:
            segs.append(pending)
            pending = None
    if pending is not None:
        if segs:  # tiny tail → previous segment
            segs[-1][1] = pending[1]
            segs[-1][2].extend(pending[2])
        else:
            segs.append(pending)
    return [(s, e, max(parts, key=lambda p: p[1])[0]) for s, e, parts in segs]


# ---------------------------------------------------------------- public API


def context_header(doc: dict[str, Any], section: dict[str, Any]) -> str:
    title, sec = doc["title"], section.get("title") or ""
    return f"{title} > {sec}" if sec and sec != title else title


def chunk_doc(
    doc: dict[str, Any],
    tok: TokenCounter,
    *,
    chunk_size: int,
    overlap: int,
    min_tokens: int,
) -> list[dict[str, Any]]:
    text = doc["text"]
    chunks: list[dict[str, Any]] = []
    for seg_start, seg_end, section in _segments(doc, tok, min_tokens):
        context = context_header(doc, section)
        budget = max(chunk_size - tok.count(context) - 2, chunk_size // 2)
        units = _units(text, seg_start, seg_end, budget, tok)
        packed = _merge_tiny(_pack(units, budget, overlap), units, text, tok, budget, min_tokens)
        for a, b, _ in packed:
            start, end = units[a].start, units[b - 1].end
            body = text[start:end]
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}@{start}-{end}",
                    "doc_id": doc["doc_id"],
                    "kind": doc["kind"],
                    "title": doc["title"],
                    "object_name": doc.get("object_name"),
                    "section_title": section.get("title") or doc["title"],
                    "section_anchor": section.get("anchor", ""),
                    "url": section.get("url", doc["url"]),
                    "char_start": start,
                    "char_end": end,
                    "n_tokens": tok.count(body),
                    "context": context,
                    "text": body,
                    "chunk_size": chunk_size,
                    "torch_version": doc.get("torch_version"),
                }
            )
    return chunks


def chunk_docs(
    docs: list[dict[str, Any]],
    tok: TokenCounter,
    *,
    chunk_size: int,
    overlap: int,
    min_tokens: int,
    max_window: int,
) -> list[dict[str, Any]]:
    if chunk_size > max_window:
        raise ValueError(
            f"chunk_size {chunk_size} exceeds the embedder window ({max_window} tokens); "
            "chunks would be truncated when embedded."
        )
    out: list[dict[str, Any]] = []
    for doc in docs:
        out.extend(
            chunk_doc(doc, tok, chunk_size=chunk_size, overlap=overlap, min_tokens=min_tokens)
        )
    return out
