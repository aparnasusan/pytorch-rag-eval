# Project Log — PyTorch Docs Assistant + Eval System

A running record of **what was built, what was decided (and why), and what went wrong and how it was fixed**.
Written so I can explain any part of the project in an interview.
`PLAN.md` = the checklist and short Change Log; **this file = the full story**.

**How to update:** after each work session, add to *Timeline*, update *What's Built*, and add any new
decision (with the alternatives considered) or problem/fix. Newest entries at the bottom of each section.

---

## 1. Elevator pitch (keep current)
A retrieval-augmented assistant that answers PyTorch questions with citations to the official docs, plus an
evaluation system that measures it on real Stack Overflow questions. It's built to answer three questions:
how good is the assistant, which changes make it better, and how to stop it from getting worse. Rigor:
a locked test set, an LLM judge checked against human labels (Cohen's kappa), confidence intervals,
and CI regression gates.

**Status (2026-11-17):** Phases 0 and 1 done. The corpus (docs, notes, API docstrings, 22 tutorials) is
chunked at 128/256/512 tokens and builds deterministically with a manifest. Next: Phase 2, the
baseline RAG, which needs the generator model choice.

---

## 2. Timeline

### 2026-11-17 — Plan review
- Reviewed `PLAN.md` and found design flaws before writing any code (see Decisions D1–D6).
- Biggest catch: **the gold labels would have broken the chunk-size experiment.** The gold labels were
  chunk IDs, but Phase 5 re-chunks the corpus, so those IDs would stop matching. Switched to
  chunking-independent text spans.

### 2026-11-17 — Phase 0: Setup
- Git repo, `src/` layout, `pyproject.toml` with optional dependency groups per phase,
  `config.yaml`, `.env.example`.
- `src/ragbench/llm.py`: a wrapper that works with any provider (LiteLLM).
- Ruff for lint and format; pytest with LiteLLM replaced by a stub, so tests need no API keys.
- Commit `d72b93b`.

### 2026-11-17 — PyTorch version pinned
- Pinned torch **2.15.0**, the latest stable release on PyPI (published 2026-10-28).
- Checked the git tag before pinning (see Problem P1).

### 2026-11-17 — Phase 1: Docs corpus
- Sparse checkout of `pytorch/docs/source` at the pinned commit.
- Wrote a parser for MyST Markdown and RST, a docstring renderer, and section and anchor extraction.
- Output: `data/corpus/docs.jsonl` with 1,638 docs (8 pages, 1,606 API objects, 24 notes),
  about 2.26M characters, 0 unresolved names. 22 tests.
- Commit `f1fca30`.

### 2026-11-17 — Tutorial selection driven by the question data
- Pulled all 10.7k Stack Overflow questions (in memory only) and counted topics with keyword matching
  to choose which tutorials to index (see D9).
- Also checked whether chunk sizes fit the embedder's input limit (see D10).

### 2026-11-17 — Phase 1 finished: tutorials, chunker, manifest
- Planned the remaining steps first (plan mode), with two decisions made from data: the tutorial list
  (D9) and the chunk sizes (D10).
- Tutorials: sparse checkout of `pytorch/tutorials/beginner_source` @ `5277794`, and a sphinx-gallery
  parser (`ingest/gallery.py`). All 22 tutorials are in the corpus.
- Chunker (`ingest/chunk.py`): section-aware, exact character offsets, token counts from the
  embedder's own tokenizer (D13–D16).
- Manifest (`ingest/manifest.py`): sources, commits, counts, token stats, sha256 of every output file.
- One entry point: `python -m ragbench.ingest [--chunk-size N] [--skip-docs]`, about 12 s for a full build.
- Verified: 0 chunks differ from their slice of the source doc, 0 exceed the embedder window, 100% of
  the text is covered, and two full rebuilds give identical checksums (after fixing P9).
- 35 tests.

---

## 3. What's built

| Component | File(s) | What it does | Key detail to mention |
|---|---|---|---|
| Config | `src/ragbench/config.py`, `config.yaml` | Loads YAML settings and the `.env` file | One file controls every pipeline setting; experiment configs override it |
| LLM wrapper | `src/ragbench/llm.py` | `complete()` / `acomplete()` through LiteLLM | Separate settings for the generator and the judge; returns tokens, cost and latency; LiteLLM is imported lazily so tests can stub it |
| Fetch | `ingest/fetch.py` | Sparse, shallow git checkout at a pinned commit | Downloads 22 MB of docs instead of the full PyTorch repo |
| Page parser | `ingest/pages.py` | MyST + RST pages → text + the list of documented objects | Follows `currentmodule` the way Sphinx does |
| RST cleaner | `ingest/rst.py` | RST → plain text with headings and code blocks | Handles roles that wrap across lines; `Shape::` treated as prose |
| API renderer | `ingest/api.py` | Name → live object → docstring text | Classes include only their own members (no inherited copies) |
| Sections | `ingest/sections.py` | Headings → anchors and character offsets | Ignores `#` lines inside code blocks |
| Build | `ingest/build_docs.py` | Writes `docs.jsonl` and `ingest_report.json` | Drops 154 objects that have no docstring |
| Tutorial parser | `ingest/gallery.py` | sphinx-gallery `.py` → text with headings and code | Turns the whole script into one RST document so heading levels match across blocks; drops hidden code and nav breadcrumbs |
| Chunker | `ingest/chunk.py` | `docs.jsonl` → `chunks*.jsonl` | Never splits a code block unless it's too big; `chunk.text` is always the exact slice `doc.text[start:end]`; refuses sizes above the embedder window |
| Manifest | `ingest/manifest.py` | `manifest.json` | Records exactly what the corpus was built from; checksums prove two builds are identical |
| Entry point | `ingest/__main__.py` | `python -m ragbench.ingest` | Docs → chunks → manifest in one command |
| Tests | `tests/test_llm.py`, `tests/test_ingest.py`, `tests/test_chunk.py` | 35 unit tests | No network or API keys; the chunker tests use a word-count tokenizer so the expected results are easy to work out by hand |

**Corpus record:** `doc_id`, `kind` (page / note / api / tutorial), `title`, `url`, `source`, `object_name`,
`text`, `sections[{title, anchor, url, char_start, char_end}]`.

**Chunk record:** `chunk_id` (`doc_id@start-end`), `doc_id`, `kind`, `title`, `object_name`,
`section_title`, `section_anchor`, `url` (link to the section), `char_start`, `char_end`, `n_tokens`,
`context` (`"title > section"`, added in front of the text at embedding time), `text`, `chunk_size`.

---

## 4. Decisions (with alternatives considered)

**D1. Gold evidence = doc spans, not chunk IDs.**
- *Problem:* the Phase 5 chunk-size sweep re-chunks the corpus, which would make chunk-ID labels invalid.
- *Decision:* store `{doc_id, char_start, char_end}`. A retrieved chunk is relevant if it overlaps a gold span.
- *Alternative:* re-label for every chunking. Rejected: too much labor and inconsistent across runs.

**D2. CI gates split into two levels.**
- *Problem:* with a 30-question smoke set, one question is about 3.3 points, so a "≤ 2-point drop" gate
  would fail on noise alone.
- *Decision:* the pull-request gate checks hard floors (valid JSON, citation validity, failure rate) and
  large drops only. A nightly run on the full dev set does paired comparisons and flags a drop only when
  the 95% confidence interval on the difference excludes zero.

**D3. Kappa measured per nugget.**
- Cohen's kappa needs categories, but nugget coverage is a fraction.
- Labeling "covered / not covered" per nugget turns 50 answers into about 150–200 labels, a much
  tighter estimate.

**D4. Paired bootstrap for comparing configs.**
- Every config runs on the same questions, so per-question differences have far lower variance than
  comparing two independent averages.

**D5. Cheap judge for experiments, strong judge for validation and final numbers.**
- Keeps the cost of the experiments down.
- The top 2–3 configs are re-scored with the strong judge before picking a winner.

**D6. `outdated` label for old Stack Overflow answers.**
- The questions are mostly from 2018–2022, and their accepted answers can use APIs that are now
  deprecated, contradicting the pinned docs.

**D7. Use the docs pages to decide which API objects go in the corpus.**
- *Alternative:* walk modules with `dir()`. Rejected: it pulls in private and undocumented objects and
  doesn't match the website.
- *Decision:* parse each page's `autosummary` / `autoclass` / `autofunction` directives. This gives exactly
  what docs.pytorch.org documents, and the correct URL for each object:
  `generated/<name>.html` or `<page>.html#<name>`.
- The text still comes from the locally installed torch's docstrings, with no scraping.

**D8. Index the docs pages' own prose, not just docstrings.**
- For example, optim's "How to use an optimizer" section exists in no docstring.

**D9. Tutorials matched to the question data (22 files).**
- Keyword topic counts over 10.7k questions:

  | Topic | Share |
  |---|---|
  | Data / DataLoader | 30% |
  | Model building | 29% |
  | Vision / fine-tuning | 27% |
  | Tensor ops | 26% |
  | Device / CUDA | 17% |
  | Save / load | 13% |
  | Loss | 11% |
  | Install / environment | 10% |
  | Autograd | 10% |

- The chosen tutorials cover each core-API topic, for example the LSTM sequence models tutorial for
  input-shape questions and transfer learning for fine-tuning.
- Left out on purpose: install, TensorFlow/Keras, Lightning, Hugging Face, ONNX and distributed. Those
  questions become refusal tests or docs-gap cases.

**D10. Chunk sweep changed to 128 / 256 / 512 (was 256 / 512 / 1024).**
- `bge-small` reads at most 512 tokens, so 1024-token chunks would be silently cut off when embedded.
  That would also change what the embedder sees, not just the chunk size.
- Token counts use the embedder's own tokenizer, so "512" means what the model actually sees.

**D11. Pin the commit the package was built from (`d0f2af8`), not the release tag's commit.**
- See Problem P1.

**D12. Commit-based pinning for tutorials.**
- `pytorch/tutorials` has only CI tags, so pin `5277794` and record it in the manifest.

**D13. Count tokens with the embedder's own tokenizer.**
- *Alternatives:* a word count, or tiktoken. Rejected: neither matches what bge-small sees, so a
  "512-token" chunk could still be cut off when embedded.
- Used the lightweight `tokenizers` package (no `transformers` dependency). The budget also reserves
  room for the context header and the two special tokens ([CLS]/[SEP]).

**D14. Section-aware chunking with whole blocks.**
- Chunks never cross a heading section, so every chunk has one precise citation URL.
- Paragraphs and code blocks are kept whole; only a block that's too big gets split, first by lines
  and then by token windows.
- Sections under 50 tokens (such as a heading with nothing under it before a subsection) merge into the
  next section, so the corpus has no chunks that are only a heading.
- Overlap is made of whole trailing blocks (≤ 64 tokens) rather than cutting mid-sentence.

**D15. `chunk_id = doc_id@char_start-char_end`.**
- Stable across rebuilds, readable, and never collides between chunk sizes. It also says exactly which
  text it covers, which ties into gold spans (D1).

**D16. Keep the context header separate from the chunk text.**
- `chunk.text` stays the exact slice of the source doc, so overlap with gold spans is exact. The
  `"title > section"` header is stored in its own field and added only when embedding.
- Used an ASCII `>` rather than `›`, which the linter flags as ambiguous and the Windows console
  can't print.

---

## 5. Problems hit and how they were solved (good interview stories)

**P1. Release tag ≠ the commit the package was built from.**
- `torch.version.git_version` reported `d0f2af8`, but the tag `v2.15.0` points to `31a77fc`.
- Checked the tags: they're lightweight tags, and the wheel was built from `v2.15.0-rc10`.
- Used the GitHub compare API: the final tag adds one commit that only changes a lint setting in
  `pyproject.toml`.
- Pinned `d0f2af8`, the exact code the docstrings come from.

**P2. The plan assumed `.rst` docs, but PyTorch moved to MyST Markdown.**
- 251 `.md` files versus 6 `.rst` files.
- Wrote a MyST parser that hands the embedded `{eval-rst}` blocks to the RST cleaner.

**P3. Git Bash rewrote `/docs/source` into `C:/Program Files/Git/docs/source`.**
- The sparse checkout came back empty without any error.
- Fixed with `MSYS_NO_PATHCONV=1`; in the code, git is called from Python so the shell never touches
  the paths.

**P4. `torch.torch.default_generator`.**
- The name resolver tried "current module + name" first, and `torch.torch` happens to exist.
- Fix: names that already start with the package name are tried as absolute first.

**P5. Inherited docstrings duplicated everywhere.**
- `inspect.getdoc` falls back to the parent class's docstring, so `Module.forward`'s text would have
  been copied onto hundreds of layers.
- Fix: read each member's own `__doc__`.

**P6. Regex merged two inline-code spans.**
- "`Iterable` with `__len__`" came out as "`Iterable with len__`".
- Cause: the RST reference-link pattern matched from the closing backtick of one span to the opening
  backtick of the next.
- Fix: the pattern now requires the reference to stand alone as a token. A regression test was added.

**P7. Roles wrapping across lines.**
- `:math:` and `:ref:` markup left in 19 docs because the cleaner worked line by line.
- Fix: clean whole paragraphs, outside code blocks only.

**P8. `Shape::` treated as code.**
- In RST, a line ending in `::` normally starts a code block. PyTorch's docs build reads `Shape::` as a
  section header and renders the math.
- Fix: a list of known section headers is rendered as prose. `Example::` stays code.

**P9. The corpus build wasn't deterministic.**
- The manifest checksum for `docs.jsonl` changed between two identical rebuilds.
- Diffed the two builds: only 4 docs differed (the `nn.Transformer*` classes and `random_split`).
- Cause: signature defaults that are functions or objects print with a memory address, e.g.
  `activation=<function relu at 0x7ff…>`, and the address changes every run.
- Fix: strip `at 0x…` from signatures (also more readable). Two full rebuilds now give identical
  sha256s for the docs and all three chunk files.
- Why it matters: checksums are how a later run proves it used the same corpus. Without this fix they
  would always look different.

**P10. Tiny chunks at small sizes (`Example:`, `` ```python ``).**
- Size 512 had none, but size 128 had 108 chunks under 10 tokens.
- Cause: a line longer than the budget was cut into full-size token windows, which left no room for a
  short lead-in to join the next chunk, so the lead-in became its own chunk.
- Fix: cut long lines into equal-size windows (leaving slack), and merge any chunk under
  `min_tokens` into the next chunk, or else the previous one, when the result fits. Down to 12 at
  size 128 and 1 at size 256.
- A test that first failed was checking the wrong thing: "Example:" had validly joined the previous
  chunk. Rewrote the test so the previous chunk is exactly full, which forces the forward merge.

**P11. Windows console encoding.**
- Printing `→` crashed the pipeline (`cp1252` can't encode it), and `’` showed up as `�`.
- Checked the raw bytes: the corpus file was correct UTF-8; only the console display was wrong.
  Changed terminal messages to ASCII.

**Verification habit:** after every build I spot-check real output (Linear, DataLoader, NLLLoss, the
optim page, the reproducibility note) and scan for leftover markup. This found P4, P6, P7 and P8.
For the chunker, I check invariants over the *whole* output rather than samples: every chunk equals
its slice of the source, nothing exceeds the window, coverage is 100%, and a rebuild gives the same
checksums. This found P9 and P10.

---

## 6. Numbers so far
| Metric | Value |
|---|---|
| Corpus docs | 1,660 (8 pages, 1,606 API objects, 24 notes, 22 tutorials) |
| Corpus size | ~2.47M chars |
| Unresolved API names | 0 |
| Dropped (no docstring) | 154 |
| Chunks @512 | 3,243 (tokens min 12 / median 223 / max 507) |
| Chunks @256 | 5,171 (median 185) |
| Chunks @128 | 12,638 (median 99) |
| Chunk invariants | 0 slice mismatches, 0 over the window, 100% text coverage |
| Full build time | ~12 s (after the one-time downloads) |
| SO dataset | 10,763 Q&As; median question ≈ late 2020 |
| Tests | 35 passing |

---

## 7. Open decisions
- Generator model and provider (needed by Phase 2). The judge must come from a different model family.
- Generator temperature, which decides what the three repeated runs measure.
- Results tracking: SQLite or MLflow.
- Whether to drop or keep outdated questions.

---

## 8. Interview talking points
- **Why a locked test set:** tuning on test data inflates results. The test set is run once, on the
  final config.
- **Why trust the LLM judge:** it comes from a different model family than the generator, runs at
  temperature 0, returns validated output, and is checked against human labels with per-nugget kappa.
- **Why spans for gold labels:** they let the chunking experiment run without re-labeling (D1).
- **Statistics:** paired bootstrap confidence intervals, and no claimed wins inside the noise (D2, D4).
- **Data provenance:** pinned to the exact commit the package was built from, with sparse checkout
  and checksums (P1).
- **Debugging rigor:** problems P2–P8 were found by inspecting real output, not assumed fixed.
- **Scoping with data:** the tutorial choice came from measuring the question topics (D9).
- **Reproducibility:** pinned commits, a manifest with checksums, and a deterministic build proven by
  rebuilding twice. A non-determinism bug (P9) was found by checking exactly that.
- **Chunking design:** section-aware, exact offsets, embedder-tokenizer budgets, and sizes kept inside
  the embedder window so the chunk-size experiment changes only one variable (D10, D13–D16).
