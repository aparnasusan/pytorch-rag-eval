# CLAUDE.md

RAG assistant over the PyTorch docs + an eval system that measures it on real Stack Overflow
questions. `PLAN.md` is the source of truth for scope, phases, open decisions and the Change Log.

## Working rules
- **Keep `PLAN.md` current:** tick `[x]` when a task is done and add a Change Log row (date + what/why)
  for every meaningful change or decision. Don't silently change a plan decision.
- **Keep `PROJECT_LOG.md` current** (the interview-reference record): after each session add to its
  Timeline, update *What's Built* / *Numbers*, and record every decision (with alternatives considered)
  and every problem → root cause → fix.
- **Locked test set:** never run experiments, tune, or look at results on `benchmark/v1/test.jsonl`.
  All experiments use the dev split; the test split is run once on the final config.
- **Pinned versions:** corpus is built from torch **2.15.0** (wheel commit `d0f2af8`, docs URLs
  `https://docs.pytorch.org/docs/2.15/`). Change the pin only as a recorded decision.
- **Gold evidence is spans, not chunk IDs:** `{doc_id, char_start, char_end}` into `docs.jsonl` text.
  A chunk is relevant if it overlaps a gold span. Never break `chunk.text == doc.text[char_start:char_end]`.
- **Judge ≠ generator family.** Both models are set per role in `config.yaml` (`llm.generator`, `llm.judge`).
- Commit only when asked; branch `main`.

## Commands (Windows; use the venv's Python)
```
.\.venv\Scripts\python.exe -m pytest -q              # tests (LiteLLM stubbed; `live` marker skipped)
.\.venv\Scripts\python.exe -m ruff format .          # format
.\.venv\Scripts\python.exe -m ruff check .           # lint
.\.venv\Scripts\python.exe -m ragbench.ingest             # full corpus: docs -> chunks -> manifest (~12 s)
.\.venv\Scripts\python.exe -m ragbench.ingest --skip-docs --chunk-size 256   # extra size for the sweep
```
Install extras per phase: `pip install -e ".[dev]"`, `".[ingest]"` (needs
`--extra-index-url https://download.pytorch.org/whl/cpu` for CPU torch), `".[rag]"`, `".[collect]"`,
`".[eval]"`, `".[app]"`.

## Layout
- `src/ragbench/config.py` — loads `config.yaml` (override with `RAGBENCH_CONFIG`) and `.env`.
- `src/ragbench/llm.py` — the only way to call an LLM: `complete()` / `acomplete()` (LiteLLM) return
  text, tokens, cost, latency. Model strings are LiteLLM format (`anthropic/…`, `openai/…`, `ollama/…`).
- `src/ragbench/ingest/` — corpus build:
  - `fetch.py` sparse git checkouts at pinned commits (into `data/raw/`)
  - `pages.py` MyST/`.rst` page parser; collects objects from `autosummary`/`auto*` directives, tracks `currentmodule`
  - `rst.py` RST → clean text (headings, ``` fences, admonitions; strips roles)
  - `api.py` resolves names → live torch objects, renders docstrings (classes: own members only)
  - `sections.py` heading sections with anchors + char offsets (ignores headings inside code)
  - `gallery.py` sphinx-gallery tutorial `.py` → text (whole script → one RST doc → `clean_rst`)
  - `build_docs.py` writes `data/corpus/docs.jsonl` + `ingest_report.json` (pages, notes, API, tutorials)
  - `chunk.py` section-aware chunker; bge-small tokenizer budgets; `chunk_id = doc_id@start-end`
  - `manifest.py` writes `data/corpus/manifest.json` (sources, commits, counts, sha256s)
  - `__main__.py` the `python -m ragbench.ingest` entry point
- `src/ragbench/{rag,collect,evals}/` — Phases 2–4 (not built yet).
- `benchmark/v1/` — dev/test JSONL + `CHANGELOG.md` (every label fix versioned with a reason).
- `experiments/` one config per experiment · `reports/` run outputs · `app/` labeling tool + demo.

## Data
- `data/` and `reports/` are git-ignored and fully rebuildable from scripts + pinned commits.
- `docs.jsonl` record: `doc_id` (`page:` / `note:` / `api:` / `tutorial:`), `kind`, `title`, `url`,
  `source`, `object_name`, `page`, `torch_version`, `text`, `sections[{title, level, anchor, url, char_start, char_end}]`.
- Stack Overflow content is CC BY-SA: attribute it; the published benchmark must be CC BY-SA 4.0.

## Gotchas
- PyTorch docs source is **MyST Markdown** (`.md` with ```` ```{eval-rst} ```` blocks), not `.rst`.
- Git Bash rewrites args starting with `/` into Windows paths (broke `git sparse-checkout set /docs/source`).
  Use `MSYS_NO_PATHCONV=1` or call git from Python (`fetch.py` does).
- `.gitattributes` forces LF so corpus checksums match across Windows and Linux CI.
- The project lives in a OneDrive folder; large caches under `data/` may sync slowly.
- **Keep builds deterministic:** two rebuilds must give identical sha256s in the manifest. Watch for
  memory addresses (`<function relu at 0x…>`), dict/set ordering, and timestamps leaking into outputs.
- Chunk sizes must stay ≤ `retrieval.embedding_max_tokens` (512 for bge-small); the chunker refuses larger.
- The Windows console is cp1252: keep terminal `print` output ASCII (files are UTF-8).
- Use `inspect.getdoc` for an object's docstring, but read members' **own** `__doc__`
  (getdoc falls back to inherited docs and duplicates e.g. `Module.forward` everywhere).
