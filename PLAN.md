# PyTorch Docs Assistant + Eval System — Project Checklist

**Goal:** A RAG assistant over the PyTorch docs, plus an eval system that measures it on real developer questions, drives measurable improvements through experiments, blocks regressions in CI, and produces public artifacts (benchmark, docs-gap report, live demo).

**Timeline:** ~5 weeks part-time (tight — Phase 3 labeling and Phase 7 are the likely overruns; items marked *(Optional)* are the first to cut) · **Est. cost:** ~$10–30 in LLM calls if sweeps use a cheap judge; more with a frontier judge throughout (embeddings/reranking run locally for free)

How to use this file: tick `[x]` when a task is done, and add a line to the **Change Log** at the bottom for every meaningful change or decision.

---

## Open Decisions
- [ ] LLM provider for generation (Claude / OpenAI / local Ollama) — needed by **Phase 2** (`ask()` needs a generator)
- [ ] Second provider for the judge (must be a different model family than the generator) — needed by Phase 4
- [ ] Judge tiers: cheap judge for Phase 5 sweeps, strong judge for kappa validation + final test run — decide in Phase 4
- [ ] Generator temperature (0 → repeats only measure provider nondeterminism; >0 → repeats measure sampling variance) — decide in Phase 4
- [ ] Outdated SO questions (answer relies on deprecated API): drop, or keep as tests with the current-docs answer — decide in Phase 3
- [ ] Results tracking: SQLite (simple) vs. MLflow (resume keyword) — decide in Phase 4

---

## Phase 0 — Setup (Week 1, day 1)
- [x] Initialize git repo and `.gitignore` (exclude `.env`, data caches, reports)
- [x] Create project layout (see **Project Structure** below)
- [x] `pyproject.toml` with dependencies
- [x] `config.yaml` for pipeline settings
- [x] `.env.example` for API keys (real `.env` never committed)
- [x] Provider-agnostic LLM wrapper `src/ragbench/llm.py` (LiteLLM; one config line switches Claude / OpenAI / Ollama)
- [x] Linter + formatter set up
- [x] Basic test scaffold (`pytest` runs green)

## Phase 1 — Corpus (Week 1, days 2–3)
**Approach: build from the PyTorch package + GitHub — no website downloading.**
- [x] Pick one PyTorch release and pin it everywhere (package version = git tag) — **2.15.0** / `v2.15.0`
- [x] Install the pinned CPU build (`pip install torch==<version>`, ~200 MB)
- [x] **API reference:** extract docstrings locally with Python's `inspect` (no web requests) — object list taken from the docs pages' `autosummary`/`auto*` directives, so the corpus matches what docs.pytorch.org documents
  - [x] `torch.Tensor` / tensor ops
  - [x] `torch.nn`
  - [x] `torch.optim`
  - [x] `torch.utils.data`
  - [x] `torch.autograd`
- [x] **Notes pages** (CUDA semantics, reproducibility, AMP, …) from `pytorch/pytorch` at the pinned commit — sparse checkout of `docs/source` (22 MB). Docs are now **MyST Markdown** (`.md` with `{eval-rst}` blocks), not `.rst`
- [ ] **Tutorials:** a few beginner tutorials from `pytorch/tutorials` repo (one git download) — mostly **sphinx-gallery `.py` files**, not `.rst`; needs its own parser (comment blocks → text, code → code blocks)
- [x] Clean docstrings / reStructuredText / MyST into plain text (keep code blocks and headings)
- [ ] Clean sphinx-gallery tutorial `.py` files
- [x] Generate the matching docs.pytorch.org URL for each item (for citations) — per-section URLs too
- [x] Store each source document's full cleaned text with a stable `doc_id` + section anchors (so gold evidence can be stored as spans independent of chunking)
- [ ] Chunker with metadata (source URL, `doc_id`, section, object name, chunk ID, **char start/end offsets into the source doc**)
- [ ] Corpus manifest (PyTorch version, git commits, item count, chunk count, checksum, build date)
- [ ] **Deliverable:** `data/corpus/docs.jsonl` (cleaned source docs) + `data/corpus/chunks.jsonl` (~2–5k chunks)

## Phase 2 — Baseline RAG (Week 1, days 4–5)
- [ ] Local embeddings (`bge-small`)
- [ ] Chroma index build script
- [ ] Top-k retriever
- [ ] Answer prompt returning structured JSON: `{answer, citations, answerable}`
- [ ] Common `Response` interface: answer, citations, retrieved chunks, tokens, cost, latency, error
- [ ] Per-question error capture (one failure never crashes a run)
- [ ] Query logging
- [ ] Baseline config saved in `experiments/baseline.yaml`
- [ ] **Deliverable:** `ask("…")` returns a cited answer

## Phase 3 — Real-Question Benchmark (Week 2)
- [ ] Load existing dataset [`shrinath-suresh/stack_overflow_pytorch`](https://huggingface.co/datasets/shrinath-suresh/stack_overflow_pytorch) (~10.8k Q&As: title, body, answer, URL)
- [ ] Enrich via Stack Exchange API using each row's URL (score, accepted status, date) — **register an app key** (~300 req/day without one) and **batch up to 100 IDs per request**
- [ ] Filter: accepted answer, high score, recent enough to match current docs
- [ ] *(Optional)* PyTorch forums collector (public Discourse JSON API) for source variety — **check forum terms first**; if redistribution isn't allowed, publish links/IDs only
- [ ] Sample ~500 candidate questions
- [ ] LLM pre-filter and pre-label
- [ ] Extract **nuggets** (3–4 key facts) from each accepted answer — FreshStack method
- [ ] Propose gold evidence automatically (doc spans that support a nugget), for human confirmation
- [ ] Streamlit labeling tool
- [ ] Human review down to ~150–200 questions, each with:
  - [ ] `case_id`, question, reference answer
  - [ ] nuggets (confirmed/edited)
  - [ ] **gold evidence spans** (confirmed): `{doc_id, char_start, char_end}` — **not chunk IDs**, so labels survive re-chunking. A chunk counts as relevant if it overlaps a gold span.
  - [ ] `answerable` flag (unanswerable → refusal tests)
  - [ ] `outdated` flag (accepted answer relies on deprecated/changed API vs. the pinned docs; handle per Open Decision)
  - [ ] slices: category (API usage / debugging / concepts / performance), source, docs section, length
- [ ] Split: **dev (~70%)** for experiments / **locked test (~30%)** for final numbers only
- [ ] Attribution for Stack Overflow content (CC BY-SA)
- [ ] `benchmark/v1/CHANGELOG.md` — every label fix versioned with a reason
- [ ] **Deliverable:** `benchmark/v1/dev.jsonl`, `benchmark/v1/test.jsonl`

## Phase 4 — Eval Harness (Weeks 2–3)
**Grading tiers:** deterministic scorers run first on every output (cheap, exact); LLM-judge scorers run second, using a cheap judge for sweeps and a strong judge for validation and final numbers.

**Runner**
- [ ] Async runner with capped concurrency, retries, timeouts
- [ ] Inference separated from scoring; outputs cached (re-grade for free)
- [ ] 3 repeated runs per question (mean + variance) — meaning depends on the generator-temperature decision

**Tier 1: deterministic scorers (run first)**
- [ ] Valid-JSON rate
- [ ] Citation validity (cited IDs exist among retrieved chunks)
- [ ] Retrieval recall@k and MRR (vs. gold evidence spans, via chunk–span overlap)
- [ ] Failure rate
- [ ] Cost per query, p50/p95 latency, tokens

**Tier 2: LLM-judge scorers**
- [ ] Judge from a different model family than the generator; temperature 0; Pydantic-validated output
- [ ] Answer correctness via **nugget coverage** (judge decides per nugget whether the answer covers it; score = fraction covered)
- [ ] Faithfulness (split into claims → fraction supported by retrieved chunks)
- [ ] Answer relevance
- [ ] Context precision
- [ ] Refusal accuracy (on unanswerable questions)

**Rigor**
- [ ] Hand-label 50 answers **at the nugget level** (covered / not covered → ~150–200 binary judgments); report Cohen's kappa vs. judge on those per-nugget labels (also per-claim for faithfulness if time allows)
- [ ] Set thresholds from the human labels
- [ ] Bootstrap 95% confidence intervals
- [ ] **Paired** comparisons between configs (same questions, cached outputs; paired bootstrap on per-question deltas)
- [ ] Per-slice breakdowns
- [ ] Run manifest in every report (model, prompt version, config, dataset version + checksum, git commit, package versions, timestamp)
- [ ] **Deliverable:** `python -m ragbench.evals run --config baseline --split dev` → scorecard + per-question results

**Headline metrics:** correctness (nugget coverage) · faithfulness · retrieval recall@5 · refusal accuracy · cost & latency

## Phase 5 — Experiments & Error Analysis (Week 3) — dev set only
Sweeps use the cheap judge; re-score the top 2–3 configs with the strong judge before picking a winner.
- [ ] Chunk size: 256 / 512 / 1024 tokens (gold spans make recall comparable across sizes)
- [ ] Hybrid search (BM25 + embeddings)
- [ ] Reranker on/off (`bge-reranker`)
- [ ] Top-k: 3 / 5 / 10
- [ ] Cheap vs. expensive generator model
- [ ] Refusal threshold tuning
- [ ] *(Optional)* local vs. API model
- [ ] Error taxonomy: retrieval miss / ranking error / hallucination / refusal error / docs gap / outdated question
- [ ] Comparison table + charts (with paired CIs; don't claim wins inside the noise)
- [ ] Pick the best config → run **once** on the locked test set
- [ ] **Deliverable:** experiment results + error-analysis write-up

## Phase 6 — Production Pieces (Week 4)
- [ ] 30-question smoke set
- [ ] `thresholds.yaml`
- [ ] Quality-gate test (pytest)
- [ ] **PR gate (smoke set, 30 Qs):** one question ≈ 3.3 points, so only catch large breaks:
  - [ ] Hard floors: valid JSON ≥ 99%, citation validity = 100%, failure rate ≤ 1%
  - [ ] Correctness drop beyond a large, noise-calibrated margin (set from repeated baseline runs, e.g. ≥ 10 points), paired vs. cached baseline
  - [ ] Cost within budget
- [ ] **Nightly / on-demand full dev-set gate** (fine-grained, baseline-relative, paired):
  - [ ] Correctness drops ≤ 2 points **and** the paired 95% CI on the delta excludes zero before flagging
  - [ ] p95 latency increases ≤ 15%
  - [ ] No slice gets significantly worse
  - [ ] Cost within budget
- [ ] GitHub Actions workflow on every PR (smoke gate)
- [ ] Nightly / on-demand full dev-set run (full gate)
- [ ] Regression report listing the questions that broke
- [ ] *(Optional)* Dashboard of runs over time
- [ ] **Deliverable:** a PR that clearly degrades quality is automatically blocked; subtle regressions are caught by the nightly run

## Phase 7 — Real-World Impact (Weeks 4–5)
- [ ] Live demo on Hugging Face Spaces (cited answers + thumbs up/down) — **rate limiting + daily spend cap** on the API key
- [ ] Online eval of demo queries (faithfulness + relevance, no reference needed)
- [ ] Feedback loop: reviewed thumbs-down failures → benchmark v2
- [ ] Docs-gap report (unanswerable real questions, grouped by topic)
- [ ] *(Optional)* Submit 1–2 small docs fixes to PyTorch on GitHub
- [ ] Publish benchmark on Hugging Face with a dataset card — **licensed CC BY-SA 4.0** (share-alike required by SO content); forum content only if its terms allow, otherwise links/IDs
- [ ] README: architecture diagram, results, findings, limitations
- [ ] README cites FreshStack (method) and the source Stack Overflow dataset + CC BY-SA attribution
- [ ] *(Optional)* cross-check faithfulness metric against RAGAS
- [ ] *(Optional)* blog post

## Final Wrap-Up
- [ ] Fill in the resume bullet with real test-set numbers
- [ ] Prepare interview talking points (dev/test split, judge validation, error taxonomy, CI gates)

---

## Project Structure
```
eval/
├── src/ragbench/
│   ├── ingest/        # extract docstrings, read repo docs, clean, chunk
│   ├── rag/           # retrievers, reranker, generator, Response interface
│   ├── collect/       # Stack Overflow + forum collection
│   ├── evals/         # scorers, judges, runner, stats, gates, manifest
│   └── llm.py         # provider-agnostic wrapper
├── benchmark/v1/      # dev.jsonl, test.jsonl, CHANGELOG.md
├── experiments/       # one config file per experiment
├── thresholds.yaml
├── reports/           # run outputs (per-question rows + manifest)
├── app/               # labeling tool + demo
├── tests/             # unit tests + quality-gate test
└── .github/workflows/eval.yml
```

## Tech Stack
Python 3.11+ · LiteLLM · `bge-small` embeddings · Chroma · `rank_bm25` · `bge-reranker` · Pydantic · JSONL + SQLite (or MLflow) · NumPy/SciPy · Streamlit · pytest · GitHub Actions · Hugging Face

## Risks
| Risk | Mitigation |
|---|---|
| Most SO questions are code debugging, not docs lookups | Collect ~500, keep ~150–200; unanswerable ones become refusal tests |
| Judge is wrong or biased | Different model family; validated against per-nugget human labels (kappa) |
| Numbers inflated by tuning on test data | Locked test set, used once |
| Labeling takes too long | LLM pre-labels; you confirm/correct in the labeling tool; forum collector optional |
| PyTorch docs change mid-project | Corpus built from a pinned release (package version + git tag), with checksum |
| Network blocking from scraping | No website scraping: docstrings read locally, notes/tutorials via standard git downloads |
| Gold labels invalidated by re-chunking | Gold evidence stored as doc spans; relevance = chunk–span overlap |
| CI gate flakes on small smoke set | Smoke gate only catches large breaks + hard floors; fine-grained paired gate runs nightly on full dev set |
| Old SO answers contradict pinned docs | `outdated` flag during labeling; explicit drop/keep decision |
| LLM cost overruns | Cached outputs; cheap judge for sweeps, strong judge only for validation/final |
| Licensing on published benchmark | CC BY-SA 4.0 for SO-derived data; forum content only if terms allow |
| Public demo burns API budget | Rate limiting + daily spend cap |

## Target Resume Bullet (fill in with real numbers)
> Built an evaluation system for a PyTorch documentation assistant on 150+ real developer questions (locked test set, LLM judge validated against human labels, κ = __). Raised answer accuracy from __% to __% and cut hallucinations __× through retrieval experiments, cut cost per query __× through model selection, and added a CI regression gate. Published the benchmark on Hugging Face and identified __ documentation gaps.

---

## Change Log
| Date | Change |
|---|---|
| 2026-11-17 | Plan created. Added from reading list: different-family judge, claim-level faithfulness, answer relevance, repeated runs, tiered grading, data-driven thresholds, dev/test split, baseline-relative CI gates, run manifest, common Response interface, JSON/citation checks, dataset changelog, demo feedback loop. |
| 2026-11-17 | Dataset search: no existing PyTorch-docs RAG benchmark (ours fills a gap). Phase 3 now starts from HF `shrinath-suresh/stack_overflow_pytorch` + Stack Exchange API enrichment instead of a custom SO collector. Adopted FreshStack nugget method: nuggets per answer, nugget-based gold-chunk proposals, nugget coverage as the correctness metric. README to cite both. |
| 2026-11-17 | Docs dataset search: no usable ready-made PyTorch docs corpus on HF. Phase 1 switched from website downloading to package + GitHub: API docstrings extracted locally from a pinned `torch` install, notes from `pytorch/pytorch` at the matching tag, tutorials from `pytorch/tutorials`. Removes network-blocking risk and gives an exact version match. |
| 2026-11-17 | Plan review fixes. **Gold labels:** stored as doc spans `{doc_id, char_start, char_end}` instead of chunk IDs so the Phase 5 chunk-size sweep doesn't invalidate them; chunks carry char offsets; corpus adds `docs.jsonl`. **CI gates:** split into a coarse PR smoke gate (hard floors + large noise-calibrated drop) and a fine-grained paired nightly gate on the full dev set (2-point rule only when the paired CI excludes zero). **Kappa:** measured per nugget, not on fractional scores. **Stats:** paired bootstrap for config comparisons. **New open decisions:** generator temperature (what repeats measure), judge tiers (cheap for sweeps / strong for validation + final), outdated-SO-question handling (new `outdated` label + error-taxonomy class). **Fixes:** generator provider decision moved to Phase 2; tiered grading made explicit in Phase 4. **Practical:** sparse checkout for `pytorch/pytorch` notes; sphinx-gallery `.py` parser for tutorials; Stack Exchange app key + 100-ID batching; CC BY-SA 4.0 license on the published benchmark; forum terms check; demo rate limit + spend cap; cost estimate caveat. **Scope:** forum collector, docs PRs, and dashboard marked optional. |
| 2026-11-17 | Phase 0 done. Git repo (`main`), src layout, `pyproject.toml` (hatchling; core deps + optional groups `ingest`/`rag`/`collect`/`eval`/`app`/`dev`), `config.yaml` (generator/judge models and `torch_version` left `null` pending Open Decisions), `.env.example`, `src/ragbench/config.py` + `llm.py` (`complete`/`acomplete` via LiteLLM; returns text, tokens, cost, latency; per-role config for generator vs. judge). Ruff for lint + format. pytest with LiteLLM stubbed (no keys needed); `live` marker reserved for real-API tests. Python 3.12 venv in `.venv`. Project stays in the OneDrive folder for now. |
| 2026-11-17 | Pinned PyTorch **2.15.0** (latest stable on PyPI, released 2026-10-28; git tag `v2.15.0` @ `31a77fc`). Recorded in `config.yaml` (`corpus.torch_version`, tag, commit) and the `ingest` extra in `pyproject.toml` (CPU build via the PyTorch wheel index). `pytorch/tutorials` has no matching release tags, so pin it by commit SHA at download time and record it in the corpus manifest. |
| 2026-11-17 | Phase 1 docs extraction. The torch 2.15.0 wheel was built from `v2.15.0-rc10` (`d0f2af8`); the final tag adds only a lint change, so the pinned commit is now `d0f2af8` (what the docstrings actually come from). PyTorch docs turned out to be **MyST Markdown**, not `.rst`, so the plan's "rst notes" step became a MyST + eval-rst parser. Design: the docs pages' `autosummary`/`auto*` directives define *which* objects are in the corpus and their URLs (`generated/<name>.html` or `<page>.html#<name>`); docstrings come from the installed torch. Docs pages' own prose (e.g. optim "How to use an optimizer") is indexed too. Class docs include own members only (no inherited copies). Objects with no docstring (154, mostly internal ops) are dropped. Result: `data/corpus/docs.jsonl` = 1,638 docs (8 pages, 1,606 API objects, 24 notes), ~2.26M chars, 0 unresolved names; report in `data/corpus/ingest_report.json`. Code: `src/ragbench/ingest/{fetch,rst,pages,api,sections,build_docs}.py`; 16 unit tests. |
