# Nexla Document MCP Server

A local **Model Context Protocol (MCP)** server that ingests a set of PDF documents, indexes them in a persistent vector store, and exposes document-intelligence tools so any MCP client (Claude Desktop, Cursor, the MCP Inspector, etc.) can ask natural-language questions and receive **grounded, source-attributed answers**.

---

## What it does

- Ingests the provided PDFs: parse → chunk → embed → persist.
- Indexes chunks in **ChromaDB on disk**, each with attribution metadata.
- Uses **hybrid retrieval**: dense vector search plus local BM25-style keyword search, fused/reranked before synthesis.
- Exposes three MCP tools:
  - `query_documents` — natural-language Q&A with grounded, cited answers
  - `list_documents` — what's indexed (pages + chunk counts)
  - `get_document_section` — raw text of a specific document page
- Returns answers with **citations** (document, page, section when detectable, snippet, similarity score), and **degrades gracefully** to ranked passages when answer synthesis is disabled or the LLM call fails.

---

## Assignment requirements coverage

| Requirement | Where it's implemented |
|---|---|
| Ingest all provided PDFs | `services/ingestion/` (`make ingest`) |
| ≥1 MCP tool for grounded Q&A | `query_documents` (+ `list_documents`, `get_document_section`) |
| Multi-document awareness | single Chroma collection; retrieval ranks across all sources |
| Source attribution | every `Citation` carries document, page, section (best-effort), snippet, score |
| MCP protocol compliance, runs locally | FastMCP server over stdio (`make run`) |
| Demonstrably working Q&A | MCP demo checklist + example interaction log below |

---

## Architecture

The codebase is layered so each module has exactly one job. The runtime request path is **routes → handlers → services → repo → vector store**, and nothing above the repo ever touches Chroma directly.

```
app/
├── server.py                     # MCP entrypoint: creates the FastMCP instance, calls register(), runs (stdio)
├── routes/documents.py           # thin MCP tool layer: defines tools via register(mcp), delegates to handlers
├── handlers/documents.py         # per-tool orchestration (query = retrieve + synthesize)
├── services/
│   ├── ingestion/
│   │   ├── pdf_loader.py          # PyMuPDF -> page-tagged text (only module that knows PDFs)
│   │   ├── chunker.py             # page-aware chunking + overlap + stable ids
│   │   └── service.py             # orchestrates parse -> chunk -> embed -> persist
│   ├── embeddings_service.py      # only module that calls the OpenAI embedding API
│   ├── retrieval/
│   │   ├── service.py             # embed query -> hybrid candidate retrieval
│   │   ├── reranker.py            # weighted fusion of dense + lexical candidates
│   │   └── helpers.py             # citations + numbered prompt context
│   └── synthesis/
│       ├── service.py             # grounded answer + hybrid fallback
│       └── prompts.py             # AI prompt text (isolated for easy tuning)
├── schemas/documents.py          # Pydantic outputs: Chunk, RetrievedChunk, Citation, Answer, ...
├── contracts/documents.py        # inbound request shapes (*Request) with validation
├── repos/
│   ├── base.py                    # BaseRepo over the shared collection
│   ├── document_repo.py           # all vector-store ops: upsert / query / list / get_section
│   └── repo_container.py          # DI container the services depend on
├── db/session.py                  # Chroma PersistentClient + collection (cached singleton)
├── tools/text.py                  # pure text helpers (no domain logic, no I/O)
└── config.py                      # single typed source of truth (Pydantic settings)
```

**Two flows:**

- **Ingestion (run once):** `pdf_loader` (PyMuPDF) → `chunker` (page-aware) → `embeddings_service` (OpenAI) → `document_repo.add_chunks` → Chroma (`.chroma/`). Chunk ids are deterministic, so re-running upserts rather than duplicating.
- **Query (per request):** tool → handler → `retrieval` (embed question → dense candidates + BM25-style keyword candidates → weighted rerank/fusion) → `synthesis` (grounded OpenAI answer, hybrid fallback) → `Answer` with citations.

---

## Setup

### 1) Prerequisites
- Python 3.12
- Poetry
- A valid OpenAI API key

### 2) Clone and install
```bash
git clone https://github.com/Abhishek0512-max/nexla-doc-mcp.git
cd nexla-doc-mcp
make install
```

### 3) Add the source PDFs
Place the provided PDFs in `data/` (this repo includes them, so you can ingest directly). They are public ACL Anthology papers: `C18-1117`, `D19-1539`, `P19-1164`, `W18-4401`, `W18-5713`.

### 4) Configure environment
Copy the example environment file and add your OpenAI key:
```bash
cp .env.example .env
# then edit .env and replace OPENAI_API_KEY=your_openai_api_key_here
```

> **Important — the key is needed at query time, not just ingestion.** Because embeddings use the OpenAI API, every incoming question is embedded per request, so a valid `OPENAI_API_KEY` is required to *answer*, not only to index. To run fully offline/key-free, swap `embeddings_service.py` to a local model (e.g. `sentence-transformers`) — the `embed_texts` / `embed_query` interface stays identical and nothing else changes.
>
> If you have `OPENAI_API_KEY` exported in your shell, it takes precedence over `.env`. If a known-good `.env` key still produces a 401, run commands as `env -u OPENAI_API_KEY make ingest` / `env -u OPENAI_API_KEY make eval`, or unset the shell variable.

### 5) Ingest the documents
```bash
make ingest
# Indexed 246 chunks from 5 document(s) across 47 pages into 'documents'.
```

### 6) Run the server
```bash
make run
```
The server starts on **stdio** and then waits silently for a client to connect — that's expected (a stdio MCP server has nothing to print until a client speaks to it). Connect it via the MCP Inspector or Claude Desktop (below). `Ctrl-C` to stop.

### 7) Tests
```bash
make test
```

### 8) Retrieval eval
```bash
make eval
```
This runs a small golden retrieval set against the indexed Chroma collection and reports `document_recall@5`, `page_recall@5`, `mrr@5`, and exact-source pass rate. It uses the same retrieval path as `query_documents`, so a valid `OPENAI_API_KEY` is required for query embeddings.

---

## Connecting a client

**MCP Inspector (interactive):**
```bash
poetry run fastmcp dev app/server.py
```

**Quick MCP demo checklist:**

1. Open the Inspector and confirm it discovers exactly three tools: `query_documents`, `list_documents`, and `get_document_section`.
2. Inspect `query_documents` and verify the advertised schema includes `question` (`minLength=1`) and optional `top_k` (`1..20`), plus read-only / non-destructive behavior.
3. Call `list_documents` to verify all 5 PDFs are indexed.
4. Call `query_documents` with `Which papers in the corpus are from Facebook AI Research, and what problems do they address?` to demonstrate multi-document retrieval with citations.
5. Call `get_document_section` with `document="W18-4401.pdf", page=3` to show page-level source inspection.

**Claude Desktop** — add to `claude_desktop_config.json` (point at the venv python directly, and set `cwd` so `.env` and the `app` package resolve), then fully quit and reopen Claude Desktop:
```json
{
  "mcpServers": {
    "nexla-doc-mcp": {
      "command": "/absolute/path/to/nexla-doc-mcp/.venv/bin/python",
      "args": ["-m", "app.server"],
      "cwd": "/absolute/path/to/nexla-doc-mcp"
    }
  }
}
```

---

## MCP Tool Documentation

### `query_documents`
Answer a natural-language question from the indexed PDFs.

- **Input:** `question: str` (required), `top_k: int | None` (optional, 1–20; overrides retrieval depth)
- **Output:** `Answer { answer: str, citations: list[Citation], synthesized: bool }`
  - `Citation { document, page, section?, snippet, score }`
  - `synthesized = true` → LLM-composed prose; `false` → ranked-passage fallback
- **Example:** `"Which of these papers are from Facebook AI Research?"`

### `list_documents`
List what has been indexed — useful to verify coverage before querying.

- **Input:** none
- **Output:** `list[DocumentInfo { document: str, pages: int, chunks: int }]`

### `get_document_section`
Fetch the raw indexed text of one page of one document, to verify or expand on a citation.

- **Input:** `document: str`, `page: int` (1-based)
- **Output:** `Section { document: str, page: int, text: str }`
- **Example:** `document="C18-1117.pdf", page=1`
- A non-existent document/page returns a clear tool error rather than an empty success.

---

## Design Decisions & Trade-offs

- **FastMCP framework.** Decorator-based tools generate spec-compliant schemas automatically, so effort goes into logic, not JSON-RPC plumbing.
- **ChromaDB on disk.** Zero-infrastructure, persists across restarts, native metadata storage/filtering, and more than sufficient for a corpus of this size. Deliberately *not* Postgres/pgvector or a hosted vector DB — this is a local take-home, and the repo abstraction makes swapping the store a one-file change (see Production Considerations).
- **OpenAI `text-embedding-3-small` embeddings.** High-quality semantic retrieval and the simplest bring-your-own-vectors path into Chroma. Trade-off: the key is needed at query time (see Setup).
- **Hybrid retrieval (dense + BM25-style lexical, fused).** Dense vectors alone blur exact tokens — paper names, dataset names like `WinoMT` / `CoNLL-2003`, and affiliations — so a lexical signal is added and weighted slightly higher (`VECTOR_WEIGHT=0.4`, `LEXICAL_WEIGHT=0.6`) for this name-heavy corpus. The pipeline over-retrieves (80 candidates) and fuses/reranks down to the final `TOP_K`. This was a deliberate upgrade from a dense-only first pass after reading Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) write-up; I added the highest-leverage pieces that fit the time box and documented the one I skipped (LLM-generated contextual chunks) as future work, because it would require a re-ingestion path and extra generation cost.
- **Hybrid answer synthesis.** When a key is configured the server composes a grounded, inline-cited answer; with synthesis off — or on any LLM/network failure — it returns the top ranked, still-cited passages. This satisfies "the server returns grounded answers" while never crashing a tool call.
- **PyMuPDF text extraction.** Fast and reliable for these text-based PDFs; deliberately text-only (richer extraction is in Future Work).
- **Page-aware chunking.** A chunk never crosses a page boundary, so every chunk carries an exact page number — attribution is structural, not best-effort.
- **Single collection, no per-document filter.** Multi-document awareness falls out for free — retrieval ranks across all sources at once.
- **Thin tool surface (3 tools).** Enough to show protocol fluency without flooding the agent's context with rarely-used tools.
- **Evaluation as a quality gate.** Retrieval is measured against a golden set rather than judged by eye (see Evaluation & Quality Gates).

---

## Known Limitations & Future Work

These are improvements to the **core RAG quality** of what's built (extraction, retrieval, evaluation depth).

- **Extraction is text-only.** PyMuPDF pulls text but is not very accurate on tables, figures, signatures, and multi-column layout. Next steps: layout-/table-aware parsing, OCR for scanned pages, and a **multimodal** path that extracts figures and stores image (or text+image) embeddings alongside the text. Enrich per-chunk metadata (section IDs, table/figure references, source coordinates) so attribution and filtering get sharper.
- **Retrieval can go further.** The current reranker is deterministic weighted fusion over dense and lexical candidates. A true **cross-encoder reranker** (Cohere rerank or a local BGE reranker) would lift harder semantic matches at the cost of another model dependency. **Contextual chunks** — Anthropic's pattern of prepending a generated chunk-specific context before embedding/BM25 indexing — is the next quality step. For multi-hop or global ("across all papers") questions, **Graph RAG** (an entity/knowledge graph plus community summaries) would beat flat top-k.
- **Section detection is best-effort.** A lightweight heuristic occasionally mis-tags a short Title-Case line (e.g., an author name) as a section heading. **Page numbers are the reliable attribution anchor**, so answers and citations are unaffected. Tightening to numbered-headings-only, or dropping the `section` field entirely, are both safe improvements.
- **API embeddings require a key at query time** (see Setup). A local-embedding mode (sentence-transformers) would enable fully offline operation.

---

## Evaluation & Quality Gates

The eval harness is intentionally **component-level** — it measures retrieval *before* synthesis, because if the right evidence isn't retrieved, the answer cannot be grounded.

- **Dataset:** `evals/retrieval_golden.jsonl` — hand-labeled questions with expected source documents/pages from the indexed PDFs.
- **Runner:** `make eval` calls the same retrieval service the MCP tool uses, so the eval exercises the real path.
- **Metrics:** document recall@k, page recall@k, mean reciprocal rank (MRR), and exact-source pass rate.
- **Current result:** on the indexed 5-paper corpus with tuned hybrid defaults, the golden set reports `document_recall@5=1.00`, `page_recall@5=1.00`, `mrr@5=1.00`, and `exact_source_pass_rate@5=1.00`.
- **Reading the score honestly.** With a small, well-separated 5-paper corpus and distinctive questions, near-perfect retrieval is *expected*. This is best read as a **regression smoke-gate** — it catches a retrieval regression on any code change — not as a benchmark of retrieval difficulty.
- **Next step — make the eval discriminating.** Add adversarial, paraphrased, and shared-keyword questions (where a term appears in several papers) so the metrics drop below 1.00 and the harness actually separates good retrieval from lucky retrieval.
- **Next step — evaluate the whole RAG system.** Extend beyond retrieval to score retrieval + augmentation + generation together, using **golden answers** plus **LLM-as-judge** for faithfulness, answer relevance, and citation coverage, and track **latency and cost** per stage as first-class metrics. Guardrail: track quality, never optimize prompts toward the metric.

---

## Production Considerations

These were considered but intentionally kept out of the 3–4 hour local take-home scope. Where Future Work makes the *core system* better, these make it *deployable at scale*.

- **Storage & ingestion at scale.** Move embedded Chroma to a served/managed vector DB (Qdrant) or pgvector, decoupled from the server. Make ingestion **incremental** (hash-based re-index of only changed PDFs) and run it as an async worker rather than in the serving path, so large or growing PDF corpora index without blocking queries.
- **MCP scaling.** Stateless Streamable HTTP behind a load balancer, a shared vector store, and a separate ingestion worker → horizontal scale with no sticky sessions.
- **MCP observability.** Log every tool call (structured logs or a `tool_calls` table): tool name, argument shape, latency, cost, and result size — so production usage is monitorable and slow/expensive calls are visible.
- **Call-intent capture.** MCP surfaces *what* tool ran with *what* arguments, but not *why*. An optional `intent` string the calling agent populates supplies that missing signal for debugging, analytics, and routing (it relies on the client filling it honestly, but even best-effort intent is valuable).
- **Context hygiene & response shaping.** Watch tool payload sizes so a single call doesn't flood the agent's context, and support caller-specified **field include/exclude** so the agent can request only the keys it needs from a rich structured response instead of the full Pydantic object.
- **Tool affordances for multi-step use.** At scale, tool naming/descriptions should also guide the agent on how to *chain* tools toward a multi-step goal (potentially with a routing/planner layer). A 3-tool local server doesn't need this, but it's the right direction for a larger tool surface.
- **Remote hardening.** If this moved from local stdio to HTTP: OAuth 2.1 / scopes, audit logs with source-side redaction, rate limits and quotas, and human approval for any write-capable tool.

---

## Example Interaction Log

Captured from local runs against the indexed assignment PDFs.

### Q1 — single document
**Question:** `What is the TRAC shared task about?`

**Answer (excerpt):** The TRAC shared task is about *Aggression Identification* — classifying social-media comments and tweets into Overtly Aggressive (OAG), Covertly Aggressive (CAG), and Non-Aggressive (NAG).

**Citations:** `W18-4401.pdf` p3, p4

### Q2 — single document
**Question:** `How are aggression labels typically defined in these papers?`

**Answer (excerpt):** Three classes — overt aggression, covert aggression, and non-aggression — broadly mapping to an explicit/implicit distinction in abusive-language typologies.

**Citations:** `W18-4401.pdf` p2, p3

### Q3 — single document
**Question:** `What challenges are mentioned for aggression detection in social media text?`

**Answer (excerpt):** Challenges include distinguishing profanity from hate speech and handling multilingual / code-mixed social data.

**Citations:** `W18-4401.pdf` p2, p10, p11

### Q4 — multi-document (demonstrates cross-document retrieval)
**Question:** `Which of these papers are from Facebook AI Research, and what problems do they address?`

**Answer (excerpt):** Two papers in the corpus are from Facebook AI Research:

1. `W18-5713.pdf`, **Retrieve and Refine: Improved Sequence Generation Models For Dialogue**, addresses dialogue generation. It combines retrieval and generation to avoid generic Seq2Seq replies while still allowing responses to be adapted to the conversation context.
2. `D19-1539.pdf`, **Cloze-driven Pretraining of Self-attention Networks**, addresses language-model pretraining. It proposes a cloze-style word reconstruction objective for bidirectional self-attention networks and reports gains on language understanding tasks.

**Citations:** `W18-5713.pdf` p1, `D19-1539.pdf` p1

---

## Vibe Coding Setup

I'd frame what I did less as "vibe coding" and more as **disciplined, AI-assisted engineering** ; The human stays the architect and owns every decision; the AI is a force multiplier for research, scaffolding, and execution, kept on-rails by an explicit harness. MCP and agentic RAG are new enough that there's no settled playbook, so much of the work was using AI to *discover* current practice and then exercising judgment about what to adopt.

## A standing harness, not ad-hoc prompts

I maintain project-level instruction files — `CLAUDE.md` for Claude Code, `AGENTS.md` for Codex-style agents, and Cursor rules — that encode the non-negotiables so every generation starts already aligned instead of being re-prompted each time: clear separation of concerns, typed boundaries between layers, small single-purpose functions, no dead scaffolding, and "write the unit tests alongside the code, not after." Keeping these portable means the same standards apply whichever agent I reach for.

Crucially, the harness is a **living document**. When the AI repeated a mistake, I encoded the correction as a new rule rather than re-explaining it each session — so the instruction files compound over time and the same class of error stops recurring. The harness is the memory the model doesn't have.

## Right tool for the cognitive task

I split the work by *type of thinking*:

- **Claude Code for spec and architecture** — decomposing the problem, weighing trade-offs, and researching the current state of MCP/RAG (FastMCP vs. the bundled SDK, contextual retrieval, transports, chunking strategies). Because the domain is new, this research-and-reason loop was where most of the real value lived. 
- **Cursor for implementation** — I standardized on a single implementation surface so I could navigate the codebase, select and refactor specific spans in place, and review every diff inline. I switched models by task economics: a fast, cheap model for boilerplate and mechanical refactors, and a stronger reasoning model for architecture, tricky bugs, and spec work. One editor keeps context tight and the loop fast.

I feel that claude is good at writing spec, codex for writing small clean code and cursor for targeted edits. Here I used claude+cursor mainly

## Human-led decomposition, spec-first

I chunked the system into phases myself — ingestion/preprocessing → chunking → embedding + retrieval → grounded synthesis → the MCP interface → tests — and ran each as the same loop: design the architecture, pressure-test the options with the AI, research what the best teams currently do, decide, implement one module, verify at runtime. The architecture spec was written down first and treated as the source of truth; the AI implemented *against* the spec and was asked to critique it, rather than inventing structure in the chat thread. The AI never set the structure; I did, then had it execute and explain so I understood every line I kept.

## Treating a new domain carefully

MCP and agentic RAG post-date the models' training data, so I treated the AI's knowledge as a **hypothesis, not a fact**. Anything load-bearing — transport choices, tool-schema conventions, current FastMCP APIs, retrieval techniques — I verified against primary sources (the official MCP specification, Anthropic's documentation, FastMCP's docs) before committing to it. Recognizing the knowledge-cutoff problem and routing around it was a core part of the workflow, not an afterthought.

## Verify, don't trust — a gate on every generation

Generated code was a draft until it passed a gate: type-checking, lint, and a runtime smoke check (`make ingest`, the in-memory client, direct calls). I treated "the static checks pass" as necessary but not sufficient — runtime correctness is a separate bar. The test suite stayed human-owned: I had the AI propose test *logic*, then reviewed and wrote the tests myself, so the safety net was never generated by the same process it was meant to catch. Generations were kept small and scoped to one module so every diff stayed reviewable and reversible.

## From baseline to hardening

Once the system worked end to end, I shifted to "make it production-grade." I researched and weighed security (notably prompt injection via document content), auditability, observability, agentic architecture, and latency, then prioritized deliberately rather than building everything in scope. I added an evaluation pass — measuring retrieval and answer quality — to drive refinement on accuracy, and used it alongside code cleanup to strengthen the design.

## Failure modes I guarded against

Working this way means knowing where AI tends to fail and watching for it. The recurring ones I actively guarded against were: **hallucinated or outdated APIs** (mitigated by primary-source verification), **confident-but-wrong runtime assumptions** (caught by the verification gate), **speculative over-engineering** (I pruned dead scaffolding the AI wanted to add), and **sycophantic agreement** with a weak idea (mitigated by asking for options and trade-offs before any code, so the model argued positions rather than rubber-stamping mine). Concrete catches included Chroma's nested `query()` return shape, the cosine-distance-to-similarity conversion, a chunker heuristic that mis-tagged author lines as section headings, and a stale `OPENAI_API_KEY` in my shell silently shadowing `.env` (shell vars outrank dotenv) — which produced confusing 401s until I isolated it. Also, the model starts performing worst when reaching context (Cursor or Claude code), I maintain a doc on side with changes and important decisions. After, it goes to 80/90% of context, I move to a new chat. 

## Honest cost/benefit

AI was a large net accelerator for research, boilerplate, scaffolding, and iteration — it collapsed the cost of exploring the unfamiliar MCP surface and wiring up the layered architecture. It was *not* free: the most time-consuming bug came from trusting a confident runtime assumption that passed static checks, and a few "helpful" abstractions had to be removed rather than added. The honest read is that AI shifts effort from typing toward judgment — and judgment is where the remaining work concentrates.

## Overall view

AI collapses the cost of research, boilerplate, and iteration, but it doesn't own the system — the engineer does. The highest leverage comes from using AI as a thinking partner for spec and a fast executor for implementation, governed by an explicit, evolving harness, while keeping decomposition, architecture, correctness, and trade-offs firmly in human hands. For a genuinely new domain like agentic MCP, that judgment — knowing which "latest and greatest" pattern actually fits the problem, and verifying it against primary sources rather than model memory — is the real differentiator, and it's what I set out to demonstrate here.
