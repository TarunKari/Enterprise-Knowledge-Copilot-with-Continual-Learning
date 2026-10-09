--- README.md (原始)


+++ README.md (修改后)
# 🚀 Enterprise LLM Platform — Fine-Tuning · RAG · Agents · Continual Learning

A production-grade, end-to-end platform that takes an open-source model (e.g. **Llama-3-8B**) from
raw domain data to a deployed, self-improving assistant:

| Phase | Capability | Key Tech |
|-------|------------|----------|
| **1** | LLM Fine-Tuning & Optimization | 4-bit NF4 quantization, LoRA/PEFT, TRL `SFTTrainer` + `DPOTrainer`, MLflow, vLLM/GGUF export |
| **2** | Advanced RAG & Vector Search | PyMuPDF/unstructured ingestion, recursive & semantic chunking, `bge-large-en-v1.5` embeddings, **pgvector HNSW**, hybrid dense+BM25 search with **Reciprocal Rank Fusion**, **Cross-Encoder reranking**, HyDE query rewriting, **RAGAS** evaluation |
| **3** | Agentic Orchestration | **LangGraph** state machine (Router → Retriever → Tool Executor → Human Gate), long-term Postgres memory, JSON-mode tool calling, Slack **human-in-the-loop approvals**, retry/escalation failure recovery |
| **4** | Deployment, Observability & Flywheel | FastAPI (`/chat`, `/ingest`, `/feedback`), Docker/K8s, LangSmith/Langfuse tracing + alerting, **Airflow DAG** that turns thumbs-up feedback into new SFT/DPO data and retrains automatically |

---

## 📖 Table of Contents

1. [Why This Project? The Big Picture](#-why-this-project-the-big-picture) — motivation + system architecture diagram
2. [Phase-by-Phase Deep Dive](#-phase-by-phase-deep-dive--how-it-works--why) — how each phase works internally and why each design choice was made
3. [Repository Layout](#-repository-layout) — annotated file tree
4. [Quick Start](#-quick-start) — install, configure, infrastructure
5. [End-to-End Workflow](#-end-to-end-workflow) — exact CLI commands for every stage
6. [Configuration Reference](#-configuration-reference-selected) — environment variables
7. [Testing & Smoke Runs](#-testing--smoke-runs) · [Hardware Guidance](#-hardware-guidance) · [Troubleshooting](#-troubleshooting)

---

## 🌟 Why This Project? The Big Picture

Most LLM tutorials stop at "prompt a chat model." Real enterprise assistants must be:

1. **Domain-expert** → the base model doesn't know your internal wikis, codebases, or SOPs. *Phase 1* teaches it, cheaply, on one GPU.
2. **Grounded & verifiable** → hallucinations are unacceptable when quoting HR policy. *Phase 2* builds a retrieval engine that finds the exact source chunks and attaches citations before the model ever answers.
3. **Autonomous but safe** → users want the assistant to *do things* (search, call APIs, send email), not just talk. *Phase 3* turns the model into a tool-using agent with a human approval gate in front of every destructive action.
4. **Self-improving** → models rot as knowledge changes. *Phase 4* closes the loop: user feedback is harvested weekly, converted into fresh training data, and used to retrain and redeploy — a continual-learning flywheel.

The four phases form one pipeline: **data → tuned model → grounded answers → acting agent → production telemetry → new training data**. Each phase's output is the next phase's input, all orchestrated from this single repository.

### Architecture at a Glance

```
                        ┌─────────────────────────────────────────────┐
                        │              USER / SLACK / WEB             │
                        └──────────────────────┬──────────────────────┘
                                               │ POST /chat
                        ┌──────────────────────▼──────────────────────┐
   PHASE 4              │        FastAPI Gateway (src/api/app.py)     │
   Deployment           │  Tracing hooks · Feedback capture           │
                        └──────────────────────┬──────────────────────┘
                                               │ invoke graph
                        ┌──────────────────────▼──────────────────────┐
                        │       LangGraph Agent (src/agent/graph.py)  │
   PHASE 3              │  Router ──► Retriever ──► Tool Executor     │
   Agentic Layer        │     ▲            │              │           │
                        │     │            ▼              ▼           │
                        │  Postgres   Hybrid RAG     Human Gate ──► Slack
                        │  memory     Pipeline      (interrupt)   Approve/Reject
                        └───────────────┬──────────┬──────────────────┘
                                        │          │ loads adapter
                        ┌───────────────▼──┐  ┌────▼───────────────────────┐
                        │ pgvector Store   │  │ Fine-Tuned Llama-3-8B      │
   PHASE 2              │ HNSW + BM25      │  │ (vLLM / GGUF served)       │
   RAG Engine           │ dense+RRF+rerank │  │ NF4 + LoRA + DPO aligned   │
                        └───────▲──────────┘  └────▲───────────────────────┘
                                │ chunks           │ checkpoints / MLflow runs
                        ┌───────┴──────────────────┴───────────────────────┐
                        │  Ingestion · Chunking · Embedding · SFT/DPO      │
                        │  Training Pipelines (src/rag, src/finetune)      │
                        └──────────────────────▲───────────────────────────┘
                                             │ JSONL datasets
                                   ┌─────────┴──────────┐
                                   │ Airflow Flywheel   │  ← thumbs up/down feedback
                                   │ (weekly retrain)   │
                                   └────────────────────┘
```

---

## 🧱 Phase-by-Phase Deep Dive — How It Works & Why

### Phase 1 — LLM Fine-Tuning & Optimization (`src/finetune/`)

**Goal:** turn a generic open model into a domain expert without buying a GPU cluster.

| Step | Module | What happens under the hood |
|------|--------|-----------------------------|
| Data curation | `data_curation.py` | Scrapes `data/raw/` documents into instruction-tuning triples `{"instruction","input","output"}` and preference pairs `{"prompt","chosen","rejected"}`. Chosen = concise/accurate; rejected = verbose/hallucinated — teaching tone *and* honesty simultaneously. |
| Quantized loading | `model_loading.py` | Loads Llama-3-8B with **bitsandbytes 4-bit NormalFloat (NF4)** + **double quantization**. NF4 is information-theoretically optimal for normally-distributed weights; double-quantizing the quant constants shrinks an 8B model from ~16 GB (fp16) to ~6 GB VRAM — trainable on a single consumer GPU. |
| LoRA injection | `model_loading.py` (PEFT) | Instead of updating all 8B weights, PEFT freezes the base model and trains small low-rank matrices on `q_proj`/`v_proj` attention projections (`r=16`, `alpha=32`). You train <1% of parameters, keep a ~100 MB adapter instead of a 16 GB checkpoint, and can hot-swap adapters per department. |
| SFT | `train_sft.py` | Hugging Face **TRL `SFTTrainer`** with `gradient_checkpointing=True` — activations aren't stored during the forward pass but recomputed in the backward pass, trading ~30% compute for massive memory savings. Checkpoints save every `SAVE_STEPS`; loss + VRAM metrics stream to **MLflow** (`mlflow_tracker.py`). |
| Preference alignment | `train_dpo.py` | Loads the best SFT checkpoint and runs **DPOTrainer**. Direct Preference Optimization folds the RLHF reward-model objective directly into a classification-style loss over chosen/rejected pairs — no separate reward model, no PPO instability, far cheaper. `beta` (default 0.1) controls how aggressively you deviate from the reference policy. |
| Export | `export_inference.py` | `merge` bakes LoRA weights back into the base model for maximum raw throughput; `gguf` converts via llama.cpp for CPU/edge serving; `vllm` launches a vLLM OpenAI-compatible server with PagedAttention + continuous batching for high-QPS production. |

**Why this order?** SFT teaches *what to say*; DPO refines *how to say it* (concise, non-hallucinating). Merging last keeps every intermediate stage cheap.

### Phase 2 — Advanced RAG & Vector Search (`src/rag/`)

**Goal:** ground every answer in verifiable enterprise evidence. Retrieval quality — not model size — determines answer quality.

Pipeline stages (each a module, each a CLI):

1. **Ingestion** (`ingestion.py`) — PyMuPDF/unstructured extract text from PDF/DOCX/MD; every chunk carries metadata (`source`, `author`, `department`, `last_updated`) so retrieval can be permission-filtered and citations generated.
2. **Chunking** (`chunking.py`) — **Recursive character splitting** respects document structure (sections → paragraphs → sentences), plus an optional **semantic** strategy that splits where embedding similarity drops. ~500-token chunks with 50-token overlap prevent facts being severed at boundaries.
3. **Embedding & indexing** (`vector_store.py`) — `bge-large-en-v1.5` maps chunks to 1024-d vectors stored in **PostgreSQL + pgvector** with an **HNSW index** (`m=16`, `ef_construction=64`) — sub-millisecond approximate nearest-neighbor search over millions of rows, in the same database as your metadata (no second vector DB to operate).
4. **Query rewriting** (`pipeline.py`) — a lightweight LLM expands the user query into variants and/or runs **HyDE**: generate a *hypothetical answer*, embed *that*, and search — hypothetical documents sit closer in embedding space to real answers than terse questions do, boosting recall.
5. **Hybrid retrieval** (`retrieval.py`) — dense cosine search (meaning) ∪ Postgres full-text/BM25 search (exact keywords, part numbers, jargon), merged by **Reciprocal Rank Fusion**: `score(d) = Σ 1/(k + rank_i(d))`, k=60. RRF needs no score calibration between the two systems and degrades gracefully if either one fails. Top-20 survive.
6. **Reranking** (`reranker.py`) — `bge-reranker-v2-m3` Cross-Encoder sees query *and* chunk tokens jointly (unlike bi-encoders which embed separately), scoring true relevance; returns the honest top-5.
7. **Context construction** (`pipeline.py`) — top-5 chunks formatted with citations `[Source: DocA, Page 2]` injected into the prompt template (`src/finetune/prompting.py`).
8. **Evaluation** (`evaluation.py`) — **RAGAS** scores *Faithfulness* (is every claim supported by context?), *Context Precision* (were the right chunks retrieved?) and *Answer Relevancy* against a golden test set — turning "feels right" into regression-testable numbers.

### Phase 3 — Agentic Orchestration (`src/agent/`)

**Goal:** move from passive Q&A to a stateful actor that plans, uses tools, remembers, and asks permission.

- **State machine** (`graph.py`) — LangGraph models the agent as a directed graph over a typed state (`messages`, `retrieved_context`, `tool_calls`, `memory`, `approval_status`). Nodes: **Router** (classify & decompose complex requests into numbered plans like "1. Search policy, 2. Check permissions, 3. Draft response"), **Retriever** (call the Phase-2 pipeline), **Tool Executor**, **Human Gate**. Conditional edges route on task type, tool outcome, and approval status — deterministic control flow wrapped around a stochastic model.
- **Memory** (`memory.py`) — short-term = conversation history carried in graph state; long-term = a Postgres `user_memory` table queried before planning, injecting known preferences ("this user prefers bullet points", "works in Finance") into every prompt.
- **Tools** (`tools.py`) — `search_knowledge_base`, `query_hr_api`, `send_email`… invoked through **JSON-mode structured outputs** so arguments are schema-valid by construction (parse failures eliminated), then executed with per-tool timeouts.
- **Human-in-the-loop** — any write/destructive tool pauses the graph at an interrupt node; a Slack Block Kit message with **Approve/Reject** buttons is posted to `SLACK_APPROVAL_WEBHOOK_URL`. Execution resumes only via the `/approvals/callback` endpoint replaying the saved graph checkpoint. Autonomy with a seatbelt.
- **Failure recovery** — tool calls wrapped in try/except; on API failure the agent reads the traceback, mutates its parameters, and retries (up to `TOOL_MAX_RETRIES=2`). Two consecutive failures escalate gracefully to the user instead of looping forever. `AGENT_MAX_ITERATIONS` bounds total graph steps.

### Phase 4 — Deployment, Observability & Continual Learning (`src/api/`, `src/observability/`, `docker/`, `src/continual/`)

- **API** (`src/api/app.py`) — FastAPI wraps the whole system: `POST /chat` (run the agent graph), `POST /ingest` (hot-add documents), `POST /feedback` (thumbs up/down), `POST /approvals/callback` (Slack resume), plus `/health`, `/metrics/summary`, `/flywheel/status`.
- **Containerization** (`docker/`) — CUDA-based image for self-hosted inference, docker-compose stack (API + Postgres/pgvector + MLflow), Kubernetes manifests for EC2/ECS-scale deployments.
- **Observability** (`src/observability/tracing.py`) — LangSmith/Langfuse callbacks trace *every* LLM call, tool execution, retrieval step, and state transition with token counts and latency; alert rules fire on p95 latency spikes, `MAX_TOKENS_PER_CALL` breaches, or tool-failure-rate surges (pushed to `ALERT_WEBHOOK_URL`).
- **The flywheel** (`src/continual/feedback_loop.py` + `docker/airflow/dags/continual_learning_dag.py`) — a weekly Airflow DAG:

  ```
  extract 👍 interactions ──► format SFT JSONL ──► pair 👍 vs 👎 ──► DPO JSONL
          │                                              │
          ▼                                              ▼
  promote new model ◄── baseline eval gate ◄── trigger SFT+DPO run in MLflow
   (auto-deploy)        (must beat incumbent)
  ```

  Highly-rated exchanges become tomorrow's training examples; low-rated ones become negative preference pairs. New model versions register in MLflow and are auto-promoted **only if they beat the incumbent on the golden/RAGAS evaluation** — improvement is enforced, never assumed. That closes the loop of continual learning.

---


## 📁 Repository Layout

```
.
├── configs/
│   └── settings.py                  # Central 12-factor config (all env-var overridable)
├── data/
│   ├── raw/                         # Drop source docs here (PDF, DOCX, MD)
│   └── processed/                   # Generated: sft.jsonl, preference.jsonl, chunks.jsonl, golden.jsonl
├── docker/
│   ├── airflow/dags/
│   │   └── continual_learning_dag.py  # Weekly flywheel: feedback → datasets → retrain → eval → deploy
│   ├── initdb/                      # Postgres init scripts (pgvector extension)
│   └── k8s/                         # Kubernetes manifests
├── notebooks/                       # Exploratory / demo notebooks
├── scripts/                         # Utility entrypoints
├── src/
│   ├── finetune/                    # PHASE 1
│   │   ├── data_curation.py         #   raw docs → SFT + preference JSONL
│   │   ├── model_loading.py         #   NF4 + double-quant loading, LoRA injection
│   │   ├── train_sft.py             #   TRL SFTTrainer + gradient checkpointing + MLflow
│   │   ├── train_dpo.py             #   TRL DPOTrainer alignment on best SFT checkpoint
│   │   ├── mlflow_tracker.py        #   run tracking, metrics callback, model registry
│   │   ├── prompting.py             #   Alpaca/ChatML formatting, RAG prompt builder
│   │   └── export_inference.py      #   LoRA merge, GGUF export, vLLM serve config
│   ├── rag/                         # PHASE 2
│   │   ├── ingestion.py             #   PDF/DOCX/MD parsing + metadata extraction
│   │   ├── chunking.py              #   recursive (~500 tok / 50 overlap) & semantic chunkers
│   │   ├── vector_store.py          #   pgvector schema, HNSW index, embedding & indexing
│   │   ├── retrieval.py             #   HyDE/query rewrite, dense + BM25, RRF fusion
│   │   ├── reranker.py              #   bge-reranker-v2-m3 Cross-Encoder top-20 → top-5
│   │   ├── pipeline.py              #   full end-to-end query pipeline w/ citations
│   │   └── evaluation.py            #   RAGAS Faithfulness / Context Precision / Answer Relevancy
│   ├── agent/                       # PHASE 3
│   │   ├── graph.py                 #   LangGraph state machine + Human Gate interrupt
│   │   ├── tools.py                 #   tool registry, JSON-mode parsing, retries
│   │   └── memory.py                #   short-term state + long-term Postgres user_memory
│   ├── api/                         # PHASE 4
│   │   └── app.py                   #   FastAPI: /chat /ingest /feedback /approvals/callback
│   ├── observability/
│   │   └── tracing.py               #   LangSmith/Langfuse tracing + latency/failure alerts
│   └── continual/
│       └── feedback_loop.py         #   feedback DB → new SFT/DPO batches (flywheel core)
├── tests/
├── requirements.txt
└── pyproject.toml                   # pip install -e . ; console entrypoints (llm-*)
```

---

## ⚡ Quick Start

### 1. Install

```bash
# CPU-only (API + RAG logic + data tooling)
pip install -e .

# GPU fine-tuning stack (CUDA machine)
pip install -r requirements.txt
```

### 2. Configure

Every setting in `configs/settings.py` is overridable via environment variables or a root `.env` file:

```bash
cp .env.example .env        # if present, otherwise export vars directly

export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ragdb"
export FT_BASE_MODEL="meta-llama/Llama-3-8B"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export SLACK_APPROVAL_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

Key knobs: `LORA_R=16`, `LORA_ALPHA=32`, `CHUNK_SIZE_TOKENS=500`, `CHUNK_OVERLAP_TOKENS=50`,
`RERANK_TOP_K=5`, `HYDE_ENABLED=true`, `GRAD_CKPT=true`.

### 3. Spin up infrastructure (Postgres + pgvector, MLflow)

```bash
docker compose up -d postgres mlflow     # or: docker run ... pgvector/pgvector:pg16
```

---

## 🧭 End-to-End Workflow

### Phase 1 — Fine-Tune the Base Model

```bash
# 1a. Curate domain data: data/raw/*.pdf|docx|md  →  SFT + preference JSONL
llm-curate --source data/raw --out data/processed
#    produces: data/processed/sft.jsonl          {"instruction","input","output"}
#              data/processed/preference.jsonl   {"prompt","chosen","rejected"}

# 1b. LoRA SFT with 4-bit NF4 + double quantization + gradient checkpointing,
#     checkpoints every N steps, loss/VRAM logged to MLflow
llm-train-sft --data data/processed/sft.jsonl --save-steps 100
llm-train-sft --smoke          # tiny CPU run to validate plumbing without a GPU

# 1c. DPO alignment starting from the best SFT LoRA adapter
llm-train-dpo --adapter checkpoints/sft/final_adapter --data data/processed/preference.jsonl
llm-train-dpo --smoke

# 1d. Merge adapter → base model; export GGUF (llama.cpp) or configure vLLM serving
llm-export merge   --base meta-llama/Llama-3-8B --adapter checkpoints/dpo/final_adapter --out models/merged
llm-export gguf    --hf-dir models/merged --out models/merged.Q4_K_M.gguf --quant Q4_K_M
llm-export vllm    --model-dir models/merged          # prints ready-to-run vLLM launch config
```

**What happens under the hood:** `bitsandbytes` NF4 + `bnb_4bit_use_double_quant=True` shrinks an
8B model's VRAM footprint ~75%; PEFT injects LoRA into `q_proj,k_proj,v_proj,o_proj,gate/up/down_proj`
(r=16, α=32); TRL's `SFTTrainer` runs with `gradient_checkpointing=True` so it fits on a single GPU.

### Phase 2 — Build the RAG Engine

```bash
# 2a. Ingest & parse documents (PDF via PyMuPDF, DOCX, MD; metadata: source/author/dept/date)
llm-ingest --source data/raw --out data/processed/corpus.jsonl

# 2b. Chunk (recursive 500-token / 50-overlap, or --strategy semantic)
llm-chunk --corpus data/processed/corpus.jsonl --strategy recursive

# 2c. Embed with bge-large-en-v1.5 and index into pgvector (HNSW m=16, ef_construction=64)
llm-embed-index --chunks data/processed/chunks.jsonl

# 2d. Query: HyDE rewrite → dense cosine + BM25 full-text → RRF top-20 → rerank top-5 → cited answer
llm-rag-query --question "What is our remote work policy?"

# 2e. Evaluate against a golden test set (RAGAS: Faithfulness, Context Precision, Answer Relevancy)
llm-rag-eval --make-template        # writes data/processed/golden.jsonl template
llm-rag-eval --golden data/processed/golden.jsonl
```

The hybrid retrieval SQL lives in `src/rag/retrieval.py`: cosine similarity over the HNSW
vector index is fused with Postgres full-text (tsvector) results via
**RRF**: `score = Σ 1/(k + rank)`, then the Cross-Encoder reranker returns the true top-5,
formatted with `[Source: DocA, Page 2]` citations by `src/rag/pipeline.py`.

### Phase 3 — Run the Agent

`src/agent/graph.py` defines the LangGraph state machine:

```
        ┌─────────┐   simple    ┌────────────┐
user ──▶│ Router  │────────────▶│ Retriever  │──▶ Tool Executor ──▶ Respond
        └─────────┘  complex▼   └────────────┘           │
             │  plan (decomposed steps)                  │ write/destructive tool
             ▼                                           ▼
      Long-term memory (Postgres user_memory)     ┌─────────────┐   approve
                                                  │ Human Gate  │──────────▶ execute
                                                  │ (interrupt) │   reject ──▶ escalate
                                                  └─────────────┘
                                       Slack Approve/Reject webhook
```

- **State schema:** `messages, retrieved_context, tool_calls, plan, memory, approval_status`
- **Tools:** `search_knowledge_base`, `query_hr_api`, `send_email` — arguments parsed in strict
  JSON mode and validated against each tool's schema (`src/agent/tools.py`).
- **Failure recovery:** try/except around every tool call; on error the agent inspects the
  traceback, adjusts parameters, retries (up to `TOOL_MAX_RETRIES=2`), then escalates to the user.

### Phase 4 — Deploy, Observe, Learn

```bash
# Serve the whole stack
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
#   POST /chat                 → run the agent graph (returns answer + citations + trace id)
#   POST /ingest               → add documents to the RAG index at runtime
#   POST /feedback             → 👍/👎 on any response (stored for the flywheel)
#   POST /approvals/callback   → resume a paused graph after Slack approval
#   GET  /health /metrics/summary /flywheel/status

# Containerize / orchestrate
docker build -f docker/Dockerfile -t llm-platform .
docker compose up -d          # api + postgres(pgvector) + mlflow + airflow
kubectl apply -f docker/k8s/  # production deployment (AWS EKS / ECS alternatives documented)
```

**Observability** (`src/observability/tracing.py`): every LLM call, tool execution, retrieval step,
and state transition is traced to LangSmith/Langfuse; `AlertMonitor` fires on high latency,
token-limit breaches, and tool-failure spikes.

**The Continual-Learning Flywheel** (`docker/airflow/dags/continual_learning_dag.py`, weekly):

```
/feedback (👍/👎) ─▶ FeedbackStore ─▶ extract highly-rated interactions
        ▲                                        │
        │                          llm-flywheel-batch (new SFT + DPO JSONL)
 auto-deploy ◀── baseline eval gate ◀── llm-train-sft/dpo (MLflow run) ──┘
```

New model versions are registered in MLflow, gated on baseline evaluation, and promoted
automatically — closing the loop.

---

## 🔧 Configuration Reference (selected)

| Env var | Default | Purpose |
|---|---|---|
| `FT_BASE_MODEL` | `meta-llama/Llama-3-8B` | Base model to fine-tune |
| `FT_BNB_QUANT_TYPE` / `FT_BNB_DOUBLE_QUANT` | `nf4` / `true` | 4-bit NormalFloat + double quantization |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | `16` / `32` / `0.05` | LoRA hyperparameters |
| `GRAD_CKPT` / `SAVE_STEPS` | `true` / `100` | Gradient checkpointing, checkpoint cadence |
| `DPO_BETA` | `0.1` | DPO temperature |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | `BAAI/bge-large-en-v1.5` / `BAAI/bge-reranker-v2-m3` | RAG models |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `500` / `50` | Chunking geometry |
| `HNSW_M` / `HNSW_EF_CONSTRUCTION` | `16` / `64` | pgvector ANN index tuning |
| `RRF_K` / `FUSE_TOP_N` / `RERANK_TOP_K` | `60` / `20` / `5` | Hybrid fusion & reranking depths |
| `HYDE_ENABLED` / `QUERY_REWRITE_VARIANTS` | `true` / `3` | Query rewriting |
| `AGENT_MAX_ITERATIONS` / `TOOL_MAX_RETRIES` | `8` / `2` | Agent loop bounds |
| `SLACK_APPROVAL_WEBHOOK_URL` | — | Human-gate notifications |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Experiment tracking |

---

## ✅ Testing & Smoke Runs

```bash
pytest                              # unit tests for chunking, curation, RRF, tools, feedback loop
llm-train-sft --smoke               # CPU plumbing check for the SFT pipeline
llm-train-dpo --smoke               # CPU plumbing check for DPO
llm-rag-eval --offline              # score pre-baked predictions without live retrieval
```

---

## 🖥 Hardware Guidance

| Task | Minimum |
|---|---|
| API / RAG logic / data tooling | Any CPU box (8 GB RAM) |
| Embeddings + reranking | 1× consumer GPU (8–12 GB) or CPU (slower) |
| LoRA SFT/DPO on Llama-3-8B (NF4 + grad ckpt) | 1× A10G/A100 24–40 GB |
| vLLM production serving | 1× A100 40 GB (bf16) or 24 GB (AWQ/GPTQ) |

---

## 🛠 Troubleshooting

- **`bitsandbytes` not found / no CUDA** → use `--no-4bit` or `--smoke`; GPU extras only needed for training.
- **`type vector does not exist`** → ensure the `pgvector/pgvector:pg16` image or run `CREATE EXTENSION vector;` (`docker/initdb/` automates this).
- **RAGAS needs an LLM-as-judge** → set `OPENAI_API_KEY`, or use the built-in offline lexical scorers in `src/rag/evaluation.py`.
- **Agent never pauses for approval** → set `AGENT_APPROVAL_REQUIRED=true` and configure `SLACK_APPROVAL_WEBHOOK_URL`.

## 📄 License

Internal enterprise project — all rights reserved by your organization. Adapt freely inside your company.
