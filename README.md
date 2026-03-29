# AI Agents for Kaggle Competitions

An autonomous ML pipeline that uses LLM agents to solve tabular Kaggle competitions end-to-end — from exploratory data analysis to Kaggle submission — without human intervention.

## How It Works

The system runs a **7-step ML pipeline** where each step is driven by a three-agent loop (Planner → Coder → Verifier). If the LLM is unavailable or all retries fail, deterministic fallbacks ensure the pipeline still completes.

```
┌─────────────────────────────────────────────────────────────┐
│                    Refinement Loop (1–3 iterations)         │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  1. EDA  │───▶│ 2. Train │───▶│ 3. Eval  │──┐          │
│   └──────────┘    └──────────┘    └──────────┘  │          │
│        ▲                                         ▼          │
│        │              ┌──────────┐               │          │
│        └──────────────│  Judge   │◀──────────────┘          │
│                       └──────────┘                          │
│                     SUFFICIENT? ──▶ break                   │
│                     NEED_REFINEMENT ──▶ next iteration      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  4. Generate Submission  │
              │  5. Submit to Kaggle     │
              │  6. Wait for Results     │
              │  7. Final Report         │
              └──────────────────────────┘
```

**Parallel candidate branches** — each refinement iteration launches multiple candidate branches concurrently (default: 3). Each branch independently runs EDA → Train → Eval, and the Judge picks the best result.

### Three-Agent Feedback Loop

Every code-generation step uses a mini feedback loop with up to 3 retry attempts:

| Agent | Role |
|-------|------|
| **Planner** | Produces a plain-English step-by-step plan (no code) |
| **Coder** | Writes and executes Python code following the plan (ReAct agent with tools) |
| **Verifier** | Validates outputs using guardrail tools; returns APPROVED or FAIL with feedback |

On FAIL, the Verifier's feedback is fed back to the Planner for the next attempt.

### RAG & Web Search

The pipeline augments LLM prompts with relevant context from two sources:

- **Notebook RAG** — a local knowledge base of curated ML notebooks and templates (`notebooks_kb/`), indexed with sentence-transformers embeddings + BM25 for hybrid retrieval via FAISS
- **Web Search** — DuckDuckGo-based search for current best practices, injected into prompts as additional context

## Project Structure

```
.
├── main.py                  # Entry point — orchestrates the full pipeline
├── config.py                # Paths, model settings, logging, data loading
├── prompts.py               # LLM prompt templates for all pipeline steps
├── steps_agent.py           # LLM-driven step implementations with fallback logic
├── steps_fallback.py        # Deterministic fallbacks (no LLM required)
├── steps_kaggle.py          # Kaggle API interactions (submit, wait for results)
├── executor.py              # Code extraction, validation, subprocess execution
├── guardrails.py            # Guardrail checks for state, outputs, and code safety
├── mini_feedback_loop.py    # Planner → Coder → Verifier LangGraph loop
├── rag/
│   ├── indexer.py           # Embedding generation, FAISS/BM25 index construction
│   ├── retriever_backend.py # Hybrid retrieval (dense + sparse + RRF fusion)
│   ├── retriever.py         # TF-IDF retrieval with section filtering
│   ├── notebook_loader.py   # Loads .ipynb, .py, .md, .txt into documents
│   ├── notebook_chunker.py  # Splits documents into labeled chunks
│   ├── rag_tools.py         # High-level RAG API: build, search, inject context
│   ├── embed_index.py       # TF-IDF index build/save/load
│   └── utils.py             # Shared types (RAGConfig, RetrievedChunk), tokenizers
├── tools/
│   ├── web_search_tool.py   # DuckDuckGo HTML search with LangChain @tool
│   └── web_context.py       # Injects web search results into pipeline state
├── notebooks_kb/            # Curated ML notebook knowledge base for RAG
│   ├── *.ipynb              # Reference notebooks (EDA, training, submission, etc.)
│   ├── *.py                 # Template scripts (CV, blending, train/val split)
│   └── *.md                 # Notes on metrics, experiment tracking, RAG queries
├── data/                    # Competition data (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submition.csv
├── artifacts/               # Generated outputs per session (gitignored)
│   ├── sessions/            # Timestamped session directories
│   └── rag/                 # Persisted RAG indexes
├── pyproject.toml
└── .env                     # API keys (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai/) API key
- A [Kaggle](https://www.kaggle.com/) account with API credentials

### Installation

```bash
git clone <repository-url>
cd ai_agents_course

uv sync
```

### Configuration

Create a `.env` file in the project root:

```
KAGGLE_USERNAME=your_kaggle_username
API_KAGGLE_KEY=your_kaggle_api_key
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key
```

Optionally set the Kaggle competition slug via environment variable (defaults to `mws-ai-agents-2026`):

```bash
export KAGGLE_COMPETITION=your-competition-slug
```

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TRAIN_SAMPLE_PCT` | `100` | Percentage of training data to use (lower for faster iterations) |
| `TRAIN_SAMPLE_FRAC` | `0.2` | Fraction held out for validation (80/20 split) |
| `RAG_ENABLED` | `True` | Enable notebook RAG context injection |
| `RAG_TOP_K` | `5` | Number of RAG chunks to retrieve |
| `WEB_SEARCH_ENABLED` | `True` | Enable web search context injection |

### Running the Pipeline

```bash
uv run python main.py
```

The pipeline creates a timestamped session directory under `artifacts/sessions/` containing:
- Generated code for each step and attempt
- Trained model artifacts (`.joblib`)
- EDA reports and visualizations
- Local evaluation metrics
- Final submission CSV
- Run logs

### Data Setup

Place your Kaggle competition data in the `data/` directory:

```
data/
├── train.csv
├── test.csv
└── sample_submition.csv
```

## Key Design Decisions

- **LLM-agnostic via OpenRouter** — swap models by changing `model_llm` in config without altering pipeline code
- **Graceful degradation** — every LLM step has a deterministic fallback, so the pipeline never crashes due to LLM failures
- **Isolated code execution** — generated code runs in subprocesses with timeouts, preventing runaway scripts from blocking the pipeline
- **Hybrid RAG** — combines dense embeddings (sentence-transformers/all-MiniLM-L6-v2) with BM25 sparse retrieval and RRF fusion for robust context retrieval
- **Parallel exploration** — multiple candidate branches run concurrently to explore different modeling strategies per iteration

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM orchestration | LangChain, LangGraph |
| LLM access | OpenRouter (ChatOpenAI) |
| ML models | CatBoost, XGBoost, LightGBM, scikit-learn |
| RAG embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (CPU) |
| Sparse retrieval | BM25 (rank-bm25) |
| Web search | DuckDuckGo (via httpx + BeautifulSoup) |
| Package management | uv |

## License

This project was developed as part of the HSE Agentic Systems course.
