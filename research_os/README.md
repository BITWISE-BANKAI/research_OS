# RESEARCHOS — Local AI Research Intelligence & RAG Chatbot

> **"ChatGPT + Semantic Scholar + NotebookLM + research analyst"**  
> Fully local · Private · Evidence-grounded · No hallucinations

---

## What is RESEARCHOS?

RESEARCHOS is a production-quality, local-first AI research intelligence platform. It lets research scholars:

- 🔍 **Search** Semantic Scholar for research papers
- ⬇️ **Index** full-text PDFs from arXiv / Open Access into a local vector database
- 💬 **Chat** with their research library — all answers backed by retrieved evidence
- ⚖️ **Compare** multiple papers side-by-side
- 🧠 **Deep Research** — generate iteratively self-evaluating multi-phase briefings

Everything runs **100% locally** using Ollama, ChromaDB, and LangChain.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama → `qwen3:latest` |
| Embeddings | Ollama → `qwen3-embedding:latest` |
| Vector DB | ChromaDB (persistent, local) |
| RAG | LangChain + LangChain-Chroma |
| Paper Discovery | Semantic Scholar API |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / Vanilla CSS (Glassmorphism) / JavaScript |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- Semantic Scholar API key (free at https://www.semanticscholar.org/product/api)

### 2. Pull Ollama models

```bash
ollama pull qwen3:latest
ollama pull qwen3-embedding:latest
ollama serve
```

### 3. Clone / navigate to project

```bash
cd research_os/
```

### 4. Create virtual environment and install dependencies

```bash
# Using uv (recommended — already in your environment)
uv venv
uv pip install -r requirements.txt

# Or standard pip
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

### 5. Configure environment

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
SEMANTIC_SCHOLAR_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:latest
OLLAMA_EMBED_MODEL=qwen3-embedding:latest
CHROMA_PERSIST_DIR=./data/chroma
TOP_K=5
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

### 6. Run the server

```bash
# From research_os/ directory
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Open in browser

```
http://localhost:8000
```

---

## How to Use

### Search & Index Papers
1. Go to **Search Papers** panel
2. Type your research topic (e.g. "LLM hallucination")
3. Click **⬇ Index Paper** on the papers you want to study
4. Indexing downloads the PDF, chunks it, embeds it locally, and stores it in ChromaDB

### RAG Chat
1. Switch to **💬 RAG Chat**
2. Choose **Research Library** (all indexed papers) or **Single Paper** mode
3. Ask any question — answers cite specific chunks and pages

### Compare Papers
1. Index at least 2 papers
2. Switch to **⚖️ Compare Papers**
3. Select papers and click Compare — get a structured methodology/findings comparison

### Deep Research
1. Index your papers
2. Switch to **🧠 Deep Research**
3. Enter a topic, select papers, set iterations and quality threshold
4. Get a Phase 0/1/2/3 research briefing that auto-evaluates and self-improves

---

## Architecture

```
Browser (index.html + script.js)
    │
    ▼
FastAPI (app.py)
    │
    ├── /api/search     → Semantic Scholar search
    ├── /api/index      → Download PDF → Chunk → Embed → ChromaDB
    ├── /api/chat       → ChromaDB retrieval → Qwen3 grounded answer
    ├── /api/compare    → Per-paper retrieval → Qwen3 comparison
    └── /api/deep-research → Iterative: Generate → Evaluate → Regenerate
```

---

## Integrating with a JavaScript Website

RESEARCHOS exposes a clean REST API. To embed it in any JS/CSS frontend:

```javascript
// Search papers
const results = await fetch('http://localhost:8000/api/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'transformer attention' })
}).then(r => r.json());

// Index a paper
await fetch('http://localhost:8000/api/index', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ paper: results.papers[0] })
});

// Chat
const answer = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'What dataset did this paper use?', mode: 'library' })
}).then(r => r.json());
// answer.answer, answer.sources, answer.retrieved_chunks

// Deep research
const briefing = await fetch('http://localhost:8000/api/deep-research', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'LLM hallucination', paper_ids: ['abc123'], max_iterations: 3 })
}).then(r => r.json());
```

Add CORS to `app.py` for cross-origin access:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

---



## Future Extensions (designed for, not implemented)

- Agent -Based chunking 
- Fine Tuning the models for better results 
- BGE-M3 / Embedding-Gemma benchmarking
- Advanced cross-encoder reranker
- Citation graph visualisation
- Research-gap agent
- Contradiction detector
- Mem0 conversational memory
- Paper recommendation engine
