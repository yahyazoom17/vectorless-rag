# 🌲 Vectorless RAG

A lightweight Retrieval-Augmented Generation (RAG) system that queries PDF documents **without vector embeddings or a vector database**. Instead of a semantic similarity search, it uses an LLM to reason over a structured document tree (built by [PageIndex](https://pageindex.ai)) to identify the most relevant sections before generating an answer.

---

## 💡 How It Works

Traditional RAG splits documents into chunks, embeds them into vectors, and retrieves the closest chunks to a query. **Vectorless RAG takes a different approach:**

1. **Index** — Upload a PDF to PageIndex, which parses it into a hierarchical tree structure (like a Table of Contents with summaries).
2. **Tree Search** — Send the query + compressed tree to an LLM. The LLM reasons step-by-step and returns the node IDs most likely to contain the answer.
3. **Retrieve** — Fetch the full text of those specific nodes from the tree.
4. **Generate** — Pass the retrieved sections as context to the LLM to produce a final, cited answer.

No embeddings. No vector stores. No similarity thresholds to tune.

```
Query
  │
  ▼
LLM Tree Search  ──►  Relevant Node IDs
                              │
                              ▼
                     Retrieve Node Text
                              │
                              ▼
                     LLM Answer Generation
                              │
                              ▼
                        Cited Answer
```

---

## 📁 Project Structure

```
vectorless-rag/
├── rag.py            # RAG pipeline using OpenRouter (cloud LLMs)
├── rag_ollama.py     # RAG pipeline using Ollama (local LLMs)
├── main.py           # Entry point/runner
├── data/
│   └── sample.pdf    # Example PDF document
├── env-example       # Template for environment variables
├── pyproject.toml    # Project metadata and dependencies
└── .python-version   # Python version pin (3.12+)
```

---

## ⚙️ Requirements

- Python 3.12+
- A [PageIndex](https://pageindex.ai) API key
- **For `rag.py`:** An [OpenRouter](https://openrouter.ai) API key
- **For `rag_ollama.py`:** [Ollama](https://ollama.com) running locally with a model pulled (e.g. `gemma4`)

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/yahyazoom17/vectorless-rag.git
cd vectorless-rag
```

### 2. Install dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

Or with pip:

```bash
pip install openai pageindex python-dotenv requests
```

### 3. Configure environment variables

```bash
cp env-example .env
```

Edit `.env` and fill in your keys:

```env
PAGEINDEX_API_KEY=your_pageindex_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here   # only needed for rag.py
```

### 4. Add your PDF

Place the PDF you want to query at:

```
data/sample.pdf
```

### 5. Run

**Using OpenRouter (cloud LLMs):**

```bash
uv run rag.py
```

**Using Ollama (local LLMs):**

```bash
# Make sure Ollama is running, and you have a model pulled
ollama pull gemma4
uv run rag_ollama.py
```

---

## 🔧 Customising Your Query

At the bottom of either script, update the `vectorless_rag(...)` call:

```python
_ = vectorless_rag(
    query="Your question about the document here",
    tree=pageindex_tree,
)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pageindex` | Parses PDFs into a hierarchical tree with node summaries |
| `openai` | OpenAI-compatible client (used for both OpenRouter and Ollama) |
| `python-dotenv` | Loads API keys from `.env` |
| `requests` | HTTP utility |

---

## 🆚 Vectorless RAG vs. Traditional RAG

| | Traditional RAG | Vectorless RAG |
|---|---|---|
| Retrieval method | Embedding similarity | LLM tree reasoning |
| Setup complexity | High (embedding model + vector DB) | Low (API only) |
| Infrastructure | Vector database required | None |
| Best for | Large unstructured corpora | Structured/hierarchical documents |
| Explainability | Low | High (LLM explains its reasoning) |
