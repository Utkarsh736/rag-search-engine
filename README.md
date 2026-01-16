# RAG Search Engine

A comprehensive command-line project exploring **multimodal search** and **Retrieval-Augmented Generation (RAG)** for movies.  
It demonstrates the progression from classic keyword search to semantic search, hybrid ranking, LLM-powered answering, and image-based retrieval using modern embedding models.

---

## 🎯 Features

- **Keyword Search**: Classic BM25-style search over movie metadata  
- **Semantic Search**: Vector-based retrieval using text embeddings  
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) combining keyword and semantic signals  
- **Query Enhancement**: Spell correction, query rewriting, and expansion  
- **LLM Reranking**: Cross-encoder and LLM-based reranking  
- **Retrieval-Augmented Generation (RAG)**: Contextual answer generation from retrieved documents  
- **Multi-Document Summarization**: Synthesis across multiple movies  
- **Citation-Aware Answers**: Inline citations referencing source documents  
- **Conversational Q&A**: Natural, chat-style interaction  
- **Multimodal Query Rewriting**: Query enhancement using image understanding  
- **Image-Based Search**: Movie retrieval via poster images using CLIP embeddings  
- **Search Evaluation**: Precision@k, Recall@k, and F1 scoring with golden datasets  

---

## 📁 Project Structure

```text
rag-search-engine/
├── cli/
│   ├── augmented_generation_cli.py      # RAG, summarization, citations, Q&A
│   ├── describe_image_cli.py            # Multimodal query rewriting (image + text)
│   ├── evaluation_cli.py                # Precision/Recall/F1 evaluation
│   ├── hybrid_search_cli.py             # Hybrid RRF search with reranking
│   ├── keyword_search_cli.py            # BM25 keyword search
│   ├── multimodal_search_cli.py         # CLIP-based image search
│   ├── semantic_search_cli.py           # Vector/embedding search
│   └── lib/
│       ├── config.py                    # Centralized configuration
│       ├── hybrid_search.py             # BM25 + embeddings + RRF logic
│       ├── keyword_search.py            # Keyword search implementation
│       ├── multimodal_search.py         # CLIP multimodal search logic
│       ├── query_enhancement.py          # Query enhancement and reranking
│       ├── reranker.py                  # Cross-encoder reranking
│       ├── search_utils.py              # Dataset loading and utilities
│       └── semantic_search.py            # Semantic search implementation
├── data/
│   ├── movies.json                      # Movie dataset
│   ├── golden_dataset.json              # Evaluation queries
│   └── paddington.jpeg                  # Example image for multimodal search
├── pyproject.toml                       # Project dependencies
└── README.md
````

---

## 🚀 Installation

### Prerequisites

* Python **3.12+**
* `uv` (recommended) or `pip`

### Clone and Install

```bash
git clone https://github.com/Utkarsh736/rag-search-engine.git
cd rag-search-engine

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from **Google AI Studio**.

---

## 📖 Usage Guide

Run all commands from the project root using `uv run`.

---

### 1️⃣ Keyword Search

```bash
uv run cli/keyword_search_cli.py search "bear in london"
```

Classic BM25-style keyword matching over movie titles and descriptions.

---

### 2️⃣ Semantic Search

```bash
uv run cli/semantic_search_cli.py search "talking teddy bear comedy"
```

Finds semantically similar movies using embeddings, even without exact keyword overlap.

---

### 3️⃣ Hybrid Search (RRF)

```bash
# Basic hybrid search
uv run cli/hybrid_search_cli.py rrf-search "bear in london" --limit 5

# With cross-encoder reranking
uv run cli/hybrid_search_cli.py rrf-search "dinosaur" --rerank-method cross_encoder

# With query enhancement
uv run cli/hybrid_search_cli.py rrf-search "scary ber atack" --enhance spell
```

Combines keyword and semantic rankings using **Reciprocal Rank Fusion**.

---

### 4️⃣ Search Evaluation

```bash
uv run cli/evaluation_cli.py --limit 4
```

Evaluates performance using Precision@k, Recall@k, and F1 on test queries.

---

### 5️⃣ Retrieval-Augmented Generation (RAG)

```bash
uv run cli/augmented_generation_cli.py rag "what dinosaur movies are available?"
```

Generates contextual answers grounded in retrieved documents.

---

### 6️⃣ Multi-Document Summarization

```bash
uv run cli/augmented_generation_cli.py summarize "action adventure movies" --limit 10
```

Produces a synthesized overview across multiple movies.

---

### 7️⃣ Citation-Aware Answers

```bash
uv run cli/augmented_generation_cli.py citations "sci-fi movies with robots"
```

Generates answers with inline citations like `[1]`, `[2]`.

---

### 8️⃣ Conversational Question Answering

```bash
# Factual question
uv run cli/augmented_generation_cli.py question "when was Jurassic Park released?"

# Analytical question
uv run cli/augmented_generation_cli.py question "which bear movies are most intense?"
```

Chat-style answers suitable for conversational interfaces.

---

### 9️⃣ Multimodal Query Rewriting

```bash
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "funny bear movie"
```

Enhances queries using image understanding from multimodal LLMs.

**Example output:**

```text
Rewritten query: family-friendly British comedy film featuring anthropomorphic bear in blue coat and red hat
Total tokens: 1247
```

---

### 🔟 Image-Based Movie Search

```bash
# Verify image embeddings
uv run cli/multimodal_search_cli.py verify_image_embedding data/paddington.jpeg

# Search movies by image
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

Find movies by poster images using CLIP embeddings and cosine similarity.

**Example output:**

```text
1. Paddington (similarity: 0.722)
   Deep in the rainforests of Peru, a young bear lives peacefully...

2. Ted (similarity: 0.685)
   In 1985, eight-year-old John Bennett makes a Christmas wish...
```

---

## 🔧 Key Components

### Multimodal Search

Located in `cli/lib/multimodal_search.py`:

* Encodes movie text as `title: description`
* Generates text and image embeddings using CLIP
* Ranks results via cosine similarity

### Hybrid Search Pipeline

Located in `cli/lib/hybrid_search.py`:

* BM25 keyword ranking
* Semantic vector search
* Reciprocal Rank Fusion (RRF)
* Optional LLM-based reranking

### RAG Pipeline

Located in `cli/augmented_generation_cli.py`:

* Hybrid document retrieval
* Context-aware generation
* Supports summaries, citations, and Q&A

---

## 🤖 Models Used

| Component        | Model                                  | Purpose                 |
| ---------------- | -------------------------------------- | ----------------------- |
| Text Embeddings  | `all-MiniLM-L6-v2`                     | Semantic search         |
| Image Embeddings | `clip-ViT-B-32`                        | Multimodal search       |
| Cross-Encoder    | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Result reranking        |
| LLM              | `gemini-2.5-flash-lite`                | RAG, Q&A, summarization |

All models are configurable via `cli/lib/config.py`.

---

## 📊 Evaluation

A golden dataset (`data/golden_dataset.json`) is provided for benchmarking.

```bash
uv run cli/evaluation_cli.py --limit 4
```

Metrics:

* **Precision@k**
* **Recall@k**
* **F1 Score**

---

## 🛠️ Development

### Code Quality

```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```

---

## 🚧 Limitations & Future Work

### Current Limitations

* Small, static dataset
* Runtime embedding computation (no caching)
* LLM API latency and rate limits
* Limited joint text + image scoring strategies

### Future Enhancements

* 🌐 Web UI (FastAPI + React)
* 🗄️ Vector databases (pgvector, Pinecone, Weaviate)
* ⚖️ Tunable text vs image similarity weighting
* 📈 Advanced metrics (NDCG, MRR)
* 🎬 Live movie database integration

---

## 📝 License

MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* Built as part of the **Boot.dev RAG Search Engine** course
* CLIP model by OpenAI
* Sentence Transformers library
* Gemini API by Google
