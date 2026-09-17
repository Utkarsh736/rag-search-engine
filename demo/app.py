#!/usr/bin/env python3
"""
Gradio demo for rag-search-engine.

Exposes the most impressive capabilities for Hugging Face Spaces / portfolio embedding:
  - Hybrid Search (BM25 + Semantic via RRF)
  - RAG answers with citations
  - Conversational Q&A
  - Image-based movie search (CLIP)

Run locally from repo root:
  uv run python demo/app.py

Or on Hugging Face Spaces (app.py at root that imports this, or set Space to use demo/app.py).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path bootstrap so both `python demo/app.py` and HF Space work
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cli"))
sys.path.insert(0, str(REPO_ROOT))

# Make PROJECT_ROOT consistent for the library code
os.environ.setdefault("RAG_PROJECT_ROOT", str(REPO_ROOT))

from lib.search_utils import load_movies, PROJECT_ROOT as LIB_PROJECT_ROOT  # noqa: E402
from lib.hybrid_search import HybridSearch  # noqa: E402
from lib.multimodal_search import MultimodalSearch  # noqa: E402
from lib.config import GEMINI_MODEL  # noqa: E402

# Override the library's PROJECT_ROOT if needed (library computes it relative to cli/lib)
# We already have data/ next to the real root.
if not Path(LIB_PROJECT_ROOT).joinpath("data", "movies.json").exists():
    # Force the correct root for cache + data
    import lib.search_utils as su
    su.PROJECT_ROOT = str(REPO_ROOT)
    su.DATA_PATH = str(REPO_ROOT / "data" / "movies.json")
    su.STOPWORDS_PATH = str(REPO_ROOT / "data" / "stopwords.txt")

load_dotenv(REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Global singletons (load models only once)
# ---------------------------------------------------------------------------
_documents: list[dict] | None = None
_hybrid: HybridSearch | None = None
_multimodal: MultimodalSearch | None = None
_gemini_client = None


def get_documents() -> list[dict]:
    global _documents
    if _documents is None:
        print("[demo] Loading movies dataset...")
        _documents = load_movies()
        print(f"[demo] Loaded {len(_documents)} movies")
    return _documents


def get_hybrid() -> HybridSearch:
    global _hybrid
    if _hybrid is None:
        print("[demo] Initializing HybridSearch (embeddings + BM25 index)...")
        # Ensure cache dir exists under the real repo root
        cache_dir = REPO_ROOT / "cache"
        cache_dir.mkdir(exist_ok=True)
        _hybrid = HybridSearch(get_documents())
        print("[demo] HybridSearch ready")
    return _hybrid


def get_multimodal() -> MultimodalSearch:
    global _multimodal
    if _multimodal is None:
        print("[demo] Initializing MultimodalSearch (CLIP)...")
        _multimodal = MultimodalSearch(documents=get_documents())
        print("[demo] MultimodalSearch ready")
    return _multimodal


def get_gemini():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_search_results(results: list[dict], score_key: str = "rrf_score") -> str:
    if not results:
        return "_No results found._"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Unknown")
        desc = (r.get("document") or r.get("description") or "")[:220]
        if len(r.get("document") or r.get("description") or "") > 220:
            desc += "…"
        score = r.get(score_key) or r.get("score") or r.get("similarity")
        score_str = f"  ·  score: {score:.4f}" if isinstance(score, (int, float)) else ""
        lines.append(f"**{i}. {title}**{score_str}\n{desc}")
    return "\n\n".join(lines)


def build_context(results: list[dict], limit: int = 5) -> str:
    parts = []
    for i, r in enumerate(results[:limit], 1):
        title = r.get("title", "Unknown")
        doc = r.get("document") or r.get("description") or ""
        parts.append(f"[{i}] {title}\n{doc}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def hybrid_search(query: str, limit: int = 5) -> str:
    if not query or not query.strip():
        return "Please enter a search query."
    try:
        hs = get_hybrid()
        results = hs.rrf_search(query.strip(), k=60, limit=int(limit))
        return format_search_results(results, score_key="rrf_score")
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"


def rag_answer(query: str, limit: int = 5, with_citations: bool = True) -> tuple[str, str]:
    """Returns (answer_markdown, sources_markdown)"""
    if not query or not query.strip():
        return "Please enter a question.", ""
    client = get_gemini()
    if client is None:
        return (
            "⚠️ **GEMINI_API_KEY is not set.**\n\n"
            "Hybrid search still works (see the Search tab). "
            "Add the key as a Hugging Face Space secret named `GEMINI_API_KEY` "
            "to enable RAG answers.",
            "",
        )
    try:
        hs = get_hybrid()
        results = hs.rrf_search(query.strip(), k=60, limit=int(limit))
        context = build_context(results, limit=int(limit))

        if with_citations:
            prompt = f"""You are a helpful movie assistant for a streaming service.
Answer the user's question using ONLY the provided documents.
Use inline citations like [1], [2] that refer to the document numbers.
Be concise (2-5 sentences) and useful.

Question: {query}

Documents:
{context}

Answer:"""
        else:
            prompt = f"""You are a helpful movie assistant for a streaming service.
Answer the user's question using the provided documents.
Be concise and useful.

Question: {query}

Documents:
{context}

Answer:"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text.strip()
        sources = format_search_results(results, score_key="rrf_score")
        return answer, sources
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}", ""


def image_search(image, limit: int = 5) -> str:
    if image is None:
        return "Please upload an image (movie poster, still, etc.)."
    try:
        # Gradio may give a path or a numpy array / PIL depending on version
        if isinstance(image, str):
            image_path = image
        else:
            # Save temporary file
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            if hasattr(image, "save"):
                image.save(tmp.name)
            else:
                from PIL import Image as PILImage
                import numpy as np
                PILImage.fromarray(np.array(image)).save(tmp.name)
            image_path = tmp.name

        ms = get_multimodal()
        results = ms.search_with_image(image_path, limit=int(limit))
        return format_search_results(results, score_key="similarity")
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; }
footer { display: none !important; }
"""

EXAMPLES_SEARCH = [
    "bear in london",
    "talking teddy bear comedy",
    "dinosaur movies",
    "sci-fi with robots",
    "intense survival thriller",
]

EXAMPLES_RAG = [
    "What dinosaur movies are available?",
    "Which bear movies are most intense?",
    "Recommend family-friendly movies with animals",
    "When was Jurassic Park released and what is it about?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Search Engine – Movie Demo",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown(
            """
# 🎬 RAG Search Engine – Interactive Demo

Explore the progression from **keyword → semantic → hybrid (RRF) → RAG** on a 5 000-movie dataset.

Built from the open-source [rag-search-engine](https://github.com/Utkarsh736/rag-search-engine) CLI project.
            """
        )

        with gr.Tabs():
            # ----- Tab 1: Hybrid Search -----
            with gr.Tab("🔍 Hybrid Search"):
                gr.Markdown(
                    "Combines **BM25 keyword** and **semantic embeddings** with "
                    "**Reciprocal Rank Fusion (RRF)**."
                )
                with gr.Row():
                    search_q = gr.Textbox(
                        label="Query",
                        placeholder="e.g. bear in london, talking teddy bear comedy…",
                        scale=4,
                    )
                    search_limit = gr.Slider(1, 15, value=5, step=1, label="Results", scale=1)
                search_btn = gr.Button("Search", variant="primary")
                search_out = gr.Markdown(label="Results")
                gr.Examples(EXAMPLES_SEARCH, inputs=search_q, label="Try these")
                search_btn.click(hybrid_search, inputs=[search_q, search_limit], outputs=search_out)
                search_q.submit(hybrid_search, inputs=[search_q, search_limit], outputs=search_out)

            # ----- Tab 2: RAG / Q&A -----
            with gr.Tab("🤖 RAG + Citations"):
                gr.Markdown(
                    "Retrieve relevant movies with hybrid search, then generate a grounded answer "
                    "using **Gemini**. Citations reference the retrieved documents."
                )
                with gr.Row():
                    rag_q = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What dinosaur movies are available?",
                        scale=4,
                    )
                    rag_limit = gr.Slider(1, 10, value=5, step=1, label="Context docs", scale=1)
                with gr.Row():
                    rag_btn = gr.Button("Ask", variant="primary")
                    cite_toggle = gr.Checkbox(value=True, label="Inline citations")
                rag_answer_md = gr.Markdown(label="Answer")
                rag_sources_md = gr.Markdown(label="Sources")
                gr.Examples(EXAMPLES_RAG, inputs=rag_q, label="Try these")
                rag_btn.click(
                    rag_answer,
                    inputs=[rag_q, rag_limit, cite_toggle],
                    outputs=[rag_answer_md, rag_sources_md],
                )
                rag_q.submit(
                    rag_answer,
                    inputs=[rag_q, rag_limit, cite_toggle],
                    outputs=[rag_answer_md, rag_sources_md],
                )

            # ----- Tab 3: Image Search -----
            with gr.Tab("🖼️ Image Search"):
                gr.Markdown(
                    "Upload a movie poster or still. **CLIP** embeds the image and finds "
                    "movies whose title+description are closest in the joint embedding space."
                )
                with gr.Row():
                    img_in = gr.Image(type="pil", label="Upload image", height=280)
                    with gr.Column():
                        img_limit = gr.Slider(1, 10, value=5, step=1, label="Results")
                        img_btn = gr.Button("Search by image", variant="primary")
                img_out = gr.Markdown(label="Similar movies")
                img_btn.click(image_search, inputs=[img_in, img_limit], outputs=img_out)

        gr.Markdown(
            """
---
**Models**: `all-MiniLM-L6-v2` · `clip-ViT-B-32` · `gemini-2.5-flash-lite`  
**Note**: First request after cold start downloads models and builds embeddings (can take 1–2 min).  
Source: [github.com/Utkarsh736/rag-search-engine](https://github.com/Utkarsh736/rag-search-engine)
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
demo = build_ui()

if __name__ == "__main__":
    # Pre-warm the hybrid index so the first user request is faster
    print("[demo] Pre-warming HybridSearch…")
    try:
        get_hybrid()
    except Exception as e:
        print(f"[demo] Pre-warm failed (will retry on first request): {e}")

    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
