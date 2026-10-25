"""Configuration for the RAG search engine"""

# Gemini model to use for all LLM operations
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Embedding model + chunking parameters.
# Recorded in cache/manifest.json. Changing any of these invalidates the
# cache; rebuild with `python scripts/build_cache.py --force`.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 4
CHUNK_OVERLAP = 1

