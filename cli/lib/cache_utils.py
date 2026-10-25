"""Cache manifest utilities.

cache/manifest.json records the exact config that produced the cached
artifacts, so a stale cache cannot silently serve wrong vectors after you
change the embedding model, chunk size, or BM25 params.

verify_manifest() is called from HybridSearch.__init__:
  - missing manifest  -> warn (or raise if RAG_STRICT_CACHE=1)
  - manifest mismatch -> always raise (that is a data-integrity bug)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from .constants import BM25_K1, BM25_B
from .search_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(PROJECT_ROOT) / "cache" / "manifest.json"


class CacheManifestError(RuntimeError):
    """Raised when cache/manifest.json does not match the current config."""


def _expected() -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
    }


def verify_manifest(strict: bool | None = None) -> dict:
    if strict is None:
        strict = os.environ.get("RAG_STRICT_CACHE") == "1"

    if not MANIFEST_PATH.exists():
        msg = (
            f"{MANIFEST_PATH} not found. "
            "Run `python scripts/build_cache.py` to build and generate it."
        )
        if strict:
            raise CacheManifestError(msg)
        logger.warning(msg)
        return {}

    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    expected = _expected()
    mismatches = [
        f"{key}: manifest={manifest.get(key)!r} current={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise CacheManifestError(
            "Cache manifest does not match current config:\n  "
            + "\n  ".join(mismatches)
            + "\nRebuild with `python scripts/build_cache.py --force`."
        )

    return manifest
