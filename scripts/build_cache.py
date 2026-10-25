#!/usr/bin/env python3
"""Build cache/ from data/movies.json and write cache/manifest.json.

Wraps the existing builders (no reimplementation):
    - InvertedIndex.build() + save()                     (BM25 artifacts)
    - ChunkedSemanticSearch.build_chunk_embeddings()     (semantic artifacts)

Usage:
    python scripts/build_cache.py                # build if missing
    python scripts/build_cache.py --force        # rebuild unconditionally
    python scripts/build_cache.py --verify-only  # verify only, no build
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cli"))

from lib.search_utils import load_movies, PROJECT_ROOT            # noqa: E402
from lib.inverted_index import InvertedIndex                       # noqa: E402
from lib.semantic_search import ChunkedSemanticSearch              # noqa: E402
from lib.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP  # noqa: E402
from lib.constants import BM25_K1, BM25_B                          # noqa: E402

CACHE_DIR = Path(PROJECT_ROOT) / "cache"
DATA_PATH = Path(PROJECT_ROOT) / "data" / "movies.json"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

REQUIRED_ARTIFACTS = [
    "index.pkl",
    "docmap.pkl",
    "term_frequencies.pkl",
    "doc_lengths.pkl",
    "chunk_embeddings.npy",
    "chunk_metadata.json",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def artifacts_present() -> bool:
    return all((CACHE_DIR / n).exists() for n in REQUIRED_ARTIFACTS)


def verify_only() -> int:
    if not MANIFEST_PATH.exists():
        print(
            f"[build_cache] ERROR: {MANIFEST_PATH} not found. "
            "Run without --verify-only to build.",
            file=sys.stderr,
        )
        return 1
    if not artifacts_present():
        missing = [n for n in REQUIRED_ARTIFACTS if not (CACHE_DIR / n).exists()]
        print(f"[build_cache] ERROR: missing artifacts: {missing}", file=sys.stderr)
        return 1
    if not DATA_PATH.exists():
        print(
            f"[build_cache] ERROR: {DATA_PATH} missing. "
            "Run scripts/download_data.py first.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    actual = sha256_of(DATA_PATH)
    if manifest.get("movies_sha256") != actual:
        print(
            f"[build_cache] ERROR: manifest.movies_sha256 does not match data/movies.json.\n"
            f"  manifest: {manifest.get('movies_sha256')}\n"
            f"  actual:   {actual}\n"
            f"  Rebuild with: python scripts/build_cache.py --force",
            file=sys.stderr,
        )
        return 1
    print(f"[build_cache] OK: {MANIFEST_PATH}")
    return 0


def build(force: bool) -> int:
    if not DATA_PATH.exists():
        print(
            f"[build_cache] ERROR: {DATA_PATH} missing. "
            "Run scripts/download_data.py first.",
            file=sys.stderr,
        )
        return 1

    if not force and artifacts_present() and MANIFEST_PATH.exists():
        if json.loads(MANIFEST_PATH.read_text()).get("movies_sha256") == sha256_of(DATA_PATH):
            print("[build_cache] cache is present and manifest matches data. Use --force to rebuild.")
            return 0
        print("[build_cache] manifest does not match data/movies.json; rebuilding.")

    documents = load_movies()
    print(f"[build_cache] loaded {len(documents)} documents")

    print("[build_cache] building BM25 inverted index...")
    idx = InvertedIndex()
    idx.build()
    idx.save()

    print("[build_cache] building chunk embeddings...")
    sem = ChunkedSemanticSearch(model_name=EMBEDDING_MODEL)
    sem.build_chunk_embeddings(documents)

    manifest = {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "movies_sha256": sha256_of(DATA_PATH),
        "num_documents": len(documents),
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"[build_cache] wrote {MANIFEST_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    return verify_only() if args.verify_only else build(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
