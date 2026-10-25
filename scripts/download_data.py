#!/usr/bin/env python3
"""Fetch data/movies.json (~26 MB) into the repo.

Mirrors the runtime download in the HF Space's demo/app.py so the dev repo
and the deployed Space share one source of truth.

Usage:
    python scripts/download_data.py                # download if missing/corrupt
    python scripts/download_data.py --force        # redownload unconditionally
    python scripts/download_data.py --verify-only  # verify only, no network
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Mirrors demo/app.py in the HF Space repo.
MOVIES_URL = (
    "https://storage.googleapis.com/qvault-webapp-dynamic-assets/"
    "course_assets/course-rag-movies.json"
)

EXPECTED_SHA256: str | None = (
    "6a6d3311d0ecd864d1d9b3eed24e44c9f56a1ff2139b40d981936a07d4706070"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "movies.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download_data] downloading {MOVIES_URL}")
    with urllib.request.urlopen(MOVIES_URL, timeout=120) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    expected = EXPECTED_SHA256
    if expected and expected.startswith("REPLACE_WITH"):
        print(
            "[download_data] ERROR: EXPECTED_SHA256 not set.\n"
            "  Run `sha256sum data/movies.json` and paste the value into this file.",
            file=sys.stderr,
        )
        return 2

    if args.verify_only:
        if not DATA_PATH.exists():
            print(f"[download_data] ERROR: {DATA_PATH} does not exist.", file=sys.stderr)
            return 1
        actual = sha256_of(DATA_PATH)
        if expected and actual != expected:
            print(
                f"[download_data] MISMATCH\n  expected {expected}\n  got      {actual}",
                file=sys.stderr,
            )
            return 1
        print(f"[download_data] OK: {DATA_PATH}")
        return 0

    if DATA_PATH.exists() and not args.force:
        actual = sha256_of(DATA_PATH)
        if expected and actual == expected:
            print(f"[download_data] {DATA_PATH} already present and verified.")
            return 0
        if expected:
            print("[download_data] hash mismatch on existing file — refetching.")

    download(DATA_PATH)
    actual = sha256_of(DATA_PATH)
    if expected and actual != expected:
        print(
            f"[download_data] FATAL: post-download hash mismatch.\n"
            f"  expected {expected}\n  got      {actual}",
            file=sys.stderr,
        )
        return 1
    if not expected:
        print(
            f"[download_data] WARNING: EXPECTED_SHA256 is unset. Recorded hash:\n"
            f"  {actual}\nPaste this into scripts/download_data.py to enable verification."
        )
    print(f"[download_data] OK: {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
