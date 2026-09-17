#!/usr/bin/env python3
"""Root entry-point for Hugging Face Spaces."""
from demo.app import demo

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
