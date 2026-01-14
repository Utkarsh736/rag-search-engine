#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify model
    subparsers.add_parser("verify", help="Verify the embedding model is loaded")
    
    # Embed text
    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    
    # Verify embeddings
    subparsers.add_parser("verify_embeddings", help="Generate and verify embeddings for all movies")
    
    # Embed query
    embedquery_parser = subparsers.add_parser("embedquery", help="Generate embedding for a search query")
    embedquery_parser.add_argument("query", type=str, help="Search query to embed")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()
        
        case "embedquery":
            embed_query_text(args.query)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
