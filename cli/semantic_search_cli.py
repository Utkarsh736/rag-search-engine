#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from lib.search_utils import load_movies


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
    
    # Search
    search_parser = subparsers.add_parser("search", help="Search for movies using semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
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
        
        case "search":
            # Create semantic search instance
            semantic_search = SemanticSearch()
            
            # Load movies and embeddings
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)
            
            # Perform search
            results = semantic_search.search(args.query, args.limit)
            
            # Print results
            for i, result in enumerate(results, 1):
                # Truncate description to ~100 chars
                desc = result['description']
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {desc}\n")
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
