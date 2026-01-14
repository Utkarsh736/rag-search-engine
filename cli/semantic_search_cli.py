#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text, 
    SemanticSearch,
    ChunkedSemanticSearch,
    chunk_text,
    semantic_chunk_text,  
)
from lib.search_utils import load_movies

def search_chunked(query: str, limit: int = 5):
    """Search movies using chunked semantic search"""
    # Load movies
    documents = load_movies()
    
    # Initialize chunked search
    chunked_search = ChunkedSemanticSearch()
    
    # Load or create chunk embeddings
    chunked_search.load_or_create_chunk_embeddings(documents)
    
    # Perform search
    results = chunked_search.search_chunks(query, limit=limit)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")


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
    
    # Chunk
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks") 

    # Semantic chunk
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text into semantic chunks by sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks")
    
    # Embed chunks
    subparsers.add_parser("embed_chunks", help="Generate chunked embeddings for all movies")

    # Search chunked
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies using chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

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
            semantic_search = SemanticSearch()
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            
            for i, result in enumerate(results, 1):
                desc = result['description']
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {desc}\n")
        
        case "chunk":
            # Get arguments
            chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else 200
            overlap = args.overlap if hasattr(args, 'overlap') else 0
            
            # Chunk the text
            chunks = chunk_text(args.text, chunk_size, overlap)
            
            # Print results
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            max_chunk_size = args.max_chunk_size if hasattr(args, 'max_chunk_size') else 4
            overlap = args.overlap if hasattr(args, 'overlap') else 0
            chunks = semantic_chunk_text(args.text, max_chunk_size, overlap)
            
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            # Load documents
            documents = load_movies()
            
            # Create chunked semantic search instance
            chunked_search = ChunkedSemanticSearch()
            
            # Load or build chunk embeddings
            embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
            
            # Print results
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked": 
            search_chunked(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
