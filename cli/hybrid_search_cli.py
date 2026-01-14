#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Normalize command
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")
    
    # Weighted search command
    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search with weighted scoring")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 search (default: 0.5)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    
    # RRF search command
    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF constant k (default: 60)")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        
        case "weighted-search":
            # Load documents
            documents = load_movies()
            
            # Initialize hybrid search
            hybrid_search = HybridSearch(documents)
            
            # Perform weighted search
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            # Print results
            for i, result in enumerate(results, 1):
                # Truncate document to 100 characters
                doc_snippet = result['document'][:100]
                if len(result['document']) > 100:
                    doc_snippet += "..."
                
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {doc_snippet}")
        
        case "rrf-search":
            # Load documents
            documents = load_movies()
            
            # Initialize hybrid search
            hybrid_search = HybridSearch(documents)
            
            # Perform RRF search
            results = hybrid_search.rrf_search(args.query, args.k, args.limit)
            
            # Print results
            for i, result in enumerate(results, 1):
                # Truncate document to 100 characters
                doc_snippet = result['document'][:100]
                if len(result['document']) > 100:
                    doc_snippet += "..."
                
                # Format ranks (show "-" if not present in that search)
                bm25_rank = result['bm25_rank'] if result['bm25_rank'] is not None else "-"
                semantic_rank = result['semantic_rank'] if result['semantic_rank'] is not None else "-"
                
                print(f"{i}. {result['title']}")
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {doc_snippet}")
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
