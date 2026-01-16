#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.query_enhancement import enhance_query_spell, enhance_query_rewrite, enhance_query_expand, rerank_individual, rerank_batch, rerank_cross_encoder
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
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method"
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM-based re-ranking method"
    )
    
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
            # Handle query enhancement
            query = args.query
            
            if hasattr(args, 'enhance') and args.enhance:
                if args.enhance == "spell":
                    enhanced_query = enhance_query_spell(query)
                    if enhanced_query != query:
                        print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
                        query = enhanced_query
                
                elif args.enhance == "rewrite":
                    enhanced_query = enhance_query_rewrite(query)
                    if enhanced_query != query:
                        print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'\n")
                        query = enhanced_query
                
                elif args.enhance == "expand":
                    enhanced_query = enhance_query_expand(query)
                    if enhanced_query != query:
                        print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'\n")
                        query = enhanced_query
            
            # Load documents
            documents = load_movies()
            
            # Initialize hybrid search
            hybrid_search = HybridSearch(documents)
            
            # Determine how many results to fetch
            fetch_limit = args.limit
            if hasattr(args, 'rerank_method') and args.rerank_method in ["individual", "batch", "cross_encoder"]:
                fetch_limit = args.limit * 5  # Get 5x more for reranking
            
            # Perform RRF search with (possibly enhanced) query
            results = hybrid_search.rrf_search(query, args.k, fetch_limit)
            
            # Apply reranking if requested
            if hasattr(args, 'rerank_method') and args.rerank_method:
                if args.rerank_method == "individual":
                    print(f"Reranking top {fetch_limit} results using individual method...")
                    results = rerank_individual(query, results)
                    results = results[:args.limit]
                    print(f"\nReciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                
                elif args.rerank_method == "batch":
                    print(f"Reranking top {fetch_limit} results using batch method...")
                    results = rerank_batch(query, results)
                    results = results[:args.limit]
                    print(f"\nReciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                
                elif args.rerank_method == "cross_encoder":
                    print(f"Reranking top {fetch_limit} results using cross_encoder method...")
                    results = rerank_cross_encoder(query, results)
                    results = results[:args.limit]
                    print(f"\nReciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            
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
                
                # Show rerank score/rank based on method
                if 'rerank_score' in result:
                    print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                elif 'rerank_rank' in result:
                    print(f"   Rerank Rank: {result['rerank_rank']}")
                elif 'cross_encoder_score' in result:
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                
                print(f"   RRF Score: {result['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {doc_snippet}")


        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
