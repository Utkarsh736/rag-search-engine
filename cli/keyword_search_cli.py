#!/usr/bin/env python3
import argparse
import sys
import string
import math

from nltk.stem import PorterStemmer

from lib.inverted_index import InvertedIndex
from lib.search_utils import load_stopwords
from lib.keyword_search import bm25_idf_command, bm25_tf_command
from lib.constants import BM25_K1

def normalize_to_tokens(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()
    stopwords = set(load_stopwords())

    clean = text.translate(translator).lower()
    tokens = [t for t in clean.split() if t]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a term in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

    args = parser.parse_args()

    match args.command:
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
            print("Built and saved inverted index.")

        case "search":
            print(f"Searching for: {args.query}")

            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)

            query_tokens = normalize_to_tokens(args.query)

            results: list[int] = []
            seen = set()

            for token in query_tokens:
                doc_ids = idx.get_documents(token)
                for doc_id in doc_ids:
                    if doc_id not in seen:
                        seen.add(doc_id)
                        results.append(doc_id)
                        if len(results) >= 5:
                            break
                if len(results) >= 5:
                    break

            for i, doc_id in enumerate(results, 1):
                movie = idx.docmap[doc_id]
                print(f"{i}. {movie['title']} ({doc_id})")
        
        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)
            
            tf = idx.get_tf(args.doc_id, args.term)
            print(tf)
        
        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)
            
            tokens = normalize_to_tokens(args.term)
            
            if len(tokens) == 0:
                print(f"Inverse document frequency of '{args.term}': 0.00")
                sys.exit(0)
            if len(tokens) > 1:
                print("Error: idf expects a single token term")
                sys.exit(1)
            
            token = tokens[0]
            
            # Calculate IDF: log((N + 1) / (n_t + 1))
            N = len(idx.docmap)
            doc_ids = idx.get_documents(token)
            n_t = len(doc_ids)
            
            idf = math.log((N + 1) / (n_t + 1))
            
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        
        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)
            
            tokens = normalize_to_tokens(args.term)
            
            if len(tokens) == 0:
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': 0.00")
                sys.exit(0)
            if len(tokens) > 1:
                print("Error: tfidf expects a single token term")
                sys.exit(1)
            
            token = tokens[0]
            
            # Calculate TF
            tf = idx.get_tf(args.doc_id, token)
            
            # Calculate IDF
            N = len(idx.docmap)
            doc_ids = idx.get_documents(token)
            n_t = len(doc_ids)
            idf = math.log((N + 1) / (n_t + 1))
            
            # Calculate TF-IDF
            tf_idf = tf * idf
            
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)

        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except FileNotFoundError:
                print("Error: index not found. Run: uv run cli/keyword_search_cli.py build")
                sys.exit(1)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

