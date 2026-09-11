#!/usr/bin/env python3

import argparse
import json
import sys
import os

sys.path.insert(0, '.')

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, PROJECT_ROOT


def calculate_precision_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int) -> float:
    """Calculate precision@k: percentage of retrieved results that are relevant"""
    # Only consider top k retrieved titles
    retrieved_top_k = retrieved_titles[:k]
    
    # Count how many retrieved titles are in the relevant set
    relevant_set = set(relevant_titles)
    matches = sum(1 for title in retrieved_top_k if title in relevant_set)
    
    # Precision = matches / k
    precision = matches / k if k > 0 else 0.0
    return precision


def calculate_recall_at_k(retrieved_titles: list[str], relevant_titles: list[str], k: int) -> float:
    """Calculate recall@k: percentage of relevant docs that were retrieved"""
    # Only consider top k retrieved titles
    retrieved_top_k = retrieved_titles[:k]
    
    # Count how many relevant titles are in the retrieved set
    retrieved_set = set(retrieved_top_k)
    matches = sum(1 for title in relevant_titles if title in retrieved_set)
    
    # Recall = matches / total_relevant
    total_relevant = len(relevant_titles)
    recall = matches / total_relevant if total_relevant > 0 else 0.0
    return recall


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score: harmonic mean of precision and recall"""
    # F1 = 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load golden dataset
    golden_dataset_path = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
    
    with open(golden_dataset_path, "r") as f:
        golden_dataset = json.load(f)
    
    # Extract test cases from the JSON
    test_cases = golden_dataset["test_cases"]
    
    # Load documents and initialize hybrid search
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    # Print header
    print(f"k={limit}\n")
    
    # Evaluate each test case
    for test_case in test_cases:
        query = test_case["query"]
        relevant_titles = test_case["relevant_docs"]
        
        # Perform RRF search
        results = hybrid_search.rrf_search(query, k=60, limit=limit)
        
        # Extract retrieved titles
        retrieved_titles = [result["title"] for result in results]
        
        # Calculate precision@k, recall@k, and F1 score
        precision = calculate_precision_at_k(retrieved_titles, relevant_titles, limit)
        recall = calculate_recall_at_k(retrieved_titles, relevant_titles, limit)
        f1 = calculate_f1_score(precision, recall)
        
        # Print results
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()
