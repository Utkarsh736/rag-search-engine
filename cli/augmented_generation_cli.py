#!/usr/bin/env python3

import argparse
import sys
import os
from dotenv import load_dotenv
from google import genai

sys.path.insert(0, '.')

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from lib.config import GEMINI_MODEL


def perform_rag(query: str):
    """Perform Retrieval Augmented Generation"""
    # Load documents and perform RRF search
    print(f"[RAG] Searching for: '{query}'")
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    # Get top 5 results using RRF
    results = hybrid_search.rrf_search(query, k=60, limit=5)
    
    # Format documents for the LLM prompt
    docs = []
    for i, result in enumerate(results, 1):
        docs.append(f"{i}. {result['title']}\n   {result['document']}")
    
    docs_text = "\n\n".join(docs)
    
    # Create prompt for LLM
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs_text}

Provide a comprehensive answer that addresses the query:"""
    
    # Call Gemini API
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    print("[RAG] Generating response...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    # Print results
    print("\nSearch Results:")
    for result in results:
        print(f"  - {result['title']}")
    
    print("\nRAG Response:")
    print(response.text)


def perform_summarization(query: str, limit: int = 5):
    """Perform multi-document summarization"""
    # Load documents and perform RRF search
    print(f"[Summarize] Searching for: '{query}'")
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    # Get top N results using RRF
    results = hybrid_search.rrf_search(query, k=60, limit=limit)
    
    # Format search results for the LLM prompt
    results_text = []
    for i, result in enumerate(results, 1):
        results_text.append(f"{i}. {result['title']}\n   {result['document']}")
    
    results_formatted = "\n\n".join(results_text)
    
    # Create prompt for multi-document summarization
    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{results_formatted}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    
    # Call Gemini API
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    print("[Summarize] Generating summary...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    # Print results
    print("\nSearch Results:")
    for result in results:
        print(f"  - {result['title']}")
    
    print("\nLLM Summary:")
    print(response.text)


def perform_citations(query: str, limit: int = 5):
    """Perform citation-aware answer generation"""
    # Load documents and perform RRF search
    print(f"[Citations] Searching for: '{query}'")
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    # Get top N results using RRF
    results = hybrid_search.rrf_search(query, k=60, limit=limit)
    
    # Format documents with numbered citations
    documents_text = []
    for i, result in enumerate(results, 1):
        documents_text.append(f"[{i}] {result['title']}\n{result['document']}")
    
    documents_formatted = "\n\n".join(documents_text)
    
    # Create prompt for citation-aware generation
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents_formatted}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    
    # Call Gemini API
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    print("[Citations] Generating answer with citations...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    # Print results
    print("\nSearch Results:")
    for result in results:
        print(f"  - {result['title']}")
    
    print("\nLLM Answer:")
    print(response.text)


def perform_question_answering(question: str, limit: int = 5):
    """Perform conversational question answering"""
    # Load documents and perform RRF search
    print(f"[Question] Searching for: '{question}'")
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    
    # Get top N results using RRF
    results = hybrid_search.rrf_search(question, k=60, limit=limit)
    
    # Format documents as context
    context_text = []
    for i, result in enumerate(results, 1):
        context_text.append(f"{i}. {result['title']}\n{result['document']}")
    
    context = "\n\n".join(context_text)
    
    # Create prompt for question answering
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    
    # Call Gemini API
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)
    
    print("[Question] Generating answer...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    # Print results
    print("\nSearch Results:")
    for result in results:
        print(f"  - {result['title']}")
    
    print("\nAnswer:")
    print(response.text)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # RAG command
    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Multi-document summarization of search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize (default: 5)"
    )

    # Citations command
    citations_parser = subparsers.add_parser(
        "citations", help="Generate answer with source citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citation-aware answer")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use (default: 5)"
    )

    # Question command
    question_parser = subparsers.add_parser(
        "question", help="Conversational question answering"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            perform_rag(query)
        case "summarize":
            query = args.query
            limit = args.limit
            perform_summarization(query, limit)
        case "citations":
            query = args.query
            limit = args.limit
            perform_citations(query, limit)
        case "question":
            question = args.question
            limit = args.limit
            perform_question_answering(question, limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
