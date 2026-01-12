#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Load the JSON file
            with open("data/movies.json", "r") as f:
                data = json.load(f)
            
            # Get the list of movies
            movies = data["movies"]

            # Search: append matching movies to results
            results = []
            for movie in movies:
                if args.query in movie["title"]:
                    results.append(movie)

            # Sort by ID ascending
            results.sort(key=lambda m: m["id"])

            # Truncate to max 5
            results = results[:5]

            print(f"Searching for: {args.query}")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
