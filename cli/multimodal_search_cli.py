#!/usr/bin/env python3

import argparse
import sys

sys.path.insert(0, '.')

from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify image embedding command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify image embedding generation"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )

    # Image search command
    search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies using an image"
    )
    search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            
            # Print results
            for i, result in enumerate(results, 1):
                # Truncate description to first 100 characters
                description = result['description'][:100]
                if len(result['description']) > 100:
                    description += "..."
                
                print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
                print(f"   {description}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
