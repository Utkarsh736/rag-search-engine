#!/usr/bin/env python3

import argparse
import mimetypes
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, '.')

from lib.config import GEMINI_MODEL


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal query rewriting using image + text"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image"
    )

    args = parser.parse_args()

    # Determine MIME type of the image
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read image file in binary mode
    with open(args.image, "rb") as f:
        img = f.read()

    # Set up Gemini client
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    client = genai.Client(api_key=api_key)

    # System prompt for multimodal query rewriting
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    # Build parts list with prompt, image, and query
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    # Generate content using Gemini
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts
    )

    # Print results
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()

