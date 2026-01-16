import os
from dotenv import load_dotenv
from google import genai


def enhance_query_spell(query: str) -> str:
    """Use Gemini to fix spelling errors in a query"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    # System prompt for spell correction
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    
    # Get response from Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Extract and clean the corrected query
    corrected = response.text.strip()
    
    # Remove quotes if Gemini added them
    if corrected.startswith('"') and corrected.endswith('"'):
        corrected = corrected[1:-1]
    if corrected.startswith("'") and corrected.endswith("'"):
        corrected = corrected[1:-1]
    
    return corrected


def enhance_query_rewrite(query: str) -> str:
    """Use Gemini to rewrite a vague query to be more specific and searchable"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    # System prompt for query rewriting
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    
    # Get response from Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Extract and clean the rewritten query
    rewritten = response.text.strip()
    
    # Remove quotes if Gemini added them
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1]
    if rewritten.startswith("'") and rewritten.endswith("'"):
        rewritten = rewritten[1:-1]
    
    return rewritten


def enhance_query_expand(query: str) -> str:
    """Use Gemini to expand a query with related terms and synonyms"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    # System prompt for query expansion
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    
    # Get response from Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Extract and clean the expanded query
    expanded = response.text.strip()
    
    # Remove quotes if Gemini added them
    if expanded.startswith('"') and expanded.endswith('"'):
        expanded = expanded[1:-1]
    if expanded.startswith("'") and expanded.endswith("'"):
        expanded = expanded[1:-1]
    
    return expanded
