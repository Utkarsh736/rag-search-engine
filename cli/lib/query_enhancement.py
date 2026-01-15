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

