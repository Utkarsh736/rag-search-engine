import os
import time
import json
from dotenv import load_dotenv
from google import genai
from lib.config import GEMINI_MODEL
from sentence_transformers import CrossEncoder


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
        model=GEMINI_MODEL,
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
        model=GEMINI_MODEL,
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
        model=GEMINI_MODEL,
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


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    """Use Gemini to individually re-rank each result"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    reranked_results = []
    
    for result in results:
        # System prompt for re-ranking
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {result.get("title", "")} - {result.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
        
        try:
            # Get response from Gemini
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            
            # Extract score
            score_text = response.text.strip()
            
            # Try to parse the score
            try:
                rerank_score = float(score_text)
            except ValueError:
                # If parsing fails, try to extract first number
                import re
                numbers = re.findall(r'\d+\.?\d*', score_text)
                rerank_score = float(numbers[0]) if numbers else 5.0
            
            # Add rerank score to result
            result['rerank_score'] = rerank_score
            reranked_results.append(result)
            
            # Sleep to avoid rate limits
            time.sleep(3)
            
        except Exception as e:
            print(f"Warning: Failed to rerank {result.get('title', 'Unknown')}: {e}")
            result['rerank_score'] = 5.0  # Default score on error
            reranked_results.append(result)
    
    # Sort by rerank score (descending)
    reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked_results


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    """Use Gemini to batch re-rank all results in one call"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    # Build the document list string
    doc_list_str = ""
    for idx, result in enumerate(results):
        doc_list_str += f"{result['id']}. {result['title']} - {result.get('document', '')[:200]}...\n"
    
    # System prompt for batch re-ranking
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    
    try:
        # Get response from Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        # Extract and parse JSON response
        response_text = response.text.strip()
        
        # Clean up response if needed (remove markdown code blocks)
        if response_text.startswith("```"):
            # Remove ```json or ``` prefix and suffix
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        # Parse JSON
        ranked_ids = json.loads(response_text)
        
        # Create a mapping of ID to result
        id_to_result = {result['id']: result for result in results}
        
        # Reorder results based on ranked IDs
        reranked_results = []
        for rank, doc_id in enumerate(ranked_ids, 1):
            if doc_id in id_to_result:
                result = id_to_result[doc_id]
                result['rerank_rank'] = rank
                reranked_results.append(result)
        
        # Add any results that weren't ranked (at the end)
        ranked_id_set = set(ranked_ids)
        for result in results:
            if result['id'] not in ranked_id_set:
                result['rerank_rank'] = len(reranked_results) + 1
                reranked_results.append(result)
        
        return reranked_results
        
    except Exception as e:
        print(f"Warning: Batch reranking failed: {e}")
        print(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        # Return original order with default ranks
        for rank, result in enumerate(results, 1):
            result['rerank_rank'] = rank
        return results


def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    """Use a cross-encoder model to re-rank results"""
    # Create pairs of [query, document]
    pairs = []
    for result in results:
        doc_text = f"{result.get('title', '')} - {result.get('document', '')}"
        pairs.append([query, doc_text])
    
    # Initialize cross-encoder model (does this once per query)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    
    # Compute scores for all pairs
    scores = cross_encoder.predict(pairs)
    
    # Add scores to results
    for i, result in enumerate(results):
        result['cross_encoder_score'] = float(scores[i])
    
    # Sort by cross-encoder score (descending)
    results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
    
    return results

def evaluate_results(query: str, results: list[dict]) -> list[int]:
    """Use Gemini to evaluate search result relevance"""
    # Load API key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    # Create Gemini client
    client = genai.Client(api_key=api_key)
    
    # Format results for the prompt
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']} - {result.get('document', '')[:200]}")
    
    # System prompt for evaluation
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    
    try:
        # Get response from Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        
        # Extract and parse JSON response
        response_text = response.text.strip()
        
        # Clean up response if needed (remove markdown code blocks)
        if response_text.startswith("```"):
            # Remove ```json or ``` prefix and suffix
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        
        # Parse JSON
        scores = json.loads(response_text)
        
        # Validate scores are 0-3
        scores = [max(0, min(3, int(score))) for score in scores]
        
        return scores
        
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        print(f"Response was: {response_text if 'response_text' in locals() else 'No response'}")
        # Return default scores (all 2s - "relevant")
        return [2 for _ in results]

