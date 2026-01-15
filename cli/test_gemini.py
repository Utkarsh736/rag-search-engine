import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Generate content
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)

# Print the response text
print("\nResponse:")
print(response.text)

# Print token usage
print(f"\nPrompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")

