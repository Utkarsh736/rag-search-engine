import os
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies, PROJECT_ROOT


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self):
        """Initialize the semantic search model"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text: str):
        """Generate an embedding for a single text input"""
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty or contain only whitespace")
        
        embeddings = self.model.encode([text])
        return embeddings[0]
    
    def build_embeddings(self, documents: list[dict]):
        """Generate embeddings for all documents and save to cache"""
        self.documents = documents
        
        # Build document map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Create string representations for each movie
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        
        # Generate embeddings with progress bar
        print("Generating embeddings...")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        
        # Save to cache
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        embeddings_path = os.path.join(cache_dir, "movie_embeddings.npy")
        np.save(embeddings_path, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
        """Load embeddings from cache or create them if not found"""
        self.documents = documents
        
        # Build document map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Check if cached embeddings exist
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        embeddings_path = os.path.join(cache_dir, "movie_embeddings.npy")
        
        if os.path.exists(embeddings_path):
            print("Loading cached embeddings...")
            self.embeddings = np.load(embeddings_path)
            
            # Verify cache is valid
            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                print("Cache size mismatch, rebuilding...")
        
        # Build embeddings from scratch
        return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search for documents using semantic similarity"""
        # Check if embeddings are loaded
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Calculate similarity scores for all documents
        results = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            doc = self.documents[i]
            
            results.append({
                'score': similarity,
                'title': doc['title'],
                'description': doc['description']
            })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top results
        return results[:limit]


def verify_model():
    """Verify the embedding model is loaded correctly"""
    semantic_search = SemanticSearch()
    
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    """Generate and display embedding for input text"""
    semantic_search = SemanticSearch()
    
    embedding = semantic_search.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    """Verify embeddings are generated correctly for all movies"""
    semantic_search = SemanticSearch()
    
    # Load movie documents
    documents = load_movies()
    
    # Load or create embeddings
    embeddings = semantic_search.load_or_create_embeddings(documents)
    
    # Print verification info
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    """Generate and display embedding for a search query"""
    semantic_search = SemanticSearch()
    
    # Generate embedding using existing method
    embedding = semantic_search.generate_embedding(query)
    
    # Print query and embedding info
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
