import os
import json
import re
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

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    """Split text into fixed-size chunks by word count with optional overlap"""
    # Split text into words
    words = text.split()
    
    # Handle edge cases
    if not words:
        return []
    
    chunks = []
    i = 0
    
    while i < len(words):
        # Get chunk_size words starting from position i
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap) words
        # If this is the last chunk, break to avoid infinite loop
        if i + chunk_size >= len(words):
            break
        
        i += chunk_size - overlap
    
    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    """Split text into chunks by sentence boundaries with optional overlap"""
    # Split text into sentences using regex
    # (?<=[.!?]) is a positive lookbehind: split after punctuation
    # \s+ matches one or more whitespace characters
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    # Remove empty strings
    sentences = [s for s in sentences if s.strip()]
    
    # Handle edge cases
    if not sentences:
        return []
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get max_chunk_size sentences starting from position i
        chunk_sentences = sentences[i:i + max_chunk_size]
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)
        
        # Break if this is the last chunk
        if i + max_chunk_size >= len(sentences):
            break
        
        # Move forward by (max_chunk_size - overlap) sentences
        i += max_chunk_size - overlap
    
    return chunks



class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the semantic search model"""
        self.model = SentenceTransformer(model_name)
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[dict]):
        """Generate embeddings for document chunks"""
        self.documents = documents
        
        # Build document map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        all_chunks = []  # List of chunk strings
        chunk_metadata = []  # List of metadata dicts
        
        # Process each document
        for movie_idx, doc in enumerate(documents):
            # Skip empty descriptions
            description = doc.get('description', '').strip()
            if not description:
                continue
            
            # Chunk the description (4 sentences per chunk, 1 sentence overlap)
            chunks = semantic_chunk_text(description, max_chunk_size=4, overlap=1)
            
            # Add chunks and metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': movie_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks)
                })
        
        # Generate embeddings for all chunks
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        # Save to cache
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_path = os.path.join(cache_dir, "chunk_embeddings.npy")
        np.save(embeddings_path, self.chunk_embeddings)
        
        # Save metadata as JSON
        metadata_path = os.path.join(cache_dir, "chunk_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """Load chunk embeddings from cache or create them"""
        self.documents = documents
        
        # Build document map
        for doc in documents:
            self.document_map[doc['id']] = doc
        
        # Check if cache exists
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        embeddings_path = os.path.join(cache_dir, "chunk_embeddings.npy")
        metadata_path = os.path.join(cache_dir, "chunk_metadata.json")
        
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            print("Loading cached chunk embeddings...")
            
            # Load embeddings
            self.chunk_embeddings = np.load(embeddings_path)
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata['chunks']
            
            return self.chunk_embeddings
        
        # Build from scratch
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """Search using chunk embeddings and aggregate results by movie"""
        # Check if chunk embeddings are loaded
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Store chunk scores
        chunk_scores = []
        
        # Calculate similarity for each chunk
        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            
            # Get movie index from metadata
            movie_idx = self.chunk_metadata[chunk_idx]['movie_idx']
            
            chunk_scores.append({
                'chunk_idx': chunk_idx,
                'movie_idx': movie_idx,
                'score': similarity
            })
        
        # Aggregate scores by movie (keep highest score per movie)
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score['movie_idx']
            score = chunk_score['score']
            
            # Update if this is the first score or a higher score
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        # Convert to list and sort by score (descending)
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top limit movies
        top_movies = sorted_movies[:limit]
        
        # Format results
        SCORE_PRECISION = 4
        results = []
        
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            
            result = {
                'id': doc['id'],
                'title': doc['title'],
                'document': doc['description'][:100],
                'score': round(score, SCORE_PRECISION),
                'metadata': doc.get('metadata', {})
            }
            results.append(result)
        
        return results

    

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



