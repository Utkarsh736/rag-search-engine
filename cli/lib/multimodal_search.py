"""Multimodal search using CLIP for image and text embeddings"""

import numpy as np
from numpy.linalg import norm
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    """Multimodal search using CLIP model"""
    
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        """Initialize with a CLIP model and optional documents"""
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []
        
        # Create text representations for each document
        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in self.documents
        ]
        
        # Generate text embeddings for all documents
        if self.texts:
            print("Generating text embeddings for documents...")
            self.text_embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True
            )
        else:
            self.text_embeddings = []
    
    def embed_image(self, image_path: str):
        """Generate embedding for an image at the given path"""
        # Load image using PIL
        image = Image.open(image_path)
        
        # Encode the image (pass as single-item list, get first result)
        embedding = self.model.encode([image])[0]
        
        return embedding
    
    def search_with_image(self, image_path: str, limit: int = 5):
        """Search for similar documents using an image"""
        # Generate embedding for the query image
        image_embedding = self.embed_image(image_path)
        
        # Calculate cosine similarity with each text embedding
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            # Cosine similarity: dot product / (magnitude_a * magnitude_b)
            similarity = np.dot(image_embedding, text_embedding) / (
                norm(image_embedding) * norm(text_embedding)
            )
            
            results.append({
                'id': i,
                'title': self.documents[i]['title'],
                'description': self.documents[i]['description'],
                'similarity': float(similarity)
            })
        
        # Sort by similarity (descending) and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str):
    """Verify that image embedding generation works"""
    # Create multimodal search instance
    multimodal_search = MultimodalSearch()
    
    # Generate embedding for the image
    embedding = multimodal_search.embed_image(image_path)
    
    # Print embedding shape
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    """Search for movies using an image"""
    from lib.search_utils import load_movies
    
    # Load movie dataset
    documents = load_movies()
    
    # Create multimodal search instance with documents
    multimodal_search = MultimodalSearch(documents=documents)
    
    # Search with the image
    results = multimodal_search.search_with_image(image_path)
    
    return results
