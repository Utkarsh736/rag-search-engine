import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        
        # Initialize semantic search with chunked embeddings
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        
        # Initialize keyword search index
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
    
    def _bm25_search(self, query, limit):
        """Perform BM25 keyword search"""
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query, alpha, limit=5):
        """Combine semantic and keyword search with weighted scoring"""
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")
    
    def rrf_search(self, query, k, limit=10):
        """Combine semantic and keyword search using Reciprocal Rank Fusion"""
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

