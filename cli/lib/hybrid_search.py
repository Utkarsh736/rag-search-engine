import os

from .inverted_index import InvertedIndex 
from .semantic_search import ChunkedSemanticSearch
from .search_utils import PROJECT_ROOT


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores using min-max normalization"""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # If all scores are the same, return list of 1.0s
    if min_score == max_score:
        return [1.0] * len(scores)
    
    # Min-max normalization: (x - min) / (max - min)
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized

def rrf_score(rank: int, k: int = 60) -> float:
    """Calculate Reciprocal Rank Fusion score"""
    return 1.0 / (k + rank)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        
        # Initialize semantic search with chunked embeddings
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        
        # Initialize keyword search index
        self.idx = InvertedIndex()
        
        # Check if index cache exists, otherwise build
        cache_dir = os.path.join(PROJECT_ROOT, "cache")
        index_path = os.path.join(cache_dir, "index.pkl")
        
        if not os.path.exists(index_path):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()
    
    def _bm25_search(self, query, limit):
        """Perform BM25 keyword search"""
        # Make sure index is loaded
        if not self.idx.index:
            self.idx.load()
        
        # Get BM25 results (returns list of tuples: [(doc_id, score), ...])
        bm25_results = self.idx.bm25_search(query, limit)
        
        # Convert to format matching semantic results
        results = []
        for doc_id, score in bm25_results:
            doc = self.idx.docmap[doc_id]
            results.append({
                'id': doc_id,
                'title': doc['title'],
                'document': doc.get('description', ''),
                'score': score
            })
        
        return results
    
    def weighted_search(self, query, alpha, limit=5):
        """Combine semantic and keyword search with weighted scoring"""
        # Get BM25 results (use fixed limit for better overlap)
        bm25_results = self._bm25_search(query, 1000)
        
        # Get semantic results (use fixed limit for better overlap)
        semantic_results = self.semantic_search.search_chunks(query, 1000)
        
        # Extract scores for normalization
        bm25_scores = [result['score'] for result in bm25_results]
        semantic_scores = [result['score'] for result in semantic_results]
        
        # Normalize scores
        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_semantic = normalize_scores(semantic_scores)
        
        # Create mapping of document ID to scores
        doc_scores = {}
        
        # Add BM25 scores
        for i, result in enumerate(bm25_results):
            doc_id = result['id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'id': doc_id,
                    'title': result['title'],
                    'document': result['document'],
                    'bm25_score': normalized_bm25[i],
                    'semantic_score': 0.0
                }
            else:
                doc_scores[doc_id]['bm25_score'] = normalized_bm25[i]
        
        # Add semantic scores
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'id': doc_id,
                    'title': result['title'],
                    'document': result['document'],
                    'bm25_score': 0.0,
                    'semantic_score': normalized_semantic[i]
                }
            else:
                doc_scores[doc_id]['semantic_score'] = normalized_semantic[i]
        
        # Calculate hybrid scores: alpha * semantic + (1 - alpha) * bm25
        for doc_id in doc_scores:
            doc = doc_scores[doc_id]
            doc['hybrid_score'] = alpha * doc['semantic_score'] + (1 - alpha) * doc['bm25_score']
            doc['hybrid_score'] = alpha * doc['bm25_score'] + (1 - alpha) * doc['semantic_score']
        
        # Sort by hybrid score (descending)
        sorted_results = sorted(doc_scores.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        # Return top limit results
        return sorted_results[:limit]
    
    def rrf_search(self, query, k, limit=10):
        """Combine semantic and keyword search using Reciprocal Rank Fusion"""
        # Get BM25 results (1000 for better coverage)
        bm25_results = self._bm25_search(query, 1000)
        
        # Get semantic results (1000 for better coverage)
        semantic_results = self.semantic_search.search_chunks(query, 1000)
        
        # Create rank mappings (rank starts at 1)
        bm25_ranks = {result['id']: rank + 1 for rank, result in enumerate(bm25_results)}
        semantic_ranks = {result['id']: rank + 1 for rank, result in enumerate(semantic_results)}
        
        # Get all unique document IDs from both searches
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        
        # Calculate RRF scores for each document
        doc_scores = {}
        
        for doc_id in all_doc_ids:
            # Get ranks (use a large number if not present in that search)
            bm25_rank = bm25_ranks.get(doc_id, None)
            semantic_rank = semantic_ranks.get(doc_id, None)
            
            # Calculate RRF score (sum scores from both searches)
            total_rrf = 0.0
            if bm25_rank is not None:
                total_rrf += rrf_score(bm25_rank, k)
            if semantic_rank is not None:
                total_rrf += rrf_score(semantic_rank, k)
            
            # Get document details (from whichever search it appeared in)
            if doc_id in self.semantic_search.document_map:
                doc = self.semantic_search.document_map[doc_id]
            else:
                doc = self.idx.docmap[doc_id]
            
            doc_scores[doc_id] = {
                'id': doc_id,
                'title': doc['title'],
                'document': doc.get('description', ''),
                'rrf_score': total_rrf,
                'bm25_rank': bm25_rank,
                'semantic_rank': semantic_rank
            }
        
        # Sort by RRF score (descending)
        sorted_results = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        
        # Return top limit results
        return sorted_results[:limit]

