import string
from nltk.stem import PorterStemmer
from lib.inverted_index import InvertedIndex
from lib.constants import BM25_K1
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    # Create translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Clean and tokenize the query
    clean_query = query.translate(translator).lower()
    query_tokens = [token for token in clean_query.split() if token]

    # Remove stopwords from query
    query_tokens = [token for token in query_tokens if token not in stopwords]

    # Stem query tokens
    query_tokens = [stemmer.stem(token) for token in query_tokens]

    results = []
    for movie in movies:
        # Clean and tokenize the title
        clean_title = movie['title'].translate(translator).lower()
        title_tokens = [token for token in clean_title.split() if token]

        # Remove stop words from title
        title_tokens = [token for token in title_tokens if token not in stopwords]

        # Stem title tokens
        title_tokens = [stemmer.stem(token) for token in title_tokens]

        # Check if ANY query token matches ANY part of ANY title token
        match_found = False
        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in  title_token:
                    match_found = True
                    break
            if match_found:
                break
        
        if match_found:
            results.append(movie)

    return results

def bm25_idf_command(term: str) -> float:
    """Get BM25 IDF score for a term"""
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1) -> float:
    """Get BM25 TF score for a term in a document"""
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)
