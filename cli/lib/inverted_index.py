import os
import pickle
import string
import math
from collections import Counter
from nltk.stem import PorterStemmer
from .search_utils import load_movies, load_stopwords, PROJECT_ROOT
from .constants import BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.stemmer = PorterStemmer()
        self.stopwords = load_stopwords()
        self.translator = str.maketrans('', '', string.punctuation)

    def __add_document(self, doc_id, text):
        """Tokenize text and add each token to index with doc_id"""
        # Clean, tokenize, remove stopwords, stem
        clean_text = text.translate(self.translator).lower()
        tokens = [token for token in clean_text.split() if token]
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = [self.stemmer.stem(token) for token in tokens]

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        # Add each token to index
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

            # Increment term frequency
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        """Return sorted list of doc IDs for a given token"""
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def get_tf(self, doc_id, term):
        """Return term frequency for a token in a document"""
        # Tokenize the term (should be single token)
        clean_term = term.translate(self.translator).lower()
        tokens = [t for t in clean_term.split() if t]
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]

        if len(tokens) == 0:
            return 0
        if len(tokens) > 1:
            raise ValueError("get_tf expects a single token term")

        token = tokens[0]

        # Return count from Counter (defaults to 0 if not found)
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][token]

    def get_bm25_idf(self, term: str) -> float:
        """Calculate BM25 IDF score for a term"""
        # Tokenize the term
        clean_term = term.translate(self.translator).lower()
        tokens = [t for t in clean_term.split() if t]
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.stemmer.stem(t) for t in tokens]

        if len(tokens) == 0:
            return 0.0
        if len(tokens) > 1:
            raise ValueError("get_bm25_idf expects a single token term")

        token = tokens[0]

        # Calculate BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        N = len(self.docmap)  # Total documents
        doc_ids = self.get_documents(token)
        df = len(doc_ids)  # Document frequency

        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        """Calculate BM25 saturated term frequency"""
        # Get raw TF
        tf = self.get_tf(doc_id, term)

        # Apply BM25 saturation formula
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)
        return bm25_tf

    def build(self):
        """Build index and docmap from all movies"""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            # Concatenate title and description
            text = f"{movie['title']} {movie['description']}"

            # Add to docmap
            self.docmap[doc_id] = movie

            # Add to index
            self.__add_document(doc_id, text)


    def save(self):
        """Save index and docmap to disk using pickle"""
        cache_dir = os.path.join(PROJECT_ROOT, "cache")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Save index
        index_path = os.path.join(cache_dir, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)

        # Save docmap
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        # Save term_frequencies
        tf_path = os.path.join(cache_dir, "term_frequencies.pkl")
        with open(tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        index_path = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
        docmap_path = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")
        tf_path = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")

        if not os.path.exists(index_path) or not os.path.exists(docmap_path):
            raise FileNotFoundError("Index files not found. Run the build command first.")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

