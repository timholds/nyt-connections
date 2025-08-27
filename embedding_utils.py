import json
import numpy as np
import pickle
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime

class EmbeddingStore:
    """
    Optimized embedding storage for both words and themes with fast lookups.
    Supports the dual-structure format produced by get_embeddings.py.
    """
    
    def __init__(self):
        # Word embeddings
        self.word_embeddings = {}
        self.word_vectors = None
        self.words = []
        self.word_to_idx = {}
        
        # Theme embeddings
        self.theme_embeddings = {}
        self.theme_vectors = None
        self.themes = []
        self.theme_to_idx = {}
        
        # Metadata
        self.dimension = None
        self.metadata = {}
    
    @classmethod
    def from_json(cls, json_file: str) -> 'EmbeddingStore':
        """Load from JSON format with both words and themes."""
        store = cls()
        print(f"Loading embeddings from {json_file}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle metadata if present
        if 'metadata' in data:
            store.metadata = data['metadata']
            store.dimension = store.metadata.get('dimension')
        
        # Load word embeddings
        if 'words' in data:
            store.word_embeddings = data['words']
            store.words = sorted(list(data['words'].keys()))
            store.word_to_idx = {w: i for i, w in enumerate(store.words)}
            
            # Convert to numpy array
            vectors_list = [data['words'][w] for w in store.words]
            store.word_vectors = np.array(vectors_list, dtype=np.float64)
            
            if store.dimension is None:
                store.dimension = store.word_vectors.shape[1]
        
        # Load theme embeddings
        if 'themes' in data:
            store.theme_embeddings = data['themes']
            store.themes = sorted(list(data['themes'].keys()))
            store.theme_to_idx = {t: i for i, t in enumerate(store.themes)}
            
            # Convert to numpy array
            vectors_list = [data['themes'][t] for t in store.themes]
            store.theme_vectors = np.array(vectors_list, dtype=np.float64)
        
        print(f"Loaded {len(store.words)} word embeddings and {len(store.themes)} theme embeddings")
        print(f"Embedding dimension: {store.dimension}")
        
        return store
    
    def save_json(self, json_file: str):
        """Save in JSON format with metadata."""
        data = {
            'metadata': {
                **self.metadata,
                'dimension': self.dimension,
                'num_words': len(self.words),
                'num_themes': len(self.themes),
                'saved_at': datetime.now().isoformat()
            },
            'words': self.word_embeddings,
            'themes': self.theme_embeddings
        }
        
        with open(json_file, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved embeddings to {json_file}")
    
    def save_optimized(self, pickle_file: str):
        """Save in optimized pickle format for faster loading."""
        data = {
            'metadata': {
                **self.metadata,
                'dimension': self.dimension,
                'saved_at': datetime.now().isoformat()
            },
            'words': self.words,
            'word_vectors': self.word_vectors,
            'themes': self.themes,
            'theme_vectors': self.theme_vectors
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved optimized embeddings to {pickle_file}")
    
    @classmethod
    def load_optimized(cls, pickle_file: str) -> 'EmbeddingStore':
        """Load from optimized pickle format (much faster)."""
        store = cls()
        
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Load metadata
        store.metadata = data.get('metadata', {})
        
        # Load words
        store.words = data['words']
        store.word_vectors = data['word_vectors']
        store.word_to_idx = {w: i for i, w in enumerate(store.words)}
        
        # Load themes
        store.themes = data['themes']
        store.theme_vectors = data['theme_vectors']
        store.theme_to_idx = {t: i for i, t in enumerate(store.themes)}
        
        # Get dimension from vectors or metadata
        if 'dimension' in data:
            store.dimension = data['dimension']
        elif 'metadata' in data and 'dimension' in data['metadata']:
            store.dimension = data['metadata']['dimension']
        else:
            store.dimension = store.word_vectors.shape[1]
        
        # Reconstruct embeddings dicts for compatibility
        store.word_embeddings = {
            word: store.word_vectors[i].tolist() 
            for i, word in enumerate(store.words)
        }
        store.theme_embeddings = {
            theme: store.theme_vectors[i].tolist() 
            for i, theme in enumerate(store.themes)
        }
        
        print(f"Loaded {len(store.words)} word embeddings and {len(store.themes)} theme embeddings")
        
        return store
    
    @classmethod
    def load(cls, file_path: str) -> 'EmbeddingStore':
        """Load from either JSON or pickle format based on extension."""
        if file_path.endswith('.pkl'):
            return cls.load_optimized(file_path)
        elif file_path.endswith('.json'):
            return cls.from_json(file_path)
        else:
            # Try to detect format
            try:
                return cls.load_optimized(file_path)
            except:
                return cls.from_json(file_path)
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word. Returns None if not found."""
        idx = self.word_to_idx.get(word)
        if idx is None:
            return None
        return self.word_vectors[idx]
    
    def get_theme_vector(self, theme: str) -> Optional[np.ndarray]:
        """Get embedding vector for a theme. Returns None if not found."""
        idx = self.theme_to_idx.get(theme)
        if idx is None:
            return None
        return self.theme_vectors[idx]
    
    def get_word_vectors_batch(self, words: List[str]) -> np.ndarray:
        """
        Get word embeddings for multiple words at once.
        Missing words are replaced with zero vectors.
        """
        vectors = []
        for word in words:
            vec = self.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
            else:
                vectors.append(np.zeros(self.dimension, dtype=np.float64))
        return np.array(vectors)
    
    def has_word(self, word: str) -> bool:
        """Check if word exists in embeddings."""
        return word in self.word_to_idx
    
    def has_theme(self, theme: str) -> bool:
        """Check if theme exists in embeddings."""
        return theme in self.theme_to_idx
    
    def word_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words."""
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def theme_similarity(self, theme1: str, theme2: str) -> float:
        """Calculate cosine similarity between two themes."""
        vec1 = self.get_theme_vector(theme1)
        vec2 = self.get_theme_vector(theme2)
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def group_similarity(self, words: List[str]) -> float:
        """Calculate average pairwise similarity within a group of words."""
        if len(words) < 2:
            return 0.0
        
        vectors = self.get_word_vectors_batch(words)
        # Skip zero vectors (missing words)
        non_zero_mask = np.any(vectors != 0, axis=1)
        valid_vectors = vectors[non_zero_mask]
        
        if len(valid_vectors) < 2:
            return 0.0
        
        # Compute pairwise similarities efficiently
        norms = np.linalg.norm(valid_vectors, axis=1, keepdims=True)
        normalized = valid_vectors / (norms + 1e-8)
        similarity_matrix = normalized @ normalized.T
        
        # Get upper triangle (excluding diagonal)
        n = len(valid_vectors)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarity_matrix[upper_tri_indices]
        
        return np.mean(pairwise_sims)
    
    def group_to_theme_similarity(self, words: List[str], theme: str) -> float:
        """
        Calculate similarity between a group of words and a theme.
        Uses average of word vectors compared to theme vector.
        """
        theme_vec = self.get_theme_vector(theme)
        if theme_vec is None:
            return 0.0
        
        word_vecs = self.get_word_vectors_batch(words)
        # Skip zero vectors
        non_zero_mask = np.any(word_vecs != 0, axis=1)
        valid_vecs = word_vecs[non_zero_mask]
        
        if len(valid_vecs) == 0:
            return 0.0
        
        # Average word vectors
        group_vec = np.mean(valid_vecs, axis=0)
        
        # Cosine similarity with theme
        norm1 = np.linalg.norm(group_vec)
        norm2 = np.linalg.norm(theme_vec)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(group_vec, theme_vec) / (norm1 * norm2)
    
    def find_best_theme(self, words: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Find themes most similar to a group of words.
        Returns list of (theme, similarity_score) tuples.
        """
        scores = []
        for theme in self.themes:
            sim = self.group_to_theme_similarity(words, theme)
            scores.append((theme, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def cluster_words(self, words: List[str], n_clusters: int = 4) -> Dict[int, List[str]]:
        """Cluster words into groups using K-means."""
        from sklearn.cluster import KMeans
        
        # Get embeddings for the words
        valid_words = []
        valid_vectors = []
        for word in words:
            vec = self.get_word_vector(word)
            if vec is not None:
                valid_words.append(word)
                valid_vectors.append(vec)
            else:
                print(f"Warning: '{word}' not in embeddings")
        
        if len(valid_words) < n_clusters:
            raise ValueError(f"Not enough words with embeddings ({len(valid_words)}) for {n_clusters} clusters")
        
        # Cluster
        X = np.array(valid_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Group words by cluster
        clusters = {}
        for word, label in zip(valid_words, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(word)
        
        return clusters
    
    def analyze_puzzle(self, words: List[str], solution: Dict = None):
        """Analyze a NYT Connections puzzle."""
        print(f"\n=== Analyzing puzzle with {len(words)} words ===")
        print(f"Words: {', '.join(words)}")
        
        # Try clustering
        print("\n--- Clustering Analysis ---")
        try:
            clusters = self.cluster_words(words, n_clusters=4)
            for i, cluster_words in clusters.items():
                similarity = self.group_similarity(cluster_words)
                best_themes = self.find_best_theme(cluster_words, top_n=3)
                print(f"\nCluster {i+1} (similarity: {similarity:.3f}): {', '.join(cluster_words)}")
                print(f"  Possible themes:")
                for theme, score in best_themes:
                    print(f"    - {theme}: {score:.3f}")
        except Exception as e:
            print(f"Clustering failed: {e}")
        
        # If solution is provided, analyze it
        if solution:
            print("\n--- Solution Analysis ---")
            for group in solution['groups']:
                group_words = group['words']
                reason = group['reason']
                word_sim = self.group_similarity(group_words)
                theme_sim = self.group_to_theme_similarity(group_words, reason)
                
                print(f"\n{reason.upper()}:")
                print(f"  Words: {', '.join(group_words)}")
                print(f"  Word similarity: {word_sim:.3f}")
                print(f"  Theme match: {theme_sim:.3f}")
    
    def memory_usage_mb(self) -> float:
        """Return memory usage of vectors in MB."""
        total = 0
        if self.word_vectors is not None:
            total += self.word_vectors.nbytes
        if self.theme_vectors is not None:
            total += self.theme_vectors.nbytes
        return total / (1024 * 1024)
    
    def stats(self) -> Dict:
        """Return statistics about the embeddings."""
        return {
            'num_words': len(self.words),
            'num_themes': len(self.themes),
            'dimension': self.dimension,
            'memory_mb': self.memory_usage_mb(),
            'metadata': self.metadata
        }