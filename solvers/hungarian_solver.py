from typing import List, Optional, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from .base import BaseSolver
from .models import GroupSolution, PuzzleSolution
from embedding_utils import EmbeddingStore


class HungarianSolver(BaseSolver):
    """Solver that uses Hungarian algorithm with embeddings for optimal grouping."""
    
    def __init__(self, embeddings_path: str = "embeddings.json"):
        """
        Initialize the Hungarian solver with embeddings.
        
        Args:
            embeddings_path: Path to the embeddings file
        """
        super().__init__()
        self.embeddings = EmbeddingStore.load(embeddings_path)
        
    def get_system_prompt(self, current_words: Optional[List[str]] = None) -> str:
        """Not needed for Hungarian solver as it doesn't use LLM."""
        return ""
    
    def get_user_message(self, words: List[str]) -> str:
        """Not needed for Hungarian solver as it doesn't use LLM."""
        return ""
    
    def compute_similarity_matrix(self, words: List[str]) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix between words.
        
        Args:
            words: List of 16 words
            
        Returns:
            16x16 similarity matrix
        """
        vectors = self.embeddings.get_word_vectors_batch(words)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Add small epsilon to avoid division by zero
        normalized = vectors / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = normalized @ normalized.T
        
        return similarity_matrix
    
    def create_bipartite_cost_matrix(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Transform similarity matrix into a bipartite cost matrix for Hungarian algorithm.
        We create a 16x4 matrix where each column represents a cluster assignment.
        
        The idea: for each word (row), we compute the cost of assigning it to each cluster (column).
        We'll use a greedy initialization for cluster centers then optimize.
        
        Args:
            similarity_matrix: 16x16 similarity matrix
            
        Returns:
            16x4 cost matrix (negative similarity for maximization)
        """
        n_words = 16
        n_groups = 4
        group_size = 4
        
        # Initialize cluster centers using greedy diverse selection
        selected_centers = []
        available = list(range(n_words))
        
        # Select first center randomly (or first word)
        first_center = 0
        selected_centers.append(first_center)
        available.remove(first_center)
        
        # Select remaining centers to be maximally dissimilar
        for _ in range(n_groups - 1):
            max_min_dist = -1
            best_candidate = None
            
            for candidate in available:
                # Find minimum similarity to existing centers
                min_sim = min(similarity_matrix[candidate, center] for center in selected_centers)
                if min_sim > max_min_dist:
                    max_min_dist = min_sim
                    best_candidate = candidate
            
            selected_centers.append(best_candidate)
            available.remove(best_candidate)
        
        # Create cost matrix: cost[i, j] = negative similarity of word i to center j
        # We need to expand this to handle 4 words per group
        # Create a 16x16 matrix where columns j*4:(j+1)*4 represent group j
        cost_matrix = np.zeros((n_words, n_words))
        
        for group_idx, center in enumerate(selected_centers):
            # Each group gets 4 columns
            for offset in range(group_size):
                col_idx = group_idx * group_size + offset
                # Cost is negative similarity to the group center
                cost_matrix[:, col_idx] = -similarity_matrix[:, center]
        
        return cost_matrix
    
    def solve_with_hungarian(self, words: List[str]) -> List[List[str]]:
        """
        Use Hungarian algorithm to find optimal word groupings.
        
        Args:
            words: List of 16 words
            
        Returns:
            List of 4 groups, each containing 4 words
        """
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(words)
        
        # Alternative approach: Use iterative Hungarian algorithm
        # We'll find the best partition by treating it as finding 4 groups of 4
        groups = self.iterative_hungarian_grouping(words, similarity_matrix)
        
        return groups
    
    def iterative_hungarian_grouping(self, words: List[str], similarity_matrix: np.ndarray) -> List[List[str]]:
        """
        Iteratively apply Hungarian algorithm to create groups.
        
        Strategy:
        1. Find the 4 most cohesive groups of 4 words each
        2. Use Hungarian algorithm to optimize assignments
        
        Args:
            words: List of 16 words
            similarity_matrix: 16x16 similarity matrix
            
        Returns:
            List of 4 groups
        """
        n_words = 16
        n_groups = 4
        group_size = 4
        
        # Convert similarity to distance (cost) matrix
        # Use 1 - similarity for distance (higher similarity = lower cost)
        cost_matrix = 1.0 - similarity_matrix
        
        # Approach: Create a bipartite graph between words and group slots
        # We need to assign each word to exactly one group slot
        # Create extended cost matrix for bipartite matching
        extended_cost = np.full((n_words, n_words), np.inf)
        
        # Initialize with K-means++ style center selection
        centers = self.select_diverse_centers(similarity_matrix, n_groups)
        
        # Create cost matrix where cost[i, j] represents assigning word i to slot j
        # Slots 0-3 belong to group 0, slots 4-7 to group 1, etc.
        for group_idx, center_idx in enumerate(centers):
            for slot in range(group_size):
                col_idx = group_idx * group_size + slot
                # Cost of assigning each word to this slot based on distance to center
                for word_idx in range(n_words):
                    extended_cost[word_idx, col_idx] = cost_matrix[word_idx, center_idx]
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(extended_cost)
        
        # Convert assignments to groups
        groups = [[] for _ in range(n_groups)]
        for word_idx, slot_idx in zip(row_indices, col_indices):
            group_idx = slot_idx // group_size
            groups[group_idx].append(words[word_idx])
        
        # Refine groups using local search
        groups = self.refine_groups(groups, words, similarity_matrix)
        
        return groups
    
    def select_diverse_centers(self, similarity_matrix: np.ndarray, n_centers: int) -> List[int]:
        """
        Select diverse centers using K-means++ initialization strategy.
        
        Args:
            similarity_matrix: Similarity matrix
            n_centers: Number of centers to select
            
        Returns:
            List of center indices
        """
        n_words = similarity_matrix.shape[0]
        centers = []
        
        # Choose first center randomly (or pick the word with highest average similarity)
        avg_similarities = np.mean(similarity_matrix, axis=1)
        first_center = np.argmax(avg_similarities)
        centers.append(first_center)
        
        # Select remaining centers
        for _ in range(n_centers - 1):
            # For each point, find its maximum similarity to existing centers
            max_similarities = np.zeros(n_words)
            for i in range(n_words):
                if i not in centers:
                    max_similarities[i] = max(similarity_matrix[i, c] for c in centers)
                else:
                    max_similarities[i] = np.inf
            
            # Select point with minimum maximum similarity (most dissimilar)
            next_center = np.argmin(max_similarities)
            centers.append(next_center)
        
        return centers
    
    def refine_groups(self, groups: List[List[str]], words: List[str], similarity_matrix: np.ndarray) -> List[List[str]]:
        """
        Refine groups using local search to improve cohesion.
        
        Args:
            groups: Initial grouping
            words: All words
            similarity_matrix: Similarity matrix
            
        Returns:
            Refined groups
        """
        word_to_idx = {word: i for i, word in enumerate(words)}
        
        def group_cohesion(group: List[str]) -> float:
            """Calculate average pairwise similarity within a group."""
            if len(group) < 2:
                return 0.0
            
            indices = [word_to_idx[w] for w in group]
            total_sim = 0
            count = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    total_sim += similarity_matrix[indices[i], indices[j]]
                    count += 1
            return total_sim / count if count > 0 else 0.0
        
        # Try swapping words between groups to improve overall cohesion
        improved = True
        max_iterations = 10
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    # Try swapping each word from group i with each word from group j
                    for wi in range(len(groups[i])):
                        for wj in range(len(groups[j])):
                            # Calculate current cohesion
                            current_cohesion = group_cohesion(groups[i]) + group_cohesion(groups[j])
                            
                            # Swap words
                            groups[i][wi], groups[j][wj] = groups[j][wj], groups[i][wi]
                            
                            # Calculate new cohesion
                            new_cohesion = group_cohesion(groups[i]) + group_cohesion(groups[j])
                            
                            # Keep swap if it improves cohesion
                            if new_cohesion > current_cohesion:
                                improved = True
                            else:
                                # Revert swap
                                groups[i][wi], groups[j][wj] = groups[j][wj], groups[i][wi]
        
        return groups
    
    def find_group_themes(self, groups: List[List[str]]) -> List[Tuple[str, float]]:
        """
        Find best matching themes for each group.
        
        Args:
            groups: List of word groups
            
        Returns:
            List of (theme, confidence) tuples
        """
        themes = []
        for group in groups:
            # Get top themes for this group (use lowercase for embedding lookup)
            group_lower = [w.lower() for w in group]
            best_themes = self.embeddings.find_best_theme(group_lower, top_n=1)
            if best_themes:
                themes.append(best_themes[0])
            else:
                # Fallback: describe based on similarity
                cohesion = self.embeddings.group_similarity(group_lower)
                themes.append((f"Group with similarity {cohesion:.2f}", cohesion))
        
        return themes
    
    def solve(self, words: List[str], use_api: bool = False, model: str = None) -> PuzzleSolution:
        """
        Solve using Hungarian algorithm with embeddings.
        
        Args:
            words: List of 16 words to group
            use_api: Not used for this solver
            model: Not used for this solver
            
        Returns:
            PuzzleSolution with groups and reasoning
        """
        # Validate input
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        if len(set(words)) != 16:
            raise ValueError("All 16 words must be unique")
        
        # Convert words to lowercase for embedding lookup (NYT Connections words are typically lowercase)
        words_lower = [w.lower() for w in words]
        
        # Check that all words have embeddings
        missing_words = [words[i] for i, w in enumerate(words_lower) if not self.embeddings.has_word(w)]
        if missing_words:
            print(f"Warning: Missing embeddings for: {missing_words}")
            print("Results may be suboptimal.")
        
        # Solve using Hungarian algorithm (using lowercase for embeddings)
        groups = self.solve_with_hungarian(words_lower)
        
        # Map back to original case for output
        word_map = {w.lower(): w for w in words}
        groups = [[word_map[w] for w in group] for group in groups]
        
        # Find themes for each group
        themes = self.find_group_themes(groups)
        
        # Create solution object
        group_solutions = []
        for group, (theme, confidence) in zip(groups, themes):
            group_lower = [w.lower() for w in group]
            cohesion = self.embeddings.group_similarity(group_lower)
            reason = f"{theme} (cohesion: {cohesion:.2f}, confidence: {confidence:.2f})"
            group_solutions.append(GroupSolution(words=group, reason=reason))
        
        # Sort groups by cohesion (best groups first)
        group_solutions.sort(key=lambda g: self.embeddings.group_similarity([w.lower() for w in g.words]), reverse=True)
        
        return PuzzleSolution(groups=group_solutions)