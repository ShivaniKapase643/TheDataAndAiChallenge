"""
Semantic Search Engine - Embedding-based candidate-JD similarity.

Uses sentence-transformers (BGE-small-en-v1.5) to encode both JD and
candidate profiles into the same embedding space, then measures cosine
similarity to find semantically relevant candidates.
"""

import numpy as np
from typing import List
from tqdm import tqdm
from config import EMBEDDING_MODEL, BATCH_SIZE


class SemanticSearchEngine:
    """Handles embedding generation and similarity computation."""
    
    def __init__(self, model_path: str = None):
        """Initialize the embedding model."""
        from sentence_transformers import SentenceTransformer
        
        path = model_path if model_path else EMBEDDING_MODEL
        self.model = SentenceTransformer(path)
        self._jd_embedding = None
    
    def encode_jd(self, jd_text: str) -> np.ndarray:
        """Encode the job description text."""
        self._jd_embedding = self.model.encode(
            jd_text, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return self._jd_embedding
    
    def encode_candidates(self, candidate_texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode all candidate texts in batches."""
        embeddings = self.model.encode(
            candidate_texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
        )
        return embeddings
    
    def compute_similarities(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between JD embedding and all candidate embeddings.
        
        Since embeddings are normalized, cosine similarity = dot product.
        """
        if self._jd_embedding is None:
            raise ValueError("JD embedding not set. Call encode_jd() first.")
        
        # Dot product of normalized vectors = cosine similarity
        similarities = candidate_embeddings @ self._jd_embedding
        return similarities.astype(float)
    
    def normalize_scores(self, similarities: np.ndarray) -> np.ndarray:
        """Min-max normalize similarity scores to [0, 1]."""
        s_min = similarities.min()
        s_max = similarities.max()
        
        if s_max - s_min < 1e-9:
            return np.full_like(similarities, 0.5)
        
        return (similarities - s_min) / (s_max - s_min)
    
    def rank_candidates(
        self, 
        jd_text: str, 
        candidate_texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Full pipeline: encode JD, encode candidates, return normalized similarity scores.
        
        Returns array of scores in [0, 1] where higher = more similar to JD.
        """
        self.encode_jd(jd_text)
        candidate_embeddings = self.encode_candidates(candidate_texts, show_progress)
        similarities = self.compute_similarities(candidate_embeddings)
        return self.normalize_scores(similarities)
