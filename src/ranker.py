"""
Main Ranking Pipeline - Orchestrates the full candidate ranking process.

Pipeline:
1. Parse JD → Extract requirements
2. Stream candidates → Filter honeypots
3. Compute feature scores
4. Compute semantic similarity scores
5. Compute behavioral scores
6. Combine with hybrid weights → Final ranking
7. Generate reasoning → Output
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from config import (
    WEIGHT_FEATURE,
    WEIGHT_SEMANTIC,
    WEIGHT_BEHAVIORAL,
    DEFAULT_TOP_N,
)
from src.jd_parser import JDParser
from src.scorer import CandidateScorer, behavioral_score
from src.semantic_search import SemanticSearchEngine
from src.honeypot_detector import is_honeypot
from src.reasoning_gen import generate_reasoning
from src.utils import build_candidate_text, stream_jsonl


class CandidateRanker:
    """Main ranking pipeline that combines all scoring components."""
    
    def __init__(self, jd_data: dict, model_path: str = None):
        """
        Initialize the ranker with a job description.
        
        Args:
            jd_data: Job description as a dict
            model_path: Optional custom model path for embeddings
        """
        self.jd_parser = JDParser(jd_data)
        self.scorer = CandidateScorer(self.jd_parser)
        self.semantic_engine = SemanticSearchEngine(model_path)
        
        # Pre-encode JD
        jd_text = self.jd_parser.get_embedding_text()
        self.semantic_engine.encode_jd(jd_text)
        
        print(f"[Ranker] JD parsed: {self.jd_parser.requirements['title']}")
        print(f"[Ranker] Required skills: {len(self.jd_parser.get_required_skills())}")
        print(f"[Ranker] Experience range: {self.jd_parser.get_experience_range()}")
    
    def rank_from_file(
        self, 
        candidates_path: str, 
        top_n: int = DEFAULT_TOP_N,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Rank candidates from a JSONL file.
        
        Args:
            candidates_path: Path to candidates JSONL file
            top_n: Number of top candidates to return
            show_progress: Show progress bars
        
        Returns:
            List of ranked candidate dicts with scores and reasoning
        """
        # Step 1: Load and filter candidates
        print("[Ranker] Loading candidates...")
        candidates = []
        honeypot_count = 0
        
        for candidate in stream_jsonl(candidates_path):
            if is_honeypot(candidate):
                honeypot_count += 1
                continue
            candidates.append(candidate)
        
        print(f"[Ranker] Valid candidates: {len(candidates)}, Honeypots filtered: {honeypot_count}")
        
        return self._rank_candidates(candidates, top_n, show_progress)
    
    def rank_candidates(
        self,
        candidates: List[Dict],
        top_n: int = DEFAULT_TOP_N,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Rank a list of candidate dicts.
        
        Args:
            candidates: List of candidate dicts
            top_n: Number of top candidates to return
            show_progress: Show progress bars
        
        Returns:
            List of ranked candidate dicts with scores and reasoning
        """
        # Filter honeypots
        valid_candidates = []
        honeypot_count = 0
        
        for candidate in candidates:
            if is_honeypot(candidate):
                honeypot_count += 1
            else:
                valid_candidates.append(candidate)
        
        print(f"[Ranker] Valid: {len(valid_candidates)}, Honeypots: {honeypot_count}")
        
        return self._rank_candidates(valid_candidates, top_n, show_progress)
    
    def _rank_candidates(
        self,
        candidates: List[Dict],
        top_n: int,
        show_progress: bool
    ) -> List[Dict]:
        """Internal ranking logic."""
        if not candidates:
            return []
        
        # Step 2: Compute feature scores
        print("[Ranker] Computing feature scores...")
        feature_scores = []
        all_score_details = []
        
        iterator = tqdm(candidates, desc="Feature scoring") if show_progress else candidates
        for candidate in iterator:
            score_detail = self.scorer.score(candidate)
            feature_scores.append(score_detail["feature_score"])
            all_score_details.append(score_detail)
        
        # Step 3: Compute semantic similarity scores
        print("[Ranker] Computing semantic similarity...")
        candidate_texts = [build_candidate_text(c) for c in candidates]
        semantic_scores = self.semantic_engine.rank_candidates(
            self.jd_parser.get_embedding_text(),
            candidate_texts,
            show_progress=show_progress
        )
        
        # Step 4: Compute behavioral scores
        print("[Ranker] Computing behavioral scores...")
        behavioral_scores = [behavioral_score(c) for c in candidates]
        
        # Step 5: Combine scores
        print("[Ranker] Combining scores...")
        feature_arr = np.array(feature_scores)
        semantic_arr = np.array(semantic_scores)
        behavioral_arr = np.array(behavioral_scores)
        
        final_scores = (
            WEIGHT_FEATURE * feature_arr +
            WEIGHT_SEMANTIC * semantic_arr +
            WEIGHT_BEHAVIORAL * behavioral_arr
        )
        
        # Step 6: Normalize final scores to [0.5, 1.0] for top candidates
        sorted_indices = np.argsort(final_scores)[::-1]
        top_indices = sorted_indices[:top_n]
        
        # Normalize
        top_scores = final_scores[top_indices]
        if len(top_scores) > 1:
            s_min, s_max = top_scores.min(), top_scores.max()
            rng = s_max - s_min if s_max != s_min else 1.0
            normalized = 0.5 + 0.5 * (top_scores - s_min) / rng
        else:
            normalized = np.array([1.0])
        
        # Step 7: Build results with reasoning
        print("[Ranker] Generating explanations...")
        results = []
        
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized), 1):
            candidate = candidates[idx]
            score_detail = all_score_details[idx]
            
            # Enrich score detail for reasoning
            score_detail["semantic_score"] = float(semantic_arr[idx])
            score_detail["behavioral_score"] = float(behavioral_arr[idx])
            score_detail["final_score"] = float(norm_score)
            
            reasoning = generate_reasoning(candidate, rank, score_detail)
            
            results.append({
                "candidate_id": candidate.get("candidate_id", f"unknown_{idx}"),
                "rank": rank,
                "score": round(float(norm_score), 4),
                "reasoning": reasoning,
                "scores_breakdown": {
                    "feature": round(float(feature_arr[idx]), 4),
                    "semantic": round(float(semantic_arr[idx]), 4),
                    "behavioral": round(float(behavioral_arr[idx]), 4),
                    "skills": round(score_detail["skills_score"], 4),
                    "career": round(score_detail["career_score"], 4),
                    "experience": round(score_detail["experience_score"], 4),
                    "location": round(score_detail["location_score"], 4),
                    "education": round(score_detail["education_score"], 4),
                },
            })
        
        print(f"[Ranker] Done! Top {len(results)} candidates ranked.")
        return results
