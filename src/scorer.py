"""
Feature-Based Scorer - Multi-dimensional candidate scoring.

Scores candidates across skills, career relevance, experience fit,
location match, and education alignment.
"""

import re
from typing import Dict, List, Tuple
from rapidfuzz import fuzz
from src.utils import normalize_text, clean_skills_list, safe_get
from src.jd_parser import JDParser
from config import (
    WEIGHT_SKILLS,
    WEIGHT_CAREER,
    WEIGHT_EXPERIENCE,
    WEIGHT_LOCATION,
    WEIGHT_EDUCATION,
    BEHAVIORAL_WEIGHT_ACTIVITY,
    BEHAVIORAL_WEIGHT_COMPLETENESS,
    BEHAVIORAL_WEIGHT_RESPONSIVENESS,
)


class CandidateScorer:
    """Scores candidates against parsed JD requirements."""
    
    def __init__(self, jd_parser: JDParser):
        self.jd = jd_parser.requirements
        self.required_skills = jd_parser.get_required_skills()
        self.preferred_skills = jd_parser.get_preferred_skills()
        self.all_jd_skills = self.required_skills + self.preferred_skills
        self.exp_range = jd_parser.get_experience_range()
        self.seniority = jd_parser.get_seniority()
    
    def score(self, candidate: dict) -> Dict[str, float]:
        """
        Score a candidate and return detailed breakdown.
        
        Returns dict with individual scores and combined feature score.
        """
        profile = candidate.get("profile", {})
        
        skills_score = self._score_skills(profile)
        career_score = self._score_career(profile)
        experience_score = self._score_experience(profile)
        location_score = self._score_location(profile)
        education_score = self._score_education(profile)
        
        # Weighted combination
        combined = (
            WEIGHT_SKILLS * skills_score +
            WEIGHT_CAREER * career_score +
            WEIGHT_EXPERIENCE * experience_score +
            WEIGHT_LOCATION * location_score +
            WEIGHT_EDUCATION * education_score
        )
        
        return {
            "feature_score": min(combined, 1.0),
            "skills_score": skills_score,
            "career_score": career_score,
            "experience_score": experience_score,
            "location_score": location_score,
            "education_score": education_score,
        }
    
    def _score_skills(self, profile: dict) -> float:
        """Score skill match using fuzzy matching."""
        candidate_skills = clean_skills_list(profile.get("skills", []))
        
        if not candidate_skills or not self.all_jd_skills:
            return 0.0
        
        # Score required skills (higher weight)
        required_matched = 0
        for req_skill in self.required_skills:
            best_match = max(
                (fuzz.ratio(req_skill, cs) for cs in candidate_skills),
                default=0
            )
            if best_match >= 75:  # 75% fuzzy match threshold
                required_matched += 1
            elif best_match >= 60:
                required_matched += 0.5
        
        required_score = required_matched / max(len(self.required_skills), 1)
        
        # Score preferred skills (lower weight)
        preferred_matched = 0
        for pref_skill in self.preferred_skills:
            best_match = max(
                (fuzz.ratio(pref_skill, cs) for cs in candidate_skills),
                default=0
            )
            if best_match >= 75:
                preferred_matched += 1
            elif best_match >= 60:
                preferred_matched += 0.5
        
        preferred_score = preferred_matched / max(len(self.preferred_skills), 1)
        
        # Required skills weighted 70%, preferred 30%
        return min(0.7 * required_score + 0.3 * preferred_score, 1.0)
    
    def _score_career(self, profile: dict) -> float:
        """Score career relevance based on title and experience history."""
        score = 0.0
        
        # Current title similarity to JD title
        current_title = normalize_text(
            profile.get("current_title", "") or profile.get("title", "")
        )
        jd_title = normalize_text(self.jd["title"])
        
        if current_title and jd_title:
            title_sim = fuzz.token_sort_ratio(current_title, jd_title) / 100.0
            score += 0.5 * title_sim
        
        # Career history relevance
        experience = profile.get("experience", []) or profile.get("career_history", [])
        if isinstance(experience, list) and experience:
            career_relevance = 0.0
            for i, exp in enumerate(experience[:5]):
                if isinstance(exp, dict):
                    exp_title = normalize_text(
                        exp.get("title", "") or exp.get("role", "")
                    )
                    if exp_title and jd_title:
                        sim = fuzz.token_sort_ratio(exp_title, jd_title) / 100.0
                        # More recent roles weighted higher
                        weight = 1.0 / (i + 1)
                        career_relevance += sim * weight
            
            # Normalize
            max_possible = sum(1.0 / (i + 1) for i in range(min(len(experience), 5)))
            if max_possible > 0:
                score += 0.5 * (career_relevance / max_possible)
        
        return min(score, 1.0)
    
    def _score_experience(self, profile: dict) -> float:
        """Score experience years fit against JD requirements."""
        years_exp = profile.get("years_of_experience", 0)
        if isinstance(years_exp, str):
            try:
                years_exp = float(re.findall(r"\d+\.?\d*", str(years_exp))[0])
            except (IndexError, ValueError):
                years_exp = 0
        
        years_exp = float(years_exp) if years_exp else 0
        
        min_exp, max_exp = self.exp_range
        
        if min_exp == 0 and max_exp == 0:
            return 0.5  # No experience requirement specified
        
        if min_exp <= years_exp <= max_exp:
            return 1.0  # Perfect fit
        elif years_exp < min_exp:
            # Under-experienced: gradual penalty
            gap = min_exp - years_exp
            return max(0.0, 1.0 - (gap / max(min_exp, 1)) * 0.5)
        else:
            # Over-experienced: slight penalty (they might still be good)
            gap = years_exp - max_exp
            return max(0.3, 1.0 - (gap / max(max_exp, 1)) * 0.3)
    
    def _score_location(self, profile: dict) -> float:
        """Score location match."""
        jd_location = self.jd.get("location", "")
        if not jd_location:
            return 0.7  # No location preference, give neutral score
        
        candidate_location = normalize_text(
            profile.get("location", "") or 
            profile.get("city", "") or
            profile.get("address", "")
        )
        
        if not candidate_location:
            return 0.4  # Unknown location
        
        # Exact or fuzzy city match
        if jd_location in candidate_location or candidate_location in jd_location:
            return 1.0
        
        sim = fuzz.token_sort_ratio(jd_location, candidate_location) / 100.0
        if sim >= 0.8:
            return 0.9
        elif sim >= 0.5:
            return 0.6
        
        # Check for remote-friendly
        if "remote" in candidate_location or "remote" in jd_location:
            return 0.8
        
        return 0.3
    
    def _score_education(self, profile: dict) -> float:
        """Score education relevance."""
        jd_education = self.jd.get("education", [])
        if not jd_education:
            return 0.6  # No education requirement specified
        
        candidate_edu = profile.get("education", [])
        if not candidate_edu:
            return 0.3
        
        max_score = 0.0
        for edu in candidate_edu:
            if isinstance(edu, dict):
                degree = normalize_text(edu.get("degree", "") or edu.get("qualification", ""))
                field = normalize_text(edu.get("field", "") or edu.get("field_of_study", ""))
                edu_text = f"{degree} {field}".strip()
            elif isinstance(edu, str):
                edu_text = normalize_text(edu)
            else:
                continue
            
            for req_edu in jd_education:
                sim = fuzz.token_sort_ratio(edu_text, req_edu) / 100.0
                max_score = max(max_score, sim)
        
        return min(max_score, 1.0)


def behavioral_score(candidate: dict) -> float:
    """
    Calculate behavioral signals score based on platform activity,
    profile completeness, and responsiveness.
    """
    profile = candidate.get("profile", {})
    behavioral = candidate.get("behavioral", {}) or candidate.get("signals", {})
    
    # Profile completeness
    completeness = _calculate_completeness(profile)
    
    # Activity signals
    activity = _calculate_activity(behavioral)
    
    # Responsiveness
    responsiveness = _calculate_responsiveness(behavioral)
    
    combined = (
        BEHAVIORAL_WEIGHT_COMPLETENESS * completeness +
        BEHAVIORAL_WEIGHT_ACTIVITY * activity +
        BEHAVIORAL_WEIGHT_RESPONSIVENESS * responsiveness
    )
    
    return min(max(combined, 0.0), 1.0)


def _calculate_completeness(profile: dict) -> float:
    """Score how complete a candidate's profile is."""
    fields_present = 0
    total_fields = 7
    
    if profile.get("current_title") or profile.get("title"):
        fields_present += 1
    if profile.get("summary") or profile.get("headline"):
        fields_present += 1
    if profile.get("skills"):
        fields_present += 1
    if profile.get("experience") or profile.get("career_history"):
        fields_present += 1
    if profile.get("education"):
        fields_present += 1
    if profile.get("location") or profile.get("city"):
        fields_present += 1
    if profile.get("years_of_experience"):
        fields_present += 1
    
    return fields_present / total_fields


def _calculate_activity(behavioral: dict) -> float:
    """Score platform activity/recency."""
    if not behavioral:
        return 0.5  # Neutral if no data
    
    # Look for activity indicators
    activity_score = 0.5
    
    if behavioral.get("last_active") or behavioral.get("last_login"):
        activity_score = 0.8
    
    if behavioral.get("profile_views", 0) > 0:
        activity_score = min(activity_score + 0.1, 1.0)
    
    if behavioral.get("applications_sent", 0) > 0:
        activity_score = min(activity_score + 0.1, 1.0)
    
    return activity_score


def _calculate_responsiveness(behavioral: dict) -> float:
    """Score responsiveness signals."""
    if not behavioral:
        return 0.5
    
    response_rate = behavioral.get("response_rate", None)
    if response_rate is not None:
        if isinstance(response_rate, (int, float)):
            return min(float(response_rate), 1.0)
    
    # Check if they respond to messages
    if behavioral.get("messages_replied", 0) > 0:
        total = behavioral.get("messages_received", 1)
        return min(behavioral["messages_replied"] / max(total, 1), 1.0)
    
    return 0.5
