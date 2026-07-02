"""
Reasoning Generator - Produces human-readable explanations for ranking decisions.

Every ranked candidate gets a justification explaining why they were ranked
at their position, making the system explainable and trustworthy for recruiters.
"""

from src.utils import safe_get, clean_skills_list


def generate_reasoning(candidate: dict, rank: int, scores: dict) -> str:
    """
    Generate a human-readable reasoning for a candidate's ranking.
    
    Args:
        candidate: Full candidate dict
        rank: Final rank position (1 = best)
        scores: Dict with feature_score, semantic_score, behavioral_score, 
                skills_score, career_score, experience_score, etc.
    
    Returns:
        A concise explanation string.
    """
    profile = candidate.get("profile", {})
    parts = []
    
    # Title relevance
    title = profile.get("current_title", "") or profile.get("title", "N/A")
    career_score = scores.get("career_score", 0)
    if career_score >= 0.7:
        parts.append(f"Strong title alignment ({title})")
    elif career_score >= 0.4:
        parts.append(f"Moderate career relevance ({title})")
    
    # Skills match
    skills_score = scores.get("skills_score", 0)
    if skills_score >= 0.8:
        parts.append("Excellent skills match")
    elif skills_score >= 0.5:
        parts.append("Good skills coverage")
    elif skills_score >= 0.3:
        parts.append("Partial skills match")
    
    # Experience fit
    exp_score = scores.get("experience_score", 0)
    years = profile.get("years_of_experience", 0)
    if exp_score >= 0.9:
        parts.append(f"Ideal experience level ({years}y)")
    elif exp_score >= 0.6:
        parts.append(f"Acceptable experience ({years}y)")
    
    # Semantic understanding
    semantic_score = scores.get("semantic_score", 0)
    if semantic_score >= 0.8:
        parts.append("High semantic relevance to role")
    elif semantic_score >= 0.5:
        parts.append("Moderate profile-JD alignment")
    
    # Location
    loc_score = scores.get("location_score", 0)
    if loc_score >= 0.9:
        location = profile.get("location", "") or profile.get("city", "")
        if location:
            parts.append(f"Location match ({location})")
    
    # Behavioral signals
    behavioral_score = scores.get("behavioral_score", 0)
    if behavioral_score >= 0.8:
        parts.append("Active profile with strong engagement")
    
    if not parts:
        parts.append("Ranked based on overall profile assessment")
    
    # Combine
    reasoning = "; ".join(parts) + f". Overall score: {scores.get('final_score', 0):.3f}"
    
    return reasoning


def generate_summary_stats(results: list) -> dict:
    """Generate summary statistics for the ranking results."""
    if not results:
        return {}
    
    scores = [r.get("final_score", 0) for r in results]
    
    return {
        "total_ranked": len(results),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "score_std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
    }
