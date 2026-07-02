"""
Honeypot Detector - Identifies and filters suspicious/fake candidate profiles.

Honeypot profiles are synthetic or low-quality entries that should be excluded
from the final ranking to maintain result quality.
"""

import re
from src.utils import safe_get, clean_skills_list
from config import (
    MAX_SKILLS_THRESHOLD,
    MIN_EXPERIENCE_FOR_SENIOR,
    MAX_TITLE_CHANGES_PER_YEAR,
    KEYWORD_DENSITY_THRESHOLD,
)


def is_honeypot(candidate: dict) -> bool:
    """
    Determine if a candidate profile is a honeypot (fake/suspicious).
    
    Returns True if the profile should be filtered out.
    """
    flags = 0
    
    profile = candidate.get("profile", {})
    if not profile:
        return True  # No profile data at all
    
    # Check 1: Excessive skills count
    skills = profile.get("skills", [])
    if isinstance(skills, list) and len(skills) > MAX_SKILLS_THRESHOLD:
        flags += 2
    
    # Check 2: Inconsistent experience claims
    flags += _check_experience_inconsistency(profile)
    
    # Check 3: Keyword stuffing in summary
    flags += _check_keyword_stuffing(profile)
    
    # Check 4: Suspicious career timeline
    flags += _check_career_timeline(profile)
    
    # Check 5: Generic/copy-paste descriptions
    flags += _check_generic_content(profile)
    
    # Check 6: Missing critical fields
    flags += _check_missing_fields(profile)
    
    # Threshold: 3+ flags means honeypot
    return flags >= 3


def _check_experience_inconsistency(profile: dict) -> int:
    """Check if experience claims are inconsistent with career history."""
    flags = 0
    
    years_exp = profile.get("years_of_experience", 0)
    if isinstance(years_exp, str):
        try:
            years_exp = float(re.findall(r"\d+\.?\d*", years_exp)[0])
        except (IndexError, ValueError):
            years_exp = 0
    
    title = (profile.get("current_title", "") or "").lower()
    
    # Senior title with very low experience
    senior_keywords = ["senior", "lead", "principal", "director", "vp", "head", "chief"]
    if any(kw in title for kw in senior_keywords) and years_exp < MIN_EXPERIENCE_FOR_SENIOR:
        flags += 1
    
    # Claims 20+ years but title is junior
    junior_keywords = ["intern", "trainee", "fresher", "entry"]
    if any(kw in title for kw in junior_keywords) and years_exp > 15:
        flags += 2
    
    # Unrealistically high experience (50+ years)
    if years_exp > 50:
        flags += 2
    
    return flags


def _check_keyword_stuffing(profile: dict) -> int:
    """Detect keyword stuffing in summary/description."""
    summary = profile.get("summary", "") or profile.get("headline", "")
    if not summary:
        return 0
    
    words = summary.lower().split()
    if len(words) < 5:
        return 0
    
    # Check for repeated buzzwords
    buzzwords = [
        "expert", "guru", "ninja", "rockstar", "passionate", "dynamic",
        "results-driven", "self-motivated", "team player", "innovative"
    ]
    
    buzzword_count = sum(1 for w in words if w in buzzwords)
    density = buzzword_count / len(words)
    
    if density > KEYWORD_DENSITY_THRESHOLD:
        return 2
    
    # Check for comma-separated skill dumps in summary
    comma_count = summary.count(",")
    if comma_count > 20 and len(summary) < 500:
        return 1
    
    return 0


def _check_career_timeline(profile: dict) -> int:
    """Check for impossible career timelines."""
    experience = profile.get("experience", []) or profile.get("career_history", [])
    if not isinstance(experience, list) or len(experience) < 2:
        return 0
    
    years_exp = profile.get("years_of_experience", 0)
    if isinstance(years_exp, str):
        try:
            years_exp = float(re.findall(r"\d+\.?\d*", years_exp)[0])
        except (IndexError, ValueError):
            years_exp = 0
    
    # Too many roles for the stated experience
    if years_exp > 0 and len(experience) / max(years_exp, 1) > MAX_TITLE_CHANGES_PER_YEAR:
        return 1
    
    return 0


def _check_generic_content(profile: dict) -> int:
    """Detect generic/template content that suggests a fake profile."""
    summary = (profile.get("summary", "") or "").lower()
    
    generic_phrases = [
        "lorem ipsum",
        "insert description here",
        "to be updated",
        "n/a",
        "test candidate",
        "sample profile",
        "placeholder",
    ]
    
    for phrase in generic_phrases:
        if phrase in summary:
            return 3  # Instant flag
    
    return 0


def _check_missing_fields(profile: dict) -> int:
    """Check if critical fields are missing (suspicious for real candidates)."""
    flags = 0
    
    # No title AND no skills AND no experience
    has_title = bool(profile.get("current_title", ""))
    has_skills = bool(profile.get("skills", []))
    has_experience = bool(profile.get("experience", []) or profile.get("career_history", []))
    
    if not has_title and not has_skills and not has_experience:
        flags += 2
    
    return flags
