"""
Shared utility functions for the candidate ranking system.
"""

import json
import re
from typing import Any, Generator


def stream_jsonl(filepath: str) -> Generator[dict, None, None]:
    """Stream JSONL file line by line to handle large files efficiently."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_json(filepath: str) -> dict:
    """Load a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip extra whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_years_from_text(text: str) -> float:
    """Extract years of experience from text like '5+ years' or '3-5 years'."""
    if not text:
        return 0.0
    matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text.lower())
    if matches:
        return float(matches[0])
    range_match = re.findall(r"(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)", text.lower())
    if range_match:
        return (float(range_match[0][0]) + float(range_match[0][1])) / 2
    return 0.0


def safe_get(data: dict, *keys, default=None) -> Any:
    """Safely navigate nested dictionaries."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def clean_skills_list(skills: list) -> list:
    """Clean and normalize a list of skills."""
    if not skills:
        return []
    cleaned = []
    for skill in skills:
        if isinstance(skill, str):
            s = normalize_text(skill)
            if s and len(s) > 1:
                cleaned.append(s)
        elif isinstance(skill, dict):
            name = skill.get("name", "") or skill.get("skill", "")
            s = normalize_text(name)
            if s and len(s) > 1:
                cleaned.append(s)
    return cleaned


def build_candidate_text(candidate: dict) -> str:
    """Build a rich text representation of a candidate for embedding."""
    parts = []
    
    profile = candidate.get("profile", {})
    
    # Current title
    title = profile.get("current_title", "") or profile.get("title", "")
    if title:
        parts.append(f"Title: {title}")
    
    # Summary/headline
    summary = profile.get("summary", "") or profile.get("headline", "")
    if summary:
        parts.append(f"Summary: {summary[:500]}")
    
    # Skills
    skills = profile.get("skills", [])
    if isinstance(skills, list):
        skill_names = clean_skills_list(skills)
        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names[:30])}")
    
    # Experience/career history
    experience = profile.get("experience", []) or profile.get("career_history", [])
    if isinstance(experience, list):
        for exp in experience[:5]:  # Top 5 most recent roles
            if isinstance(exp, dict):
                exp_title = exp.get("title", "") or exp.get("role", "")
                exp_company = exp.get("company", "") or exp.get("organization", "")
                exp_desc = exp.get("description", "") or exp.get("summary", "")
                if exp_title:
                    exp_text = f"{exp_title} at {exp_company}" if exp_company else exp_title
                    if exp_desc:
                        exp_text += f" - {exp_desc[:200]}"
                    parts.append(exp_text)
    
    # Education
    education = profile.get("education", [])
    if isinstance(education, list):
        for edu in education[:3]:
            if isinstance(edu, dict):
                degree = edu.get("degree", "") or edu.get("qualification", "")
                field = edu.get("field", "") or edu.get("field_of_study", "")
                school = edu.get("school", "") or edu.get("institution", "")
                if degree or field:
                    edu_text = f"{degree} in {field}" if field else degree
                    if school:
                        edu_text += f" from {school}"
                    parts.append(edu_text)
    
    # Location
    location = profile.get("location", "") or profile.get("city", "")
    if location:
        parts.append(f"Location: {location}")
    
    return " | ".join(parts) if parts else "No profile information available"
