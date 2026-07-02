"""
Job Description Parser - Extracts structured requirements from a JD.
"""

import re
from typing import Dict, List
from src.utils import normalize_text, extract_years_from_text


class JDParser:
    """Parses a job description into structured requirements."""
    
    def __init__(self, jd_data: dict):
        """
        Initialize with JD data. Accepts either:
        - A dict with structured fields (title, skills, experience, etc.)
        - A dict with a 'description' field containing raw text
        """
        self.raw = jd_data
        self.requirements = self._parse()
    
    def _parse(self) -> Dict:
        """Extract structured requirements from JD data."""
        req = {
            "title": "",
            "required_skills": [],
            "preferred_skills": [],
            "min_experience": 0.0,
            "max_experience": 0.0,
            "location": "",
            "education": [],
            "seniority": "",
            "description_text": "",
            "industry": "",
        }
        
        # Extract title
        req["title"] = (
            self.raw.get("title", "") or 
            self.raw.get("job_title", "") or 
            self.raw.get("role", "")
        )
        
        # Extract skills
        required = self.raw.get("required_skills", []) or self.raw.get("skills", [])
        if isinstance(required, list):
            req["required_skills"] = [normalize_text(s) for s in required if s]
        elif isinstance(required, str):
            req["required_skills"] = [normalize_text(s.strip()) for s in required.split(",") if s.strip()]
        
        preferred = self.raw.get("preferred_skills", []) or self.raw.get("nice_to_have", [])
        if isinstance(preferred, list):
            req["preferred_skills"] = [normalize_text(s) for s in preferred if s]
        
        # Extract experience
        exp = self.raw.get("experience", "") or self.raw.get("years_of_experience", "")
        if isinstance(exp, (int, float)):
            req["min_experience"] = float(exp)
            req["max_experience"] = float(exp) + 5
        elif isinstance(exp, str):
            # Try to parse range like "3-5 years"
            range_match = re.findall(r"(\d+)\s*[-–]\s*(\d+)", exp)
            if range_match:
                req["min_experience"] = float(range_match[0][0])
                req["max_experience"] = float(range_match[0][1])
            else:
                years = extract_years_from_text(exp)
                req["min_experience"] = years
                req["max_experience"] = years + 5
        elif isinstance(exp, dict):
            req["min_experience"] = float(exp.get("min", 0))
            req["max_experience"] = float(exp.get("max", req["min_experience"] + 5))
        
        # Extract location
        req["location"] = normalize_text(
            self.raw.get("location", "") or 
            self.raw.get("city", "") or 
            self.raw.get("preferred_location", "")
        )
        
        # Extract education
        education = self.raw.get("education", []) or self.raw.get("qualifications", [])
        if isinstance(education, list):
            req["education"] = [normalize_text(e) for e in education if e]
        elif isinstance(education, str):
            req["education"] = [normalize_text(education)]
        
        # Infer seniority
        title_lower = req["title"].lower()
        if any(w in title_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
            req["seniority"] = "senior"
        elif any(w in title_lower for w in ["junior", "jr.", "entry", "associate"]):
            req["seniority"] = "junior"
        else:
            req["seniority"] = "mid"
        
        # Build full description text for embedding
        desc = self.raw.get("description", "") or self.raw.get("job_description", "")
        if desc:
            req["description_text"] = desc
        else:
            # Build from structured data
            parts = [req["title"]]
            if req["required_skills"]:
                parts.append(f"Required: {', '.join(req['required_skills'])}")
            if req["preferred_skills"]:
                parts.append(f"Preferred: {', '.join(req['preferred_skills'])}")
            req["description_text"] = ". ".join(parts)
        
        # Industry
        req["industry"] = normalize_text(
            self.raw.get("industry", "") or self.raw.get("domain", "")
        )
        
        return req
    
    def get_embedding_text(self) -> str:
        """Get text suitable for embedding the JD."""
        parts = []
        if self.requirements["title"]:
            parts.append(f"Role: {self.requirements['title']}")
        if self.requirements["required_skills"]:
            parts.append(f"Required Skills: {', '.join(self.requirements['required_skills'])}")
        if self.requirements["preferred_skills"]:
            parts.append(f"Preferred Skills: {', '.join(self.requirements['preferred_skills'])}")
        if self.requirements["min_experience"]:
            parts.append(f"Experience: {self.requirements['min_experience']}-{self.requirements['max_experience']} years")
        if self.requirements["description_text"]:
            parts.append(self.requirements["description_text"][:800])
        return " | ".join(parts)
    
    def get_required_skills(self) -> List[str]:
        return self.requirements["required_skills"]
    
    def get_preferred_skills(self) -> List[str]:
        return self.requirements["preferred_skills"]
    
    def get_experience_range(self) -> tuple:
        return (self.requirements["min_experience"], self.requirements["max_experience"])
    
    def get_seniority(self) -> str:
        return self.requirements["seniority"]
