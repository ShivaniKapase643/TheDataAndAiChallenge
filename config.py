"""
Configuration for the Intelligent Candidate Discovery system.
All scoring weights and thresholds are centralized here.
"""

# ── Model Configuration ──
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 64

# ── Hybrid Scoring Weights ──
WEIGHT_FEATURE = 0.45      # Feature-based scoring weight
WEIGHT_SEMANTIC = 0.40     # Semantic similarity weight
WEIGHT_BEHAVIORAL = 0.15   # Behavioral signals weight

# ── Feature Scoring Sub-Weights ──
WEIGHT_SKILLS = 0.30       # Skills match
WEIGHT_CAREER = 0.25       # Career/title relevance
WEIGHT_EXPERIENCE = 0.20   # Experience years fit
WEIGHT_LOCATION = 0.15     # Location preference
WEIGHT_EDUCATION = 0.10    # Education relevance

# ── Honeypot Detection Thresholds ──
MAX_SKILLS_THRESHOLD = 50          # Flag if candidate lists 50+ skills
MIN_EXPERIENCE_FOR_SENIOR = 3     # Minimum years for senior-level claims
MAX_TITLE_CHANGES_PER_YEAR = 3    # Suspicious if more than 3 roles/year
KEYWORD_DENSITY_THRESHOLD = 0.4   # Flag if keyword density exceeds 40%

# ── Behavioral Scoring ──
BEHAVIORAL_WEIGHT_ACTIVITY = 0.4       # Platform activity recency
BEHAVIORAL_WEIGHT_COMPLETENESS = 0.35  # Profile completeness
BEHAVIORAL_WEIGHT_RESPONSIVENESS = 0.25 # Response rate signals

# ── Output Configuration ──
DEFAULT_TOP_N = 100
SCORE_NORMALIZATION_MIN = 0.0
SCORE_NORMALIZATION_MAX = 1.0
