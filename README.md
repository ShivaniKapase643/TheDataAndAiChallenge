<<<<<<< HEAD
# 🎯 Intelligent Candidate Discovery - RedRob INDIA.RUNS Challenge

An AI-powered candidate ranking system that goes beyond keyword matching to understand who truly fits a role. Built for the RedRob INDIA.RUNS Data & AI Challenge.

## 🧠 How It Works

The system uses a **hybrid scoring architecture** combining:

1. **Semantic Understanding** (40%) — Uses sentence-transformers (BGE-small-en-v1.5) to encode job descriptions and candidate profiles into the same embedding space, measuring true semantic similarity.

2. **Feature-Based Scoring** (45%) — Multi-dimensional scoring across:
   - Skills match (30%) — Weighted overlap between required and candidate skills
   - Career relevance (25%) — Title similarity + career progression signals
   - Experience fit (20%) — Years of experience vs. JD requirements
   - Location match (15%) — Proximity to preferred location
   - Education fit (10%) — Degree relevance to role requirements

3. **Behavioral Signals** (15%) — Platform activity, profile completeness, responsiveness indicators

4. **Honeypot Detection** — Filters out suspicious/fake profiles using heuristic rules

## 📁 Project Structure

```
├── src/
│   ├── ranker.py              # Main ranking pipeline
│   ├── scorer.py              # Feature-based scoring logic
│   ├── semantic_search.py     # Embedding-based similarity
│   ├── honeypot_detector.py   # Fake profile detection
│   ├── jd_parser.py           # Job description understanding
│   ├── reasoning_gen.py       # Explainable ranking justifications
│   └── utils.py               # Shared utilities
├── app.py                     # Gradio web demo
├── run_ranking.py             # CLI entry point for full ranking
├── config.py                  # Configuration and weights
├── requirements.txt           # Dependencies
├── sample_data/
│   ├── sample_candidates.jsonl  # Sample candidate data
│   └── job_description.json     # Sample JD
├── output/
│   └── ranked_candidates.xlsx   # Output file
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Ranking (CLI)
```bash
python run_ranking.py --candidates data/candidates.jsonl --jd data/job_description.json --output output/ranked_candidates.xlsx --top-n 100
```

### Run Web Demo
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

## 🏗️ Architecture

```
JD Input → JD Parser → Requirement Extraction
                              ↓
Candidates (JSONL) → Honeypot Filter → Feature Scorer ─┐
                                    → Semantic Encoder ─┤→ Hybrid Combiner → Ranked Output
                                    → Behavioral Score ─┘
```

## 📊 Scoring Formula

```
final_score = (0.45 × feature_score + 0.40 × semantic_score + 0.15 × behavioral_score) × honeypot_filter
```

Where `honeypot_filter` is 0 (filtered) or 1 (valid candidate).

## 🛡️ Honeypot Detection

The system identifies and filters suspicious profiles based on:
- Excessive skill count (50+ skills listed)
- Inconsistent career timelines
- Keyword-stuffed summaries
- Unrealistic experience claims
- Copy-pasted generic descriptions

## 📦 Technologies Used

| Technology | Purpose |
|-----------|---------|
| sentence-transformers | Semantic embeddings (BGE-small-en-v1.5) |
| NumPy | Vector operations and scoring |
| pandas | Data manipulation |
| scikit-learn | Cosine similarity, text processing |
| Gradio | Web demo interface |
| openpyxl | XLSX output generation |
| rapidfuzz | Fuzzy string matching for skills |

## 🎓 Why This Approach?

Traditional ATS systems rely on exact keyword matches, missing candidates who use different terminology for the same skills. Our system:

- **Understands context**: "ML Engineer" and "Machine Learning Developer" are semantically equivalent
- **Weighs career evidence**: Actual job history matters more than listed skills
- **Penalizes gaming**: Detects keyword stuffing and inflated profiles
- **Explains decisions**: Every ranking comes with human-readable justification

## 📝 Output Format

The ranked output XLSX contains:
| Column | Description |
|--------|-------------|
| candidate_id | Unique identifier |
| rank | Position (1 = best fit) |
| score | Normalized score (0-1) |
| reasoning | Human-readable justification |

## 👥 Team

- **Team Name:** [Your Team Name]
- **Team Leader:** [Your Name]

---

Built for the RedRob INDIA.RUNS Data & AI Challenge 2026
=======
# TheDataAndAiChallenge
>>>>>>> 7bd69c09a17f8f9ce461a9fd2f4d5c08db8892ca
