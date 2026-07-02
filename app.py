"""
Gradio Web Demo - Interactive candidate ranking interface.

Upload a JSONL file of candidates and a job description to see ranked results
with full score breakdown and explainable reasoning.
"""

import csv
import io
import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.jd_parser import JDParser
from src.scorer import CandidateScorer, behavioral_score
from src.honeypot_detector import is_honeypot
from src.reasoning_gen import generate_reasoning
from src.utils import build_candidate_text
from config import WEIGHT_FEATURE, WEIGHT_SEMANTIC, WEIGHT_BEHAVIORAL

# ── Lazy model loading ──
_model = None


def _load_model():
    global _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def rank_candidates_demo(candidates_file, jd_text, top_n=100):
    """Main ranking function for the Gradio demo."""
    _load_model()
    
    # Parse JD
    try:
        jd_data = json.loads(jd_text)
    except json.JSONDecodeError:
        # Treat as plain text JD
        jd_data = {"title": "Target Role", "description": jd_text}
    
    jd_parser = JDParser(jd_data)
    scorer = CandidateScorer(jd_parser)
    
    # Load candidates
    if hasattr(candidates_file, "read"):
        content = candidates_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
    else:
        with open(candidates_file, encoding="utf-8") as f:
            content = f.read()
    
    candidates = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = [parsed]
    except json.JSONDecodeError:
        pass
    
    if not candidates:
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not candidates:
        return "❌ No valid candidates found.", None, "0", "0", "0", "0.000"
    
    # Filter honeypots
    valid = []
    honeypot_count = 0
    for c in candidates:
        if is_honeypot(c):
            honeypot_count += 1
        else:
            valid.append(c)
    
    if not valid:
        return "❌ All candidates filtered as honeypots.", None, str(len(candidates)), str(honeypot_count), "0", "0.000"
    
    # Feature scores
    feature_scores = []
    score_details = []
    for c in valid:
        detail = scorer.score(c)
        feature_scores.append(detail["feature_score"])
        score_details.append(detail)
    
    # Semantic scores
    jd_embed_text = jd_parser.get_embedding_text()
    jd_emb = _model.encode(jd_embed_text, normalize_embeddings=True)
    
    candidate_texts = [build_candidate_text(c) for c in valid]
    cand_embs = _model.encode(candidate_texts, normalize_embeddings=True, batch_size=32)
    sims = (cand_embs @ jd_emb).astype(float)
    
    s_min, s_max = sims.min(), sims.max()
    semantic_scores = (sims - s_min) / (s_max - s_min + 1e-9)
    
    # Behavioral scores
    beh_scores = [behavioral_score(c) for c in valid]
    
    # Combine
    feature_arr = np.array(feature_scores)
    semantic_arr = np.array(semantic_scores)
    behavioral_arr = np.array(beh_scores)
    
    final_scores = (
        WEIGHT_FEATURE * feature_arr +
        WEIGHT_SEMANTIC * semantic_arr +
        WEIGHT_BEHAVIORAL * behavioral_arr
    )
    
    # Sort and get top N
    sorted_idx = np.argsort(final_scores)[::-1]
    top_idx = sorted_idx[:int(top_n)]
    
    top_final = final_scores[top_idx]
    if len(top_final) > 1:
        t_min, t_max = top_final.min(), top_final.max()
        rng = t_max - t_min if t_max != t_min else 1.0
        normalized = 0.5 + 0.5 * (top_final - t_min) / rng
    else:
        normalized = np.array([1.0])
    
    # Build results
    results = []
    for rank, (idx, norm_score) in enumerate(zip(top_idx, normalized), 1):
        c = valid[idx]
        detail = score_details[idx]
        detail["semantic_score"] = float(semantic_arr[idx])
        detail["behavioral_score"] = float(behavioral_arr[idx])
        detail["final_score"] = float(norm_score)
        
        reasoning = generate_reasoning(c, rank, detail)
        profile = c.get("profile", {})
        
        results.append({
            "candidate_id": c.get("candidate_id", f"cand_{idx}"),
            "rank": rank,
            "score": round(float(norm_score), 4),
            "title": profile.get("current_title", "") or profile.get("title", "N/A"),
            "years_exp": profile.get("years_of_experience", 0),
            "feature": round(float(feature_arr[idx]), 3),
            "semantic": round(float(semantic_arr[idx]), 3),
            "behavioral": round(float(behavioral_arr[idx]), 3),
            "reasoning": reasoning,
        })
    
    # Build markdown table
    lines = [
        "| Rank | ID | Title | YOE | Feature | Semantic | Behavioral | Score |",
        "|------|-----|-------|-----|---------|----------|------------|-------|",
    ]
    for r in results[:25]:
        lines.append(
            f"| **{r['rank']}** | {r['candidate_id'][:12]} | "
            f"{r['title'][:25]} | {r['years_exp']} | "
            f"{r['feature']:.3f} | {r['semantic']:.3f} | "
            f"{r['behavioral']:.2f} | **{r['score']}** |"
        )
    
    if len(results) > 25:
        lines.append(f"\n*... and {len(results) - 25} more in the downloadable file*")
    
    # Save CSV for download
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    writer = csv.writer(tmp)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    tmp.close()
    
    avg_score = sum(r["score"] for r in results) / len(results)
    
    return (
        "\n".join(lines),
        tmp.name,
        str(len(candidates)),
        str(honeypot_count),
        str(len(results)),
        f"{avg_score:.3f}",
    )


# ── Gradio Interface ──
CSS = """
.hero-title { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
.hero-title span { color: #7c3aed; }
.rank-button { background: linear-gradient(90deg, #8b5cf6, #7c3aed) !important; 
               color: white !important; font-weight: bold !important; }
"""

SAMPLE_JD = json.dumps({
    "title": "Senior Backend Engineer",
    "required_skills": ["Python", "Django", "PostgreSQL", "REST APIs", "Docker"],
    "preferred_skills": ["Kubernetes", "AWS", "Redis", "Celery", "GraphQL"],
    "experience": "5-8 years",
    "location": "Bangalore",
    "education": ["B.Tech in Computer Science", "M.Tech"],
    "description": "We are looking for a Senior Backend Engineer to design and build scalable microservices. You will work with Python/Django, manage PostgreSQL databases, deploy on AWS with Docker/K8s, and mentor junior engineers."
}, indent=2)

with gr.Blocks(title="Intelligent Candidate Discovery", css=CSS) as demo:
    gr.HTML("""
    <div style="padding: 20px;">
        <div class="hero-title">🎯 Intelligent Candidate <span>Discovery</span></div>
        <p style="color: #6b7280;">AI-powered candidate ranking that goes beyond keyword matching.
        Upload candidates + job description to get ranked results with explanations.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Inputs")
            file_input = gr.File(
                label="Candidates (JSONL/JSON)",
                file_types=[".jsonl", ".json"],
            )
            jd_input = gr.Textbox(
                label="Job Description (JSON)",
                value=SAMPLE_JD,
                lines=15,
            )
            top_n_slider = gr.Slider(
                minimum=10, maximum=500, value=100, step=10,
                label="Top N candidates to rank"
            )
            rank_btn = gr.Button("🚀 Rank Candidates", elem_classes="rank-button", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results")
            with gr.Row():
                kpi_total = gr.Textbox(label="Total Candidates", interactive=False)
                kpi_honeypot = gr.Textbox(label="Honeypots Filtered", interactive=False)
                kpi_ranked = gr.Textbox(label="Ranked", interactive=False)
                kpi_avg = gr.Textbox(label="Avg Score", interactive=False)
            
            output_table = gr.Markdown(value="Upload candidates and click Rank to see results.")
            output_file = gr.File(label="⬇️ Download Ranked CSV", interactive=False)
    
    rank_btn.click(
        fn=rank_candidates_demo,
        inputs=[file_input, jd_input, top_n_slider],
        outputs=[output_table, output_file, kpi_total, kpi_honeypot, kpi_ranked, kpi_avg],
    )

if __name__ == "__main__":
    demo.launch()
