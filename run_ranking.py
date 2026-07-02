"""
CLI Entry Point - Run the full ranking pipeline.

Usage:
    python run_ranking.py --candidates data/candidates.jsonl --jd data/job_description.json --output output/ranked_candidates.xlsx --top-n 100
"""

import argparse
import json
import sys
import os
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ranker import CandidateRanker
from src.utils import load_json


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Candidate Discovery - Rank candidates against a job description"
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates JSONL file"
    )
    parser.add_argument(
        "--jd", "-j",
        required=True,
        help="Path to job description JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="output/ranked_candidates.xlsx",
        help="Output path for ranked results (XLSX format)"
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=100,
        help="Number of top candidates to return (default: 100)"
    )
    parser.add_argument(
        "--model-path", "-m",
        default=None,
        help="Custom path to sentence-transformer model"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.candidates):
        print(f"Error: Candidates file not found: {args.candidates}")
        sys.exit(1)
    
    if not os.path.exists(args.jd):
        print(f"Error: Job description file not found: {args.jd}")
        sys.exit(1)
    
    # Load JD
    print(f"Loading job description from: {args.jd}")
    jd_data = load_json(args.jd)
    
    # Initialize ranker
    print("Initializing ranking engine...")
    ranker = CandidateRanker(jd_data, model_path=args.model_path)
    
    # Run ranking
    print(f"Ranking candidates from: {args.candidates}")
    results = ranker.rank_from_file(args.candidates, top_n=args.top_n)
    
    if not results:
        print("No candidates were ranked. Check your input data.")
        sys.exit(1)
    
    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Create DataFrame for XLSX output
    output_data = []
    for r in results:
        output_data.append({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": r["reasoning"],
        })
    
    df = pd.DataFrame(output_data)
    
    if args.output.endswith(".xlsx"):
        df.to_excel(args.output, index=False, engine="openpyxl")
    elif args.output.endswith(".csv"):
        df.to_csv(args.output, index=False)
    else:
        df.to_excel(args.output + ".xlsx", index=False, engine="openpyxl")
    
    print(f"\n{'='*60}")
    print(f"Ranking complete!")
    print(f"{'='*60}")
    print(f"Total ranked: {len(results)}")
    print(f"Top score: {results[0]['score']:.4f}")
    print(f"Output saved to: {args.output}")
    print(f"\nTop 5 candidates:")
    print(f"{'-'*60}")
    for r in results[:5]:
        print(f"  #{r['rank']} | {r['candidate_id']} | Score: {r['score']:.4f}")
        print(f"       {r['reasoning'][:80]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
