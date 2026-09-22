import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rag import TranscriptIndex, answer_question, _client

# Mock Golden Eval Set
# In a real scenario, you'd populate this with your actual expected transcript excerpts.
GOLDEN_SET = [
    {
        "question": "What was the biggest challenge in Q3?",
        "expected_quote": "We saw a massive supply chain disruption in early Q3.",
        "expected_transcript_id": "expert_1"
    },
    {
        "question": "How did pricing change?",
        "expected_quote": "Pricing increased by 15% across the board.",
        "expected_transcript_id": "expert_2"
    }
]

def run_evals():
    print("Initializing Qdrant index...")
    index = TranscriptIndex()
    client = _client()
    
    total = len(GOLDEN_SET)
    metrics = {
        "recall_at_k": 0,
        "quote_match_rate": 0,
        "fully_grounded": 0
    }
    
    print(f"Running evals on {total} questions...")
    for i, item in enumerate(GOLDEN_SET):
        q = item["question"]
        print(f"\n--- Question {i+1}: {q} ---")
        
        # 1. Recall @ K
        res = index.search(q, top_k=15)
        retrieved_texts = [seg["text"].lower() for seg, score in res]
        expected_lower = item["expected_quote"].lower()
        
        recall = any(expected_lower in t for t in retrieved_texts)
        if recall:
            metrics["recall_at_k"] += 1
            print("Recall: SUCCESS")
        else:
            print("Recall: FAILED")
            
        # 2. Quote Match Rate & Groundedness
        ans = answer_question(client, index, q, top_k=15)
        
        # Check if the expected quote is in the generated output quotes
        quote_match = any(expected_lower in q_dict["quote"].lower() for q_dict in ans.get("supporting_quotes", []))
        if quote_match:
            metrics["quote_match_rate"] += 1
            print("Quote Match: SUCCESS")
        else:
            print("Quote Match: FAILED")
            
        # Groundedness
        judge = ans.get("judge_result", {})
        if judge.get("is_faithful", True):
            metrics["fully_grounded"] += 1
            print("Groundedness: PASSED (Faithful)")
        else:
            print(f"Groundedness: FAILED (Unsupported: {judge.get('unsupported_claims')})")
            
    print("\n========================")
    print("FINAL EVALUATION METRICS")
    print("========================")
    print(f"Recall @ K:       {metrics['recall_at_k']} / {total} ({(metrics['recall_at_k']/total)*100:.1f}%)")
    print(f"Quote Match Rate: {metrics['quote_match_rate']} / {total} ({(metrics['quote_match_rate']/total)*100:.1f}%)")
    print(f"Fully Grounded:   {metrics['fully_grounded']} / {total} ({(metrics['fully_grounded']/total)*100:.1f}%)")

if __name__ == "__main__":
    run_evals()
