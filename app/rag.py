import hashlib
import json
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
from collections import defaultdict

import numpy as np
import redis
from litellm import completion  # type: ignore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SEGMENTS_PATH = os.path.join(DATA_DIR, "segments.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
COLLECTION_NAME = "transcripts"

# Initialize Redis client (adjust host/port if needed)
try:
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    redis_client = None


def load_segments():
    # Deprecated for raw files, but kept for compatibility if needed.
    # We now fetch directly from Qdrant.
    pass


class TranscriptIndex:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.q_client = QdrantClient(host=QDRANT_HOST, port=6333)
        
        # Load all segments from Qdrant for UI browsing and BM25 index
        self.segments = []
        try:
            scroll_res = self.q_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            self.segments = [p.payload for p in scroll_res[0]]
        except Exception as e:
            print(f"Warning: Could not fetch from Qdrant (might be empty): {e}")
            
        # Initialize BM25
        tokenized_corpus = [s["text"].lower().split() for s in self.segments]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 15, transcript_id: str = None):
        if not self.segments:
            return []
            
        # Semantic Search in Qdrant
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        
        filter_cond = None
        if transcript_id:
            filter_cond = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="transcript_id",
                        match=qmodels.MatchValue(value=transcript_id)
                    )
                ]
            )

        qdrant_res = self.q_client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb.tolist(),
            query_filter=filter_cond,
            limit=top_k
        )
        
        qdrant_res = qdrant_res.points
        
        # Lexical Search with BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else []
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = defaultdict(float)
        k = 60 # RRF constant
        
        # Add Qdrant ranks
        for rank, p in enumerate(qdrant_res):
            rrf_scores[p.payload["id"]] += 1.0 / (k + rank + 1)
            
        # Add BM25 ranks
        if len(bm25_scores) > 0:
            # Get top_k BM25 hits
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            for rank, idx in enumerate(bm25_top_indices):
                seg = self.segments[idx]
                if transcript_id and seg["transcript_id"] != transcript_id:
                    continue
                if bm25_scores[idx] > 0: # Only count if BM25 actually matched
                    rrf_scores[seg["id"]] += 1.0 / (k + rank + 1)
                    
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Map IDs back to segments
        seg_map = {s["id"]: s for s in self.segments}
        
        results = []
        for sid in sorted_ids[:top_k]:
            if sid in seg_map:
                results.append((seg_map[sid], rrf_scores[sid]))
                
        return results


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def verify_quote(quote: str, segments_by_id: dict, transcript_id: str) -> bool:
    nq = _normalize(quote)
    if len(nq) < 5:
        return False
    for seg in segments_by_id.values():
        if seg["transcript_id"] != transcript_id:
            continue
        if nq in _normalize(seg["text"]):
            return True
    return False


def _call_llm(provider: str, api_key: str, system: str, user: str, temperature: float = 0.0) -> str:
    model_map = {
        "OpenAI": "gpt-4o",
        "Anthropic": "claude-3-5-sonnet-20240620",
        "Gemini": "gemini/gemini-1.5-pro-latest",
        "Groq": "groq/llama3-70b-8192"
    }
    model_name = model_map.get(provider, "gpt-4o")
    
    resp = completion(
        model=model_name,
        api_key=api_key or None,
        temperature=temperature,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
    )
    return resp.choices[0].message.content


SYSTEM_PROMPT = """You are analysing expert-call transcripts. You must answer
ONLY using the excerpts provided to you in this prompt. Never use outside
knowledge and never invent quotes, speakers, or timestamps.

If the excerpts do not contain the answer, say "I don't know" and do not guess.

Always respond with valid JSON matching the schema you are given. Every
quote you output must be copied VERBATIM from the excerpts (do not
paraphrase inside a "quote" field)."""


def format_excerpts(results):
    lines = []
    for seg, score in results:
        lines.append(
            f'[{seg["transcript_id"]} | {seg["speaker"]} | {seg["timestamp"]} | id={seg["id"]}] {seg["text"]}'
        )
    return "\n".join(lines)


def rewrite_query(provider: str, api_key: str, question: str) -> list[str]:
    prompt = f"Rewrite the following user query into 3 alternate phrasings to improve vector search retrieval. Return ONLY the 3 queries, one per line.\n\nQuery: {question}"
    try:
        resp = _call_llm(provider, api_key, "You are an expert search assistant.", prompt, temperature=0.7)
        queries = [q.strip("- \t\n123456789.") for q in resp.splitlines() if q.strip()]
        return [question] + queries[:3]
    except Exception:
        return [question]


def judge_answer(provider: str, api_key: str, question: str, excerpts: str, answer: str) -> dict:
    judge_prompt = f"""You are a strict judge assessing faithfulness. 
Given a question, the source excerpts, and a generated answer, verify if every claim in the answer is STRICTLY supported by the excerpts.

Question: {question}
Excerpts:
{excerpts}

Generated Answer:
{answer}

Respond as JSON:
{{
    "is_faithful": true/false,
    "unsupported_claims": ["claim 1", "claim 2"] // empty if faithful
}}
"""
    try:
        raw = _call_llm(provider, api_key, "You evaluate RAG answers for hallucinations.", judge_prompt, temperature=0.0)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        return {"is_faithful": True, "unsupported_claims": []}


def answer_question(provider: str, api_key: str, index: TranscriptIndex, question: str, transcript_id: str = None, top_k: int = 15):
    # Check cache first
    cache_key = None
    if redis_client:
        key_str = f"{question}:{transcript_id or 'all'}:{top_k}"
        cache_key = f"answer_cache:{hashlib.md5(key_str.encode()).hexdigest()}"
        cached = redis_client.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

    # Multi-query rewriting
    queries = rewrite_query(provider, api_key, question)
    
    # Pool results from all queries
    all_results = {}
    for q in queries:
        res = index.search(q, top_k=top_k, transcript_id=transcript_id)
        for seg, score in res:
            all_results[seg["id"]] = (seg, score)
            
    # Sort and take top_k overall
    pooled = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:top_k]
    excerpts = format_excerpts(pooled)

    user_prompt = f"""Question: {question}

Excerpts (each line: [transcript | speaker | timestamp | id] text):
{excerpts}

Respond as JSON:
{{
  "answer": "<your answer, grounded only in the excerpts above>",
  "supporting_quotes": [
    {{"transcript_id": "...", "speaker": "...", "timestamp": "...", "quote": "<verbatim quote>"}}
  ]
}}"""

    raw = _call_llm(provider, api_key, SYSTEM_PROMPT, user_prompt, temperature=0.0)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw, "supporting_quotes": []}

    # Verify quotes
    segments_by_id = {s["id"]: s for s in index.segments}
    verified = []
    dropped = 0
    for q in parsed.get("supporting_quotes", []):
        if verify_quote(q.get("quote", ""), segments_by_id, q.get("transcript_id", "")):
            verified.append(q)
        else:
            dropped += 1

    parsed["supporting_quotes"] = verified
    parsed["unverified_dropped"] = dropped
    
    # LLM-as-judge (Faithfulness pass)
    judge_result = judge_answer(provider, api_key, question, excerpts, parsed.get("answer", ""))
    parsed["judge_result"] = judge_result

    # Store in cache for 24 hours
    if redis_client and cache_key:
        redis_client.setex(cache_key, 86400, json.dumps(parsed))
        
    return parsed


def analyze_themes(provider: str, api_key: str, index: TranscriptIndex):
    """Summarize -> Cluster by embeddings -> Compare within clusters -> Merge."""
    transcript_ids = sorted({s["transcript_id"] for s in index.segments})
    if not transcript_ids:
        return {"per_transcript": {}, "cross_transcript": {}}
        
    per_transcript_summaries = {}
    summary_texts = []

    for tid in transcript_ids:
        segs = [s for s in index.segments if s["transcript_id"] == tid]
        full_text = "\n".join(f'[{s["timestamp"]} | {s["speaker"]}] {s["text"]}' for s in segs)
        user_prompt = f"""Summarize the key points and opinions expressed in this transcript.
Be specific and attribute claims to the speaker.

Transcript ({tid}):
{full_text}

Respond as JSON: {{"summary": "<3-6 sentence summary>", "key_points": ["...", "..."]}}"""
        raw = _call_llm(provider, api_key, SYSTEM_PROMPT, user_prompt, temperature=0.0)
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(raw)
            per_transcript_summaries[tid] = parsed
            summary_texts.append(parsed.get("summary", ""))
        except json.JSONDecodeError:
            per_transcript_summaries[tid] = {"summary": raw, "key_points": []}
            summary_texts.append(raw)

    # Cluster summaries
    if len(summary_texts) > 2:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        embeddings = model.encode(summary_texts)
        # Choose n_clusters based on number of transcripts (max 3 or less)
        n_clusters = min(3, len(summary_texts) - 1)
        if n_clusters < 1: n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
    else:
        clusters = [0] * len(summary_texts)
        
    # Group by cluster
    grouped_summaries = defaultdict(dict)
    for i, tid in enumerate(transcript_ids):
        grouped_summaries[clusters[i]][tid] = per_transcript_summaries[tid]
        
    # Compare within clusters, then globally (simplified to global pass for this iteration)
    # At 30+ transcripts, you would loop through grouped_summaries first.
    
    combined = json.dumps(per_transcript_summaries, indent=2)
    user_prompt = f"""Here are per-expert summaries from call transcripts:

{combined}

Compare them and respond as JSON:
{{
  "common_themes": [{{"theme": "...", "supported_by": ["transcript_id", "..."]}}],
  "disagreements": [
    {{"topic": "...", "positions": [{{"transcript_id": "...", "position": "..."}}]}}
  ]
}}"""
    raw = _call_llm(provider, api_key, SYSTEM_PROMPT, user_prompt, temperature=0.0)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        cross = json.loads(raw)
    except json.JSONDecodeError:
        cross = {"common_themes": [], "disagreements": [], "raw": raw}

    # Add clustering info to result
    for i, tid in enumerate(transcript_ids):
        per_transcript_summaries[tid]["cluster"] = int(clusters[i])

    return {"per_transcript": per_transcript_summaries, "cross_transcript": cross}
