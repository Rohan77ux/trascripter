from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

q_client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")
q_emb = model.encode(["What was the biggest challenge in Q3?"], normalize_embeddings=True)[0]

try:
    res = q_client.query_points(
        collection_name="transcripts",
        query=q_emb.tolist(),
        query_filter=None,
        limit=15
    )
    print("Found points:", len(res.points))
except Exception as e:
    import traceback
    traceback.print_exc()
