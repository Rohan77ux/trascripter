import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

try:
    q_client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode(["What was the biggest challenge in Q3?"], normalize_embeddings=True)[0]
    
    qdrant_res = q_client.search(
        collection_name="transcripts",
        query_vector=q_emb.tolist(),
        query_filter=None,
        limit=15
    )
    print("Search successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
