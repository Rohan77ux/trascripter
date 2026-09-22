import os
from dotenv import load_dotenv
load_dotenv()

import pika
import sys
import time
import uuid
import tempfile

from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from ingest import parse_transcript

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost:9000")
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
COLLECTION_NAME = "transcripts"
BUCKET_NAME = "transcripts"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
q_client = QdrantClient(host=QDRANT_HOST, port=6333)

minio_client = Minio(
    MINIO_HOST,
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

# Ensure collection exists
try:
    q_client.get_collection(COLLECTION_NAME)
except Exception:
    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=384, # all-MiniLM-L6-v2 output size
            distance=qmodels.Distance.COSINE
        )
    )

def process_file(ch, method, properties, body):
    object_name = body.decode('utf-8')
    print(f" [x] Received {object_name} from queue")
    
    # Download from MinIO to a temporary file
    ext = os.path.splitext(object_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        
    try:
        minio_client.fget_object(BUCKET_NAME, object_name, tmp_path)
        
        # Parse using existing regex logic
        segs = parse_transcript(tmp_path)
        
        points = []
        texts = [s["text"] for s in segs]
        
        if texts:
            embeddings = embed_model.encode(texts, normalize_embeddings=True)
            
            for i, seg in enumerate(segs):
                points.append(
                    qmodels.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, seg["id"])),
                        vector=embeddings[i].tolist(),
                        payload=seg
                    )
                )
                
            q_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
        print(f" [x] Processed {len(segs)} segments from MinIO object {object_name}")
    except Exception as e:
        print(f" [!] Error processing {object_name}: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    connection = None
    while not connection:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        except pika.exceptions.AMQPConnectionError:
            print("Waiting for RabbitMQ...")
            time.sleep(2)
            
    channel = connection.channel()
    channel.queue_declare(queue='ingestion-queue', durable=True)
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='ingestion-queue', on_message_callback=process_file)
    
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    main()
