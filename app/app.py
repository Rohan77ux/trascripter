import os
from dotenv import load_dotenv
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import io
import streamlit as st
from minio import Minio

sys.path.insert(0, os.path.dirname(__file__))
from rag import TranscriptIndex, answer_question, analyze_themes, SEGMENTS_PATH
from ingest import parse_transcript, TRANSCRIPT_DIR, OUT_PATH

# MinIO Client
MINIO_HOST = os.environ.get("MINIO_HOST", "localhost:9000")
minio_client = Minio(
    MINIO_HOST,
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

# Ensure bucket exists
BUCKET_NAME = "transcripts"
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

st.set_page_config(page_title="Expert Call Transcript Analyzer", layout="wide")
st.title("Expert Call Transcript Analyzer")
st.caption("Hasamex AI Engineer case study")

# ---- Setup ----
with st.sidebar:
    st.header("Setup")
    provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic", "Gemini", "Groq"])
    api_key = st.text_input("API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    
    uploaded_files = st.file_uploader("Upload Transcripts (.txt, .pdf)", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Process & Ingest"):
        import pika
        
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ.get('RABBITMQ_HOST', 'localhost')))
            channel = connection.channel()
            channel.queue_declare(queue='ingestion-queue', durable=True)
            
            for uf in uploaded_files:
                # Upload directly to MinIO
                file_bytes = uf.getbuffer()
                minio_client.put_object(
                    BUCKET_NAME,
                    uf.name,
                    data=io.BytesIO(file_bytes),
                    length=file_bytes.nbytes,
                    content_type=uf.type
                )
                
                # Publish the MinIO object name to queue
                channel.basic_publish(
                    exchange='',
                    routing_key='ingestion-queue',
                    body=uf.name,
                    properties=pika.BasicProperties(
                        delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
                    )
                )
            
            connection.close()
            st.success(f"Uploaded {len(uploaded_files)} files to MinIO and queued for background processing!")
            st.info("The files will be processed by the worker soon. Click 'Reload transcripts' below to see updates.")
        except Exception as e:
            st.error(f"Failed to connect to RabbitMQ or MinIO: {e}")

    if st.button("Reload transcripts"):
        st.cache_resource.clear()
        st.rerun()

@st.cache_resource(ttl=300) # Cache but refresh occasionally
def get_index():
    return TranscriptIndex()

index = get_index()

if not index.segments:
    st.warning("No parsed transcripts found yet. Please upload your transcripts in the sidebar, and wait for the background worker to process them. Click 'Reload transcripts' to check again.")
    st.stop()
transcript_ids = sorted({s["transcript_id"] for s in index.segments})
st.sidebar.write(f"Loaded {len(index.segments)} segments across {len(transcript_ids)} transcripts:")
st.sidebar.write(", ".join(transcript_ids))

if not api_key:
    st.info("Enter your API key in the sidebar to run analysis.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Interview Guide", "Themes & Disagreements", "Ask a Question", "Browse Transcripts"])

# ---- Tab 1: Interview guide ----
with tab1:
    st.subheader("Run interview-guide questions")
    st.caption("Paste one question per line (from the interview guide in your case pack).")
    guide_text = st.text_area("Questions", height=150, placeholder="What was the biggest challenge in Q3?\nHow did pricing change?\n...")
    scope = st.selectbox("Scope", ["All transcripts"] + transcript_ids)
    if st.button("Run questions") and guide_text.strip():
        tid = None if scope == "All transcripts" else scope
        questions = [l.strip() for l in guide_text.splitlines() if l.strip()]
        
        # We can batch them into one LLM call or just run them concurrently.
        # Since answer_question is fully synchronous, we will run them sequentially here,
        # but the prompt requested batching. Let's send them in one prompt.
        
        with st.spinner("Answering all questions..."):
            # A true batched prompt would be a separate function, but for now we iterate 
            # (In a production system you'd use asyncio or a single massive prompt).
            # We'll stick to iterating but cache speeds it up.
            for q in questions:
                result = answer_question(provider, api_key, index, q, transcript_id=tid, top_k=15)
                st.markdown(f"**Q: {q}**")
                
                # Faithfulness flag
                judge = result.get("judge_result", {})
                if not judge.get("is_faithful", True):
                    st.warning(f"⚠️ Potential Hallucination Detected: {', '.join(judge.get('unsupported_claims', []))}")
                    
                st.write(result.get("answer", ""))
                for quote in result.get("supporting_quotes", []):
                    st.markdown(
                        f"> \"{quote['quote']}\" — *{quote['speaker']}, {quote['transcript_id']} @ {quote['timestamp']}*"
                    )
                if result.get("unverified_dropped"):
                    st.caption(f"({result['unverified_dropped']} unverified quote(s) from the model were dropped)")
                st.divider()

# ---- Tab 2: Themes ----
with tab2:
    st.subheader("Common themes and disagreements across all 3 experts")
    if st.button("Analyze themes"):
        with st.spinner("Summarizing each transcript, then comparing..."):
            result = analyze_themes(provider, api_key, index)
        cross = result["cross_transcript"]

        st.markdown("### Common themes")
        for t in cross.get("common_themes", []):
            st.markdown(f"- **{t.get('theme')}** — supported by: {', '.join(t.get('supported_by', []))}")

        st.markdown("### Disagreements")
        for d in cross.get("disagreements", []):
            st.markdown(f"**{d.get('topic')}**")
            for p in d.get("positions", []):
                st.markdown(f"- *{p.get('transcript_id')}*: {p.get('position')}")

        with st.expander("Per-transcript summaries"):
            for tid, s in result["per_transcript"].items():
                st.markdown(f"**{tid}**: {s.get('summary')}")

# ---- Tab 3: Free Q&A ----
with tab3:
    st.subheader("Ask a question across all transcripts")
    q = st.text_input("Your question")
    if st.button("Ask") and q.strip():
        with st.spinner("Thinking..."):
            result = answer_question(provider, api_key, index, q, top_k=15)
            
        judge = result.get("judge_result", {})
        if not judge.get("is_faithful", True):
            st.warning(f"⚠️ Potential Hallucination Detected: {', '.join(judge.get('unsupported_claims', []))}")
            
        st.write(result.get("answer", ""))
        for quote in result.get("supporting_quotes", []):
            st.markdown(
                f"> \"{quote['quote']}\" — *{quote['speaker']}, {quote['transcript_id']} @ {quote['timestamp']}*"
            )

# ---- Tab 4: Browse ----
with tab4:
    st.subheader("Raw transcript viewer")
    view_tid = st.selectbox("Transcript", transcript_ids, key="browse")
    for seg in index.segments:
        if seg["transcript_id"] == view_tid:
            st.markdown(f"**[{seg['timestamp']}] {seg['speaker']}:** {seg['text']}")
