# Expert Call Transcript Analyzer — Advanced RAG

**Repository:** [https://github.com/Rohan77ux/trascripter](https://github.com/Rohan77ux/trascripter)
## What this does
Analyzes expert-call transcripts (.txt, .pdf) by using an advanced Retrieval-Augmented Generation (RAG) pipeline:
1. **Answers interview-guide questions** per expert or across all experts.
2. **Extracts exact, verbatim quotes** with source timestamps.
3. **Identifies common themes and disagreements** across the calls.
4. **Interactive Q&A** with multi-query retrieval and faithfulness checks.

## Architecture

This project has been upgraded from a simple in-memory script to a robust, scalable microservices architecture:

- **Streamlit (Frontend)**: Handles UI, file uploads, and displaying results.
- **MinIO (Object Storage)**: Acts as a local S3-compatible bucket. Uploaded transcripts are streamed directly to MinIO.
- **RabbitMQ (Message Queue)**: Handles asynchronous task queuing for ingestion. When a file is uploaded, a message is published to the queue.
- **Python Worker (Ingestion)**: Runs in the background, pulls messages from RabbitMQ, downloads the file from MinIO, extracts text chunks, generates embeddings using `sentence-transformers`, and upserts them into Qdrant.
- **Qdrant (Vector DB)**: Stores the vector embeddings for fast semantic search.
- **Redis (Cache)**: Caches LLM answers and themes to reduce API costs and improve latency.
- **LiteLLM (LLM Gateway)**: Abstracts API calls so you can seamlessly drop in API keys from **OpenAI**, **Anthropic**, **Google (Gemini)**, or **Groq** directly from the UI.

## Advanced RAG Features

- **Hybrid Search**: Combines BM25 (exact keyword match) and semantic embeddings (meaning) using Reciprocal Rank Fusion to ensure names, acronyms, and concepts are all accurately retrieved.
- **Query Rewriting**: The user's query is automatically rewritten into 3 alternate phrasings to catch wording mismatches during retrieval.
- **Quote Verification**: The generated answer is required to provide exact quotes. The code checks these quotes against the raw transcript text. If the LLM hallucinates a fake quote, it is dropped entirely.
- **LLM-as-Judge (Faithfulness Check)**: A second LLM pass grades the generated answer against the retrieved excerpts to ensure the model didn't subtly overstate or hallucinate details beyond what is supported by the text.

## Setup & Execution

### 1. Install Dependencies (Optional for IDEs)
While Docker handles its own dependencies, you can install them locally for your IDE:

```bash
pip install -r requirements.txt
```

### 2. Launch the Platform
The entire infrastructure (Streamlit, Background Worker, Qdrant, MinIO, RabbitMQ, Redis) is fully containerized. To start everything, simply run:

```bash
docker-compose up -d --build
```

### 3. Access the Application
- **Streamlit UI:** Open your browser to `http://localhost:8501`.
- **MinIO Console:** Available at `http://localhost:9001` (User: `minioadmin`, Pass: `minioadmin`)

## Usage
1. In the sidebar, select your preferred **AI Provider** (e.g., Anthropic, OpenAI, Groq, Gemini).
2. Enter your API Key.
3. Upload your transcript files (`.txt` or `.pdf`) using the file uploader.
4. Click **Process & Ingest**.
5. Wait for the background worker to process the files (check the worker terminal for logs).
6. Click **Reload transcripts** in the sidebar.
7. Start analyzing themes, asking questions, or generating interview guides!

## Transcript Format Assumptions
The ingestion logic in `app/ingest.py` expects lines roughly formatted like this:
```
[00:01:23] Jane Doe: We saw a big shift in Q3...
```
If your transcripts use a radically different format, adjust the `LINE_PATTERNS` regex list at the top of `app/ingest.py`.
