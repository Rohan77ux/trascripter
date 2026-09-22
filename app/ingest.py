"""
Parses raw transcript .txt files into structured segments:
{id, transcript_id, transcript_name, speaker, timestamp, text}

Drop your 3 transcripts into data/transcripts/ as .txt files
(e.g. expert_1.txt, expert_2.txt, expert_3.txt) and run:

    python app/ingest.py

This writes data/segments.json, which the app reads.

WHY A SEPARATE INGEST STEP:
Keeping parsing separate from retrieval/generation means you can inspect
data/segments.json and fix the regex below in 30 seconds if the real
transcript format differs slightly from what's assumed here, without
touching the RAG or UI code at all.
"""

import json
import os
import re
import sys

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
OUT_PATH = os.path.join(DATA_DIR, "segments.json")

# Try a few common transcript line formats, in order.
# Add/adjust a pattern here if your real transcripts look different --
# each pattern must capture (timestamp, speaker, text) or (speaker, text).
LINE_PATTERNS = [
    # [00:01:23] Speaker Name: text
    re.compile(r"^\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<speaker>[^:]+):\s*(?P<text>.+)$"),
    # Speaker Name (00:01:23): text
    re.compile(r"^(?P<speaker>[^(:\n]+)\((?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\):\s*(?P<text>.+)$"),
    # 00:01:23 - Speaker Name: text
    re.compile(r"^(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(?P<speaker>[^:]+):\s*(?P<text>.+)$"),
    # Speaker Name: text   (no timestamp)
    re.compile(r"^(?P<speaker>[A-Za-z][A-Za-z0-9 ._'-]{0,40}):\s*(?P<text>.+)$"),
]


def get_lines_from_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if not PdfReader:
            raise RuntimeError("pypdf is required to parse PDFs.")
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f if ln.strip()]


def parse_transcript(path: str, transcript_name: str = None):
    name = transcript_name or os.path.splitext(os.path.basename(path))[0]
    segments = []
    lines = get_lines_from_file(path)

    current = None
    for line in lines:
        matched = False
        for pat in LINE_PATTERNS:
            m = pat.match(line)
            if m:
                gd = m.groupdict()
                if current:
                    segments.append(current)
                current = {
                    "speaker": gd.get("speaker", "Unknown").strip(),
                    "timestamp": gd.get("ts"),
                    "text": gd["text"].strip(),
                }
                matched = True
                break
        if not matched:
            # Continuation of the previous speaker's turn (wrapped line)
            if current:
                current["text"] += " " + line
            else:
                current = {"speaker": "Unknown", "timestamp": None, "text": line}
    if current:
        segments.append(current)

    out = []
    for i, seg in enumerate(segments):
        out.append(
            {
                "id": f"{name}_seg_{i:03d}",
                "transcript_id": name,
                "transcript_name": name,
                "speaker": seg["speaker"],
                "timestamp": seg["timestamp"] or "unknown",
                "text": seg["text"],
            }
        )
    return out


def main():
    if not os.path.isdir(TRANSCRIPT_DIR):
        print(f"Put your .txt transcripts in {TRANSCRIPT_DIR} first.")
        sys.exit(1)

    files = [f for f in sorted(os.listdir(TRANSCRIPT_DIR)) if f.endswith(".txt")]
    if not files:
        print(f"No .txt files found in {TRANSCRIPT_DIR}.")
        sys.exit(1)

    all_segments = []
    for fname in files:
        path = os.path.join(TRANSCRIPT_DIR, fname)
        segs = parse_transcript(path)
        print(f"Parsed {len(segs)} segments from {fname}")
        all_segments.extend(segs)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(all_segments)} total segments to {OUT_PATH}")
    print("Spot-check a few entries -- if speaker/timestamp look wrong,")
    print("adjust LINE_PATTERNS in this file to match your real transcript format.")


if __name__ == "__main__":
    main()
