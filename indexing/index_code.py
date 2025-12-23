import sys
from pathlib import Path

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
from typing import Dict, Iterable, List
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import (
    CHUNKS_JSONL,
    QDRANT_URL,
    QDRANT_COLLECTION,
    BATCH_SIZE,
)

from indexing.embeddings import embed_text
from indexing.qdrant_utils import init_qdrant_collection


def iter_chunks(jsonl_path: Path) -> Iterable[Dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping bad JSON line: {e}")
                continue
            yield obj



def index_chunks():
    if not CHUNKS_JSONL.is_file():
        print(f"[ERROR] Chunks file not found: {CHUNKS_JSONL}")
        return
    
    # Count total chunks for progress bar
    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        total_chunks = sum(1 for line in f if line.strip())

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # Create/reset collection
    init_qdrant_collection(client)

    batch_points: List[qm.PointStruct] = []
    point_id = 0
    num_chunks = 0

    for chunk in tqdm(
        iter_chunks(CHUNKS_JSONL),
        total=total_chunks,
        desc="Indexing chunks",
    ):
        num_chunks += 1

        text = chunk["text"]
        repo_name = chunk.get("repo_name", "unknown")
        file_path = chunk.get("file_path", "")
        chunk_id = chunk.get("chunk_id", 0)
        subchunk_id = chunk.get("subchunk_id", 0)
        start_line = chunk.get("start_line", 1)
        end_line = chunk.get("end_line", start_line)
        language = chunk.get("language", "unknown")

        # print("Embedding chars:", len(text))
        # print("First 200 chars:", repr(text[:200]))

        # Get embedding from Ollama
        try:
            emb = embed_text(text)
        except Exception as e:
            print("[EMBED SKIP]", len(text), file_path)
            continue

        # Construct payload with tree-based metadata
        payload = {
            "repo_name": repo_name,
            "project_id": repo_name,  # TODO: temporary placeholder
            "file_path": file_path,
            "chunk_id": chunk_id,
            "subchunk_id": subchunk_id,
            "start_line": start_line,
            "end_line": end_line,
            "language": language,
            "type": "code",
        }

        point = qm.PointStruct(
            id=point_id,
            vector=emb,
            payload=payload,
        )

        batch_points.append(point)
        point_id += 1

        # When batch is full, upsert to Qdrant
        if len(batch_points) >= BATCH_SIZE:
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)
            print(f"\n[UPSERT] {point_id} vectors indexed")
            batch_points = []

    # Flush any remaining points
    if batch_points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch_points)

    print(f"[INFO] Indexed {num_chunks} chunks as {point_id} Qdrant points.")




if __name__ == "__main__":
    index_chunks()
