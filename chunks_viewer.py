from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_COLLECTION, CLEANED_ROOT
from pathlib import Path



def read_code_fragment(full_path: Path, start: int, end: int) -> str:
    if not full_path.is_file():
        return f"[ERROR] File not found: {full_path}"

    try:
        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_idx = max(start - 1, 0)
        end_idx = min(end, len(lines))
        snippet = "\n".join(lines[start_idx:end_idx])
        return snippet
    except Exception as e:
        return f"[ERROR reading file] {e}"


def view_chunks(limit):
    client = QdrantClient(url=QDRANT_URL)

    print(f"[INFO] Retrieving first {limit} points from '{QDRANT_COLLECTION}'...\n")

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    for point in points:
        p = point.payload

        repo = p.get("repo_name")
        file_path = p.get("file_path")
        chunk_id = p.get("chunk_id")
        start = p.get("start_line")
        end = p.get("end_line")

        full_path = Path(CLEANED_ROOT) / file_path

        print("============================================================")
        print(f"POINT ID:       {point.id}")
        print(f"REPO:           {repo}")
        print(f"FILE:           {file_path}")
        print(f"CHUNK ID:       {chunk_id}")
        print(f"LINES:          {start}-{end}")
        print(f"DISK PATH:      {full_path}")
        print("------------------------------------------------------------")

        # Load the actual code snippet
        snippet = read_code_fragment(full_path, start, end)

        print(snippet)
        print("============================================================\n")


if __name__ == "__main__":
    view_chunks(limit=10)
