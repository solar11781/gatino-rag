import sys
from pathlib import Path

from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_COLLECTION, CLEANED_ROOT
from index_code import embed_text


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


def search_code(query: str, limit: int = 5):
    client = QdrantClient(url=QDRANT_URL)

    # Embed the query
    q_emb = embed_text(query)

    # Vector search in Qdrant
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=q_emb,
        limit=limit,
    )

    print(f"\n[SEARCH] Top {limit} results for query: '{query}'\n")

    if not result.points:
        print("No results found.")
        return

    for rank, point in enumerate(result.points, start=1):
        p = point.payload or {}

        repo = p.get("repo_name")
        file_path = p.get("file_path")
        chunk_id = p.get("chunk_id")
        start_line = p.get("start_line")
        end_line = p.get("end_line")
        language = p.get("language")
        score = point.score

        full_path = Path(CLEANED_ROOT) / file_path

        print("============================================================")
        print(f"RESULT #{rank}")
        print(f"SCORE:      {score:.4f}")
        print(f"REPO:       {repo}")
        print(f"FILE:       {file_path}")
        print(f"LANG:       {language}")
        print(f"CHUNK ID:   {chunk_id}")
        print(f"LINES:      {start_line}-{end_line}")
        print(f"DISK PATH:  {full_path}")
        print("------------------------------------------------------------")

        snippet = read_code_fragment(full_path, start_line, end_line)
        print(snippet)
        print("============================================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter search query: ").strip()

    if not query:
        print("Empty query, exiting.")
    else:
        search_code(query, limit=5)
