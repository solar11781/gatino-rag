import json
from pathlib import Path
from typing import List, Dict, Iterable

from tqdm import tqdm

from config import CLEANED_ROOT, CHUNKS_JSONL, CODE_EXTS, MIN_LINES_PER_CHUNK
from splitters import get_function_spans_for_ext, fallback_line_chunks, pull_comments_up


def iter_code_files(root: Path) -> Iterable[Path]:
    for repo_dir in root.iterdir():
        if not repo_dir.is_dir():
            continue

        for path in repo_dir.rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in CODE_EXTS:
                yield path

def chunk_file(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return []

    lines = text.splitlines()
    if not lines:
        return []

    ext = path.suffix.lower()

    # Try language-specific function-based splitter
    spans = get_function_spans_for_ext(ext, lines)

    # If found spans, pull comments above into the chunk
    # Otherwise fallback to line chunks
    if spans:
        spans = pull_comments_up(lines, spans)
    else:
        spans = fallback_line_chunks(lines)

    chunks: List[Dict] = []

    # Find repo name and file path
    try:
        rel_to_root = path.relative_to(CLEANED_ROOT)
        repo_name = rel_to_root.parts[0]
        file_rel = str(rel_to_root)
    except ValueError:
        repo_name = "unknown"
        file_rel = str(path)

    for cid, span in enumerate(spans):
        start = span["start"]
        end = span["end"]
        chunk_lines = lines[start:end]

        # Skip very small chunks
        if len(chunk_lines) < MIN_LINES_PER_CHUNK:
            continue

        chunk_text = "\n".join(chunk_lines)

        chunk = {
            "repo_name": repo_name,
            "file_path": file_rel,
            "chunk_id": cid,
            "start_line": start + 1,
            "end_line": end,
            "language": ext.lstrip("."),
            "text": chunk_text,
        }
        chunks.append(chunk)

    return chunks

def main():
    if not CLEANED_ROOT.is_dir():
        print(f"[ERROR] CLEANED_ROOT does not exist: {CLEANED_ROOT}")
        return

    files = list(iter_code_files(CLEANED_ROOT))
    print(f"Found {len(files)} code files under {CLEANED_ROOT}")

    total_chunks = 0

    with CHUNKS_JSONL.open("w", encoding="utf-8") as out_f:
        for file_path in tqdm(files, desc="Chunking code files"):
            chunks = chunk_file(file_path)
            total_chunks += len(chunks)
            for ch in chunks:
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Done. Created {total_chunks} code chunks.")
    print(f"Saved to: {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
