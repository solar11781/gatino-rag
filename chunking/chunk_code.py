import sys
from pathlib import Path

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path
from typing import List, Dict, Iterable
from tqdm import tqdm

from config import (
    CLEANED_ROOT,
    CHUNKS_JSONL,
    CODE_EXTS,
    MIN_LINES_PER_CHUNK,
    USE_TREE_SITTER,
    TREE_SITTER_FALLBACK_TO_RULES,
    MAX_CHUNK_CHARS,
    SUBCHUNK_OVERLAP_LINES,
)

from chunking.manual_chunker import (
    get_function_spans_for_ext,
    fallback_line_chunks,
    pull_comments_up,
)

from chunking.ts_chunker import (
    tree_sitter_spans_for_text,
    ts_structural_subspans_for_span,
)


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

# Split a [start:end] line-span into smaller spans so that each chunk's text length <= MAX_CHUNK_CHARS.
def split_span_by_max_chars(lines: List[str], start: int, end: int) -> List[Dict[str, int]]:
    out = []
    i = start

    while i < end:
        j = i
        while j < end:
            candidate = "\n".join(lines[i : j + 1])
            if len(candidate) <= MAX_CHUNK_CHARS:
                j += 1
            else:
                break

        # If even 1 line is too long, force a 1-line chunk
        if j == i:
            out.append({"start": i, "end": i + 1})
            i += 1
            continue

        out.append({"start": i, "end": j})

        # Move forward with a small overlap (keeps continuity)
        i = max(i + 1, j - SUBCHUNK_OVERLAP_LINES)

    return out

# Ensure every span produces text <= MAX_CHUNK_CHARS.
def enforce_max_chars_on_spans(lines: List[str], spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
    
    fixed: List[Dict[str, int]] = []
    for sp in spans:
        s, e = sp["start"], sp["end"]
        txt = "\n".join(lines[s:e])
        if len(txt) <= MAX_CHUNK_CHARS:
            fixed.append(sp)
        else:
            # line-based splitting by MAX_CHUNK_CHARS
            fixed.extend(split_span_by_max_chars(lines, s, e))
    return fixed


def chunk_file(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return []

    lines = text.splitlines()
    if not lines:
        return []

    # Drop files containing any single line exceeding MAX_CHUNK_CHARS
    for ln in lines:
        if len(ln) > MAX_CHUNK_CHARS:
            return []

    ext = path.suffix.lower()


    # Tree-sitter span extraction
    spans: List[Dict[str, int]] = []

    if USE_TREE_SITTER:
        ts_res = tree_sitter_spans_for_text(ext=ext, text=text)

        # If Tree-sitter returns spans, use them.
        # If it fails or returns empty, optionally fallback to rule-based.
        if ts_res.spans:
            spans = ts_res.spans
        else:
            print(f"[TS] No spans for {path} ({ts_res.error})")
            if not TREE_SITTER_FALLBACK_TO_RULES:
                return []

    # Rule-based splitters fallback
    if not spans:
        spans = get_function_spans_for_ext(ext, lines)

    # Pull comments only for the first produced chunk/subchunk.
    if not spans:
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

        # Guard against out-of-range spans
        start = max(0, min(start, len(lines)))
        end = max(0, min(end, len(lines)))

        chunk_lines = lines[start:end]

        # Skip very small chunks
        if len(chunk_lines) < MIN_LINES_PER_CHUNK:
            continue

        chunk_text = "\n".join(chunk_lines)

        # If parent is small enough => one piece [start:end]
        # If too big => try structural Tree-sitter statement splits
        # If structural split fails => fallback to line-based max-char split
        children_spans = [{"start": start, "end": end}]
        if len(chunk_text) > MAX_CHUNK_CHARS:
            # Try structural split (if TS enabled and the ext is supported)
            ts_struct = ts_structural_subspans_for_span(ext=ext, text=text, start_line=start, end_line=end)
            if ts_struct.spans:
                children_spans = ts_struct.spans
            else:
                # fallback to line-based size splitting
                children_spans = split_span_by_max_chars(lines, start, end)

        # Pull comments only for the first piece to avoid mixing
        if children_spans:
            first = pull_comments_up(lines, [children_spans[0]])[0]
            children_spans[0] = first
        children_spans = enforce_max_chars_on_spans(lines, children_spans)

        # Emit one record per child span
        for sub_id, sp in enumerate(children_spans):
            s = max(0, min(sp["start"], len(lines)))
            e = max(0, min(sp["end"], len(lines)))

            sub_lines = lines[s:e]
            if len(sub_lines) < MIN_LINES_PER_CHUNK:
                continue

            sub_text = "\n".join(sub_lines)

            # Drop whitespace-only / comment-only chunks
            if not sub_text.strip():
                continue

            chunk = {
                "repo_name": repo_name,
                "file_path": file_rel,
                "chunk_id": cid,         # parent function/method/chunk id
                "subchunk_id": sub_id,   # 0 if a chunk does not split
                "start_line": s + 1,
                "end_line": e,
                "language": ext.lstrip("."),
                "text": sub_text,
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
