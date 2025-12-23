import sys
from pathlib import Path

# Allow running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from config import (
    MAX_CHUNK_CHARS,
    CHUNKS_JSONL,
)

bad = []

with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        text = obj.get("text", "")
        l = len(text)

        if not text.strip():
            bad.append((i, "EMPTY", obj["file_path"]))
            continue

        if l > MAX_CHUNK_CHARS:
            bad.append((
                i,
                l,
                obj["file_path"],
                obj.get("start_line"),
                obj.get("end_line"),
            ))

print("Bad chunks:", len(bad))
if len(bad) > 0:
    print(bad[:5])
