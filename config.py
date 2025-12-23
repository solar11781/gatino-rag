from pathlib import Path


# Path to cleaned dataset
CLEANED_ROOT = Path(__file__).resolve().parent / "dataset"

# Output file for code chunks
CHUNKS_JSONL = Path("code_chunks.jsonl")

CODE_EXTS = {
    ".py",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".cs", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp",
    ".go",
    ".php",
    ".rb",
    ".rs",
    ".sh", ".bat",
}

# Line-based fallback chunking
MAX_LINES_PER_CHUNK = 80
MIN_LINES_PER_CHUNK = 1

MAX_CHUNK_CHARS = 1500

# When split a long chunk into smaller ones, overlap a bit to keeps context
SUBCHUNK_OVERLAP_LINES = 2

# Tree-sitter chunking
USE_TREE_SITTER = True

# If Tree-sitter fails for a language/file, fallback to rule-based splitters.
TREE_SITTER_FALLBACK_TO_RULES = True


# Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "code_chunks"

# Ollama / embedding model
EMBED_MODEL = "nomic-embed-text"

EMBED_DIM = 768

# Batch size for Qdrant upserts
BATCH_SIZE = 50