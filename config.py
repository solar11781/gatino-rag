from pathlib import Path


# Path to cleaned dataset
CLEANED_ROOT = Path(r"D:\Desktop\torture\new project\indexing\dummy_dataset")
# CLEANED_ROOT = Path(r"D:\Desktop\torture\new project\indexing\dataset")

# Output file for code chunks
CHUNKS_JSONL = Path("code_chunks.jsonl")
# CHUNKS_JSONL = Path("code_chunks_test.jsonl")

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

# Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "code_chunks"
# QDRANT_COLLECTION = "code"

# Ollama / embedding model
EMBED_MODEL = "mxbai-embed-large"
EMBED_DIM = 1024

# Batch size for Qdrant upserts
BATCH_SIZE = 50