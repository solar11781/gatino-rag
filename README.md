# Codebase RAG

A local, script-based pipeline for building a **semantic search / RAG system over codebases**.

This project takes raw source code from one or more repositories, **splits it into meaningful chunks**, **embeds those chunks locally**, and **indexes them into a vector database** so they can later be searched or used in RAG systems.

## Pipeline Overview

The codebase dataset is expected to be organized by repository, where each top-level folder represents a repo.

1. **Chunking** – Turning raw source code into structured, size-bounded text chunks that are suitable for embedding and indexing.

The process begins by scanning all files under `CLEANED_ROOT` and selecting only those with extensions listed in `CODE_EXTS`. For each file, the chunker records the repository name and the file path relative to `CLEANED_ROOT`.

Files are read as UTF-8 with errors ignored and split into lines. Empty files produce no output. If any single line exceeds the maximum allowed chunk size, the entire file is skipped to avoid processing minified or otherwise un-splittable content that would break downstream embedding.

For valid files, the chunker attempts to extract high-level structural spans such as functions, methods, or classes using Tree-sitter. If Tree-sitter fails, returns no spans, or is disabled, the system falls back to rule-based splitters using simple language heuristics. As a final fallback, files may be split using a purely line-based strategy.

Each identified span is treated as a parent chunk. If the span is already within the size limit, it becomes a single chunk. If it is too large, the chunker attempts to split it further using Tree-sitter subspans when possible. When no structural boundaries are available, the span is split by lines with a small overlap between adjacent subchunks to preserve context.

Before finalizing output, the chunker performs comment handling. Leading comments and blank lines that logically belong to a function or class are attached to the first subchunk of that span so documentation remains paired with the code it describes. Because this step can increase chunk size, size constraints are re-applied afterward. Whitespace-only chunks are discarded.

Each resulting chunk is written as a JSON object to `code_chunks.jsonl` and includes metadata such as:

- repository name
- file path
- chunk and subchunk IDs
- start and end line numbers
- language
- chunk text

Result: a clean, structured chunk dataset that preserves code structure and comments while remaining safe for embedding models.

2. **Indexing** – Convert the chunks into vector embeddings using Ollama and store embeddings with metadata in Qdrant

The indexer reads `code_chunks.jsonl` line by line, skipping invalid or empty entries so partial corruption does not interrupt processing. A connection to Qdrant is established and the target collection is created or recreated at the start of each run to ensure a clean index. The collection is configured with a fixed vector size matching the embedding model and uses cosine similarity.

Chunks are processed sequentially. For each chunk, the raw code text is sent to Ollama to generate an embedding. If embedding fails for any reason, the chunk is skipped and indexing continues.

Each embedding is stored alongside its metadata payload. For efficiency, vectors are accumulated into batches before being upserted into Qdrant, with any remaining vectors flushed at the end of the run.

Result: a fully populated Qdrant collection containing embeddings and metadata for every valid chunk, ready to be queried by search or RAG components.

3. **Search & Validation** – Helper scripts to inspect generated chunks, validate outputs, and perform basic vector similarity search against the indexed codebase.

## Project Structure

```
indexing/
│
├── chunking/                 # Code for chunking
│   ├── chunk_code.py         # Main chunking entrypoint
│   ├── manual_chunker.py     # Rule-based fallback + comment handling
│   ├── ts_chunker.py         # Tree-sitter span extraction
│   └── __init__.py
│
├── indexing/                 # Code for indexing
│   ├── index_code.py         # Main indexing entrypoint
│   ├── embeddings.py         # Ollama embedding wrapper
│   ├── qdrant_utils.py       # Qdrant collection helpers
│   └── __init__.py
│
├── helper_scripts/
│   ├── search.py             # Simple vector search
│   ├── check_oversized_chunks.py       # Check for any oversized chunks after chunking
│   ├── check_ts_languages.py           # Check for supported langauges by tree sitter
│   ├── chunks_viewer.py      # View chunks on QdrantDB
│   └── __init__.py
│
├── dataset/
├── code_chunks.jsonl         # Generated chunked code records
├── config.py                 # Global configuration
├── README.md
├── requirements.txt          # Dependencies
└── venv/                     # Local virtual environment
├── .gitignore
```

## How to Run the Project

### Prerequisites

- **Python 3.11.x** (developed and tested with [3.11.9](https://www.python.org/downloads/release/python-3119/))
- [**Docker Desktop**](https://www.docker.com/products/docker-desktop/) (for running Qdrant)
- [**Ollama**](https://ollama.com/download) (for local embeddings)

---

### 1. Install and run Qdrant (Docker)

```bash
docker pull qdrant/qdrant
```

```bash
docker run -p 6333:6333 -p 6334:6334 ^
  -v "%cd%/qdrant_storage:/qdrant/storage" ^
  qdrant/qdrant
```

Qdrant will be available at: http://localhost:6333

### 2. Install Ollama and pull the embedding model

Install Ollama for your OS and make sure the service is running.

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

### 3. Set up Python environment and install dependencies

From the project root:

### Create a virtual environment

```bash
py -3.11 -m venv venv
```

### Activate it (Windows)

```bash
venv\Scripts\activate
```

### Upgrade pip

```
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the project

Edit `config.py`:

- `CLEANED_ROOT` – path to your cleaned dataset (default: `dataset/` folder in the project root)
- `CHUNKS_JSONL` – output chunk file (default: `code_chunks.jsonl`)
- `MAX_CHUNK_CHARS` – max characters per chunk
- `EMBED_MODEL` – `nomic-embed-text`
- `EMBED_DIM` – `768`
- `QDRANT_URL` – `http://localhost:6333`
- `QDRANT_COLLECTION` – collection name

### 5. Run the pipeline

All commands below must be run **from the project root directory**.

#### Chunk the codebase

```bash
python chunking/chunk_code.py
```

This generates `code_chunks.jsonl` containing chunked code records.

#### Index chunks into Qdrant

```bash
python indexing/index_code.py
```

This step:

- Creates or resets the Qdrant collection
- Embeds each chunk using Ollama
- Upserts vectors and metadata into Qdrant

## Helper Scripts

Helper scripts are optional and run manually for inspection or validation:

#### Checks for generated chunks that exceeds the configured size limits

```bash
python helper_scripts/check_oversized_chunks.py
```

#### Prints which languages are supported by the current Tree-sitter setup

```bash
python helper_scripts/check_ts_languages.py
```

#### Displays or inspects chunks stored in Qdrant

```bash
python helper_scripts/chunks_viewer.py
```

#### Runs a simple vector similarity search against the indexed codebase.

```bash
python helper_scripts/search.py
```

## Notes

- Tree-sitter versions are pinned for compatibility
