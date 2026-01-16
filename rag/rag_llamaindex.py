from pathlib import Path
import sys
import re
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple

# --- Add project root to PYTHONPATH ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, OLLAMA_EMBED_URL, OLLAMA_LLM_URL, LLM_MODEL

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# -----------------------------
# Rich output
# -----------------------------
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---- Config ----


# Debug toggles
DEBUG_CONTEXT = False          # print approximate context_str
DEBUG_RENDERED_PROMPT = False  # print rendered prompt text
DEBUG_FILES = False            # print retrieved file list


# -----------------------------
# Helpers
# -----------------------------
def detect_repo_from_query(q: str) -> Optional[str]:
    """
    If user includes repo name in query (e.g. 'repo:ChessCode' or 'in ChessCode'), auto filter.
    """
    m = re.search(r"(?:repo:|in\s+)([a-zA-Z0-9\-_]+)", q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def guess_lang_from_path(file_path: str) -> str:
    fp = (file_path or "").lower()
    if fp.endswith(".py"):
        return "python"
    if fp.endswith(".rb"):
        return "ruby"
    if fp.endswith(".js"):
        return "javascript"
    if fp.endswith(".ts"):
        return "typescript"
    if fp.endswith(".tsx"):
        return "tsx"
    if fp.endswith(".jsx"):
        return "jsx"
    if fp.endswith(".json"):
        return "json"
    if fp.endswith(".yml") or fp.endswith(".yaml"):
        return "yaml"
    if fp.endswith(".md"):
        return "markdown"
    if fp.endswith(".html"):
        return "html"
    if fp.endswith(".css"):
        return "css"
    if fp.endswith(".sh"):
        return "bash"
    return "text"


def fmt_score(score):
    if score is None:
        return "NA"
    try:
        return f"{float(score):.4f}"
    except Exception:
        return str(score)


def normalize_query_tokens(q: str) -> List[str]:
    """
    Extract meaningful tokens from query for evidence search.
    Prefer long tokens; keep unique.
    """
    q = (q or "").lower()
    toks = re.findall(r"[a-zA-Z0-9_\.]+", q)
    toks = [t for t in toks if len(t) >= 3]  # allow 3 for functions like "api"
    toks = sorted(set(toks), key=len, reverse=True)
    return toks


def build_query_engine(top_k: int = 40, repo_filter: Optional[str] = None) -> Tuple[Any, PromptTemplate]:
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        text_key="text",
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_EMBED_URL,
    )
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_LLM_URL,
        request_timeout=120.0,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    # STRICT prompt: require evidence, never guess
    prompt = PromptTemplate(
        """
You are a senior software engineer assistant.

STRICT RULES (NO HALLUCINATION):
- Use ONLY the provided context.
- If the context does NOT explicitly contain the answer, output exactly:
  - Answer: Not found in provided context.
  - Evidence: (none)
  - Files: (none)
- If you answer, you MUST include an exact code quote in Evidence copied verbatim from context.
- NEVER guess file names, function names, or behavior.

Context:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer format:
- Answer: ...
- Evidence: "..."   (MUST be an exact quote from context)
- Files: repo/path:start-end
"""
    )

    filters = None
    if repo_filter:
        filters = MetadataFilters(filters=[MetadataFilter(key="repo_name", value=repo_filter)])

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=prompt,
        response_mode="compact",
        filters=filters,
    )
    return query_engine, prompt


def choose_repo_interactively(source_nodes):
    """
    If results cover multiple repos, ask user to pick repo -> requery.
    """
    repos = []
    for sn in source_nodes:
        meta = sn.node.metadata or {}
        r = meta.get("repo_name")
        if r:
            repos.append(r)

    if not repos:
        return None

    counts = Counter(repos)
    if len(counts) <= 1:
        return None

    print("\nFound results from multiple repos:")
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i, (r, c) in enumerate(items, start=1):
        print(f"[{i}] {r} (hits={c})")

    choice = input("Choose repo number (Enter to skip): ").strip()
    if not choice:
        return None

    try:
        idx = int(choice)
        if 1 <= idx <= len(items):
            return items[idx - 1][0]
    except Exception:
        return None

    return None


# -----------------------------
# DEBUG: context_str inspection
# -----------------------------
def debug_context(res, max_nodes: int = 10, max_chars: int = 8000):
    """Print an approximate context_str (what nodes look like)"""
    if not res.source_nodes:
        print("\n[DEBUG] No source_nodes => context_str is empty.\n")
        return

    parts = []
    for i, sn in enumerate(res.source_nodes[:max_nodes], start=1):
        node = sn.node
        meta = node.metadata or {}
        repo = meta.get("repo_name", "")
        fp = meta.get("file_path", "")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        score = getattr(sn, "score", None)

        header = f"\n### Node {i} | score={fmt_score(score)} | {repo}/{fp}:{sl}-{el}\n"
        parts.append(header + (node.text or ""))

    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n\n...[TRUNCATED]..."

    print("\n" + "=" * 30 + " DEBUG CONTEXT_STR " + "=" * 30)
    print(ctx)
    print("=" * 80 + "\n")


def debug_rendered_prompt(q: str, res, prompt: PromptTemplate, max_nodes: int = 8, max_chars: int = 12000):
    """Render prompt with a locally-built context_str and print it"""
    ctx = "\n\n".join([(sn.node.text or "") for sn in res.source_nodes[:max_nodes]]) if res.source_nodes else ""
    rendered = prompt.format(context_str=ctx, query_str=q)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n\n...[TRUNCATED]..."
    print("\n" + "=" * 30 + " DEBUG RENDERED PROMPT " + "=" * 30)
    print(rendered)
    print("=" * 82 + "\n")


# -----------------------------
# Evidence extractor (Upgraded)
# -----------------------------
STOP_TOKENS = {
    "the", "and", "with", "this", "that", "what", "which", "where", "when", "from",
    "trong", "project", "file", "repo", "hàm", "ham", "nào", "nao", "xử", "xu", "lý", "ly",
    "kiểm", "kiem", "tra", "nước", "nuoc", "đi", "di", "hợp", "hop", "lệ", "le",
    "valid", "move", "check", "logic", "giải", "giai", "thích", "thich"
}

def score_line_match(line: str, tokens: List[str]) -> int:
    """
    Score a line by how many important tokens appear.
    Higher = better evidence.
    """
    l = line.lower()
    score = 0
    for t in tokens:
        if t in STOP_TOKENS or len(t) < 3:
            continue
        # if t.lower() in l:
        #     score += 2 if len(t) >= 6 else 1

        # normalize token to alnum root
        t_clean = re.sub(r"[^a-z0-9]", "", t.lower())
        if not t_clean:
            continue

        # exact or substring match (strong for long tokens)
        if t_clean in l:
            score += 2 if len(t_clean) >= 6 else 1
            continue

        # partial-root match: allow long-token prefixes to match morphological variants
        # e.g., "authentication" -> matches "authenticated" or "is_authenticated"
        if len(t_clean) >= 6:
            prefix = t_clean[:6]
            if prefix in l:
                score += 2
                continue


    return score


def extract_signature_tokens(q: str) -> List[str]:
    """
    Pull likely identifiers: function names, class names, file names, snake_case, etc.
    """
    ql = (q or "").lower()
    toks = normalize_query_tokens(ql)

    # add common code intent tokens
    if "valid" in ql and "move" in ql:
        toks = ["is_legal", "valid_move", "legal_move", "islegal", "can_move", "move_legal"] + toks

    # prefer identifiers (contains underscore or dot or camelish)
    preferred = []
    for t in toks:
        if t in STOP_TOKENS:
            continue
        if "_" in t or "." in t:
            preferred.append(t)
        elif re.match(r"^[a-z][a-z0-9]{2,}$", t):
            preferred.append(t)

    # de-dup preserving order
    out = []
    seen = set()
    for t in preferred:
        if t not in seen:
            out.append(t)
            seen.add(t)

    return out[:20]


def extract_evidence(question: str, source_nodes, max_nodes: int = 25) -> Optional[Dict[str, Any]]:
    """
    Upgraded hard extraction:
    - Prefer actual definitions: Ruby `def`, Python `def`, JS/TS `function` / method patterns.
    - Require stronger match: line must match >=2 meaningful tokens OR match a definition pattern.
    - Reduce false positives from UI/icon files.
    """
    q = question or ""
    ql = q.lower()
    tokens = extract_signature_tokens(q)

    best = None  # (score, evidence_line, files, answer_hint)

    for sn in source_nodes[:max_nodes]:
        node = sn.node
        meta = node.metadata or {}
        fp = meta.get("file_path", "")
        repo = meta.get("repo_name", "")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        text = node.text or ""
        lines = text.splitlines()

        fp_l = fp.lower()
        is_ruby = fp_l.endswith(".rb")
        is_py = fp_l.endswith(".py")
        is_js_ts = fp_l.endswith(".js") or fp_l.endswith(".ts") or fp_l.endswith(".tsx") or fp_l.endswith(".jsx")

        # Heuristic: deprioritize obvious UI/icon assets if query is about "valid move"
        if ("valid" in ql and "move" in ql) and ("pieceicon" in fp_l or "icon" in fp_l):
            continue

        # --- 1) Prefer explicit definitions ---
        # Ruby: def <name>
        if is_ruby:
            for t in tokens:
                if t in STOP_TOKENS or len(t) < 3:
                    continue
                pat = re.compile(rf"^\s*def\s+{re.escape(t)}\b", re.IGNORECASE)
                for ln in lines:
                    if pat.search(ln):
                        return {
                            "answer": f"Found Ruby method definition `def {t}`",
                            "evidence": ln.strip(),
                            "files": f"{repo}/{fp}:{sl}-{el}",
                        }

        # Python: def <name>
        if is_py:
            for t in tokens:
                if t in STOP_TOKENS or len(t) < 3:
                    continue
                pat = re.compile(rf"^\s*def\s+{re.escape(t)}\b", re.IGNORECASE)
                for ln in lines:
                    if pat.search(ln):
                        return {
                            "answer": f"Found Python function definition `def {t}`",
                            "evidence": ln.strip(),
                            "files": f"{repo}/{fp}:{sl}-{el}",
                        }

        # JS/TS: function <name> OR <name>(...) {  OR <name> = (...) =>
        if is_js_ts:
            for t in tokens:
                if t in STOP_TOKENS or len(t) < 3:
                    continue
                pats = [
                    re.compile(rf"^\s*function\s+{re.escape(t)}\b"),
                    re.compile(rf"^\s*{re.escape(t)}\s*\("),
                    re.compile(rf"^\s*{re.escape(t)}\s*=\s*\(?.*\)?\s*=>"),
                ]
                for ln in lines:
                    if any(p.search(ln) for p in pats):
                        return {
                            "answer": f"Found JS/TS function/method definition `{t}`",
                            "evidence": ln.strip(),
                            "files": f"{repo}/{fp}:{sl}-{el}",
                        }

        # --- 2) Otherwise, pick best-scoring evidence line ---
        # Require at least 2 token hits (or 1 very strong token)
        for ln in lines:
            s = score_line_match(ln, tokens)
            if s <= 0:
                continue

            # gate: avoid weak single-hit lines unless token is long
            # If only 1 short token matched => too weak
            # We approximate by requiring score >= 2
            if s < 2:
                continue

            files = f"{repo}/{fp}:{sl}-{el}"
            if best is None or s > best[0]:
                best = (s, ln.rstrip(), files, f"Found evidence line matching query tokens (score={s})")

    if best:
        return {
            "answer": best[3],
            "evidence": best[1],
            "files": best[2],
        }

    return None


def print_sources(res, top_n: int = 5):
    if not res.source_nodes:
        return

    console.print("\n[bold cyan]Sources (top):[/bold cyan]\n")

    for sn in res.source_nodes[:top_n]:
        node = sn.node
        meta = node.metadata or {}
        fp = meta.get("file_path", "unknown")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")
        repo = meta.get("repo_name", "")
        score = getattr(sn, "score", None)

        lang = guess_lang_from_path(fp)
        text = node.text or ""

        header = f"score={fmt_score(score)} | {repo}/{fp}:{sl}-{el}"
        console.print(Panel(Text(header, style="bold white"), style="bright_black"))

        syntax = Syntax(
            text,
            lexer=lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(syntax)
        console.print("\n")


def main():
    top_k = 40
    repo_filter = None

    query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=repo_filter)

    print(f"RAG ready | collection={QDRANT_COLLECTION}")
    print(f"Embedding={EMBED_MODEL} @ {OLLAMA_EMBED_URL}")
    print(f"LLM={LLM_MODEL} @ {OLLAMA_LLM_URL}")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q or q.lower() == "exit":
            break

        detected_repo = detect_repo_from_query(q)
        if detected_repo:
            query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=detected_repo)
            print(f"\nAuto repo filter detected: {detected_repo}\n")

        res = query_engine.query(q)

        # # DEBUG: quickly show retrieval hit count + first nodes
        # try:
        #     hits = len(res.source_nodes) if getattr(res, "source_nodes", None) else 0
        # except Exception:
        #     hits = 0
        # print(f"\n[DEBUG] retrieval hits: {hits}")
        # if hits:
        #     for i, sn in enumerate(res.source_nodes[:5], start=1):
        #         meta = sn.node.metadata or {}
        #         print(f"[DEBUG] node{i}: repo={meta.get('repo_name')}, file={meta.get('file_path')}, "
        #               f"lines={meta.get('start_line','?')}-{meta.get('end_line','?')}, score={getattr(sn,'score',None)}")
        #     print()

        # if multiple repos -> ask user to choose -> requery
        if res.source_nodes and not detected_repo:
            picked_repo = choose_repo_interactively(res.source_nodes)
            if picked_repo:
                print(f"\nRe-querying with repo filter = {picked_repo}\n")
                query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=picked_repo)
                res = query_engine.query(q)

        # Debug: show retrieved files quickly
        if DEBUG_FILES and res.source_nodes:
            print("\n[DEBUG] Retrieved files (top 10):")
            for sn in res.source_nodes[:10]:
                meta = sn.node.metadata or {}
                print("-", meta.get("repo_name", ""), meta.get("file_path", ""),
                      f"{meta.get('start_line','?')}-{meta.get('end_line','?')}")
            print()

        # Debug: show context_str
        if DEBUG_CONTEXT:
            debug_context(res, max_nodes=10, max_chars=8000)

        # Debug: show prompt rendered
        if DEBUG_RENDERED_PROMPT:
            debug_rendered_prompt(q, res, prompt, max_nodes=8, max_chars=12000)

        # HARD GATE: require evidence from snippets
        extracted = None
        if res.source_nodes:
            extracted = extract_evidence(q, res.source_nodes, max_nodes=25)

        if not extracted:
            print("\nAnswer:")
            print("- Answer: Not found in provided context.")
            print("- Evidence: (none)")
            print("- Files: (none)")
        else:
            print("\nAnswer:")
            print(f"- Answer: {extracted['answer']}")
            print(f'- Evidence: "{extracted["evidence"]}"')
            print(f"- Files: {extracted['files']}")

        # Print sources
        print_sources(res, top_n=5)

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
