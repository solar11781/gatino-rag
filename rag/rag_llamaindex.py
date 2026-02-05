from pathlib import Path
import sys
import re
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple

# --- Add project root to PYTHONPATH ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_MODEL, CODE_EXTS,OLLAMA_EMBED_URL, OLLAMA_LLM_URL, LLM_MODEL

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

# Retrieval strictness
MIN_SIMILARITY_SCORE = 0.25  # if top score below this => "Not found"


# -----------------------------
# Helpers
# -----------------------------
def detect_repo_from_query(q: str) -> Optional[str]:
    """
    Match explicit repo syntax:
    - repo:RepoName
    - repo=RepoName
    - in repo RepoName
    """
    if not q:
        return None

    patterns = [
        r"\brepo\s*[:=]\s*([a-zA-Z0-9\-_]+)",
        r"\bin\s+repo\s+([a-zA-Z0-9\-_]+)",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip(",.;")
    return None


LANG_ALIASES = {
    "python": "py",
    "py": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "tsx": "tsx",
    "jsx": "jsx",
    "java": "java",
    "kotlin": "kt",
    "kt": "kt",
    "c#": "cs",
    "csharp": "cs",
    "c_sharp": "cs",
    "cs": "cs",
    "go": "go",
    "golang": "go",
    "php": "php",
    "ruby": "rb",
    "rb": "rb",
    "rust": "rs",
    "rs": "rs",
    "bash": "sh",
    "shell": "sh",
    "sh": "sh",
    "bat": "bat",
    "yaml": "yaml",
    "yml": "yml",
    "json": "json",
}

def detect_language_from_query(q: str) -> Optional[str]:
    if not q:
        return None

    ql = q.lower()
    # prefer explicit "written in <lang>" / "in <lang>"
    m = re.search(r"\b(?:written in|in|using)\s+([a-zA-Z\#_]+)\b", ql)
    if m:
        lang = m.group(1).strip()
        return LANG_ALIASES.get(lang)

    # fallback to direct mentions
    for k, v in LANG_ALIASES.items():
        if re.search(rf"\b{re.escape(k)}\b", ql):
            return v

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


def build_query_engine(
    top_k: int = 40,
    repo_filter: Optional[str] = None,
    lang_filter: Optional[str] = None,
) -> Tuple[Any, PromptTemplate]:
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
    filter_list = []
    if repo_filter:
        filter_list.append(MetadataFilter(key="repo_name", value=repo_filter))
    if lang_filter:
        filter_list.append(MetadataFilter(key="language", value=lang_filter))
    if filter_list:
        filters = MetadataFilters(filters=filter_list)

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
    "valid", "move", "check", "logic", "giải", "giai", "thích", "thich", "written"
}

GENERIC_QUERY_TOKENS = {
    "auth", "authentication", "authorize", "authorization",
    "token", "jwt", "bearer",
    "database", "db", "sql", "orm",
    "api", "route", "routes", "router", "endpoint",
    "middleware", "config", "configuration", "module",
    # languages (avoid treating as specific identifiers)
    "python", "py", "javascript", "js", "typescript", "ts", "tsx", "jsx",
    "java", "kotlin", "kt", "csharp", "c_sharp", "cs", "go", "golang",
    "php", "ruby", "rb", "rust", "rs", "bash", "shell", "sh",
}

INTENT_KEYWORDS = {
    "auth": ["auth", "authentication", "authenticated", "authorize", "authorization", "security", "login", "signin", "permission"],
    "token": ["token", "jwt", "bearer", "session"],
    "db": ["db", "database", "datasource", "connection", "sequelize", "mongoose", "sql", "orm", "repository"],
    "routes": ["route", "routes", "router", "controller", "endpoint", "api"],
    "middleware": ["middleware", "middlewares", "interceptor", "filter"],
}

INTENT_PATH_HINTS = {
    "auth": ["auth", "authentication", "security", "jwt", "token", "login", "account", "accounts", "user", "users"],
    "token": ["token", "jwt", "session"],
    "db": ["repository", "dao", "db", "database", "datasource", "entity", "model", "migration", "schema"],
    "routes": ["route", "routes", "router", "controller", "endpoint", "api"],
    "middleware": ["middleware", "middlewares", "interceptor", "filter"],
}

BAD_PATH_SEGMENTS = {
    "test", "tests", "__tests__", "spec", "mock", "fixture", "sample",
    "openapi", "swagger", "kubernetes", "k8s", "docs", "doc",
}

def score_line_match(line: str, tokens: List[str]) -> int:
    """
    Score a line by how many important tokens appear.
    Higher = better evidence.
    """
    l = line.lower()
    l_clean = re.sub(r"[^a-z0-9]", "", l)
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
        if t_clean in l or (t_clean and t_clean in l_clean):
            score += 2 if len(t_clean) >= 6 else 1
            continue

        # partial-root match: allow long-token prefixes to match morphological variants
        # e.g., "authentication" -> matches "authenticated" or "is_authenticated"
        if len(t_clean) >= 6:
            prefix = t_clean[:6]
            if prefix in l or (prefix and prefix in l_clean):
                score += 2
                continue


    return score


def extract_signature_tokens(q: str, repo_filter: Optional[str] = None) -> List[str]:
    """
    Pull likely identifiers: function names, class names, file names, snake_case, etc.
    """
    ql = (q or "").lower()
    if repo_filter:
        # remove repo name tokens from query to reduce false matches
        repo_tokens = re.split(r"[^a-zA-Z0-9_]+", repo_filter.lower())
        for t in repo_tokens:
            if t:
                ql = ql.replace(t, " ")
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

def _has_specific_tokens(q: str) -> bool:
    toks = normalize_query_tokens(q or "")
    for t in toks:
        if t in STOP_TOKENS or t in GENERIC_QUERY_TOKENS:
            continue
        if len(t) >= 4:
            return True
    return False

def _detect_intent(q: str) -> Optional[str]:
    ql = (q or "").lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in ql for k in kws):
            return intent
    return None

def _is_code_file_path(fp: str) -> bool:
    if not fp or "." not in fp:
        return False
    ext = "." + fp.lower().rsplit(".", 1)[-1]
    return ext in CODE_EXTS


def _path_has_bad_segment(fp: str) -> bool:
    parts = [p.lower() for p in (fp or "").replace("\\", "/").split("/")]
    for p in parts:
        if not p:
            continue
        # exact match or substring (e.g., "employeemanagementtests")
        if p in BAD_PATH_SEGMENTS:
            return True
        if any(bad in p for bad in BAD_PATH_SEGMENTS):
            return True
    return False

def _filter_by_intent(
    source_nodes,
    intent: Optional[str],
    prefer_code: bool = True,
    allow_fallback: bool = True,
    generic_query: bool = False,
):
    if not source_nodes or not intent:
        return source_nodes

    kws = INTENT_KEYWORDS.get(intent, [])
    path_hints = INTENT_PATH_HINTS.get(intent, [])
    if not kws:
        return source_nodes

    filtered = []
    for sn in source_nodes:
        meta = sn.node.metadata or {}
        fp = meta.get("file_path", "")
        fp_l = fp.lower()
        text_l = (sn.node.text or "").lower()

        if _path_has_bad_segment(fp_l):
            continue
        if prefer_code and not _is_code_file_path(fp_l):
            continue

        # Global intent-specific exclusions
        if intent == "auth":
            if "/client/" in fp_l or fp_l.endswith(".d.ts"):
                continue
        if intent == "middleware":
            if fp_l.endswith("settings.py"):
                continue

        if generic_query and path_hints:
            # avoid config-heavy settings files for auth/token queries
            if intent in {"auth", "token", "middleware"} and fp_l.endswith("settings.py"):
                continue
            if intent == "auth":
                # Avoid client model typings for auth; prefer server code/config
                if "/client/" in fp_l or fp_l.endswith(".d.ts"):
                    continue
                if "/server/" not in fp_l and "/backend/" not in fp_l:
                    continue

            if any(k in fp_l for k in path_hints) or ("views.py" in fp_l) or ("serializers.py" in fp_l):
                filtered.append(sn)
            continue

        if any(k in fp_l for k in kws) or any(k in text_l for k in kws):
            filtered.append(sn)

    if filtered:
        return filtered
    return source_nodes if allow_fallback else []


def extract_evidence(
    question: str,
    source_nodes,
    repo_filter: Optional[str] = None,
    intent: Optional[str] = None,
    max_nodes: int = 25,
) -> Optional[Dict[str, Any]]:
    """
    Upgraded hard extraction:
    - Prefer actual definitions: Ruby `def`, Python `def`, JS/TS `function` / method patterns.
    - Require stronger match: line must match >=2 meaningful tokens OR match a definition pattern.
    - Reduce false positives from UI/icon files.
    """
    q = question or ""
    ql = q.lower()
    tokens = extract_signature_tokens(q, repo_filter=repo_filter)
    intent_keywords = INTENT_KEYWORDS.get(intent or "", [])
    is_generic = not _has_specific_tokens(q) and bool(intent_keywords)

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
        is_java_like = fp_l.endswith(".java") or fp_l.endswith(".kt") or fp_l.endswith(".cs")

        # For definition queries, skip non-code files to avoid config/doc hits
        if _is_definition_query(q):
            ext = "." + fp_l.split(".")[-1] if "." in fp_l else ""
            if ext and ext not in CODE_EXTS:
                continue

        # Avoid config-heavy settings files for auth/token/middleware intent
        if intent in {"auth", "token", "middleware"} and fp_l.endswith("settings.py"):
            continue

        # Skip obvious doc links for middleware intent
        if intent == "middleware":
            if any("docs.scrapy.org" in ln.lower() for ln in lines):
                # still allow if file is actually middlewares.py
                if "middlewares.py" not in fp_l:
                    continue

        if intent == "auth":
            if "/client/" in fp_l or fp_l.endswith(".d.ts"):
                continue

        # Generic intent: try direct keyword evidence even if tokens don't match
        if is_generic and intent_keywords:
            for ln in lines:
                ln_l = ln.lower()
                if any(k in ln_l for k in intent_keywords):
                    return {
                        "answer": f"Found {intent} evidence line",
                        "evidence": ln.strip(),
                        "files": f"{repo}/{fp}:{sl}-{el}",
                    }

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

        # Java/Kotlin/C#: method definition
        if is_java_like:
            for t in tokens:
                if t in STOP_TOKENS or len(t) < 3:
                    continue
                pat = re.compile(
                    rf"^\s*(public|private|protected|static|final|abstract|synchronized|\s)*\s*"
                    rf"[A-Za-z0-9_<>\[\]]+\s+{re.escape(t)}\s*\(",
                    re.IGNORECASE,
                )
                for ln in lines:
                    if pat.search(ln):
                        return {
                            "answer": f"Found Java/Kotlin/C# method definition `{t}`",
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

            if is_generic:
                ln_l = ln.lower()
                if not any(k in ln_l for k in intent_keywords) and not any(k in fp_l for k in intent_keywords):
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


def _parse_strict_response(text: str) -> Optional[Dict[str, str]]:
    """
    Parse the STRICT prompt response format:
    - Answer: ...
    - Evidence: "..."
    - Files: repo/path:start-end
    """
    if not text:
        return None

    answer = None
    evidence = None
    files = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("- answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("- evidence:"):
            evidence = line.split(":", 1)[1].strip()
            # strip surrounding quotes if present
            if evidence.startswith('"') and evidence.endswith('"') and len(evidence) >= 2:
                evidence = evidence[1:-1].strip()
        elif line.lower().startswith("- files:"):
            files = line.split(":", 1)[1].strip()

    if not answer:
        return None

    return {"answer": answer, "evidence": evidence or "", "files": files or ""}


def _evidence_in_sources(evidence: str, source_nodes) -> bool:
    if not evidence:
        return False
    for sn in source_nodes or []:
        if evidence in (sn.node.text or ""):
            return True
    return False


def _parse_file_spec(files: str) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
    if not files:
        return None
    if ":" not in files:
        return (files.strip(), None, None)
    path_part, line_part = files.rsplit(":", 1)
    path_part = path_part.strip()
    line_part = line_part.strip()
    if "-" in line_part:
        s, e = line_part.split("-", 1)
        try:
            return (path_part, int(s), int(e))
        except Exception:
            return (path_part, None, None)
    return (path_part, None, None)


def _evidence_matches_file(evidence: str, files: str, source_nodes) -> bool:
    if not evidence or not files:
        return False
    parsed = _parse_file_spec(files)
    if not parsed:
        return False
    file_path, _, _ = parsed
    for sn in source_nodes or []:
        meta = sn.node.metadata or {}
        repo = meta.get("repo_name", "")
        fp = meta.get("file_path", "")
        full = f"{repo}/{fp}".strip("/")
        if file_path.strip("/") == full and evidence in (sn.node.text or ""):
            return True
    return False


def _passes_score_cutoff(source_nodes, min_score: float) -> bool:
    if not source_nodes:
        return False
    scores = [getattr(sn, "score", None) for sn in source_nodes]
    scores = [s for s in scores if s is not None]
    if not scores:
        return True  # no scores provided; don't block
    return max(scores) >= min_score


def _filter_source_nodes(source_nodes, repo_filter: Optional[str], lang_filter: Optional[str]):
    if not source_nodes:
        return []
    out = []
    for sn in source_nodes:
        meta = sn.node.metadata or {}
        if repo_filter and meta.get("repo_name") != repo_filter:
            continue
        if lang_filter and meta.get("language") != lang_filter:
            continue
        out.append(sn)
    return out


def _is_definition_query(q: str) -> bool:
    if not q:
        return False
    ql = q.lower()
    return any(k in ql for k in ["defined", "definition", "define", "implement", "implemented", "implementation"])


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
    lang_filter = None

    query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=repo_filter, lang_filter=lang_filter)

    print(f"RAG ready | collection={QDRANT_COLLECTION}")
    print(f"Embedding={EMBED_MODEL} @ {OLLAMA_EMBED_URL}")
    print(f"LLM={LLM_MODEL} @ {OLLAMA_LLM_URL}")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q or q.lower() == "exit":
            break

        detected_repo = detect_repo_from_query(q)
        detected_lang = detect_language_from_query(q)
        repo_filter = detected_repo
        lang_filter = detected_lang

        if repo_filter or lang_filter:
            query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=repo_filter, lang_filter=lang_filter)
            if repo_filter:
                print(f"\nAuto repo filter detected: {repo_filter}\n")
            if lang_filter:
                print(f"Auto language filter detected: {lang_filter}\n")
        else:
            query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=None, lang_filter=None)

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
                repo_filter = picked_repo
                query_engine, prompt = build_query_engine(top_k=top_k, repo_filter=repo_filter, lang_filter=lang_filter)
                res = query_engine.query(q)
                detected_repo = picked_repo

        # Apply local filtering for safety (even if filters were used in retrieval)
        if res.source_nodes:
            res.source_nodes = _filter_source_nodes(res.source_nodes, repo_filter, lang_filter)

        intent = _detect_intent(q)
        if res.source_nodes and intent:
            allow_fallback = _has_specific_tokens(q) or intent in {"auth", "token"}
            res.source_nodes = _filter_by_intent(
                res.source_nodes,
                intent,
                prefer_code=True,
                allow_fallback=allow_fallback,
                generic_query=not _has_specific_tokens(q),
            )

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

        # If top scores are too low, return Not found
        if not _passes_score_cutoff(res.source_nodes, MIN_SIMILARITY_SCORE):
            res.source_nodes = []

        # If query asks for definition, prefer definition-based extraction first
        extracted = None
        if _is_definition_query(q) and res.source_nodes:
            extracted = extract_evidence(q, res.source_nodes, repo_filter=repo_filter, intent=intent, max_nodes=25)

        # Prefer STRICT LLM output, then verify evidence is actually in sources and files match.
        if not extracted:
            llm_text = getattr(res, "response", None) or str(res) or ""
            parsed = _parse_strict_response(llm_text)
            if parsed and parsed.get("answer") and parsed.get("evidence"):
                if parsed["answer"].lower().startswith("not found"):
                    extracted = None
                elif _evidence_in_sources(parsed["evidence"], res.source_nodes) and _evidence_matches_file(
                    parsed["evidence"], parsed.get("files", ""), res.source_nodes
                ):
                    extracted = parsed

        # Optional fallback: heuristic evidence from snippets (only if LLM failed)
        if not extracted and res.source_nodes and (intent or _has_specific_tokens(q)):
            extracted = extract_evidence(q, res.source_nodes, repo_filter=repo_filter, intent=intent, max_nodes=25)

        if not extracted:
            print("\nAnswer:")
            print("- Answer: Not found in provided context.")
            print("- Evidence: (none)")
            print("- Files: (none)")
            res.source_nodes = []
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
