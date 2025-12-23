from typing import List, Dict
from config import MAX_LINES_PER_CHUNK, MIN_LINES_PER_CHUNK

# ============================================================
# FUNCTION-BASED CHUNKERS
# Each returns a list of spans: {"start": int, "end": int} (0-based indices)
# ============================================================

# ---------- Python ----------
PY_PREFIXES = ("def ", "class ", "async def ")

# Start a new chunk at lines beginning with 'def ', 'class ', or 'async def '
def split_by_functions_py(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(PY_PREFIXES):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- JavaScript / TypeScript ----------
JS_TS_PREFIXES = (
    "function ",
    "async function ",
    "export function ",
    "export async function ",
    "class ",
)

# Simple heuristic for arrow functions
def is_arrow_function_line(line: str) -> bool:
    return "=" in line and "=>" in line and "(" in line and ")" in line

# Start a chunk at lines that look like functions/classes or arrow functions
def split_by_functions_js_ts(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(p) for p in JS_TS_PREFIXES) or is_arrow_function_line(line):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- Java / C# / Kotlin / Swift ----------
OOP_CLASS_PREFIXES = (
    "public class ",
    "class ",
    "public interface ",
    "interface ",
    "public enum ",
    "enum ",
    "public record ",
    "record ",
)

OOP_METHOD_PREFIXES = (
    "public ",
    "private ",
    "protected ",
    "internal ",
    "static ",
    "final ",
    "abstract ",
    "override ",
)

# Rough check for method declarations in Java-like languages by looking for a visibility/modifier at the start, plus '(...)' and ending with '{'
def is_java_like_method_line(line: str) -> bool:
    stripped = line.strip()
    if not any(stripped.startswith(p) for p in OOP_METHOD_PREFIXES):
        return False
    if "(" in stripped and ")" in stripped and stripped.endswith("{"):
        return True
    return False

# Start a chunk at class declarations or typical method signatures
def split_by_functions_java_like(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(p) for p in OOP_CLASS_PREFIXES) or is_java_like_method_line(line):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- Go ----------
def split_by_functions_go(lines: List[str]) -> List[Dict[str, int]]:
    # Start a chunk at 'func ' or 'type '
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("func ") or stripped.startswith("type "):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- PHP ----------
def split_by_functions_php(lines: List[str]) -> List[Dict[str, int]]:
    # Start a chunk at 'function ' or 'class '
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("function ") or stripped.startswith("class "):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- Ruby ----------
def split_by_functions_ruby(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- Rust ----------
def split_by_functions_rust(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith("fn ")
            or stripped.startswith("impl ")
            or stripped.startswith("struct ")
            or stripped.startswith("enum ")
        ):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ---------- Shell scripts (.sh, .bat) ----------
def is_shell_function_line(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("function "):
        return True
    # Simple check for `name() {`
    if "()" in stripped and stripped.endswith("{"):
        return True
    return False

def split_by_functions_shell(lines: List[str]) -> List[Dict[str, int]]:
    markers = []
    for i, line in enumerate(lines):
        if is_shell_function_line(line):
            markers.append(i)

    if not markers:
        return []

    spans = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(lines)
        spans.append({"start": start, "end": end})
    return spans


# ============================================================
# FALLBACK: LINE-BASED CHUNKING
# ============================================================

def fallback_line_chunks(lines: List[str]) -> List[Dict[str, int]]:
    spans = []
    n = len(lines)
    start = 0
    while start < n:
        end = min(start + MAX_LINES_PER_CHUNK, n)
        if end - start < MIN_LINES_PER_CHUNK and start != 0:
            break
        spans.append({"start": start, "end": end})
        start = end
    return spans


# ============================================================
# CHOOSE SPLITTER BASED ON EXT
# ============================================================

def get_function_spans_for_ext(ext: str, lines: List[str]) -> List[Dict[str, int]]:
    if ext == ".py":
        return split_by_functions_py(lines)
    if ext in {".js", ".jsx", ".ts", ".tsx"}:
        return split_by_functions_js_ts(lines)
    if ext in {".java", ".cs", ".kt", ".swift"}:
        return split_by_functions_java_like(lines)
    if ext == ".go":
        return split_by_functions_go(lines)
    if ext == ".php":
        return split_by_functions_php(lines)
    if ext == ".rb":
        return split_by_functions_ruby(lines)
    if ext == ".rs":
        return split_by_functions_rust(lines)
    if ext in {".sh", ".bat"}:
        return split_by_functions_shell(lines)

    return []  # no function-based splitter for this extension


# ============================================================
# PULL COMMENTS UP INTO THE FUNCTION CHUNK
# ============================================================

#Comments/blank lines immediately above a function/class become part of that chunk
def pull_comments_up(lines: List[str], spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
    def is_comment_or_blank(line: str) -> bool:
        stripped = line.strip()
        return (
            stripped.startswith("#") or
            stripped.startswith("//") or
            stripped.startswith("/*") or
            stripped == ""
        )

    new_spans = []
    for span in spans:
        start = span["start"]

        # Move start upwards to include any comment/blank lines
        while start > 0 and is_comment_or_blank(lines[start - 1]):
            start -= 1

        new_spans.append({"start": start, "end": span["end"]})

    return new_spans
