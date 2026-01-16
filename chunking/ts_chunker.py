from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from tree_sitter_languages import get_parser


EXT_TO_TS_LANG: Dict[str, str] = {
    # Python
    ".py": "python",

    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",

    # JVM / .NET
    ".java": "java",
    ".cs": "c_sharp",
    ".kt": "kotlin",

    # Backend / Systems
    ".go": "go",
    ".php": "php",
    ".rb": "ruby",
    ".rs": "rust",

    # Shell
    ".sh": "bash",
    ".bat": "bash",

    # C / C++
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}


# If a language file parses but collect zero nodes, fallback to rule-based splitting
TS_NODE_TYPES: Dict[str, Set[str]] = {
    "python": {
        "function_definition",
        "async_function_definition",
        "class_definition",
    },

    "javascript": {
        "function_declaration",
        "method_definition",
        "generator_function_declaration",
        "arrow_function",
        "function_expression",
	    "class_declaration",
    },

    "typescript": {
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
	    "class_declaration",
    },

    "tsx": {
        "function_declaration",
        "method_definition",
        "arrow_function",
        "function_expression",
        "class_declaration",
    },

    "java": {
        "method_declaration",
        "constructor_declaration",
    },

    "c_sharp": {
        "method_declaration",
        "constructor_declaration",
    },

    "go": {
        "function_declaration",
        "method_declaration",
    },

    "php": {
        "function_definition",
        "method_declaration",
    },

    "ruby": {
        "method",
    },

    "rust": {
        "function_item",
    },

    "bash": {
        "function_definition",
    },

    "kotlin": {
        "function_declaration",
    },

    "c": {
        "function_definition",
    },

    "cpp": {
        "function_definition",
    },
}


@dataclass
class TSChunkResult:
    spans: List[Dict[str, int]]
    error: Optional[str] = None


def _walk_collect_nodes(root, wanted_types: Set[str]):
    stack = [root]
    out = []
    while stack:
        node = stack.pop()
        if node.type in wanted_types:
            out.append(node)
        for child in reversed(node.children):
            stack.append(child)
    return out

def _dedupe_and_reduce_spans(spans: List[Dict[str, int]]) -> List[Dict[str, int]]:
    if not spans:
        return []

    spans_sorted = sorted(spans, key=lambda s: (s["start"], -s["end"]))

    # Remove exact duplicates
    deduped: List[Dict[str, int]] = []
    seen = set()
    for sp in spans_sorted:
        key = (sp["start"], sp["end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sp)

    # Drop fully-contained spans (keep outermost)
    reduced: List[Dict[str, int]] = []
    for sp in deduped:
        if not reduced:
            reduced.append(sp)
            continue
        last = reduced[-1]
        # If sp is fully inside last, skip it
        if sp["start"] >= last["start"] and sp["end"] <= last["end"]:
            continue
        reduced.append(sp)

    return reduced


def tree_sitter_spans_for_text(ext: str, text: str) -> TSChunkResult:
    ts_lang = EXT_TO_TS_LANG.get(ext)
    if not ts_lang:
        return TSChunkResult(spans=[], error=f"No Tree-sitter language mapping for ext={ext}")

    wanted = TS_NODE_TYPES.get(ts_lang)
    if not wanted:
        return TSChunkResult(spans=[], error=f"No node type config for Tree-sitter language '{ts_lang}'")

    # Get parser from tree_sitter_languages (prebuilt bundle)
    try:
        parser = get_parser(ts_lang)
    except Exception as e:
        return TSChunkResult(spans=[], error=f"Failed to get parser for '{ts_lang}': {e}")

    # Tree-sitter parses bytes
    src_bytes = text.encode("utf-8", errors="ignore")

    try:
        tree = parser.parse(src_bytes)
    except Exception as e:
        return TSChunkResult(spans=[], error=f"Tree-sitter parse failed for '{ts_lang}': {e}")

    root = tree.root_node

    nodes = _walk_collect_nodes(root, wanted)

    # Convert node ranges -> line spans
    # node.start_point and node.end_point are tuples (row, col), where row is 0-based.
    # End-exclusive line indexing => end_row + 1.
    spans: List[Dict[str, int]] = []
    for n in nodes:
        # Lift JS/TS arrow/function expressions to their declaration
        if ts_lang in {"javascript", "typescript", "tsx"} and n.type in {"arrow_function", "function_expression"}:
            p = n.parent
            while p is not None and p.type not in {
                "variable_declarator",
                "lexical_declaration",
                "variable_declaration",
                "export_statement",
                "expression_statement",
            }:
                p = p.parent
            if p is not None:
                n = p  # lift span to include the declaration line
        start_row = int(n.start_point[0])
        end_row = int(n.end_point[0]) + 1
        
        # Guard against weird empty nodes
        if end_row <= start_row:
            continue
        spans.append({"start": start_row, "end": end_row})

    spans = _dedupe_and_reduce_spans(spans)

    return TSChunkResult(spans=spans, error=None)


# ============================================================
# STRUCTURAL SUBCHUNKING (inside a function/method)
# Split oversized functions by statement (if/for/while/try/switch blocks)
# If not, fallback to line-based split
# ============================================================

TS_STATEMENT_TYPES = {
    "python": {
        "if_statement", "for_statement", "while_statement", "try_statement",
        "with_statement", "match_statement",
    },
    "javascript": {
        "if_statement", "for_statement", "for_in_statement", "for_of_statement",
        "while_statement", "do_statement", "switch_statement", "try_statement",
        "block",
    },
    "typescript": {
        "if_statement", "for_statement", "for_in_statement", "for_of_statement",
        "while_statement", "do_statement", "switch_statement", "try_statement",
        "block",
    },
    "tsx": {
        "if_statement", "for_statement", "for_in_statement", "for_of_statement",
        "while_statement", "do_statement", "switch_statement", "try_statement",
        "block",
    },
    "java": {
        "if_statement", "for_statement", "while_statement", "do_statement", "switch_expression", "switch_statement", "try_statement", "block"
    },
    "c_sharp": {
        "if_statement", "for_statement", "foreach_statement", "while_statement", "do_statement", "switch_statement", "try_statement", "block"
    },
    "go": {
        "if_statement", "for_statement", "switch_statement", "type_switch_statement", "block"
    },
    "php": {
        "if_statement", "for_statement", "foreach_statement", "while_statement", "do_statement", "switch_statement", "try_statement", "compound_statement"
    },
    "ruby": {
        "if", "while", "for", "case", "begin"
    },
    "rust": {
        "if_expression", "for_expression", "while_expression", "match_expression", "block"
    },
    "kotlin": {
        "if_expression", "for_statement", "while_statement", "when_expression", "block"
    },
    "c": {
        "if_statement", "for_statement", "while_statement", "switch_statement", "compound_statement"
    },
    "cpp": {
        "if_statement", "for_statement", "while_statement", "switch_statement", "compound_statement"
    },
}

# Split a big function-span [start_line:end_line] into smaller subspans based on statement/block boundaries
def ts_structural_subspans_for_span(ext: str, text: str, start_line: int, end_line: int) -> TSChunkResult:
    ts_lang = EXT_TO_TS_LANG.get(ext)
    if not ts_lang:
        return TSChunkResult(spans=[], error=f"No Tree-sitter language mapping for ext={ext}")

    wanted_stmt = TS_STATEMENT_TYPES.get(ts_lang)
    if not wanted_stmt:
        return TSChunkResult(spans=[], error=f"No statement type config for Tree-sitter language '{ts_lang}'")

    try:
        parser = get_parser(ts_lang)
    except Exception as e:
        return TSChunkResult(spans=[], error=f"Failed to get parser for '{ts_lang}': {e}")

    src_bytes = text.encode("utf-8", errors="ignore")
    try:
        tree = parser.parse(src_bytes)
    except Exception as e:
        return TSChunkResult(spans=[], error=f"Tree-sitter parse failed for '{ts_lang}': {e}")

    root = tree.root_node

    # Collect candidate statement nodes
    nodes = _walk_collect_nodes(root, wanted_stmt)

    # Convert to spans BUT only those fully inside the parent function span
    spans: List[Dict[str, int]] = []
    for n in nodes:
        s = int(n.start_point[0])
        e = int(n.end_point[0]) + 1
        if e <= s:
            continue
        # Keep only nodes that lie within the parent span
        if s >= start_line and e <= end_line:
            spans.append({"start": s, "end": e})

    spans = _dedupe_and_reduce_spans(spans)

    # If too few boundaries, not useful
    if len(spans) < 2:
        return TSChunkResult(spans=[], error="Not enough structural boundaries inside span")

    return TSChunkResult(spans=spans, error=None)
