from tree_sitter_languages import get_parser

CANDIDATES = [
    "python",
    "javascript",
    "typescript",
    "tsx",
    "java",
    "c_sharp",
    "go",
    "php",
    "ruby",
    "rust",
    "bash",
    "kotlin",
    "swift",
    "c",
    "cpp",
    "c_plus_plus",
    "vue",
]

for lang in CANDIDATES:
    try:
        get_parser(lang)
        print(f"[OK] {lang}")
    except Exception as e:
        print(f"[NO] {lang} -> {e}")