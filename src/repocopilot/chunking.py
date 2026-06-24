from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import ast


@dataclass
class Chunk:
    kind: str  # "class" | "function" | "module" | "text"
    name: str  # es: "MyClass" / "foo" / "module"
    text: str


def chunk_text_charwise(text: str, maxc: int, overlap: int) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + maxc)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        i = max(0, end - overlap)
    return chunks


def _collect_import_block(tree: ast.AST, src: str) -> str:
    # include import statements at top for context (best effort)
    imports = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node)
            if seg:
                imports.append(seg)
        else:
            # stop when first non-import appears (heuristic)
            break
    return ("\n".join(imports).strip() + "\n\n") if imports else ""


def chunk_python_ast(src: str) -> List[Chunk]:
    try:
        tree = ast.parse(src)
    except Exception:
        # fallback
        return [Chunk(kind="text", name="text", text=src)]

    import_block = _collect_import_block(tree, src)

    out: List[Chunk] = []
    body = getattr(tree, "body", [])

    # top-level defs/classes = semantic chunks
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, node)
            if seg:
                out.append(
                    Chunk(kind="function", name=node.name, text=import_block + seg)
                )
        elif isinstance(node, ast.ClassDef):
            seg = ast.get_source_segment(src, node)
            if seg:
                out.append(Chunk(kind="class", name=node.name, text=import_block + seg))

    # fallback: if no semantic nodes found, return module as one chunk
    if not out:
        out.append(Chunk(kind="module", name="module", text=src))

    return out


def chunk_file(path: Path, text: str, maxc: int, overlap: int) -> List[str]:
    if path.suffix.lower() == ".py":
        chunks = chunk_python_ast(text)
        # convert to plain strings
        return [c.text for c in chunks if c.text.strip()]
    # fallback universale: charwise
    return chunk_text_charwise(text, maxc=maxc, overlap=overlap)
