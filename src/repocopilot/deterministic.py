from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .schemas import AnswerWithCitations, Source
from .indexer import iter_files


def _repo_root() -> Path:
    return Path(".").resolve()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _make_source(ref: str, path: Path, chunk_id: str, excerpt: str) -> Source:
    return Source(ref=ref, chunk_id=chunk_id, path=str(path), excerpt=excerpt[:2000])


def _parse_project_scripts(pyproject_text: str) -> Dict[str, str]:
    """
    Parsing semplice della sezione [project.scripts] in pyproject.toml.
    """
    lines = pyproject_text.splitlines()
    in_section = False
    scripts: Dict[str, str] = {}

    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        if raw.startswith("[") and raw.endswith("]"):
            in_section = raw == "[project.scripts]"
            continue

        if in_section and "=" in raw:
            k, v = raw.split("=", 1)
            k = k.strip()
            v = v.split("#", 1)[0].strip().strip('"').strip("'")
            scripts[k] = v

    return scripts


def _needs_entrypoint_context(q: str) -> bool:
    ql = q.lower()
    hints = (
        "entrypoint",
        "entry point",
        "console script",
        "project.scripts",
        "entry_points",
        "console_scripts",
    )
    return any(h in ql for h in hints)


def try_deterministic_entrypoint(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    if not _needs_entrypoint_context(question):
        return None

    pyproject = _repo_root() / "pyproject.toml"
    if not pyproject.exists():
        return None

    text = _read_text(pyproject)
    scripts = _parse_project_scripts(text)
    if not scripts:
        ans = AnswerWithCitations(
            answer_md="Non ho trovato la sezione `[project.scripts]` in `pyproject.toml`.",
            citations=[],
            confidence="bassa",
            open_questions=[
                "Il progetto usa setuptools entry_points (setup.py/setup.cfg) invece di [project.scripts]?"
            ],
        )
        src = _make_source("S1", pyproject, "pyproject.toml::deterministic", text)
        return ans, [src]

    script_name = (
        "repocopilot" if "repocopilot" in scripts else sorted(scripts.keys())[0]
    )
    entry = scripts[script_name]
    quote = f'{script_name} = "{entry}"'

    ans = AnswerWithCitations(
        answer_md=(
            "L'entrypoint (console script) della CLI è definito in `pyproject.toml` nella sezione `[project.scripts]`:\n\n"
            f"- `{quote}`"
        ),
        citations=[{"ref": "S1", "quote": quote}],
        confidence="alta",
        open_questions=[],
    )

    # excerpt mirato sulla sezione [project.scripts]
    m = re.search(r"(?ms)^\[project\.scripts\]\s*$.*?(?=^\[|\Z)", text)
    excerpt = m.group(0)[:2000] if m else text[:2000]
    src = _make_source("S1", pyproject, "pyproject.toml::deterministic", excerpt)
    return ans, [src]


def try_deterministic_default_models(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if not (
        ("modelli" in q or "models" in q)
        and ("default" in q or "di default" in q or "predefiniti" in q)
    ):
        return None

    cfg = _repo_root() / "src" / "repocopilot" / "config.py"
    if not cfg.exists():
        return None

    text = _read_text(cfg)
    m_chat = re.search(
        r'chat_model:\s*str\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', text
    )
    m_emb = re.search(
        r'embed_model:\s*str\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', text
    )
    if not (m_chat and m_emb):
        return None

    chat_model = m_chat.group(1)
    embed_model = m_emb.group(1)

    chat_line = next(
        (
            ln.strip()
            for ln in text.splitlines()
            if "chat_model" in ln and "os.getenv" in ln
        ),
        "",
    )
    emb_line = next(
        (
            ln.strip()
            for ln in text.splitlines()
            if "embed_model" in ln and "os.getenv" in ln
        ),
        "",
    )

    ans = AnswerWithCitations(
        answer_md=(
            "I modelli di default sono definiti in `src/repocopilot/config.py`:\n\n"
            f"- Chat: `{chat_model}`\n"
            f"- Embeddings: `{embed_model}`"
        ),
        citations=[
            {"ref": "S1", "quote": chat_line[:200]},
            {"ref": "S1", "quote": emb_line[:200]},
        ],
        confidence="alta",
        open_questions=[],
    )
    src = _make_source("S1", cfg, "config.py::deterministic", text)
    return ans, [src]


def try_deterministic_cli_commands(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if not (
        ("comandi" in q or "commands" in q)
        and ("cli" in q or "command" in q or "implement" in q or "implementati" in q)
    ):
        return None

    cli = _repo_root() / "src" / "repocopilot" / "cli.py"
    if not cli.exists():
        return None

    text = _read_text(cli)
    cmds = re.findall(
        r"@app\.command\(\)\s*\r?\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text
    )
    if not cmds:
        ans = AnswerWithCitations(
            answer_md="Non ho trovato comandi Typer del tipo `@app.command()` in `src/repocopilot/cli.py`.",
            citations=[],
            confidence="bassa",
            open_questions=["I comandi sono definiti in un file diverso da cli.py?"],
        )
        src = _make_source("S1", cli, "cli_commands::deterministic", text)
        return ans, [src]

    seen = set()
    cmds_unique = []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            cmds_unique.append(c)

    excerpt_lines = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "@app.command" in ln:
            excerpt_lines.append(ln.strip())
            if i + 1 < len(lines):
                excerpt_lines.append(lines[i + 1].strip())
    excerpt = "\n".join(excerpt_lines[:120]) or text[:2000]

    ans = AnswerWithCitations(
        answer_md="Comandi CLI implementati (decorator `@app.command()`):\n\n"
        + "\n".join(f"- `{c}`" for c in cmds_unique),
        citations=[{"ref": "S1", "quote": "@app.command()"}],
        confidence="alta",
        open_questions=[],
    )
    src = _make_source("S1", cli, "cli_commands::deterministic", excerpt)
    return ans, [src]


def try_deterministic_doctor(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if "doctor" not in q:
        return None

    cli = _repo_root() / "src" / "repocopilot" / "cli.py"
    if not cli.exists():
        return None

    text = _read_text(cli)
    m = re.search(
        r"def\s+doctor\s*\(.*?\):(?P<body>.*?)(?=\n@app\.command|\nif __name__|\Z)",
        text,
        flags=re.S,
    )
    excerpt = m.group(0)[:2000] if m else text[:2000]

    ans = AnswerWithCitations(
        answer_md=(
            "Il comando `doctor` verifica che l'endpoint Ollama (OpenAI-compatible) funzioni e controlla:\n\n"
            "- chiamata a `/v1/models`\n"
            "- embeddings (embed di test)\n"
            "- chat completion (chat di test)\n"
        ),
        citations=[
            {"ref": "S1", "quote": "models = list_models()"},
            {"ref": "S1", "quote": 'embed(["ping"])'},
            {"ref": "S1", "quote": "out = chat("},
        ],
        confidence="alta",
        open_questions=[],
    )
    src = _make_source("S1", cli, "doctor::deterministic", excerpt)
    return ans, [src]


def try_deterministic_embeddings_where(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    # if "embedding" not in q and "embeddings" not in q:
    #    return None
    if not (
        ("embedding" in q or "embeddings" in q)
        and ("dove" in q or "where" in q or "file" in q or "quali" in q or "punti" in q)
    ):
        return None

    root = _repo_root()
    files = []
    for f in iter_files(root):
        if f.suffix.lower() == ".py":
            txt = _read_text(f)
            if "def embed(" in txt or "embed(" in txt:
                files.append(f)
        if len(files) >= 5:
            break

    if not files:
        return None

    sources = []
    for i, f in enumerate(files[:3], start=1):
        txt = _read_text(f)
        # excerpt: prime righe che contengono embed(
        lines = [
            ln.strip()
            for ln in txt.splitlines()
            if "embed(" in ln or "def embed(" in ln
        ][:30]
        excerpt = "\n".join(lines) or txt[:2000]
        sources.append(_make_source(f"S{i}", f, f"{f.name}::deterministic", excerpt))

    ans = AnswerWithCitations(
        answer_md=(
            "Ho trovato riferimenti a embeddings in questi punti:\n\n"
            + "\n".join(f"- `{s.path}`" for s in sources)
            + "\n\nIn genere: definizione in `http_llm.py` (def `embed`) e chiamate in `indexer.py` (indicizzazione) e `rag.py` (query)."
        ),
        citations=[{"ref": s.ref, "quote": "embed("} for s in sources],
        confidence="media",
        open_questions=[],
    )
    return ans, sources


def try_deterministic_embed_purpose(
    question: str,
) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if not (
        ("embed" in q or "embeddings" in q)
        and ("scopo" in q or "purpose" in q or "perché" in q or "why" in q)
    ):
        return None

    root = _repo_root()
    rag = root / "src" / "repocopilot" / "rag.py"
    idx = root / "src" / "repocopilot" / "indexer.py"
    http = root / "src" / "repocopilot" / "http_llm.py"

    def first_line_containing(path: Path, needle: str) -> str:
        if not path.exists():
            return ""
        txt = _read_text(path)
        for ln in txt.splitlines():
            if needle in ln:
                return ln.strip()
        return ""

    rag_line = first_line_containing(
        rag, "q_emb = embed("
    )  # query embedding (retrieval)
    idx_line = first_line_containing(
        idx, "embs = embed("
    )  # doc embedding (indexing)  <-- perfetto
    http_line = first_line_containing(http, "def embed(")  # definizione

    sources: List[Source] = []
    if rag.exists():
        sources.append(_make_source("S1", rag, "rag.py::embed-call", _read_text(rag)))
    if idx.exists():
        sources.append(
            _make_source("S2", idx, "indexer.py::embed-call", _read_text(idx))
        )
    if http.exists():
        sources.append(
            _make_source("S3", http, "http_llm.py::embed-def", _read_text(http))
        )

    answer_md = (
        "✅ `embed()` serve a trasformare testo in vettori (embeddings) e viene usata in due momenti:\n\n"
        "1) **Retrieval (query embedding)**: in `rag.py`, per vettorializzare la domanda (`q_emb = embed(...)`) e interrogare il vector store.\n"
        "2) **Indicizzazione (document embedding)**: in `indexer.py`, per vettorializzare i chunk dei file (`embs = embed(docs)`) e salvarli in Chroma.\n\n"
        "La chiamata HTTP verso Ollama per ottenere embeddings è implementata in `http_llm.py` (def `embed`)."
    )

    citations = []
    if rag_line:
        citations.append({"ref": "S1", "quote": rag_line[:200]})
    if idx_line:
        citations.append({"ref": "S2", "quote": idx_line[:200]})
    if http_line:
        citations.append({"ref": "S3", "quote": http_line[:200]})

    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=citations,
        confidence="alta" if len(citations) >= 2 else "media",
        open_questions=[],
    )

    return ans, sources
