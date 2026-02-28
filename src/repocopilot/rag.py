from __future__ import annotations
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from .config import settings
from .http_llm import embed, chat
from .indexer import get_client, get_collection, iter_files
from .schemas import AnswerWithCitations, Source

# _ENTRYPOINT_HINTS = ("entrypoint", "entry point", "console script", "project.scripts", "entry_points", "cli")
_ENTRYPOINT_HINTS = ("entrypoint", "entry point", "console script", "project.scripts", "entry_points", "console_scripts")
def _needs_entrypoint_context(q: str) -> bool:
    ql = q.lower()
    return any(h in ql for h in _ENTRYPOINT_HINTS)
def _parse_project_scripts(pyproject_text: str) -> Dict[str, str]:
    """
    Parsing semplice della sezione [project.scripts] in pyproject.toml.
    Non richiede librerie TOML aggiuntive.
    """
    lines = pyproject_text.splitlines()
    in_section = False
    scripts: Dict[str, str] = {}

    for line in lines:
        raw = line.strip()

        if not raw or raw.startswith("#"):
            continue

        # nuova sezione
        if raw.startswith("[") and raw.endswith("]"):
            in_section = (raw == "[project.scripts]")
            continue

        if in_section and "=" in raw:
            k, v = raw.split("=", 1)
            k = k.strip()
            v = v.strip()
            # rimuove virgolette singole/doppie
            v = v.strip('"').strip("'")
            # rimuove commento inline
            v = v.split("#", 1)[0].strip().strip('"').strip("'")
            scripts[k] = v

    return scripts
def try_deterministic_entrypoint(question: str) -> Optional[tuple[AnswerWithCitations, list[Source]]]:
    """
    Se la domanda riguarda l'entrypoint della CLI, risponde leggendo direttamente pyproject.toml.
    """
    if not _needs_entrypoint_context(question):
        return None

    pyproject = Path(".").resolve() / "pyproject.toml"
    if not pyproject.exists():
        return None

    text = pyproject.read_text(encoding="utf-8", errors="ignore")
    scripts = _parse_project_scripts(text)

    if not scripts:
        # non c'è [project.scripts]
        ans = AnswerWithCitations(
            answer_md="Non ho trovato la sezione `[project.scripts]` in `pyproject.toml`, quindi non posso determinare l'entrypoint della CLI.",
            citations=[],
            confidence="bassa",
            open_questions=["Il progetto usa setuptools entry_points (setup.cfg/setup.py) invece di [project.scripts]?"],
        )
        src = Source(ref="S1", chunk_id="pyproject.toml::deterministic", path=str(pyproject), excerpt=text[:1200])
        return ans, [src]

    # Se esiste lo script "repocopilot", usiamo quello, altrimenti prendiamo il primo
    script_name = "repocopilot" if "repocopilot" in scripts else sorted(scripts.keys())[0]
    entry = scripts[script_name]

    quote = f'{script_name} = "{entry}"'
    answer_md = (
        "L'entrypoint (console script) della CLI è definito in `pyproject.toml` nella sezione `[project.scripts]`:\n\n"
        f"- `{quote}`"
    )

    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=[{"ref": "S1", "quote": quote}],
        confidence="alta",
        open_questions=[],
    )

    # excerpt: includiamo almeno la riga citata (se possibile)
    # cerchiamo di estrarre un excerpt “mirato”
    m = re.search(r"(?ms)^\[project\.scripts\]\s*$.*?(?=^\[|\Z)", text)
    excerpt = m.group(0)[:1200] if m else text[:1200]

    src = Source(ref="S1", chunk_id="pyproject.toml::deterministic", path=str(pyproject), excerpt=excerpt)
    return ans, [src]

def _repo_root() -> Path:
    return Path(".").resolve()

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _make_source(ref: str, path: Path, chunk_id: str, excerpt: str) -> Source:
    return Source(ref=ref, chunk_id=chunk_id, path=str(path), excerpt=excerpt[:1200])

def try_deterministic_default_models(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if not (("modelli" in q or "models" in q) and ("default" in q or "di default" in q or "predefiniti" in q)):
        return None

    cfg = _repo_root() / "src" / "repocopilot" / "config.py"
    if not cfg.exists():
        return None

    text = _read_text(cfg)
    m_chat = re.search(r'chat_model:\s*str\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', text)
    m_emb = re.search(r'embed_model:\s*str\s*=\s*os\.getenv\([^,]+,\s*"([^"]+)"\)', text)
    if not (m_chat and m_emb):
        return None

    chat_model = m_chat.group(1)
    embed_model = m_emb.group(1)

    # citazioni: prendi le righe reali dal file
    chat_line = next((ln.strip() for ln in text.splitlines() if "chat_model" in ln and "os.getenv" in ln), "")
    emb_line = next((ln.strip() for ln in text.splitlines() if "embed_model" in ln and "os.getenv" in ln), "")

    answer_md = (
        "I modelli di default sono definiti in `src/repocopilot/config.py`:\n\n"
        f"- Chat: `{chat_model}`\n"
        f"- Embeddings: `{embed_model}`"
    )

    src = _make_source("S1", cfg, "config.py::deterministic", excerpt=text)
    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=[
            {"ref": "S1", "quote": chat_line[:200]},
            {"ref": "S1", "quote": emb_line[:200]},
        ],
        confidence="alta",
        open_questions=[],
    )
    return ans, [src]


# def try_deterministic_cli_commands(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
#     q = question.lower()
#     if not (("comandi" in q or "commands" in q) and ("cli" in q or "progetto" in q or "implement" in q or "implementati" in q or "available" in q)):
#         return None

#     # trova il file python più probabile che definisce l'app Typer
#     root = _repo_root()
#     candidates: List[Path] = []
#     for f in iter_files(root):
#         if f.suffix.lower() != ".py":
#             continue
#         txt = _read_text(f)
#         if "typer.Typer" in txt and "@app.command" in txt:
#             candidates.append(f)

#     if not candidates:
#         return None

#     # scegli il candidato con più occorrenze di @app.command
#     best = max(candidates, key=lambda p: _read_text(p).count("@app.command"))

#     text = _read_text(best)
#     # estrae nomi funzione dopo @app.command()
#     cmds = re.findall(r"@app\.command\(\)\s*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)
#     if not cmds:
#         return None

#     cmds_unique = list(dict.fromkeys(cmds))  # preserva ordine e rimuove duplicati

#     # excerpt mirato: righe con decorator + def
#     lines = text.splitlines()
#     excerpt_lines = []
#     for i, ln in enumerate(lines):
#         if "@app.command" in ln:
#             excerpt_lines.append(ln)
#             if i + 1 < len(lines):
#                 excerpt_lines.append(lines[i + 1])
#     excerpt = "\n".join(excerpt_lines[:80])

#     answer_md = "Comandi CLI trovati (decorator `@app.command()`):\n\n" + "\n".join(f"- `{c}`" for c in cmds_unique)

#     src = _make_source("S1", best, "cli_commands::deterministic", excerpt=excerpt or text[:1200])
#     ans = AnswerWithCitations(
#         answer_md=answer_md,
#         citations=[{"ref": "S1", "quote": "@app.command()"}],
#         confidence="alta",
#         open_questions=[],
#     )
#     return ans, [src]
# def try_deterministic_cli_commands(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
#     q = question.lower()
#     if not (("comandi" in q or "commands" in q) and ("cli" in q or "progetto" in q or "implement" in q or "implementati" in q or "available" in q)):
#         return None

#     root = _repo_root()

#     # 1) Cerca in tutti i .py occorrenze di @app.command()
#     best = None
#     best_count = 0
#     best_text = ""

#     for f in iter_files(root):
#         if f.suffix.lower() != ".py":
#             continue
#         txt = _read_text(f)
#         c = txt.count("@app.command")
#         if c > best_count:
#             best = f
#             best_count = c
#             best_text = txt

#     if not best or best_count == 0:
#         return None

#     # 2) Estrai i nomi dei comandi: @app.command() seguito da def <name>(
#     cmds = re.findall(r"@app\.command\(\)\s*\ndef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", best_text)
#     if not cmds:
#         return None

#     cmds_unique = list(dict.fromkeys(cmds))

#     # excerpt mirato: solo le righe con decorator+def
#     lines = best_text.splitlines()
#     excerpt_lines = []
#     for i, ln in enumerate(lines):
#         if "@app.command" in ln:
#             excerpt_lines.append(ln.strip())
#             if i + 1 < len(lines):
#                 excerpt_lines.append(lines[i + 1].strip())

#     excerpt = "\n".join(excerpt_lines[:120])

#     answer_md = "Comandi CLI implementati (decorator `@app.command()`):\n\n" + "\n".join(
#         f"- `{c}`" for c in cmds_unique
#     )

#     src = _make_source("S1", best, "cli_commands::deterministic", excerpt=excerpt or best_text[:1200])
#     ans = AnswerWithCitations(
#         answer_md=answer_md,
#         citations=[{"ref": "S1", "quote": "@app.command()"}],
#         confidence="alta",
#         open_questions=[],
#     )
#     return ans, [src]
def try_deterministic_cli_commands(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if not (("comandi" in q or "commands" in q) and ("cli" in q or "command" in q or "implement" in q or "implementati" in q)):
        return None

    cli = Path(".").resolve() / "src" / "repocopilot" / "cli.py"
    if not cli.exists():
        return None

    text = cli.read_text(encoding="utf-8", errors="ignore")

    # trova pattern: @app.command() seguito da def <name>(
    cmds = re.findall(r"@app\.command\(\)\s*\r?\n\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)
    if not cmds:
        ans = AnswerWithCitations(
            answer_md="Non ho trovato comandi Typer del tipo `@app.command()` in `src/repocopilot/cli.py`.",
            citations=[],
            confidence="bassa",
            open_questions=["I comandi sono definiti in un file diverso da cli.py?"],
        )
        src = Source(ref="S1", chunk_id="cli_commands::deterministic", path=str(cli), excerpt=text[:1200])
        return ans, [src]

    # unique preservando ordine
    seen = set()
    cmds_unique = []
    for c in cmds:
        if c not in seen:
            seen.add(c)
            cmds_unique.append(c)

    # excerpt mirato: solo righe decorator + def
    excerpt_lines = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "@app.command" in ln:
            excerpt_lines.append(ln.strip())
            if i + 1 < len(lines):
                excerpt_lines.append(lines[i + 1].strip())
    excerpt = "\n".join(excerpt_lines[:120]) or text[:1200]

    answer_md = "Comandi CLI implementati (decorator `@app.command()`):\n\n" + "\n".join(
        f"- `{c}`" for c in cmds_unique
    )

    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=[{"ref": "S1", "quote": "@app.command()"}],
        confidence="alta",
        open_questions=[],
    )
    src = Source(ref="S1", chunk_id="cli_commands::deterministic", path=str(cli), excerpt=excerpt)
    return ans, [src]
def try_deterministic_doctor(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if "doctor" not in q:
        return None

    cli = _repo_root() / "src" / "repocopilot" / "cli.py"
    if not cli.exists():
        return None

    text = _read_text(cli)
    # prendi righe significative dalla funzione doctor
    wanted = []
    for ln in text.splitlines():
        s = ln.strip()
        if "models = list_models()" in s:
            wanted.append(s)
        if "embed([\"ping\"])" in s or "embed(['ping'])" in s:
            wanted.append(s)
        if "chat(" in s and "ping" in text:
            # prendiamo una riga chat “indicativa”, non tutte
            pass

    # fallback: estrai blocco doctor (semplice)
    m = re.search(r"def\s+doctor\s*\(.*?\):(?P<body>.*?)(?=\n@app\.command|\nif __name__|\Z)", text, flags=re.S)
    excerpt = m.group(0)[:1200] if m else text[:1200]

    answer_md = (
        "Il comando `doctor` fa un check di connettività verso Ollama (OpenAI-compatible) e verifica:\n\n"
        "- chiamata a `/v1/models` (listaggio modelli)\n"
        "- embeddings (chiamata `embed([\"ping\"])`)\n"
        "- chat completion (chiamata `chat(...)`)\n"
    )

    src = _make_source("S1", cli, "doctor::deterministic", excerpt=excerpt)
    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=[
            {"ref": "S1", "quote": "models = list_models()"},
            {"ref": "S1", "quote": "embed([\"ping\"])"},
            {"ref": "S1", "quote": "out = chat("},
        ],
        confidence="alta",
        open_questions=[],
    )
    return ans, [src]


def try_deterministic_embeddings_where(question: str) -> Optional[Tuple[AnswerWithCitations, List[Source]]]:
    q = question.lower()
    if "embedding" not in q and "embeddings" not in q:
        return None

    root = _repo_root()
    hits: List[tuple[Path, str]] = []

    # cerca definizione def embed e call-site embed(
    for f in iter_files(root):
        if f.suffix.lower() != ".py":
            continue
        txt = _read_text(f)
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith("def embed(") or "embed(" in s:
                if "chromadb" in s:
                    continue
                hits.append((f, s[:200]))
        if len(hits) >= 8:
            break

    if not hits:
        return None

    # crea sources (max 3 file)
    sources: List[Source] = []
    seen = set()
    for f, _ in hits:
        if f in seen:
            continue
        seen.add(f)
        excerpt = "\n".join([line for (ff, line) in hits if ff == f][:20])
        sources.append(_make_source(f"S{len(sources)+1}", f, f"{f.name}::deterministic", excerpt=excerpt))
        if len(sources) >= 3:
            break

    answer_md = "Ho trovato riferimenti a embeddings in questi punti:\n\n" + "\n".join(
        f"- `{s.path}`" for s in sources
    )
    answer_md += "\n\nIn particolare: la definizione è in `http_llm.py` (def `embed`), e viene chiamata in `indexer.py` (indicizzazione) e `rag.py` (query)."

    ans = AnswerWithCitations(
        answer_md=answer_md,
        citations=[{"ref": s.ref, "quote": "embed("} for s in sources],
        confidence="media",
        open_questions=[],
    )
    return ans, sources
def retrieve(question: str) -> List[Source]:
    client = get_client()
    col = get_collection(client)

    # Query expansion: aumenta la probabilità di tirare dentro pyproject.toml
    q = question
    if _needs_entrypoint_context(question):
        q = question + "\nKeywords: pyproject.toml [project.scripts] console_scripts entry_points"


    q_emb = embed([q])[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(settings.top_k, 10),
        include=["documents", "metadatas", "distances"],  # <-- rimosso "ids"
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    sources: List[Source] = []
    # for idx, (chunk_id, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
    #     ref = f"S{idx}"
    #     sources.append(Source(
    #         ref=ref,
    #         chunk_id=chunk_id,
    #         path=meta.get("path", "unknown"),
    #         excerpt=(doc[:900] if doc else ""),
    #     ))
    # return sources
    for chunk_id, doc, meta in zip(ids, docs, metas):
      sources.append(Source(
          ref="",
          chunk_id=chunk_id,
          path=meta.get("path", "unknown"),
          excerpt=(doc[:900] if doc else ""),
      ))

    # Fallback deterministico: se serve, aggiungi pyproject.toml come fonte
    if _needs_entrypoint_context(question):
        root = Path(".").resolve()
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and not any(s.path.endswith("pyproject.toml") for s in sources):
            text = pyproject.read_text(encoding="utf-8", errors="ignore")
            sources.insert(0, Source(
                ref="",
                chunk_id="pyproject.toml::manual",
                path=str(pyproject),
                excerpt=text[:1200],
            ))

    # Assegna ref S1..Sk e tronca top_k
    sources = sources[:settings.top_k]
    for i, s in enumerate(sources, start=1):
        s.ref = f"S{i}"
    return sources

def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text

def answer_with_citations(question: str) -> Tuple[AnswerWithCitations, List[Source]]:
    # deterministici (prima!)
    for resolver in (
        try_deterministic_entrypoint,
        try_deterministic_cli_commands,
        try_deterministic_default_models,
        try_deterministic_doctor,
        try_deterministic_embeddings_where,
    ):
        out = resolver(question)
        if out is not None:
            return out
        
    sources = retrieve(question)

    ctx = []
    for s in sources:
        ctx.append(
            f"[{s.ref}] chunk_id={s.chunk_id}\n"
            f"path={s.path}\n"
            f"excerpt:\n{s.excerpt}\n"
        )
    sources_block = "\n".join(ctx)

    # system = (
    #     "Sei RepoCopilot, assistente offline per analisi codebase.\n"
    #     "REGOLE:\n"
    #     "- Usa SOLO le fonti fornite.\n"
    #     "- Se mancano informazioni, dillo.\n"
    #     "- Rispondi in JSON valido e SOLO JSON.\n"
    #     "- Ogni affermazione importante deve avere almeno una citation.\n\n"
    #     "SCHEMA JSON:\n"
    #     "{"
    #     "\"answer_md\": string, "
    #     "\"citations\": [{\"ref\": \"S1|S2|...\", \"quote\": string}], "
    #     "\"confidence\": \"alta|media|bassa\", "
    #     "\"open_questions\": [string]"
    #     "}"
    # )
    system = (
        "Sei RepoCopilot, assistente offline per analisi codebase.\n"
        "REGOLE:\n"
        "- Usa SOLO le fonti fornite.\n"
        "- Se mancano informazioni, dillo.\n"
        "- Rispondi in JSON valido e SOLO JSON.\n"
        "- Ogni affermazione importante deve avere almeno una citation.\n"
        "- citations[].ref deve essere uno tra " + ", ".join([s.ref for s in sources]) + "\n\n"
        "SCHEMA JSON:\n"
        "{"
        "\"answer_md\": string, "
        "\"citations\": [{\"ref\": \"S1|S2|...\", \"quote\": string}], "
        "\"confidence\": \"alta|media|bassa\", "
        "\"open_questions\": [string]"
        "}"
    )

    user = (
        f"DOMANDA:\n{question}\n\n"
        f"FONTI:\n{sources_block}\n\n"
        "Produci la risposta nello schema richiesto."
    )
    for attempt in range(2):
        raw = chat(system=system, user=user, temperature=0.0 if attempt == 0 else 0.0)
        json_text = _extract_json(raw)

        try:
            data = json.loads(json_text)
            model = AnswerWithCitations.model_validate(data)
            return model, sources
        except Exception:
            # secondo tentativo più rigido
            system = system + "\nATTENZIONE: l'output precedente NON era JSON valido. Ora outputta SOLO JSON valido."

    fallback = AnswerWithCitations(
        answer_md=(raw.strip() or "Il modello non ha restituito una risposta valida (output vuoto o non-JSON)."),
        citations=[],
        confidence="bassa",
        open_questions=["Output non in JSON valido anche dopo retry."],
    )
    return fallback, sources
    # raw = chat(system=system, user=user)
    # json_text = _extract_json(raw)

    # try:
    #     data = json.loads(json_text)
    #     model = AnswerWithCitations.model_validate(data)
    #     return model, sources
    # except Exception:
    #     fallback = AnswerWithCitations(
    #         answer_md=raw.strip(),
    #         citations=[],
    #         confidence="bassa",
    #         open_questions=["Output non in JSON valido. Riprovare o ridurre la domanda."],
    #     )
    #     return fallback, sources