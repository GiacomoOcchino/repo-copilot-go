from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

from .config import settings
from .http_llm import embed, chat
from .indexer import get_client, get_collection, iter_files
from .schemas import AnswerWithCitations, Source

from .deterministic import (
    try_deterministic_entrypoint,
    try_deterministic_cli_commands,
    try_deterministic_default_models,
    try_deterministic_doctor,
    try_deterministic_embed_purpose,
    try_deterministic_embeddings_where,
)


# -----------------------
# JSON parsing (robusto)
# -----------------------
def _extract_json(text: str) -> str:
    # fenced ```json { ... } ```
    m = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1)
    # first {...}
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return text


def _repair_json(js: str) -> str:
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    ob = js.count("{") - js.count("}")
    osq = js.count("[") - js.count("]")
    if osq > 0:
        js += "]" * osq
    if ob > 0:
        js += "}" * ob
    return js


def _parse_to_model(raw_text: str) -> AnswerWithCitations:
    # caso patologico: il modello restituisce più oggetti JSON separati (come nel tuo answer.md)
    if raw_text.strip().startswith('{\n  "ref"') or raw_text.strip().startswith(
        '{\n "ref"'
    ):
        return AnswerWithCitations(
            answer_md=raw_text.strip(),
            citations=[],
            confidence="bassa",
            open_questions=[
                "Il modello ha restituito oggetti JSON separati, non un singolo oggetto conforme allo schema."
            ],
        )

    try:
        data = json.loads(raw_text)
    except Exception:
        js = _extract_json(raw_text)
        js = _repair_json(js)
        data = json.loads(js)

    # normalizza answer_md se lista
    if isinstance(data.get("answer_md"), list):
        data["answer_md"] = "\n".join(str(x) for x in data["answer_md"])

    if "citations" not in data or data["citations"] is None:
        data["citations"] = []

    return AnswerWithCitations.model_validate(data)


# -----------------------
# Evidence injection (riga di codice, non docstring)
# -----------------------
def _repo_root() -> Path:
    return Path(".").resolve()


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _extract_lines_around_regex(
    text: str, rx: re.Pattern, radius: int = 10, max_chars: int = 1200
) -> str:
    lines = text.splitlines()
    hit: Optional[int] = None
    for i, ln in enumerate(lines):
        if rx.search(ln):
            hit = i
            break
    if hit is None:
        return ""
    a = max(0, hit - radius)
    b = min(len(lines), hit + radius + 1)
    snippet = "\n".join(lines[a:b])
    if not any(rx.search(ln) for ln in snippet.splitlines()):
        return ""
    return snippet[:max_chars]


def inject_retrieve_evidence(question: str) -> List[Source]:
    ql = question.lower()
    if not any(
        k in ql
        for k in (
            "retrieve",
            "col.query",
            "embed",
            "embedding",
            "embeddings",
            "query_embeddings",
        )
    ):
        return []

    root = _repo_root()
    evidences: List[Source] = []

    rx_embed_call = re.compile(r"^\s*q_emb\s*=\s*embed\(", re.IGNORECASE)
    rx_query_call = re.compile(r"^\s*res\s*=\s*col\.query\(", re.IGNORECASE)
    rx_embed_def = re.compile(r"^\s*def\s+embed\s*\(", re.IGNORECASE)

    # find file with both call-sites
    target = None
    target_text = ""
    for f in iter_files(root):
        if f.suffix.lower() != ".py":
            continue
        txt = _read_text(f)
        if rx_embed_call.search(txt) and rx_query_call.search(txt):
            target, target_text = f, txt
            break

    if target:
        embed_call = _extract_lines_around_regex(
            target_text, rx_embed_call, radius=8, max_chars=900
        )
        query_call = _extract_lines_around_regex(
            target_text, rx_query_call, radius=10, max_chars=1100
        )
        if embed_call:
            evidences.append(
                Source(
                    ref="",
                    chunk_id="evidence::embed-call",
                    path=str(target),
                    excerpt=embed_call,
                )
            )
        if query_call:
            evidences.append(
                Source(
                    ref="",
                    chunk_id="evidence::col.query",
                    path=str(target),
                    excerpt=query_call,
                )
            )

    # embed definition
    for f in iter_files(root):
        if f.suffix.lower() != ".py":
            continue
        txt = _read_text(f)
        if not rx_embed_def.search(txt):
            continue
        embed_def = _extract_lines_around_regex(
            txt, rx_embed_def, radius=18, max_chars=1600
        )
        if embed_def:
            evidences.append(
                Source(
                    ref="",
                    chunk_id="evidence::embed-def",
                    path=str(f),
                    excerpt=embed_def,
                )
            )
            break

    return evidences


# -----------------------
# Excerpt shaping
# -----------------------
def strip_prompt_noise(text: str) -> str:
    noise_patterns = (
        "Sei RepoCopilot",
        "REGOLE",
        "SCHEMA JSON",
        "Schema JSON",
        "QUOTE_CANDIDATES",
        "ISTRUZIONI",
        "ATTENZIONE",
    )
    out = []
    for ln in text.splitlines():
        if any(p in ln for p in noise_patterns):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def focus_excerpt(
    excerpt: str, patterns: List[str], radius: int = 10, max_chars: int = 2000
) -> str:
    lines = excerpt.splitlines()
    for i, ln in enumerate(lines):
        if any(p in ln for p in patterns):
            a = max(0, i - radius)
            b = min(len(lines), i + radius + 1)
            snippet = "\n".join(lines[a:b])
            return snippet[:max_chars]
    return excerpt[:max_chars]


# -----------------------
# Retrieval (Chroma)
# -----------------------


def retrieve(
    question: str,
    repo_path: Optional[str] = None,
    onboarding: bool = False,
) -> List[Source]:
    """
    Retrieval su Chroma per domande generiche e onboarding.

    - repo_path: path del repo indicizzato (solo per euristiche di ranking; l'indice è già globale/locale)
    - onboarding: se True, priorizza file tipici (README, composer.json, .env, public/index.php, docker-compose...)
    """
    client = get_client()
    col = get_collection(client)

    ql = question.lower()
    q = question

    # --------------------------
    # Query expansion (mirata)
    # --------------------------
    if onboarding:
        q += (
            "\nKeywords: readme quickstart install run setup prerequisites "
            "composer.json composer.lock vendor autoload .env .env.example "
            "docker-compose docker compose apache nginx php-fpm "
            "public/index.php index.php routes config "
        )

    # euristiche utili per domande tecniche comuni
    if any(
        k in ql
        for k in ("entrypoint", "entry point", "avvio", "start", "run", "bootstrap")
    ):
        q += "\nKeywords: index.php public/index.php bootstrap app.php routes router front controller"

    if any(k in ql for k in ("config", "env", "variabili", "environment")):
        q += "\nKeywords: .env getenv $_ENV $_SERVER config.php settings.php parameters.yml"

    if any(k in ql for k in ("database", "db", "mysql", "postgres", "sqlite")):
        q += "\nKeywords: database mysql postgres sqlite doctrine pdo connection dsn migrate migration"

    # --------------------------
    # Embedding query + Chroma
    # --------------------------
    q_emb = embed([q])[0]
    n = max(settings.top_k, 12)

    res = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]

    sources: List[Source] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        sources.append(
            Source(
                ref="",
                chunk_id=chunk_id,
                path=meta.get("path", "unknown"),
                excerpt=(doc[:2200] if doc else ""),
            )
        )

    # --------------------------
    # Dedup (chunk_id, path)
    # --------------------------
    seen = set()
    deduped: List[Source] = []
    for s in sources:
        key = (s.chunk_id, s.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    sources = deduped

    # --------------------------
    # Rerank onboarding (prefer file "di avvio")
    # --------------------------
    if onboarding:
        prefer = (
            "readme",
            "composer.json",
            "composer.lock",
            ".env",
            ".env.example",
            "docker-compose",
            "public/index.php",
            "index.php",
            "config",
            "routes",
            "bootstrap",
        )

        def score_path(p: str) -> Tuple[int, int, str]:
            pl = p.lower().replace("\\", "/")
            # 0 = preferito, 1 = altro
            pref = 0 if any(x in pl for x in prefer) else 1
            # penalizza vendor se per sbaglio entra
            vend = 1 if "/vendor/" in pl else 0
            return (pref, vend, pl)

        sources.sort(key=lambda s: score_path(s.path))

    # --------------------------
    # (Opzionale) Focus excerpt
    # --------------------------
    if "focus_excerpt" in globals() and "strip_prompt_noise" in globals():
        if onboarding:
            patterns = [
                "README",
                "composer",
                ".env",
                "docker",
                "php -S",
                "public/index.php",
                "index.php",
            ]
        else:
            patterns = [
                "def ",
                "class ",
                "function ",
                "route",
                "controller",
                "config",
                "env",
            ]

        for s in sources:
            s.excerpt = strip_prompt_noise(focus_excerpt(s.excerpt, patterns))

    # --------------------------
    # Limit + assign refs
    # --------------------------
    sources = sources[: settings.top_k]
    for i, s in enumerate(sources, start=1):
        s.ref = f"S{i}"

    return sources


# -----------------------
# Citation verification
# -----------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _verify_and_filter_citations(
    ans: AnswerWithCitations, sources: List[Source]
) -> AnswerWithCitations:
    source_map = {s.ref: _norm(s.excerpt) for s in sources}
    kept = []
    for c in ans.citations:
        ex = source_map.get(c.ref, "")
        if ex and c.quote:
            if _norm(c.quote) in ex:
                kept.append(c)
    ans.citations = kept
    return ans


# -----------------------
# Main ASK (Strategy A)
# -----------------------
def pick_citations(
    question: str, sources: List[Source], max_cits: int = 3
) -> List[dict]:
    ql = question.lower()

    # Priorità alta: call-site reali
    high = ["q_emb = embed(", "res = col.query("]
    mid = ["col.query(", "query_embeddings", "n_results=", "include=["]
    low = ["def embed(", "def retrieve("]

    # righe da evitare (meta)
    banned = ["needles =", "Keywords:", "if any(k in ql", "evidence::"]

    if any(
        k in ql for k in ("retrieve", "col.query", "embed", "embedding", "embeddings")
    ):
        priority_sets = [high, mid, low]
    else:
        priority_sets = [low, mid]

    def scan(needles: List[str]) -> List[dict]:
        out = []
        for s in sources:
            for ln in s.excerpt.splitlines():
                l = ln.strip()
                if not l or any(b in l for b in banned):
                    continue
                if any(n in l for n in needles):
                    out.append({"ref": s.ref, "quote": l[:200]})
                    break
            if len(out) >= max_cits:
                break
        return out

    cits: List[dict] = []
    for needles in priority_sets:
        for c in scan(needles):
            if c not in cits:
                cits.append(c)
            if len(cits) >= max_cits:
                return cits

    return cits


def answer_with_citations(
    question: str, rag_only: bool = False, onboarding: bool = False
) -> Tuple[AnswerWithCitations, List[Source], str]:
    # deterministici prima
    if not rag_only:
        for resolver in (
            try_deterministic_entrypoint,
            try_deterministic_cli_commands,
            try_deterministic_default_models,
            try_deterministic_doctor,
            try_deterministic_embed_purpose,
            try_deterministic_embeddings_where,
        ):
            out = resolver(question)
            if out is not None:
                ans, srcs = out
                return ans, srcs, ""

    sources = retrieve(question)

    # budget: evidenze + 2 chunk
    if any(s.chunk_id.startswith("evidence::") for s in sources):
        evid = [s for s in sources if s.chunk_id.startswith("evidence::")]
        other = [s for s in sources if not s.chunk_id.startswith("evidence::")]
        sources = evid + other[:2]

    sources_block = "\n".join(
        f"[{s.ref}] path={s.path} (chunk_id={s.chunk_id})\nexcerpt:\n{s.excerpt}\n"
        for s in sources
    )

    if onboarding:
        system = (
            "Sei un Senior Engineer e devi fare onboarding di un nuovo dev su questo repository.\n"
            "Regole:\n"
            "- Usa SOLO le fonti fornite.\n"
            "- Se un'informazione non è presente nelle fonti, scrivi chiaramente: NON TROVATO.\n"
            "- Non fare overview generiche del progetto.\n"
            "- Struttura l'output come una 'scheda di onboarding' con sezioni.\n"
            "- Ogni sezione deve citare almeno 1 riferimento [Sx] quando possibile.\n"
        )
        user = (
            f"DOMANDA:\n{question}\n\n"
            f"FONTI:\n{sources_block}\n\n"
            "OUTPUT OBBLIGATORIO (Markdown):\n"
            "## Quickstart\n"
            "- Prerequisiti: ... [Sx]\n"
            "- Install: ... (comandi) [Sx]\n"
            "- Run: ... (comandi/URL) [Sx]\n"
            "## Config (env vars / file)\n"
            "- ... [Sx]\n"
            "## Entrypoints e flusso principale\n"
            "- ... [Sx]\n"
            "## Dipendenze e tooling (Composer)\n"
            "- ... [Sx]\n"
            "## Gap / Cose mancanti\n"
            "- NON TROVATO: ... (cosa manca e dove ti aspetteresti di trovarlo)\n"
        )
    else:
        system = (
            "Sei RepoCopilot. Usa SOLO le fonti fornite.\n"
            "Rispondi in Markdown.\n"
            "DEVI produrre ESATTAMENTE 6 bullet numerati (1..6).\n"
            "Ogni bullet deve citare almeno un file/funzione tra le fonti.\n"
            "Non aggiungere introduzioni o conclusioni."
        )

        user = (
            f"DOMANDA:\n{question}\n\n"
            f"FONTI:\n{sources_block}\n\n"
            "FORMATO OBBLIGATORIO:\n"
            "1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n6. ...\n"
            "Contenuto richiesto: flusso end-to-end del comando ask (cli.py -> answer_with_citations -> retrieve -> embed -> col.query -> chat -> output file).\n"
        )

    raw = chat(system=system, user=user, temperature=0.0, max_tokens=900)

    ans = AnswerWithCitations(
        answer_md=raw.strip(),
        citations=pick_citations(question, sources),
        confidence="media" if sources else "bassa",
        open_questions=[],
    )

    # verifica citazioni (se non verificabili, le scarta)
    ans = _verify_and_filter_citations(ans, sources)

    return ans, sources, raw
