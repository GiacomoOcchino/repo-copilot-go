from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from .http_llm import ollama_chat_structured
from .schemas import PrNotes, Source


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def _sources_map(sources: List[Source]) -> Dict[str, str]:
    return {s.ref: s.excerpt for s in sources}


def _filter_bad_citations(notes: PrNotes, sources: List[Source]) -> PrNotes:
    """
    Guardrail: una citation è valida solo se quote è contenuta nell'excerpt della sorgente.
    """
    smap = _sources_map(sources)
    good = []
    for c in notes.citations:
        ex = smap.get(c.ref, "")
        if c.quote and c.quote in ex:
            good.append(c)
    notes.citations = good
    if notes.confidence == "alta" and len(good) == 0:
        notes.confidence = "media"
        notes.open_questions.append(
            "Nessuna citazione verificabile trovata negli snippet forniti."
        )
    return notes


def generate_pr_notes(
    diff_text: str, file_snippets: List[Tuple[str, str]], repo_root: Path
) -> Tuple[PrNotes, List[Source]]:
    """
    diff_text: testo completo del git diff
    file_snippets: lista (path, excerpt) per i file toccati
    """
    sources: List[Source] = []

    # S1: diff (tagliato per non esplodere)
    diff_excerpt = diff_text[:3500]
    sources.append(
        Source(
            ref="S1", chunk_id="git_diff::raw", path="git diff", excerpt=diff_excerpt
        )
    )

    # S2..: snippet file
    for i, (path, excerpt) in enumerate(file_snippets, start=2):
        if not excerpt.strip():
            continue
        sources.append(
            Source(
                ref=f"S{i}",
                chunk_id="file_context::hunks",
                path=path,
                excerpt=excerpt[:2200],
            )
        )
        if len(sources) >= 10:
            break

    allowed_refs = ", ".join(s.ref for s in sources)

    system = (
        "Sei RepoCopilot PR Notes generator.\n"
        "Regole:\n"
        "- Usa SOLO le fonti fornite.\n"
        "- Non inventare rischi o test: devono derivare dal diff.\n"
        "- Se il diff è piccolo, sii breve e specifico.\n"
        "- Restituisci SOLO JSON valido e NIENTE testo extra.\n"
        "- citations[].quote deve essere una sottostringa ESATTA presente nell'excerpt della source.\n"
        f"- citations[].ref deve essere uno tra: {allowed_refs}\n"
        "\n"
        "Schema JSON:\n"
        "{"
        '"title": string,'
        '"summary": [string],'
        '"files_changed": [string],'
        '"risks": [{"severity":"alta|media|bassa","description":string}],'
        '"suggested_tests": [string],'
        '"rollout_plan": [string],'
        '"rollback_plan": [string],'
        '"open_questions": [string],'
        '"citations": [{"ref":"S1|S2|...","quote":string}],'
        '"confidence":"alta|media|bassa"'
        "}"
    )

    # costruiamo blocco fonti
    sources_block = []
    for s in sources:
        sources_block.append(f"[{s.ref}] path={s.path}\nexcerpt:\n{s.excerpt}\n")

    user = (
        "Genera PR notes basate sul diff e contesto file.\n\n"
        "FONTI:\n" + "\n".join(sources_block) + "\n"
        "Richieste:\n"
        "- summary: 3-7 bullet chiari\n"
        "- risks: pochi ma concreti, legati a ciò che cambia\n"
        "- suggested_tests: test realistici\n"
        "- rollout/rollback: indicazioni pratiche\n"
    )

    schema = PrNotes.model_json_schema()

    raw = ollama_chat_structured(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        schema=schema,
        temperature=0.0,
    )
    # raw ora dovrebbe già essere JSON valido conforme allo schema
    try:
        notes = PrNotes.model_validate_json(raw)
    except Exception:
        notes = PrNotes(
            title="PR Notes",
            summary=["Il modello non ha restituito JSON valido (anche con schema)."],
            files_changed=[],
            risks=[],
            suggested_tests=[],
            rollout_plan=[],
            rollback_plan=[],
            open_questions=["Verificare modello Ollama / schema / prompt."],
            citations=[],
            confidence="bassa",
        )

    notes = _filter_bad_citations(notes, sources)
    if notes.files_changed == [".gitignore"]:
        # migliora suggerimenti tipici
        if not notes.suggested_tests:
            notes.suggested_tests = [
                "Esegui `git status` per verificare che i file attesi siano ancora tracciati.",
                "Esegui `git check-ignore -v <path>` per verificare che le nuove regole ignorino solo ciò che vuoi.",
            ]
        # rischi più realistici
        if not notes.risks:
            notes.risks = [
                {
                    "severity": "media",
                    "description": "Rischio di ignorare file che dovrebbero restare versionati (regole troppo ampie).",
                }
            ]
    return notes, sources
