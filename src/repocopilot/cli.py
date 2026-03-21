from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

from .http_llm import list_models, embed, chat
from .indexer import index_repo
from .rag import answer_with_citations
from .config import settings
from .git_utils import get_git_diff, parse_unified_diff_files, extract_file_context
from .pr_notes import generate_pr_notes

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def doctor():
    """Verifica che Ollama (OpenAI-compatible) risponda e che chat+embeddings funzionino."""
    try:
        models = list_models()
        console.print(Panel.fit("✅ Endpoint OK: /v1/models", title="Doctor"))
        console.print(f"Modelli visibili: {len(models.get('data', []))}")
    except Exception as e:
        console.print(Panel.fit(f"❌ Errore /v1/models: {e}", title="Doctor"))
        raise typer.Exit(code=1)

    try:
        _ = embed(["ping"])
        console.print("✅ Embeddings OK")
    except Exception as e:
        console.print(f"❌ Embeddings KO: {e}")
        raise typer.Exit(code=1)

    try:
        out = chat(system="Rispondi con una sola parola: OK", user="ping")
        console.print(f"✅ Chat OK (risposta: {out[:50]!r})")
    except Exception as e:
        console.print(f"❌ Chat KO: {e}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"Base URL: {settings.base_url}\nChat model: {settings.chat_model}\nEmbed model: {settings.embed_model}\n"
            f"Index dir: {settings.index_dir}\nCollection: {settings.collection_name}",
            title="Config",
        )
    )


@app.command()
def index(
    path: str = typer.Argument("."),
    reset: bool = typer.Option(
        False, "--reset", help="Elimina l'indice esistente e ricrea la collection."
    ),
):
    """Indicizza docs+codice del repository in un vector store locale."""
    n_files, n_chunks = index_repo(path, reset=reset)
    console.print(
        Panel.fit(
            f"Indicizzati {n_files} file, {n_chunks} chunk.\nIndex dir: {settings.index_dir}",
            title="Index",
        )
    )


@app.command()
def ask(
    question: str,
    out_dir: str = typer.Option("out", "--out"),
    rag_only: bool = typer.Option(False, "--rag-only"),
    onboarding: bool = typer.Option(False, "--onboarding"),
):
    """Q/A sulla codebase con citazioni (Markdown + JSON)."""
    typer.echo(f"[DEBUG] rag_only={rag_only}")
    ans, sources, raw = answer_with_citations(
        question, rag_only=rag_only, onboarding=onboarding
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sources_debug.txt").write_text(
        "\n\n".join(
            [
                f"[{s.ref}] {s.path} (chunk_id={s.chunk_id})\n{s.excerpt}"
                for s in sources
            ]
        ),
        encoding="utf-8",
    )
    (Path(out_dir) / "raw_llm.txt").write_text(raw or "", encoding="utf-8")

    (Path(out_dir) / "answer.json").write_text(
        ans.model_dump_json(indent=2), encoding="utf-8"
    )

    md = ["# RepoCopilot Answer", "", ans.answer_md.strip(), "", "## Citations"]
    if ans.citations:
        for c in ans.citations:
            md.append(f"- **{c.ref}**: `{c.quote}`")
    else:
        md.append("- (nessuna citazione estratta)")

    md.append("\n## Sources (retrieved)")
    for s in sources:
        md.append(f"- **{s.ref}** → `{s.path}` (chunk: `{s.chunk_id}`)")
        # print(s)

    (Path(out_dir) / "answer.md").write_text("\n".join(md), encoding="utf-8")

    console.print(
        Panel.fit(f"Scritti: {out_dir}/answer.md e {out_dir}/answer.json", title="Ask")
    )
    console.print(ans.answer_md)


@app.command("pr-notes")
def pr_notes(
    range_spec: str = typer.Option(
        "HEAD~1..HEAD", "--range", help="Range git diff, es: HEAD~1..HEAD"
    ),
    out_dir: str = typer.Option("out_pr", "--out", help="Cartella output"),
):
    """Genera PR notes (summary, rischi, test plan) a partire da git diff."""
    repo_root = Path(".").resolve()

    diff_text = get_git_diff(range_spec)
    if not diff_text.strip():
        console.print(
            Panel.fit("Diff vuoto: nessuna modifica da analizzare.", title="PR Notes")
        )
        raise typer.Exit(code=0)

    file_hunks = parse_unified_diff_files(diff_text)
    files_changed = sorted({fh.path for fh in file_hunks})
    file_snippets = []
    for fh in file_hunks:
        snippet = extract_file_context(repo_root, fh)
        file_snippets.append((fh.path, snippet))

    notes, sources = generate_pr_notes(
        diff_text=diff_text, file_snippets=file_snippets, repo_root=repo_root
    )
    notes.files_changed = files_changed
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "pr_notes.json").write_text(
        notes.model_dump_json(indent=2), encoding="utf-8"
    )

    md = []
    md.append(f"# {notes.title}")
    md.append("")
    md.append("## Summary")
    md.extend([f"- {s}" for s in notes.summary] or ["- (vuoto)"])
    md.append("")
    md.append("## Files changed")
    md.extend([f"- `{p}`" for p in notes.files_changed] or ["- (non specificato)"])
    md.append("")
    md.append("## Risks")
    if notes.risks:
        for r in notes.risks:
            md.append(f"- **{r.severity}**: {r.description}")
    else:
        md.append("- (nessun rischio indicato)")
    md.append("")
    md.append("## Suggested tests")
    md.extend(
        [f"- {t}" for t in notes.suggested_tests] or ["- (nessun test suggerito)"]
    )
    md.append("")
    md.append("## Rollout plan")
    md.extend([f"- {x}" for x in notes.rollout_plan] or ["- (vuoto)"])
    md.append("")
    md.append("## Rollback plan")
    md.extend([f"- {x}" for x in notes.rollback_plan] or ["- (vuoto)"])
    md.append("")
    md.append("## Open questions")
    md.extend([f"- {q}" for q in notes.open_questions] or ["- (nessuna)"])
    md.append("")
    md.append("## Citations")
    if notes.citations:
        for c in notes.citations:
            md.append(f"- **{c.ref}**: `{c.quote}`")
    else:
        md.append("- (nessuna citazione verificabile)")
    md.append("")
    md.append("## Sources (used)")
    for s in sources:
        md.append(f"- **{s.ref}** → `{s.path}`")

    (Path(out_dir) / "pr_notes.md").write_text("\n".join(md), encoding="utf-8")

    console.print(
        Panel.fit(
            f"Scritti: {out_dir}/pr_notes.md e {out_dir}/pr_notes.json",
            title="PR Notes",
        )
    )


if __name__ == "__main__":
    app()
