from __future__ import annotations
import json
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

from .http_llm import list_models, embed, chat
from .indexer import index_repo
from .rag import answer_with_citations
from .config import settings

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

    console.print(Panel.fit(
        f"Base URL: {settings.base_url}\nChat model: {settings.chat_model}\nEmbed model: {settings.embed_model}\n"
        f"Index dir: {settings.index_dir}\nCollection: {settings.collection_name}",
        title="Config"
    ))

@app.command()
def index(
    path: str = typer.Argument("."),
    reset: bool = typer.Option(False, "--reset", help="Elimina l'indice esistente e ricrea la collection.")
):
    """Indicizza docs+codice del repository in un vector store locale."""
    n_files, n_chunks = index_repo(path, reset=reset)
    console.print(Panel.fit(
        f"Indicizzati {n_files} file, {n_chunks} chunk.\nIndex dir: {settings.index_dir}",
        title="Index"
    ))

@app.command()
def ask(question: str, out_dir: str = typer.Option("out", "--out")):
    """Q/A sulla codebase con citazioni (Markdown + JSON)."""
    ans, sources = answer_with_citations(question)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    (Path(out_dir) / "answer.json").write_text(ans.model_dump_json(indent=2), encoding="utf-8")

    md = ["# RepoCopilot Answer", "", ans.answer_md.strip(), "", "## Citations"]
    if ans.citations:
        for c in ans.citations:
            md.append(f"- **{c.ref}**: `{c.quote}`")
    else:
        md.append("- (nessuna citazione estratta)")

    md.append("\n## Sources (retrieved)")
    for s in sources:
        md.append(f"- **{s.ref}** → `{s.path}` (chunk: `{s.chunk_id}`)")

    (Path(out_dir) / "answer.md").write_text("\n".join(md), encoding="utf-8")

    console.print(Panel.fit(f"Scritti: {out_dir}/answer.md e {out_dir}/answer.json", title="Ask"))
    console.print(ans.answer_md)

if __name__ == "__main__":
    app()