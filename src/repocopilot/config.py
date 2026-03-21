from pydantic import BaseModel
import os


class Settings(BaseModel):
    # Ollama OpenAI-compatible
    base_url: str = os.getenv("REPOCOPILOT_BASE_URL", "http://localhost:11434/v1")
    chat_model: str = os.getenv("REPOCOPILOT_CHAT_MODEL", "llama3.2")
    embed_model: str = os.getenv("REPOCOPILOT_EMBED_MODEL", "nomic-embed-text")

    # Storage
    index_dir: str = os.getenv("REPOCOPILOT_INDEX_DIR", ".repocopilot/index")
    collection_name: str = os.getenv("REPOCOPILOT_COLLECTION", "repo_chunks")

    # Indexing
    include_ext: tuple[str, ...] = (".md", ".txt", ".py", ".toml", ".php", ".html")
    exclude_dirs: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        "out",
        "out_pr",
        "out_pr_old",
        ".repocopilot",
        "repocopilot.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "vendor",
        "node_modules",
    )
    exclude_files: list[str] = [
        "src/repocopilot/deterministic.py",
        "src/repocopilot/pr_notes.py",
        "src/repocopilot/git_utils.py",
    ]
    max_chars_per_chunk: int = 1800
    overlap_chars: int = 150

    # Retrieval
    top_k: int = 10


settings = Settings()
