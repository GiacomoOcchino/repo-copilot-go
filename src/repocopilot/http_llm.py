from __future__ import annotations
import httpx
from typing import List, Any, Dict
from .config import settings

def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.base_url, timeout=60.0)

def list_models() -> Dict[str, Any]:
    with _client() as c:
        r = c.get("/models")
        r.raise_for_status()
        return r.json()

def embed(texts: List[str]) -> List[List[float]]:
    payload = {"model": settings.embed_model, "input": texts}
    with _client() as c:
        r = c.post("/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]

def chat(system: str, user: str,temperature: float = 0.0) -> str:
    payload = {
        "model": settings.chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    with _client() as c:
        r = c.post("/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    

def ollama_chat_structured(messages: List[Dict[str, str]], schema: Dict[str, Any], temperature: float = 0.0) -> str:
    """
    Chiama l'endpoint Ollama nativo /api/chat usando 'format' (JSON schema),
    che forza l'output a rispettare lo schema (Structured Outputs).
    """
    # settings.base_url di solito è http://localhost:11434/v1
    root = settings.base_url
    if root.endswith("/v1"):
        root = root[:-3]

    payload: Dict[str, Any] = {
        "model": settings.chat_model,
        "messages": messages,
        "stream": False,
        "format": schema,                 # <-- schema JSON qui
        "options": {"temperature": temperature},
    }

    with httpx.Client(base_url=root, timeout=120.0) as c:
        r = c.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]