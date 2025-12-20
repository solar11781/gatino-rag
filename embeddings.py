from typing import List

from ollama import Client

from config import EMBED_MODEL


# By default, this talks to http://localhost:11434.
ollama_client = Client()


def embed_text(text: str) -> List[float]:
    # Avoid extremely long inputs
    if len(text) > 8000:
        text = text[:8000]

    # Call Ollama embeddings API
    res = ollama_client.embeddings(model=EMBED_MODEL, prompt=text)
    emb = res["embedding"]

    return emb
