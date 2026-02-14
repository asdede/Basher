from fastapi import FastAPI
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = FastAPI()

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5-coder:3b"
QDRANT_URL = "http://qdrant:6333"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTIONS = ["fedora_quick_docs", "bash_docs", "man_pages", "docker_docs"]
TOP_K = 10
MIN_SCORE = 0.18  # Include borderline-relevant chunks; model can ignore noise

qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)


def _collect_chunks(q_vec, debug=False):
    """Search existing collections and return (chunks, any_searched, debug_info)."""
    chunks = []
    any_searched = False
    debug_info = []
    for collection in COLLECTIONS:
        try:
            if not qdrant.collection_exists(collection):
                if debug:
                    debug_info.append({"collection": collection, "status": "missing"})
                continue
            any_searched = True
            response = qdrant.query_points(
                collection_name=collection,
                query=q_vec,
                limit=TOP_K,
            )
            # query_points returns QueryResponse; results are in .points (list of scored records)
            hits = getattr(response, "points", None) or []
            scores = []
            added = 0
            for h in hits:
                if h.score is not None:
                    scores.append(h.score)
                if h.score is not None and h.score < MIN_SCORE:
                    continue
                p = h.payload or {}
                text = p.get("text", "")
                if not text:
                    continue
                added += 1
                if p.get("command"):
                    chunks.append(f"Command: {p['command']}\nDescription: {p.get('description', '')}\n\n{text}")
                elif p.get("source"):
                    chunks.append(f"Source: {p['source']}\n\n{text}")
                else:
                    chunks.append(text)
            if debug:
                debug_info.append({
                    "collection": collection,
                    "hits": len(hits),
                    "used": added,
                    "scores": sorted(scores, reverse=True)[:5] if scores else [],
                })
        except Exception as e:
            if debug:
                debug_info.append({"collection": collection, "status": "error", "error": str(e)})
    return chunks, any_searched, debug_info


@app.post("/ask")
def ask(query: dict, debug: bool = False):
    prompt = query["query"]

    q_vec = embedder.encode(prompt).tolist()
    chunks, any_collection_searched, debug_info = _collect_chunks(q_vec, debug=debug)

    if chunks:
        context = "\n\n".join(chunks[:15])
    elif any_collection_searched:
        context = "No relevant documentation found for this question."
    else:
        context = (
            "Documentation collections are not initialized. "
            "Populate the database by running: python -m src.db.init (see README)."
        )
    full_prompt = (
        "You are Fedora bash assistant. Use the following documentation when relevant. \n\n"
        "Documentation:\n" + context + "\n\n"
        "User question: " + prompt
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()

    if debug:
        return {
            "response": response.json()["response"],
            "context": context,
            "prompt": full_prompt,
            "retrieval": {"chunks_used": len(chunks), "collections": debug_info},
        }
    else:
        return {"response": response.json()["response"]}
