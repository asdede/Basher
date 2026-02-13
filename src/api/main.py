from fastapi import FastAPI
import requests
from qdrant_client import QdrantClient

app = FastAPI()

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5-coder:3b"

@app.post("/ask")
def ask(query: dict):
    prompt = query["query"]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": f"You are a Fedora bash assistant.\n\nUser: {prompt}",
            "stream": False
        }
    )

    return {"response": response.json()["response"]}
