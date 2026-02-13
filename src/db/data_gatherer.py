from sentence_transformers import SentenceTransformer
import os

from web_crawler import crawl_urls

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def gather_docs(folder="docs"):
    """Read text files from folder, return list of text chunks."""
    all_chunks = []
    if not os.path.isdir(folder):
        return all_chunks
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                all_chunks.extend(chunk_text(text))
    return all_chunks

def get_embeddings_and_chunks(folder="docs"):
    """Load docs from folder, embed with SentenceTransformer, return (embeddings, chunks)."""
    chunks = gather_docs(folder)
    if not chunks:
        return [], []
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks)
    return embeddings, chunks

def get_embeddings_and_chunks_from_urls(
    start_urls,
    max_pages=50,
    max_depth=1,
    delay_sec=1.5,
):
    """
    Crawl start_urls (politely), chunk all text, embed, return (embeddings, chunks).
    Uses web_carwler with rate limiting and page cap so we don't overload target sites.
    """
    if not start_urls:
        return [], []
    urls = [start_urls] if isinstance(start_urls, str) else list(start_urls)
    texts, _ = crawl_urls(
        urls,
        max_depth=max_depth,
        max_pages=max_pages,
        delay_sec=delay_sec,
    )
    if not texts:
        return [], []
    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))
    if not all_chunks:
        return [], []
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_chunks)
    return embeddings, all_chunks
