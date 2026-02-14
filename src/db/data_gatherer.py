import os

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from web_crawler import scrape_urls_only, scrape_man_page_index, scrape_docker_cli_index

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

def get_embeddings_and_chunks_from_urls(url_list, delay_sec=1.5):
    """
    Scrape exactly the given URLs (one page each), chunk, embed, return (embeddings, chunks).
    """
    if not url_list:
        return [], []
    urls = list(url_list)
    texts, _ = scrape_urls_only(urls, delay_sec=delay_sec)
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


# Max words per command doc when building embedding text (avoids huge vectors)
MAN_PAGES_EMBED_WORDS = 1500


def get_embeddings_and_payloads_for_man_pages(index_url, delay_sec=1.5):
    """
    Scrape man page index + detail pages, one point per command.
    Returns (embeddings, payloads) where payloads are dicts with "command", "description", "text".
    """
    entries = scrape_man_page_index(
        index_url,
        fetch_details=True,
        delay_sec=delay_sec,
    )
    if not entries:
        return [], []

    model = SentenceTransformer(MODEL_NAME)
    payloads = []
    texts_to_embed = []

    for e in entries:
        payload = {
            "command": e["command"],
            "description": e["description"],
            "text": e["text"],
        }
        payloads.append(payload)
        # Embed a summary: command + description + (truncated) full doc
        parts = [f"Command: {e['command']}", f"Description: {e['description']}"]
        if e["text"]:
            words = e["text"].split()[:MAN_PAGES_EMBED_WORDS]
            parts.append(" ".join(words))
        texts_to_embed.append("\n\n".join(parts))

    embeddings = model.encode(texts_to_embed)
    return embeddings, payloads


# Max words per Docker CLI doc when building embedding text (same as man_pages)
DOCKER_CLI_EMBED_WORDS = 1500


def get_embeddings_and_payloads_for_docker_cli(index_url, delay_sec=1.5):
    """
    Scrape Docker CLI reference (sidebar nav only, no TOC), one point per command.
    Returns (embeddings, payloads) where payloads have "command", "description", "text", "source" (url).
    Same structure as man_pages so the API formats them identically.
    """
    entries = scrape_docker_cli_index(
        index_url,
        fetch_details=True,
        delay_sec=delay_sec,
    )
    if not entries:
        return [], []

    model = SentenceTransformer(MODEL_NAME)
    payloads = []
    texts_to_embed = []

    for e in entries:
        payload = {
            "command": e["command"],
            "description": e.get("description", ""),
            "text": e.get("text", ""),
            "source": e.get("source", e.get("detail_url", "")),
        }
        payloads.append(payload)
        parts = [f"Command: {e['command']}", f"Description: {e.get('description', '')}"]
        if e.get("text"):
            words = e["text"].split()[:DOCKER_CLI_EMBED_WORDS]
            parts.append(" ".join(words))
        texts_to_embed.append("\n\n".join(parts))

    embeddings = model.encode(texts_to_embed)
    return embeddings, payloads


def _html_file_to_text(filepath):
    """Read an HTML file and return plain text (tags stripped)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style"]):
        s.extract()
    return " ".join(soup.get_text(separator=" ").split())


def _adoc_file_to_text(filepath):
    """Read an AsciiDoc file and return plain text (markup stripped)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""
    lines = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith(":") or s == "---" or s == "===" or s == "'''":
            continue
        if s.startswith("=") and len(s) > 1 and s[1] in "= ":
            lines.append(s.lstrip("=#").strip())
            continue
        if s.startswith("*") or s.startswith(".") or s.startswith("["):
            continue
        lines.append(s)
    return " ".join(" ".join(lines).split())


def get_embeddings_and_payloads_from_html_folder(folder):
    """
    Walk a folder for *.html files (or *.adoc if no HTML), extract text, chunk, embed.
    Returns (embeddings, payloads) with payloads having "text" and "source" (relative path).
    Supports AsciiDoc source (e.g. Fedora Quick Docs) or built HTML.
    """
    if not os.path.isdir(folder):
        return [], []

    chunks = []
    sources = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".html"):
                path = os.path.join(root, f)
                text = _html_file_to_text(path)
                if not text:
                    continue
                rel = os.path.relpath(path, folder)
                for chunk in chunk_text(text):
                    chunks.append(chunk)
                    sources.append(rel)
    if not chunks:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".adoc"):
                    path = os.path.join(root, f)
                    text = _adoc_file_to_text(path)
                    if not text:
                        continue
                    rel = os.path.relpath(path, folder)
                    for chunk in chunk_text(text):
                        chunks.append(chunk)
                        sources.append(rel)

    if not chunks:
        return [], []

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks)
    payloads = [{"text": c, "source": s} for c, s in zip(chunks, sources)]
    return embeddings, payloads
