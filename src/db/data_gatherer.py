"""
Document loading, chunking, and embedding for RAG.

Chunking strategies (set CHUNK_STRATEGY env):
  - fixed: 500-word windows (default).
  - semantic: split by paragraphs, embed, merge while similarity is high (no LLM).
  - agentic: one LLM call per document to output coherent chunks (uses OLLAMA_URL).
"""
import os
import re

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from .web_crawler import scrape_urls_only, scrape_man_page_index, scrape_docker_cli_index

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension

# Chunking strategy: "fixed" (default), "semantic", or "agentic"
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "fixed").strip().lower()
if CHUNK_STRATEGY not in ("fixed", "semantic", "agentic"):
    CHUNK_STRATEGY = "fixed"

# Semantic chunking: merge segments while similarity >= threshold
SEMANTIC_SIMILARITY_THRESHOLD = float(os.environ.get("SEMANTIC_CHUNK_SIMILARITY", "0.5"))
SEMANTIC_MIN_WORDS = int(os.environ.get("SEMANTIC_CHUNK_MIN_WORDS", "80"))
SEMANTIC_MAX_WORDS = int(os.environ.get("SEMANTIC_CHUNK_MAX_WORDS", "500"))

# Agentic chunking: Ollama endpoint (used during ingest)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHUNK_MODEL = os.environ.get("OLLAMA_CHUNK_MODEL", "qwen2.5-coder:1.5b")

_embedder_cache = None


def _get_embedder():
    """Lazy-load embedder for semantic chunking (avoids loading when using fixed/agentic only)."""
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer(MODEL_NAME)
    return _embedder_cache


def chunk_text(text, chunk_size=500):
    """Fixed-size chunking by word count (original behavior)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def _paragraphs(text):
    """Split text into paragraphs (double newline or single newline)."""
    text = (text or "").strip()
    if not text:
        return []
    # Prefer double newline; fallback to single newline so we get segments
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_semantic(
    text,
    similarity_threshold=None,
    min_words=None,
    max_words=None,
    embedder=None,
):
    """
    Semantic chunking: split into paragraphs, embed, merge consecutive paragraphs
    while similarity is high. Creates chunks that follow topic boundaries.
    """
    threshold = similarity_threshold if similarity_threshold is not None else SEMANTIC_SIMILARITY_THRESHOLD
    min_w = min_words if min_words is not None else SEMANTIC_MIN_WORDS
    max_w = max_words if max_words is not None else SEMANTIC_MAX_WORDS
    model = embedder or _get_embedder()

    paragraphs = _paragraphs(text)
    if not paragraphs:
        return chunk_text(text, max_w)  # fallback to fixed

    if len(paragraphs) == 1:
        # Single paragraph: respect max_words with fixed split if needed
        words = paragraphs[0].split()
        if len(words) <= max_w:
            return [paragraphs[0]] if paragraphs[0] else []
        out = []
        for i in range(0, len(words), max_w):
            out.append(" ".join(words[i : i + max_w]))
        return out

    embeddings = model.encode(paragraphs)
    chunks = []
    current = [paragraphs[0]]
    current_words = len(paragraphs[0].split())

    for i in range(1, len(paragraphs)):
        prev_emb = embeddings[i - 1]
        curr_emb = embeddings[i]
        # Cosine similarity (embeddings are normalized by default in many models)
        sim = float((prev_emb @ curr_emb) / (max(1e-9, (prev_emb ** 2).sum() ** 0.5 * (curr_emb ** 2).sum() ** 0.5)))
        add_to_current = sim >= threshold and (current_words + len(paragraphs[i].split()) <= max_w)

        if add_to_current:
            current.append(paragraphs[i])
            current_words += len(paragraphs[i].split())
        else:
            chunk_text_str = "\n\n".join(current)
            if current_words < min_w and chunks:
                chunks[-1] = chunks[-1] + "\n\n" + chunk_text_str
            else:
                chunks.append(chunk_text_str)
            current = [paragraphs[i]]
            current_words = len(paragraphs[i].split())

    if current:
        chunk_text_str = "\n\n".join(current)
        if current_words < min_w and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + chunk_text_str
        else:
            chunks.append(chunk_text_str)

    return chunks if chunks else chunk_text(text, max_w)


def chunk_text_agentic(text, ollama_url=None, model=None):
    """
    Agentic chunking: ask an LLM to split the document into coherent chunks.
    Uses one LLM call per document; best for smaller docs or when quality matters.
    """
    url = ollama_url or OLLAMA_URL
    model_name = model or OLLAMA_CHUNK_MODEL
    text = (text or "").strip()
    if not text:
        return []

    prompt = """Split the following documentation into coherent chunks. Each chunk should be self-contained and cover one topic or procedure. Output ONLY the chunk texts, one chunk per block, separated by exactly "---" on its own line. No numbering, titles, or extra commentary.

Documentation:
"""
    prompt += text[:120000]  # avoid token limits
    try:
        resp = requests.post(
            url,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
    except Exception:
        return chunk_text(text, SEMANTIC_MAX_WORDS)  # fallback

    if not raw or "---" not in raw:
        return chunk_text(text, SEMANTIC_MAX_WORDS)

    chunks = [s.strip() for s in re.split(r"\n---\n", raw) if s.strip()]
    if not chunks:
        return chunk_text(text, SEMANTIC_MAX_WORDS)
    return chunks


def chunk_text_auto(text, chunk_size=500):
    """Dispatch to fixed, semantic, or agentic chunking based on CHUNK_STRATEGY."""
    if CHUNK_STRATEGY == "semantic":
        return chunk_text_semantic(text)
    if CHUNK_STRATEGY == "agentic":
        return chunk_text_agentic(text)
    return chunk_text(text, chunk_size)

def gather_docs(folder="docs"):
    """Read text files from folder, return list of text chunks (strategy from CHUNK_STRATEGY)."""
    all_chunks = []
    if not os.path.isdir(folder):
        return all_chunks
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                all_chunks.extend(chunk_text_auto(text))
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
    Scrape exactly the given URLs (one page each), chunk, embed.
    Returns (embeddings, payloads) with payloads {"text", "source"} (source = URL).
    """
    if not url_list:
        return [], []
    urls = list(url_list)
    texts, fetched = scrape_urls_only(urls, delay_sec=delay_sec)
    if not texts:
        return [], []
    all_chunks = []
    sources = []
    for text, src in zip(texts, fetched):
        for chunk in chunk_text_auto(text):
            all_chunks.append(chunk)
            sources.append(src)
    if not all_chunks:
        return [], []
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_chunks)
    payloads = [{"text": c, "source": s} for c, s in zip(all_chunks, sources)]
    return embeddings, payloads


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


def _strip_adoc_inline(text):
    """Light AsciiDoc inline cleanup for embedding text."""
    text = re.sub(r"link:[^\[]+\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"xref:[^\[]+\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+\[([^\]]*)\]", r"\1", text)
    text = text.replace("`+", "").replace("+`", "")
    text = text.replace("`", "")
    return text


def _adoc_file_to_text(filepath):
    """
    Read an AsciiDoc file and return plain text for embedding.
    Keeps list items and procedure steps (previously dropped when starting with * / .).
    Strips attributes, block markers, and light inline markup.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""
    lines = []
    in_fence = False
    fence_marks = {"----", "....", "****", "===="}
    for line in content.splitlines():
        raw = line.rstrip()
        s = raw.strip()
        if not s:
            continue
        # Toggle fenced blocks (source/listing); keep inner lines
        if s[:4] in fence_marks or s in fence_marks:
            in_fence = not in_fence
            continue
        if in_fence:
            lines.append(s)
            continue
        # Document/attribute lines
        if s.startswith(":") and ":" in s[1:]:
            continue
        if s.startswith("include::"):
            continue
        # Block attribute lines like [source,console] / [abstract]
        if s.startswith("[") and s.endswith("]"):
            continue
        if s in ("'''", "---", "***"):
            continue
        # Headings: = Title / == Section
        if s.startswith("="):
            title = s.lstrip("= ").strip()
            if title:
                lines.append(_strip_adoc_inline(title))
            continue
        # Unordered / ordered list items (keep the content)
        if re.match(r"^[\*\.]+\s+", s):
            item = re.sub(r"^[\*\.]+\s+", "", s)
            if item:
                lines.append(_strip_adoc_inline(item))
            continue
        lines.append(_strip_adoc_inline(s))
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

    def _walk_ext(ext):
        for root, dirs, files in os.walk(folder):
            # Skip Antora partials / deleted drafts
            dirs[:] = [
                d
                for d in dirs
                if d not in ("_partials", "partials", "partialsdelete")
                and not d.startswith(".")
            ]
            for f in files:
                if f.lower().endswith(ext):
                    yield os.path.join(root, f)

    for path in _walk_ext(".html"):
        text = _html_file_to_text(path)
        if not text:
            continue
        rel = os.path.relpath(path, folder)
        for chunk in chunk_text_auto(text):
            chunks.append(chunk)
            sources.append(rel)
    if not chunks:
        for path in _walk_ext(".adoc"):
            text = _adoc_file_to_text(path)
            if not text:
                continue
            rel = os.path.relpath(path, folder)
            for chunk in chunk_text_auto(text):
                chunks.append(chunk)
                sources.append(rel)

    if not chunks:
        return [], []

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks)
    payloads = [{"text": c, "source": s} for c, s in zip(chunks, sources)]
    return embeddings, payloads
