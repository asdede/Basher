import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .data_gatherer import (
    get_embeddings_and_chunks,
    get_embeddings_and_chunks_from_urls,
    get_embeddings_and_payloads_for_man_pages,
    get_embeddings_and_payloads_for_docker_cli,
    get_embeddings_and_payloads_from_html_folder,
    VECTOR_SIZE,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DOCPAGES_CONFIG = os.environ.get("DOCPAGES_CONFIG", "config/docPages.json")
SCRAPE_DELAY_SEC = float(os.environ.get("CRAWL_DELAY_SEC", "1.5"))


def _ensure_collection(qdrant, database):
    try:
        qdrant.get_collection(database)
    except Exception:
        qdrant.create_collection(
            collection_name=database,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def init_db_from_urls(database, url_list, qdrant):
    """Scrape the given URLs (one page each), embed, and upsert into the collection."""
    print(f"[{database}] Scraping {len(url_list)} URL(s) (delay {SCRAPE_DELAY_SEC}s)...")
    embeddings, chunks = get_embeddings_and_chunks_from_urls(
        url_list,
        delay_sec=SCRAPE_DELAY_SEC,
    )
    if not chunks:
        print(f"[{database}] No chunks from crawl. Skip.")
        return
    _ensure_collection(qdrant, database)
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload={"text": chunk})
        for i, (vec, chunk) in enumerate(zip(embeddings, chunks))
    ]
    qdrant.upsert(collection_name=database, points=points)
    print(f"[{database}] Upserted {len(points)} points.")


def init_db_from_man_pages(database, index_url, qdrant):
    """Scrape man page index + detail pages, one point per command; payload has command, description, text."""
    print(f"[{database}] Scraping man page index and detail pages (delay {SCRAPE_DELAY_SEC}s)...")
    embeddings, payloads = get_embeddings_and_payloads_for_man_pages(
        index_url,
        delay_sec=SCRAPE_DELAY_SEC,
    )
    if not payloads:
        print(f"[{database}] No man page entries. Skip.")
        return
    _ensure_collection(qdrant, database)
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload=payload)
        for i, (vec, payload) in enumerate(zip(embeddings, payloads))
    ]
    qdrant.upsert(collection_name=database, points=points)
    print(f"[{database}] Upserted {len(points)} command entries.")


def init_db_from_docker_cli(database, index_url, qdrant):
    """Scrape Docker CLI reference (sidebar nav), one point per command; payload has command, description, text, source."""
    print(f"[{database}] Scraping Docker CLI index and detail pages (delay {SCRAPE_DELAY_SEC}s)...")
    embeddings, payloads = get_embeddings_and_payloads_for_docker_cli(
        index_url,
        delay_sec=SCRAPE_DELAY_SEC,
    )
    if not payloads:
        print(f"[{database}] No Docker CLI entries. Skip.")
        return
    _ensure_collection(qdrant, database)
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload=payload)
        for i, (vec, payload) in enumerate(zip(embeddings, payloads))
    ]
    qdrant.upsert(collection_name=database, points=points)
    print(f"[{database}] Upserted {len(points)} command entries.")


def init_db_from_html_folder(database, path, qdrant):
    """Load HTML files from a folder (e.g. AsciiDoc→HTML output), extract text, chunk, embed, upsert."""
    resolved = os.path.abspath(path)
    print(f"[{database}] Reading HTML folder: {path}")
    if not os.path.isdir(resolved):
        print(f"[{database}] Path does not exist or is not a directory: {resolved}. Skip.")
        return
    embeddings, payloads = get_embeddings_and_payloads_from_html_folder(path)
    if not payloads:
        print(f"[{database}] No .html or .adoc files found in '{resolved}'. Skip.")
        return
    _ensure_collection(qdrant, database)
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload=payload)
        for i, (vec, payload) in enumerate(zip(embeddings, payloads))
    ]
    qdrant.upsert(collection_name=database, points=points)
    print(f"[{database}] Upserted {len(points)} chunks.")


def init_db_from_folder(database, docs_folder, qdrant):
    """Load docs from folder, embed, and upsert."""
    embeddings, chunks = get_embeddings_and_chunks(docs_folder)
    if not chunks:
        print(f"[{database}] No chunks from '{docs_folder}'. Skip.")
        return
    _ensure_collection(qdrant, database)
    points = [
        PointStruct(id=i, vector=vec.tolist(), payload={"text": chunk})
        for i, (vec, chunk) in enumerate(zip(embeddings, chunks))
    ]
    qdrant.upsert(collection_name=database, points=points)
    print(f"[{database}] Upserted {len(points)} points.")


def main(databases=None, docs_folder="docs"):
    qdrant = QdrantClient(url=QDRANT_URL)
    # docPages.json: { "collection_name": ["url1", "url2", ...] } — scrape only those URLs
    if os.path.isfile(DOCPAGES_CONFIG):
        config_abspath = os.path.abspath(DOCPAGES_CONFIG)
        # Paths in config are relative to project root (directory containing config/)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        with open(DOCPAGES_CONFIG, "r", encoding="utf-8") as f:
            doc_pages = json.load(f)
        for name, val in doc_pages.items():
            # Object: { "type": "html_folder", "path": "docs/quick-docs" } — path relative to project root
            if isinstance(val, dict):
                if val.get("type") == "html_folder" and val.get("path"):
                    raw_path = val["path"]
                    path = os.path.normpath(
                        os.path.join(project_root, raw_path)
                        if not os.path.isabs(raw_path)
                        else raw_path
                    )
                    init_db_from_html_folder(name, path, qdrant)
                elif val.get("type") == "docker_cli" and val.get("url"):
                    init_db_from_docker_cli(name, val["url"], qdrant)
                else:
                    print(f"[{name}] Unknown or invalid object config. Skip.")
                continue
            # List of URLs (or single string)
            urls = [val] if isinstance(val, str) else val
            if not isinstance(urls, list) or not urls:
                print(f"[{name}] Invalid or empty URL list. Skip.")
                continue
            if name == "man_pages":
                init_db_from_man_pages(name, urls[0], qdrant)
            else:
                init_db_from_urls(name, urls, qdrant)
        return
    # Fallback: init named collections from local docs folder
    if databases is None:
        databases = ["fedora_docs", "bash_docs"]
    for database in databases:
        init_db_from_folder(database, docs_folder, qdrant)


if __name__ == "__main__":
    main()
