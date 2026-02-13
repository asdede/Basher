import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from data_gatherer import (
    get_embeddings_and_chunks,
    get_embeddings_and_chunks_from_urls,
    VECTOR_SIZE,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DOCPAGES_CONFIG = os.environ.get("DOCPAGES_CONFIG", "config/docPages.json")
CRAWL_MAX_PAGES = int(os.environ.get("CRAWL_MAX_PAGES", "50"))
CRAWL_MAX_DEPTH = int(os.environ.get("CRAWL_MAX_DEPTH", "1"))
CRAWL_DELAY_SEC = float(os.environ.get("CRAWL_DELAY_SEC", "1.5"))


def _ensure_collection(qdrant, database):
    try:
        qdrant.get_collection(database)
    except Exception:
        qdrant.create_collection(
            collection_name=database,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def init_db_from_urls(database, start_urls, qdrant):
    """Crawl start_urls, embed, and upsert into the given collection."""
    print(f"[{database}] Crawling up to {CRAWL_MAX_PAGES} pages (delay {CRAWL_DELAY_SEC}s)...")
    embeddings, chunks = get_embeddings_and_chunks_from_urls(
        start_urls,
        max_pages=CRAWL_MAX_PAGES,
        max_depth=CRAWL_MAX_DEPTH,
        delay_sec=CRAWL_DELAY_SEC,
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
    # Prefer docPages.json: one collection per entry, crawl that URL
    if os.path.isfile(DOCPAGES_CONFIG):
        with open(DOCPAGES_CONFIG, "r", encoding="utf-8") as f:
            doc_pages = json.load(f)
        for name, url in doc_pages.items():
            if isinstance(url, str):
                init_db_from_urls(name, url, qdrant)
            elif isinstance(url, list):
                init_db_from_urls(name, url, qdrant)
            else:
                print(f"[{name}] Invalid URL entry in config. Skip.")
        return
    # Fallback: init named collections from local docs folder
    if databases is None:
        databases = ["fedora_docs", "bash_docs"]
    for database in databases:
        init_db_from_folder(database, docs_folder, qdrant)


if __name__ == "__main__":
    main()
