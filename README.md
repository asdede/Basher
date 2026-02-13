# Basher
Fedora's Terminal AI helper.

## DB init

Init fills Qdrant by **crawling the URLs in `config/docPages.json`** (one collection per entry), with rate limiting so we don’t overload those sites.

1. **Start Qdrant** (from project root):
   ```bash
   docker compose up -d qdrant
   ```

2. **Install Python deps** (includes crawler deps):
   ```bash
   pip install -r config/requirements-db.txt
   ```

3. **Run init from project root** (so `config/docPages.json` is found):
   ```bash
   PYTHONPATH=src/db python -m src.db.init
   ```
   Or: `cd src/db && python init.py` (then set `DOCPAGES_CONFIG` to an absolute path to `docPages.json` if needed).

   **Crawl limits (optional env):** `CRAWL_MAX_PAGES=50`, `CRAWL_MAX_DEPTH=1`, `CRAWL_DELAY_SEC=1.5`. The crawler uses a polite User-Agent and a delay between requests to avoid hammering servers.

   **Fallback:** If `config/docPages.json` is missing, init uses the `docs/` folder and collections `fedora_docs` / `bash_docs`. Add plain-text files under `docs/` in that case.

   **Qdrant URL:** Default `http://localhost:6333`. Override with `QDRANT_URL` if needed.
