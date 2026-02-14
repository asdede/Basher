# Basher
Fedora's Terminal AI helper.

## CLI

With the API running (e.g. `docker compose up -d` then start the API), use the Basher script:

```bash
./Scripts/basher.sh "How do I list directories in bash?"
# or
echo "What are useful dnf commands?" | ./Scripts/basher.sh
```

- **API URL:** Set `BASHER_API_URL` if the API is not at `http://localhost:8000` (e.g. `BASHER_API_URL=http://localhost:8000`).
- **Debug:** Set `BASHER_DEBUG=1` to print context and retrieval info below the response.
- **Prettify:** If [glow](https://github.com/charmbracelet/glow) is installed, markdown is rendered; otherwise output is wrapped and code/headers are lightly highlighted.

## First-time setup (required for the assistant to use docs)

The assistant uses Qdrant to search Fedora, Bash, and other docs. **Before using the API**, start Qdrant and populate the database (see **DB init** below). If you see *"No relevant documentation found"* or *"Documentation collections are not initialized"*, run the init step and ensure `config/docPages.json` points to sources that exist (e.g. `bash_docs` uses DevDocs; Fedora docs need a built HTML folder).

## DB init

Init fills Qdrant from **`config/docPages.json`**: either URLs to scrape (`["url1", "url2"]`) or a local HTML folder (`{"type": "html_folder", "path": "./some-dir"}`).

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
   PYTHONPATH=. python -m src.db.init
   ```
   Or: `cd src/db && python init.py` (then set `DOCPAGES_CONFIG` to an absolute path to `docPages.json` if needed).

   **Optional env:** `CRAWL_DELAY_SEC=1.5` (delay between requests). A polite User-Agent is used.

   **Fallback:** If `config/docPages.json` is missing, init uses the `docs/` folder and collections `fedora_docs` / `bash_docs`. Add plain-text files under `docs/` in that case.

   **Qdrant URL:** Default `http://localhost:6333`. Override with `QDRANT_URL` if needed.

### Fedora Quick Docs (local HTML)

To use [Fedora Quick Docs](https://pagure.io/fedora-docs/quick-docs) (AsciiDoc), clone the repo and build HTML with **podman + asciidoctor** (see the [quick-docs repo](https://pagure.io/fedora-docs/quick-docs) for the exact script: it converts `modules/.../pages/*.adoc` to HTML under an output dir). Put the built HTML in e.g. `docs/quick-docs-html` and set `fedora_quick_docs` in `config/docPages.json` to `{"type": "html_folder", "path": "./docs/quick-docs-html"}`. Run init from the project root so the path resolves.

### Bash manual (gnu.org 403)

If `bash_docs` fails with **403 Forbidden** on https://www.gnu.org/software/bash/manual/bash.html, gnu.org may be blocking automated requests. Options: (1) Remove or comment out `bash_docs` in `config/docPages.json` to skip it. (2) Download the [Bash Reference Manual](https://www.gnu.org/software/bash/manual/) HTML (single page or tar), put it in e.g. `docs/bash-manual-html`, and replace `bash_docs` with `{"type": "html_folder", "path": "./docs/bash-manual-html"}`.
