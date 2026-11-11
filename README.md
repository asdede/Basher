# Basher

Local Fedora/Linux terminal assistant: **Ollama** (chat LLM) + **Qdrant** (RAG) + **FastAPI** + **Electron agent UI** (shell tools) + **CLI** + **Next.js eval app**.

| Piece | Role |
|-------|------|
| Electron (`src/agent-ui`) | Agent UI — tools run on the host with approval |
| Next.js (`src/eval-app`) | Evaluation / scoring UI — **not** the agent window |
| CLI (`Scripts/basher.sh`) | Ask `/ask`; `--agent` is one-shot (no interactive tool loop) |

Default model: **`qwen3.5:4b`** (tool-capable). Context: **`OLLAMA_NUM_CTX=12288`**.

---

## Requirements

- Linux (developed on Fedora)
- Docker + Docker Compose v2
- Optional: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU inside Compose
- Node.js 20+ for agent-ui and eval-app
- `curl`, `jq` for the CLI

---

## Quick start

```bash
cp .env.example .env          # edit UID/GID if needed; --start sets them for you
./Scripts/basher.sh --start   # build + up (CPU by default)
./Scripts/basher.sh --status
```

GPU in Compose (optional):

```bash
BASHER_USE_GPU=1 ./Scripts/basher.sh --start
# equivalent:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Host Ollama (recommended on Fedora when `ollama serve` already uses the GPU on `127.0.0.1:11434`):

```bash
BASHER_USE_HOST_OLLAMA=1 ./Scripts/basher.sh --start
# equivalent:
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d --build qdrant
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d --build --no-deps api
```

`BASHER_USE_HOST_OLLAMA=1` takes precedence over `BASHER_USE_GPU`.

Stop / tear down:

```bash
./Scripts/basher.sh --stop    # stop containers
docker compose down           # remove containers (volumes kept)
```

Services after start:

| Service | Port |
|---------|------|
| API | http://localhost:8000 |
| Ollama | http://localhost:11434 |
| Qdrant | http://localhost:6333 |

---

## Environment

Copy `.env.example` → `.env`. Important keys:

| Variable | Meaning |
|----------|---------|
| `OLLAMA_MODEL` | Chat/agent model (default `qwen3.5:4b`) |
| `OLLAMA_NUM_CTX` | Context window (default `12288`) |
| `UID` / `GID` | API container user (eval results writable on host) |
| `BASHER_USE_GPU` | Set `1` for compose NVIDIA overlay (compose Ollama) |
| `BASHER_USE_HOST_OLLAMA` | Set `1` to use host Ollama + host-network API (preferred on Fedora) |
| `BASHER_API_URL` | CLI API base (default `http://localhost:8000`) |

Retrieval knobs (`RAG_*`) are documented in `.env.example`.

---

## RAG ingest

1. Fetch Fedora Quick Docs (gitignored under `docs/`):

   ```bash
   ./Scripts/fetch-fedora-docs.sh
   ```

2. Ingest into Qdrant (from host or API container):

   ```bash
   # all collections from config/docPages.json
   docker compose exec api python -m src.db.init

   # or one collection
   docker compose exec api python -m src.db.init --only fedora_quick_docs
   ```

   Host alternative: `QDRANT_URL=http://localhost:6333 python -m src.db.init`

Collections: `fedora_quick_docs`, `bash_docs`, `git_docs`, `docker_docs`, `man_pages`.

---

## CLI

```bash
# optional alias
alias basher='/path/to/Basher/Scripts/basher.sh'

basher how do I enable automatic updates on Fedora
BASHER_DEBUG=1 basher explain bash arrays
basher --agent "what kernel am I on?"   # one-shot; exit 3 if tools proposed (use Electron)
basher --status
```

Exit codes: `0` success, `1` usage/API error, `2` API unreachable, `3` agent proposed tools (not executed), `130` interrupted.

---

## Electron agent UI

Runs shell tools on **your** machine (approval UI). Not the eval app.

```bash
cd src/agent-ui
npm install
npm start
```

Needs the API at `http://localhost:8000` (override via whatever the UI uses for health — default localhost).

---

## Eval app (Next.js)

Scores / reviews eval runs. **Not** the agent UI.

```bash
# produce a results file (API must be up + RAG ingested)
python eval/run_evals.py

cd src/eval-app
npm install
npm run dev    # http://localhost:3000
```

See `eval/README.md` for dataset fields and output shape.

---

## Project layout

```
Scripts/          basher.sh, fetch-fedora-docs.sh
config/           docPages.json (RAG sources)
src/api/          FastAPI + agent loop
src/db/           ingest / scrapers
src/agent-ui/     Electron agent
src/eval-app/     Next.js eval UI
eval/             dataset + runner
docker-compose.yml
docker-compose.gpu.yml   # optional NVIDIA overlay
```
