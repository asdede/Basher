#!/usr/bin/env bash
# Fetch / update Fedora Quick Docs for RAG ingest (gitignored under docs/).
#
# Source (migrated from Pagure): https://forge.fedoraproject.org/docs/quick-docs
# Ingest path in config/docPages.json:
#   docs/quick-docs/modules/ROOT/pages  →  collection fedora_quick_docs
#
# Usage:
#   ./Scripts/fetch-fedora-docs.sh           # clone or pull
#   ./Scripts/fetch-fedora-docs.sh --force   # delete and re-clone
#
# After fetch, (re)build the collection:
#   docker compose exec api python -m src.db.init --only fedora_quick_docs
#   # or on the host with Qdrant up:
#   QDRANT_URL=http://localhost:6333 python -m src.db.init --only fedora_quick_docs

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/docs/quick-docs"
REPO_URL="${FEDORA_QUICK_DOCS_URL:-https://forge.fedoraproject.org/docs/quick-docs.git}"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${ROOT}/docs"

if [[ "$FORCE" -eq 1 && -e "$DEST" ]]; then
  echo "Removing existing $DEST (--force)"
  rm -rf "$DEST"
fi

if [[ -d "$DEST/.git" ]]; then
  echo "Updating $DEST ..."
  git -C "$DEST" pull --ff-only
elif [[ -e "$DEST" ]]; then
  echo "ERROR: $DEST exists but is not a git repo. Use --force to replace it." >&2
  exit 1
else
  echo "Cloning $REPO_URL → $DEST ..."
  # Pagure/Forgejo may not support shallow clone; fall back to full clone.
  if ! git clone --depth 1 "$REPO_URL" "$DEST" 2>/dev/null; then
    echo "Shallow clone failed; trying full clone..."
    git clone "$REPO_URL" "$DEST"
  fi
fi

PAGES="$DEST/modules/ROOT/pages"
if [[ ! -d "$PAGES" ]]; then
  echo "ERROR: expected pages dir missing: $PAGES" >&2
  exit 1
fi

ADOC=$(find "$PAGES" -name '*.adoc' ! -path '*/_partials/*' 2>/dev/null | wc -l | tr -d ' ')
echo "OK: $ADOC AsciiDoc pages under $PAGES"
echo
echo "Next (re-ingest Fedora collection into Qdrant):"
echo "  docker compose exec api python -m src.db.init --only fedora_quick_docs"
echo "  # or: QDRANT_URL=http://localhost:6333 python -m src.db.init --only fedora_quick_docs"
