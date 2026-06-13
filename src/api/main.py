import asyncio
import json
import os
import queue
import re
import threading
from pathlib import Path
import sys
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# So "from eval.run_evals" works: add directory that contains eval (repo root locally; /app in Docker is already on path)
if os.environ.get("ENVIRONMENT") != "DOCKER":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.run_evals import main as run_evals_main

# Import the sibling agent module. Works both in the local repo layout
# (run from repo root: `uvicorn src.api.main:app`) and inside the Docker
# image (Dockerfile.API flattens src/api/ into /app/, so we run
# `uvicorn main:app` and `agent.py` becomes a top-level module).
try:
    from src.api import agent as agent_mod
except ImportError:
    import agent as agent_mod  # type: ignore[no-redef]

app = FastAPI()

# CORS: allow eval-app (and other frontends) to call this API from another origin
_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Eval dataset path: set EVAL_DATASET_PATH or defaults to repo_root/eval/dataset.json
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DATASET_PATH = Path(os.environ.get("EVAL_DATASET_PATH", _REPO_ROOT / "eval" / "dataset.json"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_CHAT_URL = os.environ.get(
    "OLLAMA_CHAT_URL", OLLAMA_URL.replace("/api/generate", "/api/chat")
)
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")
# Ollama defaults to num_ctx=2048, which silently truncates our system
# prompt + RAG context + recent turns. 4096 is comfortable for 7b on
# 8GB VRAM (Q4_K_M ~5.5-6GB total). Drop to 2048 if you OOM, bump if you
# have more VRAM.
try:
    NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
except ValueError:
    NUM_CTX = 4096
QDRANT_URL = "http://qdrant:6333"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTIONS = ["fedora_quick_docs", "bash_docs", "man_pages", "docker_docs", "git_docs"]
TOP_K = 10
MIN_SCORE = 0.18  # Include borderline-relevant chunks; model can ignore noise

qdrant = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBED_MODEL)
agent_sessions = {}


class AskRequest(BaseModel):
    query: str = ""
    q: str = ""
    debug: bool = False
    base_model: bool = False


class AgentPriorTurn(BaseModel):
    user: str
    answer: str = ""


class AgentRunRequest(BaseModel):
    query: str
    mode: str = "interactive"
    debug: bool = False
    base_model: bool = False
    session_id: str | None = None
    # Compressed memory of all earlier exchanges (one summary string).
    prior_summary: str | None = None
    # Verbatim memory of the most recent N exchanges (default cap = 3 in
    # the client). Cheap enough to keep raw because each one is short and
    # gives the model real continuity ("yes, do that", "now from that
    # result", refs to prior commands), which a single compressed summary
    # cannot do well on a small model.
    prior_turns: list[AgentPriorTurn] | None = None
    # RAG retrieval is OFF by default for the agent loop. The whole point
    # of agent mode is that the model can probe the live system with
    # tools — a `find $HOME ...` is more reliable than a stale doc chunk
    # about "the find command". Stuffing both into a small model dilutes
    # attention and produces hallucinated paths and echo answers. Opt in
    # via `use_retrieval: true` for explanatory questions where docs
    # really do help (e.g. "explain SELinux contexts").
    use_retrieval: bool = False


class AgentToolResult(BaseModel):
    tool_call_id: str
    command: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False


class AgentToolResultsRequest(BaseModel):
    session_id: str
    results: list[AgentToolResult]
    debug: bool = False


class AgentSummarizeRequest(BaseModel):
    query: str
    final_answer: str
    prior_summary: str | None = None

@app.get("/run-evaluation")
async def run_evaluation(use_base_model: bool = False, stream: bool = True):
    """Run evaluation.
    Args:
        use_base_model: Whether to use the base model only.
        stream: If True (default), stream progress as Server-Sent Events for a progress bar.
    Returns:
        If stream=True: SSE stream with { current, total, percent, message } then { status: "complete" }.
        If stream=False: blocks until done, then { status: "ok", message: "Evaluation complete" }.
    """
    if not stream:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: run_evals_main(use_base_model=use_base_model, argv=[])
        )
        return {"status": "ok", "message": "Evaluation complete"}

    progress_queue = queue.Queue()

    def progress_callback(current: int, total: int, message: str):
        progress_queue.put(("progress", current, total, message))

    def run_eval():
        run_evals_main(
            use_base_model=use_base_model,
            progress_callback=progress_callback,
            argv=[],
        )
        progress_queue.put(("done",))

    thread = threading.Thread(target=run_eval)
    thread.start()

    async def event_stream():
        while True:
            item = await asyncio.get_event_loop().run_in_executor(None, progress_queue.get)
            if item[0] == "done":
                break
            _, current, total, msg = item
            percent = round(100 * current / total) if total else 0
            yield f"data: {json.dumps({'current': current, 'total': total, 'percent': percent, 'message': msg})}\n\n"
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
def health():
    """Health check for CLI and load balancers. Returns status 'ok' when
    the API is up, plus the model the API is currently configured to call
    (so the UI banner can stop lying about which model is in use)."""
    return {"status": "ok", "model": MODEL, "num_ctx": NUM_CTX}


def _collect_chunks(q_vec, debug=False):
    """Search existing collections and return (chunks, any_searched, debug_info)."""
    chunks = []
    any_searched = False
    debug_info = []
    for collection in COLLECTIONS:
        try:
            if not qdrant.collection_exists(collection):
                if debug:
                    debug_info.append({"collection": collection, "status": "missing"})
                continue
            any_searched = True
            response = qdrant.query_points(
                collection_name=collection,
                query=q_vec,
                limit=TOP_K,
            )
            # query_points returns QueryResponse; results are in .points (list of scored records)
            hits = getattr(response, "points", None) or []
            scores = []
            added = 0
            for h in hits:
                if h.score is not None:
                    scores.append(h.score)
                if h.score is not None and h.score < MIN_SCORE:
                    continue
                p = h.payload or {}
                text = p.get("text", "")
                if not text:
                    continue
                added += 1
                if p.get("command"):
                    chunks.append(f"Command: {p['command']}\nDescription: {p.get('description', '')}\n\n{text}")
                elif p.get("source"):
                    chunks.append(f"Source: {p['source']}\n\n{text}")
                else:
                    chunks.append(text)
            if debug:
                debug_info.append({
                    "collection": collection,
                    "hits": len(hits),
                    "used": added,
                    "scores": sorted(scores, reverse=True)[:5] if scores else [],
                })
        except Exception as e:
            if debug:
                debug_info.append({"collection": collection, "status": "error", "error": str(e)})
    return chunks, any_searched, debug_info


@app.post("/ask")
def ask(query: dict, debug: bool = False, base_model: bool = False):
    """Request body: { \"query\": \"user question\", \"debug\": bool, \"base_model\": bool }. When debug is true, response includes context and retrieval. When base_model is true, response is based on your knowledge only. debug/base_model can also be passed as query params."""
    req = AskRequest(**query)
    if req.debug:
        debug = True
    if req.base_model:
        base_model = True
    prompt = req.query or req.q
    return _ask_core(prompt=prompt, debug=debug, base_model=base_model)


def _retrieval_context(prompt: str, debug: bool = False):
    """Run the RAG pipeline for the agent's initial system prompt."""
    q_vec = embedder.encode(prompt).tolist()
    chunks, any_searched, debug_info = _collect_chunks(q_vec, debug=debug)
    if chunks:
        context = "\n\n".join(chunks[:15])
    elif any_searched:
        context = "No relevant documentation found for this question."
    else:
        context = (
            "Documentation collections are not initialized. "
            "Populate the database by running: python -m src.db.init (see README)."
        )
    return context, len(chunks), debug_info


def _run_agent_turn_for_session(
    session: dict, *, base_model: bool, debug: bool, mode: str
) -> dict:
    """Run one model turn against the session's chat history. Returns the
    API response payload (no execution — execution lives in the client)."""
    try:
        norm = agent_mod.run_agent_turn(
            session["messages"],
            chat_url=OLLAMA_CHAT_URL,
            generate_url=OLLAMA_URL,
            model=MODEL,
            num_ctx=NUM_CTX,
        )
    except agent_mod.OllamaError as e:
        # Translate upstream-Ollama failures into something the client can
        # actually act on. The most common one in practice: operator
        # changed OLLAMA_MODEL but didn't pull it in the LLM container.
        raise HTTPException(status_code=e.status, detail=str(e))
    agent_mod.append_assistant_turn(session["messages"], norm)
    session["iterations"] = session.get("iterations", 0) + 1

    capped = session["iterations"] >= agent_mod.MAX_ITERATIONS
    if capped and norm.get("tool_calls"):
        # Force termination: ask the model to stop and answer with what it has.
        norm["tool_calls"] = []
        norm["final_answer"] = (
            norm.get("final_answer")
            or "Stopped: reached the agent's max tool-call iterations. "
            "Re-run with a more specific question, or run the suggested "
            "commands manually."
        )

    steps = [{"type": "thoughts", "message": norm.get("thoughts", "")}]
    if norm.get("tool_calls"):
        steps.append({
            "type": "tool_request",
            "message": f"Requesting {len(norm['tool_calls'])} tool call(s)",
        })
    else:
        steps.append({"type": "response", "message": "Produced final answer"})

    out = {
        "session_id": session["id"],
        "mode": mode,
        "iteration": session["iterations"],
        "thoughts": norm.get("thoughts", ""),
        # Chain of thinking passes for this turn. Empty list when the
        # model only ran a single pass (e.g. tool-mediated turns where
        # the post-tool reasoning is itself the second pass — the agent
        # only fills `thinking_steps` when it forced a reflection round
        # to enforce the >= 2-passes-before-final-answer policy).
        "thinking_steps": norm.get("thinking_steps") or [],
        "tool_calls": norm.get("tool_calls", []),
        "final_answer": norm.get("final_answer"),
        "done": not bool(norm.get("tool_calls")),
        # `schema_ok` is always returned (not just in debug). The client
        # uses it to decide whether to summarize/cache the turn — feeding
        # the summarizer a failed turn produces hallucinated next_steps
        # ("show podman logs" pattern-matched off the few-shot examples).
        "schema_ok": bool(norm.get("_schema_ok", True)),
        # When the agent had to nudge the model ("thinking_only" /
        # "refusal" / "echo") and the retry succeeded, expose that fact
        # so the UI can show a dim hint. Useful for debugging prompts
        # and reassuring the user when latency was a hair higher.
        "retry_reason": norm.get("_retry_reason"),
        # When `schema_ok` is False this names the failure mode (same
        # enum as `retry_reason`: thinking_only / refusal / echo /
        # field_leak / empty / non_json). The renderer reads it to
        # render a clean "(model failed: <reason>)" header instead of
        # double-wrapping the server-side explanation.
        "failure_reason": norm.get("_failure_reason"),
        "steps": steps,
        "max_iterations": agent_mod.MAX_ITERATIONS,
    }
    if debug:
        out["raw"] = norm.get("raw", "")
        out["base_model"] = base_model
        # The exact `final_answer` the model emitted before server-side
        # sanitization replaced it with a clean English explanation.
        # Only present when sanitization actually fired.
        if norm.get("_raw_final_answer"):
            out["raw_final_answer"] = norm.get("_raw_final_answer")
    return out


@app.post("/agent/run")
def agent_run(payload: AgentRunRequest):
    """Start a Basher agent turn for `payload.query`.

    Response shape:
      { session_id, mode, iteration, thoughts, tool_calls, final_answer,
        done, steps, max_iterations }

    The client should execute any returned tool_calls (shell commands) on the
    user's machine and POST the results to /agent/tool-results to continue the
    loop. When `done` is true and `tool_calls` is empty, the loop is finished.
    """
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    session_id = payload.session_id or str(uuid4())
    mode = payload.mode if payload.mode in ("interactive", "autonomous") else "interactive"

    documentation = ""
    retrieval_meta = None
    if not payload.base_model and payload.use_retrieval:
        documentation, used, debug_info = _retrieval_context(query, debug=payload.debug)
        if payload.debug:
            retrieval_meta = {"chunks_used": used, "collections": debug_info}

    session = agent_sessions.get(session_id)
    if not session:
        prior_turns_raw = [
            {"user": t.user, "answer": t.answer}
            for t in (payload.prior_turns or [])
            if (t.user or "").strip()
        ]
        session = {
            "id": session_id,
            "iterations": 0,
            "mode": mode,
            "base_model": payload.base_model,
            "messages": agent_mod.build_initial_messages(
                query,
                documentation=documentation,
                base_model=payload.base_model,
                prior_summary=payload.prior_summary or "",
                prior_turns=prior_turns_raw,
            ),
        }
        agent_sessions[session_id] = session
    else:
        # Continuation in the same session: append a fresh user message.
        # We deliberately ignore prior_summary here — the live transcript
        # is already richer than any compressed recap.
        session["messages"].append({"role": "user", "content": query})

    out = _run_agent_turn_for_session(
        session,
        base_model=payload.base_model,
        debug=payload.debug,
        mode=mode,
    )
    if payload.debug:
        out["context"] = documentation if not payload.base_model else "(base model only)"
        if retrieval_meta is not None:
            out["retrieval"] = retrieval_meta
    return out


@app.post("/agent/tool-results")
def agent_tool_results(payload: AgentToolResultsRequest):
    """Continue an agent loop after the client executed the requested tools.

    Body: { session_id, results: [{ tool_call_id, command, stdout, stderr,
                                    exit_code, timed_out }], debug }
    Returns the same shape as /agent/run.
    """
    session = agent_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not payload.results:
        raise HTTPException(status_code=400, detail="results required")

    for r in payload.results:
        session["messages"].append(
            agent_mod.format_tool_result_message(
                r.tool_call_id,
                r.command,
                {
                    "stdout": r.stdout,
                    "stderr": r.stderr,
                    "exit_code": r.exit_code,
                    "timed_out": r.timed_out,
                },
            )
        )

    return _run_agent_turn_for_session(
        session,
        base_model=session.get("base_model", False),
        debug=payload.debug,
        mode=session.get("mode", "interactive"),
    )


@app.post("/agent/summarize")
def agent_summarize(payload: AgentSummarizeRequest):
    """Compress a finished turn into { summary, next_step }.

    The desktop UI calls this after a final_answer so it can cache a
    compact memory of what just happened (`summary`) and a likely
    follow-up (`next_step`, used when the user just hits Enter on the
    next prompt). Best-effort: if the summarizer fails we return empty
    strings rather than 500ing the client.
    """
    query = (payload.query or "").strip()
    final = (payload.final_answer or "").strip()
    if not query or not final:
        return {"summary": "", "next_step": ""}
    try:
        return agent_mod.summarize_turn(
            user_query=query,
            final_answer=final,
            chat_url=OLLAMA_CHAT_URL,
            generate_url=OLLAMA_URL,
            model=MODEL,
            prior_summary=payload.prior_summary or "",
            num_ctx=NUM_CTX,
        )
    except agent_mod.OllamaError:
        # summarize is best-effort; never 500 the client over a cache miss.
        return {"summary": "", "next_step": ""}


@app.get("/agent/session/{session_id}")
def agent_session(session_id: str):
    session = agent_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "iterations": session.get("iterations", 0),
        "mode": session.get("mode"),
        "messages": session.get("messages", []),
    }


def _ask_core(prompt: str, debug: bool = False, base_model: bool = False):
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="query required")

    context = ""
    chunks = []
    debug_info = []
    # No database use when using only base model.
    if base_model:
        full_prompt = (
            "You are Fedora Linux assistant. You help users related to fedora linux usage, docker usage, git usage, bash usage. Answer the user question based on your knowledge. \n\n"
            "User question: " + prompt
        )

    else:
        q_vec = embedder.encode(prompt).tolist()
        chunks, any_collection_searched, debug_info = _collect_chunks(q_vec, debug=debug)

        if chunks:
            context = "\n\n".join(chunks[:15])
        elif any_collection_searched:
            context = "No relevant documentation found for this question."
        else:
            context = (
                "Documentation collections are not initialized. "
                "Populate the database by running: python -m src.db.init (see README)."
            )

        full_prompt = (
            "You are Fedora Linux assistant. You help users related to fedora linux usage, docker usage, git usage, bash usage. Use the following documentation when relevant. \n\n"
            "Documentation:\n" + context + "\n\n"
            "User question: " + prompt
        )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False
        }
    )
    response.raise_for_status()

    if debug:
        return {
            "response": response.json()["response"],
            "base_model": base_model,
            "context": context if not base_model else "(base model only — no retrieval)",
            "prompt": full_prompt,
            "retrieval": {"chunks_used": len(chunks) if not base_model else 0, "collections": debug_info if not base_model else []},
        }
    return {"response": response.json()["response"]}


def _get_dataset():
    if not EVAL_DATASET_PATH.exists():
        raise HTTPException(status_code=503, detail="Eval dataset not configured or file missing")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_dataset(data: dict):
    EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "item"


@app.get("/eval/dataset")
def get_eval_dataset():
    """Return current eval dataset (sections and labels) for dropdowns. 503 if not configured."""
    return _get_dataset()


@app.post("/eval/items")
def add_eval_items(payload: dict):
    """
    Add items to eval dataset. Body: { "section_id": str, "section_label": str (optional for existing),
    "items": [ { "id": str (optional), "query": str, "review_type": "auto"|"human", ... } ] }.
    """
    section_id = (payload.get("section_id") or "").strip()
    section_label = (payload.get("section_label") or section_id or "Uncategorized").strip()
    items = payload.get("items") or []
    if not section_id:
        raise HTTPException(status_code=400, detail="section_id required")
    if not items or not isinstance(items, list):
        raise HTTPException(status_code=400, detail="items must be a non-empty array")

    data = _get_dataset()
    sections = data.get("sections") or []
    section_index = next((i for i, s in enumerate(sections) if s.get("id") == section_id), None)

    normalized = []
    for it in items:
        q = (it.get("query") or "").strip()
        if not q:
            continue
        item_id = (it.get("id") or "").strip() or f"{section_id}-{_slug(q)[:40]}"
        normalized.append({
            "id": item_id,
            "query": q,
            "review_type": it.get("review_type") or "human",
            "expected_contains": it.get("expected_contains"),
            "expected_collections": it.get("expected_collections"),
            "must_not_contain": it.get("must_not_contain"),
            "human_rubric": it.get("human_rubric"),
        })

    if not normalized:
        raise HTTPException(status_code=400, detail="No valid items (query required)")

    if section_index is not None:
        sections[section_index].setdefault("items", []).extend(normalized)
    else:
        sections.append({"id": section_id, "label": section_label, "items": normalized})

    _save_dataset(data)
    return {"ok": True, "section_id": section_id, "added": len(normalized)}
