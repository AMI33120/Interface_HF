"""
Local agentic chat UI for a llama.cpp OpenAI-compatible endpoint (e.g. Qwen3.5-35B).

Run:
    pip install -r requirements.txt
    python server.py
Then open http://127.0.0.1:8765
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ---------------------------------------------------------------------------
# Chat history -- one small local SQLite file, no server, no extra dependency.
# Every user message, tool call, tool result, and thinking trace gets saved as
# it happens so past chats can be reloaded later with full context intact.
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "data" / "history.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    conn = db_conn()
    conn.execute("""CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT,
        sandbox TEXT,
        created_at TEXT,
        updated_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        content TEXT,
        thinking TEXT,
        tool_calls TEXT,
        tool_call_id TEXT,
        created_at TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
    conn.commit()
    conn.close()


db_init()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_new_chat(sandbox: str) -> str:
    chat_id = uuid.uuid4().hex[:12]
    conn = db_conn()
    ts = now_iso()
    conn.execute("INSERT INTO chats (id, title, sandbox, created_at, updated_at) VALUES (?,?,?,?,?)",
                 (chat_id, "New chat", sandbox, ts, ts))
    conn.commit()
    conn.close()
    return chat_id


def db_save_message(chat_id: str, role: str, content: str | None, thinking: str | None = None,
                     tool_calls: list | None = None, tool_call_id: str | None = None):
    if not chat_id:
        return
    conn = db_conn()
    ts = now_iso()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, thinking, tool_calls, tool_call_id, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (chat_id, role, content, thinking, json.dumps(tool_calls) if tool_calls else None, tool_call_id, ts),
    )
    conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id))
    if role == "user":
        row = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
        if row and row["title"] == "New chat":
            title = " ".join((content or "").split())[:60] or "New chat"
            conn.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()


def db_list_chats(limit: int = 100) -> list[dict]:
    conn = db_conn()
    rows = conn.execute(
        """SELECT c.id, c.title, c.sandbox, c.created_at, c.updated_at,
           (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS msg_count
           FROM chats c ORDER BY c.updated_at DESC LIMIT ?""", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_load_convo(chat_id: str) -> list[dict]:
    """Reconstructs the OpenAI-format message list the model actually needs -- this is what
    gives it real memory of a reloaded chat, not just a nice-looking transcript."""
    conn = db_conn()
    rows = conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,)).fetchall()
    conn.close()
    msgs = []
    for r in rows:
        m: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["tool_calls"]:
            m["tool_calls"] = json.loads(r["tool_calls"])
        if r["tool_call_id"]:
            m["tool_call_id"] = r["tool_call_id"]
        msgs.append(m)
    return msgs


def db_render_events(chat_id: str) -> tuple[str, list[dict]]:
    """Turns the stored rows back into the same event shape the frontend already renders
    live (thinking/tool_call/tool_result/final), so loading history reuses the exact same
    UI code as a live turn. Returns (sandbox, events)."""
    conn = db_conn()
    chat = conn.execute("SELECT sandbox FROM chats WHERE id=?", (chat_id,)).fetchone()
    rows = conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,)).fetchall()
    conn.close()
    tool_results = {r["tool_call_id"]: r["content"] for r in rows if r["role"] == "tool" and r["tool_call_id"]}

    events = []
    for r in rows:
        if r["role"] == "user":
            events.append({"type": "user_turn", "text": r["content"] or ""})
        elif r["role"] == "assistant":
            if r["thinking"]:
                events.append({"type": "thinking", "text": r["thinking"]})
            calls = json.loads(r["tool_calls"]) if r["tool_calls"] else None
            if calls:
                for call in calls:
                    fn = call.get("function", {})
                    args = _safe_json(fn.get("arguments", "{}"))
                    cid = call.get("id")
                    events.append({"type": "tool_call", "id": cid, "name": fn.get("name"), "args": args})
                    events.append({"type": "tool_result", "id": cid, "name": fn.get("name"),
                                    "result": tool_results.get(cid, "")})
            else:
                events.append({"type": "final", "text": r["content"] or ""})
    return (chat["sandbox"] if chat else ""), events

# ---------------------------------------------------------------------------
# Global session state. This is a single-user local tool, so one in-memory
# config is intentional (no auth / multi-tenant concerns).
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "base_url": None,   # e.g. http://127.0.0.1:8080  (llama.cpp server, OpenAI-compatible)
    "model": None,      # model name/id llama.cpp reports, only used if the server wants it
    "sandbox": None,    # absolute path the model is confined to
    "auto_approve_terminal": False,  # if False, every run_terminal call pauses for user approval
    "hf_token": None,   # Bearer token -- only needed for a hosted HF Inference Endpoint, not local llama.cpp
    "max_tokens": 4096, # response token budget -- too low silently truncates "thinking" models mid-turn
}

# The real, persistent conversation -- including every tool call and tool result, not just
# final answers. This is what actually gets sent to the model, so it remembers what it already
# read/wrote across separate user messages instead of re-discovering the sandbox every turn.
# Single-user local tool -> one global transcript is intentional.
CONVO: list[dict] = []
CURRENT_CHAT_ID: str | None = None


def convo_append(role: str, content, thinking: str | None = None,
                  tool_calls: list | None = None, tool_call_id: str | None = None):
    """Appends to the in-memory transcript AND persists the same message to SQLite, so a
    server restart or a later 'load previous chat' never loses anything."""
    msg: dict[str, Any] = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    CONVO.append(msg)
    db_save_message(CURRENT_CHAT_ID, role, content, thinking=thinking,
                     tool_calls=tool_calls, tool_call_id=tool_call_id)


# Scratch state for whichever turn is currently in progress (possibly paused on an approval).
SESSION: dict[str, Any] = {
    "queue": [],          # tool calls from the current model turn not yet processed
    "awaiting": None,     # the call dict currently paused on user approval, or None
    "iteration": 0,
}

MAX_TOOL_ITERATIONS = 80  # a high safety ceiling against genuine infinite loops, not a practical
                          # limit -- normal multi-step tasks should never get near this
TERMINAL_TIMEOUT_S = 90

# Best-effort guardrail on top of path sandboxing. Not a substitute for real
# OS-level isolation (see README) but blocks the obviously catastrophic stuff.
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/(\s|$)", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",
    r"\bsudo\b", r">\s*/dev/sd", r"\bshutdown\b", r"\breboot\b", r"\bchown\s+-R\s+/",
    r"\bchmod\s+-R\s+777\s+/", r"\bmv\s+/\S+\s+/dev/null",
]


def sandbox_root() -> Path:
    if not CONFIG["sandbox"]:
        raise ValueError("No sandbox folder set yet")
    return Path(CONFIG["sandbox"]).resolve()


def resolve_in_sandbox(rel_path: str) -> Path:
    """Resolve a model-supplied path against the sandbox root, refusing escapes."""
    root = sandbox_root()
    p = Path(rel_path or ".")
    if p.is_absolute():
        # Treat an absolute path as relative-to-root instead of silently
        # overriding it (pathlib would otherwise let '/etc/passwd' win the join).
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
    target = (root / p).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Refused: '{rel_path}' resolves outside the sandbox folder ({root})")
    return target


# ---------------------------------------------------------------------------
# Tools available to the model
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders inside the sandbox, relative to its root. Always call this before editing anything if you haven't already listed the relevant directory in this conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to the sandbox root. Defaults to '.' (the root)."},
                    "recursive": {"type": "boolean", "description": "Recurse into subfolders. Defaults to true."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file's full contents, with line numbers, from inside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path relative to the sandbox root."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents (like grep) for a topic/string across the sandbox, or a subfolder of it. Use this to confirm a topic actually exists in the codebase before claiming it does, or before editing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex to search for."},
                    "path": {"type": "string", "description": "Directory to search in, relative to sandbox root. Defaults to '.'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or fully overwrite an existing one inside the sandbox. Creates parent folders as needed. Use edit_file instead if you only need to change part of an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the sandbox root."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact, unique substring in an existing file with new text. Fails loudly if old_str isn't found or isn't unique -- read the file first to get the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string", "description": "Exact existing text to replace, must appear exactly once."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Run a shell command with the sandbox folder as the working directory. Use this to install libraries (pip/npm), run scripts you wrote, or verify a file (syntax-check / run tests) after writing it. You can also write a throwaway script with write_file and execute it here if no existing tool covers what you need -- that's how you build yourself a one-off function.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run inside the sandbox root."}},
                "required": ["command"],
            },
        },
    },
]


def tool_list_files(args: dict) -> str:
    root = sandbox_root()
    start = resolve_in_sandbox(args.get("path", "."))
    recursive = args.get("recursive", True)
    if not start.exists():
        return f"ERROR: '{args.get('path', '.')}' does not exist in the sandbox."
    lines = []
    if recursive:
        for p in sorted(start.rglob("*")):
            if any(part.startswith((".git", "node_modules", "__pycache__", ".venv")) for part in p.parts):
                continue
            rel = p.relative_to(root)
            lines.append(f"{'DIR ' if p.is_dir() else 'FILE'}  {rel}")
    else:
        for p in sorted(start.iterdir()):
            rel = p.relative_to(root)
            lines.append(f"{'DIR ' if p.is_dir() else 'FILE'}  {rel}")
    return "\n".join(lines) if lines else "(empty)"


def tool_read_file(args: dict) -> str:
    target = resolve_in_sandbox(args["path"])
    if not target.exists() or not target.is_file():
        return f"ERROR: file '{args['path']}' does not exist in the sandbox."
    try:
        text = target.read_text(errors="replace")
    except Exception as e:
        return f"ERROR reading file: {e}"
    numbered = "\n".join(f"{i+1:>5}\t{line}" for i, line in enumerate(text.splitlines()))
    return numbered if numbered else "(empty file)"


def tool_search_files(args: dict) -> str:
    root = sandbox_root()
    start = resolve_in_sandbox(args.get("path", "."))
    query = args["query"]
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    hits = []
    for p in start.rglob("*") if start.is_dir() else [start]:
        if p.is_dir() or any(part.startswith((".git", "node_modules", "__pycache__", ".venv")) for part in p.parts):
            continue
        try:
            for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{p.relative_to(root)}:{i}: {line.strip()[:200]}")
        except Exception:
            continue
        if len(hits) >= 200:
            break
    if not hits:
        return f"No matches for '{query}'."
    return "\n".join(hits[:200])


def tool_write_file(args: dict) -> str:
    target = resolve_in_sandbox(args["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(args["content"])
    return f"{'Overwrote' if existed else 'Created'} '{args['path']}' ({len(args['content'])} chars)."


def tool_edit_file(args: dict) -> str:
    target = resolve_in_sandbox(args["path"])
    if not target.exists():
        return f"ERROR: file '{args['path']}' does not exist."
    text = target.read_text()
    count = text.count(args["old_str"])
    if count == 0:
        return "ERROR: old_str not found in file. Re-read the file to get exact text."
    if count > 1:
        return f"ERROR: old_str appears {count} times, must be unique. Add more surrounding context."
    target.write_text(text.replace(args["old_str"], args["new_str"], 1))
    return f"Edited '{args['path']}': replaced 1 occurrence."


def tool_run_terminal(args: dict) -> str:
    command = args["command"]
    root = sandbox_root()
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(root), capture_output=True, text=True,
            timeout=TERMINAL_TIMEOUT_S,
        )
        out = f"$ {command}\n(exit {result.returncode})\n"
        if result.stdout:
            out += f"--- stdout ---\n{result.stdout[-4000:]}\n"
        if result.stderr:
            out += f"--- stderr ---\n{result.stderr[-4000:]}\n"
        return out
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {TERMINAL_TIMEOUT_S}s."
    except Exception as e:
        return f"ERROR running command: {e}"


def is_risky_command(command: str) -> bool:
    """Flag-only now (no auto-block) -- surfaced to the user in the approval prompt."""
    return any(re.search(pat, command) for pat in DANGEROUS_PATTERNS)


def unique_dest(dest_dir: Path, filename: str) -> Path:
    """Avoids clobbering an existing file: foo.txt -> foo (1).txt -> foo (2).txt ..."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while True:
        candidate = dest_dir / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[str]:
    """Extracts a zip into dest_dir, rejecting any entry that would resolve outside it
    (a "zip slip" attack via '../' or absolute paths in the archive)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if dest_dir.resolve() not in target.parents and target != dest_dir.resolve():
                continue  # skip anything that would escape the extraction folder
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(member.filename)
    return extracted


TOOL_IMPL = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search_files": tool_search_files,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "run_terminal": tool_run_terminal,
}


SYSTEM_PROMPT = """You are a local coding agent with direct, sandboxed access to one folder on the user's \
machine through tools: list_files, read_file, search_files, write_file, edit_file, run_terminal. \
You can NEVER access anything outside that folder -- every tool enforces this, so don't try workarounds like '..' or absolute paths.

Working method, always:
1. Break the user's request into a short internal list of concrete steps before acting.
2. Before touching any file the user mentions: call list_files (if you haven't already seen the relevant \
directory in this conversation) to confirm the file actually exists, then use search_files or read_file to \
confirm the specific topic/content the user is referring to is really there. Never assume -- check.
3. Make the smallest correct change. Prefer edit_file over write_file when only part of a file needs to change.
4. After writing or editing a file, verify it: re-read it and/or run an appropriate check with run_terminal \
(e.g. `python -m py_compile file.py`, `node --check file.js`, running relevant tests). If verification fails, fix it before finishing.
5. If a task needs logic no existing tool covers, write a small script with write_file and run it with run_terminal \
-- that is how you create yourself a one-off function. Use run_terminal to install any libraries you need first.
6. Never ask the user to re-paste file contents you can just read yourself.

Multi-step tasks: keep calling tools across as many turns as the task genuinely needs -- there is \
no meaningful limit, so don't rush or truncate scope to fit a smaller number of steps. Work through \
your whole internal step list before stopping. Only produce a final answer (no tool call) once you \
have actually verified -- by re-reading the changed file(s) and/or running a real check via \
run_terminal -- that every part of the user's request is done, not just the first step of it. If a \
check fails or something is still incomplete, keep going and fix it rather than reporting it as done.

If your model supports it, put your step-by-step reasoning inside <think>...</think> tags. Keep the final \
answer (outside the think tags) separate from that reasoning.

Your FINAL reply (after any tool calls) must be a short, clearly framed summary, not a transcript of your tool \
calls. Structure it as:
- What changed: the files touched, one line each
- Why: the reasoning behind each change, briefly
- Verified: what you checked and how you confirmed it actually works
- Anything the user should know (follow-ups, risks)

Be concise. Do not restate the user's request back to them."""


def normalize_hf_token(raw: str) -> str | None:
    """Cleans up common copy-paste mistakes: surrounding whitespace/newlines/quotes, or
    pasting the whole header value ('Bearer hf_xxx') instead of just the token."""
    if not raw:
        return None
    tok = raw.strip().strip('"').strip("'").strip()
    if tok.lower().startswith("bearer "):
        tok = tok[len("bearer "):].strip()
    return tok or None


def build_client() -> httpx.AsyncClient:
    headers = {}
    if CONFIG.get("hf_token"):
        headers["Authorization"] = f"Bearer {CONFIG['hf_token']}"
    # No cookie jar: some hosted endpoints sit behind a proxy/load balancer that sets a
    # session-affinity cookie on every response. A shared client reused across many tool-loop
    # requests would otherwise re-send and re-accumulate that cookie until it exceeds the
    # proxy's header size limit ("400 Request Header Or Cookie Too Large"). We don't need
    # session affinity for stateless chat-completions calls, so just never store cookies.
    return httpx.AsyncClient(base_url=CONFIG["base_url"], timeout=None, headers=headers,
                              cookies=httpx.Cookies())


def sse(event_type: str, **payload) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


async def call_llm_stream(client: httpx.AsyncClient, messages: list[dict]):
    """POST to the llama.cpp OpenAI-compatible endpoint and yield raw SSE JSON chunks."""
    body = {
        "messages": messages,
        "stream": True,
        "tools": TOOLS_SCHEMA,
        "temperature": 0.3,
        "max_tokens": CONFIG.get("max_tokens") or 4096,
    }
    if CONFIG.get("model"):
        body["model"] = CONFIG["model"]
    # Belt-and-suspenders: clear anything the jar picked up from a prior response before
    # sending the next request, in case some intermediary sets cookies despite the empty jar.
    client.cookies.clear()
    async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
        if resp.status_code == 401 or resp.status_code == 403:
            err = await resp.aread()
            has_token = bool(CONFIG.get("hf_token"))
            hint = (
                "no HF Token is set -- if this is a hosted HF Inference Endpoint (not your own "
                "local llama.cpp server), it needs one in the sidebar."
                if not has_token else
                "an HF Token is set but the endpoint rejected it -- check it's current/valid "
                "for this specific endpoint, hasn't expired, and was pasted without an extra "
                "'Bearer ' prefix or stray whitespace."
            )
            raise RuntimeError(f"llama.cpp server returned {resp.status_code} (auth rejected): "
                                f"{hint} Raw response: {err.decode(errors='replace')[:300]}")
        if resp.status_code >= 400:
            err = await resp.aread()
            raise RuntimeError(f"llama.cpp server returned {resp.status_code}: {err.decode(errors='replace')[:500]}")
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def flush_think_buffer(state: dict, flush_all: bool = False) -> list[tuple[str, str]]:
    """Incrementally split streamed text into ('content', ...) / ('thinking', ...) events."""
    events = []
    buf = state["buf"]
    while True:
        if not state["in_think"]:
            idx = buf.find(THINK_OPEN)
            if idx == -1:
                safe_len = len(buf) if flush_all else max(0, len(buf) - len(THINK_OPEN))
                if safe_len:
                    events.append(("content", buf[:safe_len]))
                    buf = buf[safe_len:]
                break
            if idx > 0:
                events.append(("content", buf[:idx]))
            buf = buf[idx + len(THINK_OPEN):]
            state["in_think"] = True
        else:
            idx = buf.find(THINK_CLOSE)
            if idx == -1:
                safe_len = len(buf) if flush_all else max(0, len(buf) - len(THINK_CLOSE))
                if safe_len:
                    events.append(("thinking", buf[:safe_len]))
                    buf = buf[safe_len:]
                break
            if idx > 0:
                events.append(("thinking", buf[:idx]))
            buf = buf[idx + len(THINK_CLOSE):]
            state["in_think"] = False
    state["buf"] = buf
    return events


# Fallback for servers/templates that emit tool calls as plain text like
# Qwen's Hermes-style <tool_call>{"name": ..., "arguments": {...}}</tool_call>
TEXT_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_text_tool_calls(content: str) -> tuple[str, list[dict]]:
    calls = []
    for m in TEXT_TOOLCALL_RE.finditer(content):
        try:
            obj = json.loads(m.group(1))
            calls.append({
                "id": f"textcall_{len(calls)}_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": obj.get("name"), "arguments": json.dumps(obj.get("arguments", {}))},
            })
        except json.JSONDecodeError:
            continue
    cleaned = TEXT_TOOLCALL_RE.sub("", content)
    return cleaned, calls


async def model_turn(client: httpx.AsyncClient, messages: list[dict]):
    """Stream one model turn, yielding SSE lines live, then yield a final
    ('__final__', text, thinking) / ('__calls__', content, calls, thinking) / ('__truncated__',) tuple."""
    think_state = {"buf": "", "in_think": False}
    visible_content = ""  # content with <think>...</think> stripped -- the real answer text
    thinking_accum = ""   # the raw <think> text, kept separately so it can be persisted to history
    tool_call_acc: dict[int, dict] = {}
    finish_reason = None

    async for chunk in call_llm_stream(client, messages):
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        if delta.get("content"):
            think_state["buf"] += delta["content"]
            for etype, text in flush_think_buffer(think_state):
                if etype == "content":
                    visible_content += text
                else:
                    thinking_accum += text
                yield sse(etype, text=text)

        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_call_acc.setdefault(idx, {"id": tc.get("id"), "name": "", "arguments": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]

    for etype, text in flush_think_buffer(think_state, flush_all=True):
        if etype == "content":
            visible_content += text
        else:
            thinking_accum += text
        yield sse(etype, text=text)

    cleaned_visible, text_calls_raw = extract_text_tool_calls(visible_content)
    native_calls = [
        {"id": v["id"] or f"call_{i}", "name": v["name"], "args": _safe_json(v["arguments"])}
        for i, v in sorted(tool_call_acc.items())
    ]
    text_calls = [
        {"id": c["id"], "name": c["function"]["name"], "args": _safe_json(c["function"]["arguments"])}
        for c in text_calls_raw
    ]
    all_calls = native_calls + text_calls

    if not all_calls and not cleaned_visible.strip() and finish_reason == "length":
        # The model burned its whole token budget (often on <think> reasoning) and got cut off
        # before producing calls or an answer. Treating this as a real "final" would push an
        # empty turn into history and silently erase all progress for the next message.
        yield ("__truncated__",)
    elif not all_calls:
        yield ("__final__", cleaned_visible.strip(), thinking_accum or None)
    else:
        yield ("__calls__", cleaned_visible or None, all_calls, thinking_accum or None)


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


def execute_call(call: dict) -> str:
    impl = TOOL_IMPL.get(call["name"])
    if not impl:
        return f"ERROR: unknown tool '{call['name']}'"
    try:
        return impl(call["args"])
    except Exception as e:
        return f"ERROR: {e}"


def reset_session():
    """Clear scratch state for the in-progress turn. Does NOT touch CONVO -- the persistent
    transcript survives across turns on purpose."""
    SESSION["queue"] = []
    SESSION["awaiting"] = None
    SESSION["iteration"] = 0


async def drive():
    """Resumable agent loop over the persistent CONVO transcript. Reads/writes the
    module-level SESSION scratch state so a paused run_terminal approval can be resumed
    later by a separate HTTP request."""
    try:
        async with build_client() as client:
            while True:
                # Work through any tool calls already queued from the last model turn.
                while SESSION["queue"]:
                    call = SESSION["queue"].pop(0)
                    if call["name"] == "run_terminal" and not CONFIG["auto_approve_terminal"]:
                        SESSION["awaiting"] = call
                        yield sse("confirm_request", id=call["id"], name=call["name"], args=call["args"],
                                   risky=is_risky_command(call["args"].get("command", "")))
                        return  # pause here -- /api/chat/respond resumes the loop
                    yield sse("tool_call", id=call["id"], name=call["name"], args=call["args"])
                    result = execute_call(call)
                    yield sse("tool_result", id=call["id"], name=call["name"], result=result)
                    convo_append("tool", result, tool_call_id=call["id"])

                if SESSION["iteration"] >= MAX_TOOL_ITERATIONS:
                    yield sse("error", message=(
                        f"Hit the safety ceiling of {MAX_TOOL_ITERATIONS} tool-call rounds in one turn "
                        f"without reaching a verified final answer. This isn't a normal stopping point -- "
                        f"it usually means the model got stuck repeating itself rather than making "
                        f"progress. Try breaking your request into a smaller step and resending."
                    ))
                    yield sse("done")
                    reset_session()
                    return
                SESSION["iteration"] += 1

                try:
                    result = None
                    async for item in model_turn(client, CONVO):
                        if isinstance(item, str):
                            yield item  # an sse(...) line -- pass straight through
                        else:
                            result = item
                except Exception as e:
                    yield sse("error", message=f"Connection to model failed: {e}")
                    yield sse("done")
                    reset_session()
                    return

                if result[0] == "__truncated__":
                    yield sse("error", message=(
                        f"The model ran out of response tokens (currently capped at "
                        f"{CONFIG.get('max_tokens') or 4096}) before it finished reasoning or "
                        f"acting -- it produced no answer and no tool call. Try raising "
                        f"'Max response tokens' in the sidebar and resending your last message."
                    ))
                    yield sse("done")
                    reset_session()
                    return
                elif result[0] == "__final__":
                    _, text, thinking = result
                    convo_append("assistant", text, thinking=thinking)
                    yield sse("final", text=text)
                    yield sse("done")
                    reset_session()
                    return
                else:
                    _, content, calls, thinking = result
                    tool_calls_payload = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                        for c in calls
                    ]
                    convo_append("assistant", content, thinking=thinking, tool_calls=tool_calls_payload)
                    SESSION["queue"] = calls
    except GeneratorExit:
        raise
    except Exception as e:
        yield sse("error", message=f"Unexpected agent error: {e}")
        yield sse("done")
        reset_session()


def start_turn(user_text: str):
    if not CONFIG["base_url"] or not CONFIG["sandbox"]:
        async def err():
            yield sse("error", message="Set the endpoint URL and sandbox folder first.")
            yield sse("done")
        return err()
    global CURRENT_CHAT_ID
    if not CONVO:
        if not CURRENT_CHAT_ID:
            CURRENT_CHAT_ID = db_new_chat(CONFIG["sandbox"])
        CONVO.append({"role": "system", "content": SYSTEM_PROMPT})  # not persisted -- regenerated fresh on load
    convo_append("user", user_text)
    SESSION["queue"] = []
    SESSION["awaiting"] = None
    SESSION["iteration"] = 0
    return drive()


def resume_turn(call_id: str, approved: bool, edited_command, note):
    pending = SESSION.get("awaiting")
    if not pending:
        async def err():
            yield sse("error", message="Nothing is awaiting approval.")
            yield sse("done")
        return err()
    if pending["id"] != call_id:
        async def err():
            yield sse("error", message="That approval request is no longer current.")
            yield sse("done")
        return err()

    SESSION["awaiting"] = None

    async def resumed():
        if approved:
            if edited_command:
                pending["args"]["command"] = edited_command
            yield sse("tool_call", id=pending["id"], name=pending["name"], args=pending["args"])
            result = execute_call(pending)
            yield sse("tool_result", id=pending["id"], name=pending["name"], result=result)
        else:
            result = "User denied permission to run this command." + (f" Reason: {note}" if note else "")
            yield sse("tool_result", id=pending["id"], name=pending["name"], result=result, denied=True)
        convo_append("tool", result, tool_call_id=pending["id"])
        async for line in drive():
            yield line

    return resumed()


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.post("/api/config")
async def set_config(req: Request):
    global CURRENT_CHAT_ID
    body = await req.json()
    base_url = (body.get("base_url") or "").rstrip("/")
    sandbox = body.get("sandbox") or ""
    model = body.get("model") or ""

    if not base_url:
        return JSONResponse({"ok": False, "error": "Endpoint URL is required."}, status_code=400)

    sb_path = Path(sandbox).expanduser()
    if body.get("create_if_missing") and not sb_path.exists():
        sb_path.mkdir(parents=True, exist_ok=True)
    if not sb_path.exists() or not sb_path.is_dir():
        return JSONResponse({"ok": False, "error": f"Folder does not exist: {sb_path}"}, status_code=400)

    CONFIG["base_url"] = base_url
    CONFIG["model"] = model
    CONFIG["sandbox"] = str(sb_path.resolve())
    CONFIG["hf_token"] = normalize_hf_token(body.get("hf_token") or "")
    if "auto_approve_terminal" in body:
        CONFIG["auto_approve_terminal"] = bool(body["auto_approve_terminal"])
    if body.get("max_tokens"):
        try:
            CONFIG["max_tokens"] = max(256, int(body["max_tokens"]))
        except (TypeError, ValueError):
            pass
    # A (re)connect means a new/changed sandbox -- start the transcript over so the model
    # isn't carrying context about a different project or a stale connection. A new chat row
    # is created lazily once the first message is actually sent (see start_turn).
    CONVO.clear()
    CURRENT_CHAT_ID = None
    reset_session()
    return {"ok": True, "sandbox": CONFIG["sandbox"]}


@app.post("/api/settings")
async def set_settings(req: Request):
    """Lightweight setter for auto-approve/max_tokens, callable mid-session without re-connecting."""
    body = await req.json()
    if "auto_approve_terminal" in body:
        CONFIG["auto_approve_terminal"] = bool(body["auto_approve_terminal"])
    if body.get("max_tokens"):
        try:
            CONFIG["max_tokens"] = max(256, int(body["max_tokens"]))
        except (TypeError, ValueError):
            pass
    return {"ok": True, "auto_approve_terminal": CONFIG["auto_approve_terminal"], "max_tokens": CONFIG["max_tokens"]}


@app.post("/api/reset")
async def reset_conversation():
    """Manual 'New chat' -- clears the transcript but keeps the endpoint/sandbox connected.
    A new chat row is created lazily once the first message is sent."""
    global CURRENT_CHAT_ID
    CONVO.clear()
    CURRENT_CHAT_ID = None
    reset_session()
    return {"ok": True}


@app.get("/api/chats")
async def api_list_chats():
    return {"ok": True, "chats": db_list_chats(), "active_id": CURRENT_CHAT_ID}


@app.get("/api/chats/{chat_id}")
async def api_get_chat(chat_id: str):
    """Read-only preview of a past chat -- does not change the active session."""
    sandbox, events = db_render_events(chat_id)
    if not sandbox and not events:
        return JSONResponse({"ok": False, "error": "Chat not found."}, status_code=404)
    return {"ok": True, "sandbox": sandbox, "events": events}


@app.post("/api/chats/{chat_id}/load")
async def api_load_chat(chat_id: str):
    """Makes a past chat the active conversation -- the model picks up with full memory of
    everything in it, not just a visual replay."""
    global CURRENT_CHAT_ID
    conn = db_conn()
    row = conn.execute("SELECT sandbox FROM chats WHERE id=?", (chat_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse({"ok": False, "error": "Chat not found."}, status_code=404)

    CONFIG["sandbox"] = row["sandbox"]
    CURRENT_CHAT_ID = chat_id
    CONVO.clear()
    CONVO.append({"role": "system", "content": SYSTEM_PROMPT})
    CONVO.extend(db_load_convo(chat_id))
    reset_session()

    sandbox_exists = Path(row["sandbox"]).is_dir() if row["sandbox"] else False
    _, events = db_render_events(chat_id)
    return {"ok": True, "sandbox": row["sandbox"], "sandbox_exists": sandbox_exists, "events": events}


@app.delete("/api/chats/{chat_id}")
async def api_delete_chat(chat_id: str):
    global CURRENT_CHAT_ID
    conn = db_conn()
    conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()
    if CURRENT_CHAT_ID == chat_id:
        CURRENT_CHAT_ID = None
        CONVO.clear()
        reset_session()
    return {"ok": True}


@app.get("/api/config")
async def get_config():
    safe = dict(CONFIG)
    if safe.get("hf_token"):
        safe["hf_token"] = "set"
    return safe


@app.get("/api/browse")
async def browse(path: str = ""):
    """Lets the frontend folder-picker walk the real filesystem to choose a sandbox root."""
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve()
        if not target.exists():
            target = Path.home()
        entries = sorted(
            [p.name for p in target.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=str.lower,
        )
        return {"path": str(target), "parent": str(target.parent), "dirs": entries}
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    if not CONFIG["sandbox"]:
        return JSONResponse({"ok": False, "error": "Connect to a sandbox folder first."}, status_code=400)
    root = sandbox_root()
    results = []
    for f in files:
        raw_name = Path(f.filename or "upload").name  # strips any directory components the client sent
        if not raw_name:
            continue
        content = await f.read()
        if raw_name.lower().endswith(".zip"):
            zip_dest = unique_dest(root, raw_name)
            zip_dest.write_bytes(content)
            extract_dir = unique_dest(root, Path(raw_name).stem)
            try:
                members = safe_extract_zip(zip_dest, extract_dir)
                results.append({
                    "name": raw_name, "type": "zip", "saved_as": zip_dest.name,
                    "extracted_to": extract_dir.relative_to(root).as_posix(), "file_count": len(members),
                })
            except zipfile.BadZipFile:
                zip_dest.unlink(missing_ok=True)
                results.append({"name": raw_name, "type": "zip", "error": "Not a valid zip file."})
        else:
            dest = unique_dest(root, raw_name)
            dest.write_bytes(content)
            results.append({"name": raw_name, "type": "file", "saved_as": dest.name, "size": len(content)})

    # Note the upload in the persistent transcript so the model knows this material exists
    # without you having to describe every filename yourself in your next message.
    if results:
        if not CONVO:
            CONVO.append({"role": "system", "content": SYSTEM_PROMPT})
        lines = []
        for r in results:
            if r.get("error"):
                lines.append(f"- {r['name']}: upload failed ({r['error']})")
            elif r["type"] == "zip":
                lines.append(f"- {r['name']}: saved as {r['saved_as']}, extracted to {r['extracted_to']}/ "
                              f"({r['file_count']} files)")
            else:
                lines.append(f"- {r['name']}: saved as {r['saved_as']} ({r['size']} bytes)")
        note = "[The user just uploaded the following into the sandbox folder:]\n" + "\n".join(lines)
        CONVO.append({"role": "user", "content": note})

    return {"ok": True, "results": results}


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    user_text = body.get("message", "")
    return StreamingResponse(start_turn(user_text), media_type="text/event-stream")


@app.post("/api/chat/respond")
async def chat_respond(req: Request):
    body = await req.json()
    call_id = body.get("id")
    approved = bool(body.get("approved"))
    edited_command = body.get("edited_command") or None
    note = body.get("note") or None
    return StreamingResponse(resume_turn(call_id, approved, edited_command, note), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
