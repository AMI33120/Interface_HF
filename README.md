# Local Agent — chat UI for your llama.cpp endpoint

A small local web app for chatting with a model served by **llama.cpp's OpenAI-compatible
server** (what you're running for Qwen3.5-35B). It gives the model tools to read/write files
and run shell commands **inside one folder you pick**, shows its thinking live, and always
ends with a short "what changed and why" summary instead of a raw tool transcript.

## 1. Install

```bash
cd hf-local-agent
pip install -r requirements.txt
```

## 2. Make sure your llama.cpp server exposes tool calling

Qwen models need the server started with a Jinja chat template so `tool_calls` actually
populate in the response (otherwise Qwen writes tool calls as plain `<tool_call>{...}</tool_call>`
text instead — the app also handles that fallback automatically, so either way works, but
native support streams more reliably):

```bash
./llama-server -m qwen3.5-35b.gguf --host 127.0.0.1 --port 8080 --jinja -c 8192
```

## 3. Run the app

```bash
python server.py
```

Open **http://127.0.0.1:8765**.

In the sidebar:
- **Endpoint URL** — your llama.cpp server, e.g. `http://127.0.0.1:8080` (the app calls
  `POST {endpoint}/v1/chat/completions`).
- **HF Token** — leave this blank for a local llama.cpp server, it doesn't need auth. Only
  fill it in if `Endpoint URL` points at a **hosted Hugging Face Inference Endpoint**
  (`https://xxxxx.endpoints.huggingface.cloud` or similar) instead of your own machine — those
  require a Bearer token on every request. When set, the app sends
  `Authorization: Bearer <token>` on all calls to the endpoint. It's stored in server memory
  only (never written to disk, never echoed back to the browser after you set it).
- **Model name** — optional, only needed if your server requires it in the request body.
- **Sandbox folder** — click **Browse** to walk your real filesystem and pick a folder, or
  type a path directly. If it doesn't exist yet, it's created for you on Connect.

Then just chat. Enter sends, Shift+Enter for a newline, and the box grows as you type.

## What it does on every request

The model has six tools, all confined to the sandbox folder you picked (path traversal like
`../` or absolute paths outside the folder is rejected server-side, not just in the prompt):

| Tool | Purpose |
|---|---|
| `list_files` | See what's actually in the folder |
| `read_file` | Read a file, with line numbers |
| `search_files` | grep-style search for a topic/string before assuming it's there |
| `write_file` | Create or fully overwrite a file |
| `edit_file` | Exact-match find & replace inside a file |
| `run_terminal` | Run a shell command (cwd = sandbox root) — install libraries, run scripts, syntax-check |

The system prompt requires the model to work this way on every task:
1. Break the request into steps.
2. Before editing anything you mention, first `list_files` (if it hasn't already seen the
   directory) then `search_files`/`read_file` to confirm the file **and** the specific thing
   you're referring to are actually there — it won't just take your word for it.
3. Make the smallest correct change (`edit_file` over `write_file` where possible).
4. After writing, verify: re-read the file and/or run a syntax check or test via
   `run_terminal`.
5. If a task needs logic no tool covers, it writes a small script with `write_file` and runs
   it with `run_terminal` — that's how it builds itself a one-off function on demand.
6. The final reply is always a short **what changed / why / anything to know** summary, never
   a dump of the raw tool calls.

## Thinking, live

If the model wraps reasoning in `<think>...</think>` (Qwen's native format), the app splits
that out of the stream in real time into a small, collapsed-by-default "thinking" panel — a
one-line label with a subtle pulsing dot while it's actively reasoning, in light grey, that
never clutters the main reply. Click it to expand and read the full reasoning; it auto-collapses
again once tool calls or the real answer start, and it's never included in the final response
text itself.

Tool calls stream in as their own collapsible cards (name, arguments, live result) so you can
see exactly what it read, wrote, or ran, and in what order, without it cluttering the answer.

## How long it'll keep working on a task

There's no practical limit on how many tool calls it can make in one turn — it'll keep going
across as many read/write/terminal rounds as the task actually needs, and the system prompt
tells it not to stop until it has verified the whole request is done (re-reading changed files,
running a real check), not just the first step. There's still a high safety ceiling (80 rounds)
purely to catch a genuine infinite loop -- if you ever hit it, that's a sign the model got stuck
repeating itself, not a normal stopping point; try breaking the request into a smaller piece.

## Uploading files

The 📎 button next to the message box lets you upload anything — zips, code files, images,
whatever — straight into the sandbox folder, so the model can see them without you re-pasting
content or the file already needing to be there beforehand.
- **Zip files** are extracted automatically into a folder named after the zip (e.g.
  `project.zip` → `project/`), with the original zip also kept alongside it. Extraction is
  zip-slip-safe: any entry that tries to write outside its target folder (via `../` or an
  absolute path in the archive) is silently skipped rather than followed.
- **Anything else** (images, single files, etc.) is saved as-is at the sandbox root. If a file
  with the same name already exists, the upload is saved as `name (1).ext` instead of
  overwriting it.
- The model is told what just landed in the sandbox automatically (added to the conversation
  transcript), so you can immediately follow up with something like "use the logo in
  `assets/` I just uploaded" without spelling out the exact path yourself.

## Conversation memory across messages

The server keeps the real transcript — every tool call and tool result, not just your messages
and its final replies — so if it already read a file two messages ago, it won't re-read it from
scratch just because you sent a new message. Click **New chat** in the top bar to start over
with a clean transcript (your endpoint/sandbox connection stays intact).

## Previous chats (saved locally, nothing sent anywhere)

Every chat — your messages, its thinking, every tool call and result, its final replies — is
saved as it happens into a single SQLite file at `data/history.db`, right next to `server.py`.
No external database, no service, no network call: it's the Python standard library's built-in
`sqlite3`, so this costs nothing and stays a few KB to a few MB even after heavy use.

The **Previous chats** list in the sidebar shows every past conversation (auto-titled from your
first message), with how many messages and how long ago. Click one to make it the active chat —
this doesn't just replay it visually, it reloads the *actual* message history (including tool
calls and results) into the model's context, so you can keep talking with the full memory of
where you left off. The × on each item deletes that chat's history permanently.

Since chat history is tied to the sandbox folder it was created in, loading an old chat also
switches the active sandbox back to whichever folder it used — if that folder no longer exists
on disk, you'll see a note next to the connection status, and file tools will fail until you
point the sandbox path somewhere valid again.

To back up or wipe your history, the whole thing is just that one file — copy `data/history.db`
elsewhere to back it up, or delete it (with the app stopped) to start completely fresh.

## If it seems stuck just reading files and never acting

This almost always means the model ran out of response tokens mid-turn — often because a
"thinking" model spends its whole budget reasoning about file contents and never gets to the
part where it calls `edit_file` or gives you an answer. Rather than silently treating that as a
blank "done", the app now detects this (empty response + `finish_reason: length`) and shows a
clear error telling you to raise it. Fix: increase **Max response tokens** in the sidebar
(default 4096 — try 8000+ for larger files or multi-step edits) and resend your message.

## Approving terminal commands

Every `run_terminal` call pauses and shows you the exact command inline — like Claude Code's
permission prompts — before it runs. You can:
- **Approve & run** it as-is
- **Edit the command** in the textbox first, then approve (the edited version is what runs)
- **Deny** it, optionally with a reason the model sees and can react to

Nothing else happens while a command is awaiting your decision — the composer is disabled and
the model is paused mid-turn until you respond, then it continues exactly where it left off.

If you don't want to approve every single command, check **"Auto-approve terminal commands"**
in the sidebar (toggleable any time, even mid-conversation). Commands that match an obviously
destructive pattern (`rm -rf /`, `sudo`, `mkfs`, fork bombs, etc.) are still sent to you for
approval either way — they're just flagged with a red "risky pattern" badge so you know to look
twice — auto-approve does not silently wave those through.

## Honest limitations

- **Path sandboxing is enforced** for every file tool — the model cannot read/write outside
  the folder you picked, verified with path-traversal tests during development.
- **`run_terminal` is not a full OS sandbox.** It runs with the sandbox folder as its working
  directory, and every command needs your explicit approval (or auto-approve, if you enabled
  it) — but a shell command can still reference absolute paths elsewhere on your machine (e.g.
  `cat /etc/passwd`) if you approve it, since cwd restriction alone doesn't stop that. Read
  what you're approving. If you want a hard boundary regardless of what gets approved, run this
  whole app inside a container or VM with the sandbox folder as the only writable mount.
- Conversation history is the real transcript (tool calls included), not just final replies —
  see "Conversation memory across messages" above.
- Chat history in `data/history.db` isn't encrypted and has no access control beyond your
  machine's own file permissions — anyone with access to that file (or to this app while it's
  running) can read everything in it, same as any other local app data.
- Single-user, no auth — it's meant to be run locally on your own machine, not exposed on a
  network.

## Files

- `server.py` — FastAPI backend: config, folder browser, sandboxed tools, chat history, and
  the streaming agent loop that talks to your llama.cpp endpoint.
- `static/index.html` — the whole frontend (single file, no build step).
- `data/history.db` — created automatically on first run; your local chat history (see
  "Previous chats" above). Safe to delete any time the app isn't running.
