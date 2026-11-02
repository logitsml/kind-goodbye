"""Driver for the Codex CLI: `codex exec` (a new session, or a fork of a stored one) and `codex app-server` (fork a stored session
without a turn, then compact the fork).

Every call runs the given Codex binary with CODEX_HOME set to the given home, the prompt on stdin and `--json` events on stdout. A call
returns a dict: thread_id, text (the last agent message), usage, the events' types, the session file ("rollout") path, rate_limits (read
from the session file's last token_count event), meta (model, effort, sandbox, client version, ...), the tool calls of the turn and the
call's timestamps. A failed call raises CodexError with a kind: "usage" (the plan's usage window is used up), "throttle" (short-term
request throttling), "transient" (network or server trouble) or "fatal".

The Codex home is meant to be an isolated directory that holds only a link to the user's auth.json, so that no personal Codex
configuration, instructions or memories reach the model. Nothing here reads or prints auth.json.

Defaults (each can be overridden by the caller):
    KIND_GOODBYE_CODEX_BIN    the Codex binary (default: `codex` on PATH)
    KIND_GOODBYE_CODEX_HOME   the isolated Codex home (default: ~/.config/agent_api)
"""

import glob
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

ENV_PREFIX = "KIND_GOODBYE_"
DEFAULT_BIN = os.environ.get("KIND_GOODBYE_CODEX_BIN", "codex")
DEFAULT_HOME = Path(os.environ.get("KIND_GOODBYE_CODEX_HOME", "~/.config/agent_api")).expanduser()

USAGE_RE = re.compile(r"usage limit|you've hit your|limit reached|quota|rate_limit_reached|purchase more credits", re.I)  # the plan's window cap
THROTTLE_RE = re.compile(r"429|too many requests|rate limit", re.I)  # short-term request throttling (Codex retries it itself)
TRANSIENT_RE = re.compile(r"timed? ?out|connection|stream (error|disconnected)|5\d\d|overloaded|unavailable|reset by peer|EOF|temporar", re.I)
BASE_CONFIG = ["check_for_update_on_startup=false"]  # on every call: no update check (the client is pinned by path)


class CodexError(Exception):
    def __init__(self, kind, msg):
        super().__init__(f"{kind}: {msg}")
        self.kind = kind  # "usage", "throttle", "transient" or "fatal"


def classify(text):
    if USAGE_RE.search(text or ""):
        return "usage"
    if THROTTLE_RE.search(text or ""):
        return "throttle"
    if TRANSIENT_RE.search(text or ""):
        return "transient"
    return "fatal"


def env(home):
    """The subprocess environment: CODEX_HOME set to home; API-key variables dropped, so Codex runs on its own (ChatGPT) sign-in; this
    package's own KIND_GOODBYE_* settings dropped, so the model never sees them."""
    e = {k: v for k, v in os.environ.items()
         if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY") and not k.startswith(ENV_PREFIX)}
    e.setdefault("RUST_LOG", "warn")  # Codex's own retry warnings (e.g. 429 Too Many Requests) reach stderr; logging only
    return {**e, "CODEX_HOME": str(home)}


def throttle_retries(stderr):
    """Codex's internal sampling retries in a call's stderr: (count, count with status 429)."""
    n, n429 = 0, 0
    for line in (stderr or "").splitlines():
        if "retrying sampling request" in line:
            n += 1
            n429 += "429" in line
    return n, n429


def rollout_path(thread_id, home, wait=5.0):
    """The session file of a thread: $CODEX_HOME/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl (None if not found in time)."""
    pat = str(Path(home) / "sessions" / "*" / "*" / "*" / f"rollout-*-{thread_id}.jsonl")
    t0 = time.time()
    while True:
        hits = sorted(glob.glob(pat))
        if hits or time.time() - t0 > wait:
            return Path(hits[-1]) if hits else None
        time.sleep(0.2)


def read_rollout(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def last_rate_limits(rows):
    for r in reversed(rows):
        p = r.get("payload") or {}
        if r.get("type") == "event_msg" and p.get("type") == "token_count" and p.get("rate_limits"):
            return p["rate_limits"]
    return None


def last_turn_usage(rows):
    """Token usage of the session file's last turn (summed over its sampling requests). The --json stream's turn.completed reports the
    thread's cumulative total, which for a fork includes the parent's."""
    for r in reversed(rows):
        p = r.get("payload") or {}
        if r.get("type") == "token_usage_record" and p.get("turn_token_usage"):
            return p["turn_token_usage"]
    return None


def rollout_meta(rows):
    meta = {}
    for r in rows:
        p = r.get("payload") or {}
        if r.get("type") == "session_meta":
            meta.update({"client": p.get("cli_version"), "originator": p.get("originator"), "forked_from_id": p.get("forked_from_id"),
                         "history_mode": p.get("history_mode")})
        elif r.get("type") == "turn_context":
            meta.update({"model": p.get("model"), "effort": p.get("effort"), "sandbox": (p.get("sandbox_policy") or {}).get("type"),
                         "sandbox_policy": p.get("sandbox_policy"), "approval": p.get("approval_policy"),
                         "network": (p.get("permission_profile") or {}).get("network"),
                         "collaboration_effort": ((p.get("collaboration_mode") or {}).get("settings") or {}).get("reasoning_effort")})
    return meta


def tool_items(rows):
    """Tool and command use in a session file's own items (a fork's file holds only its own turn): each call's type, name and input
    (truncated), and the head of each output."""
    out = []
    for r in rows:
        p = r.get("payload") or {}
        if r.get("type") != "response_item" or p.get("type") in ("message", "reasoning", None):
            continue
        t = p.get("type")
        if t.endswith("_output"):
            o = p.get("output")
            out.append({"type": t, "call_id": p.get("call_id"), "output_head": (o if isinstance(o, str) else json.dumps(o))[:600]})
        else:
            inp = p.get("arguments", p.get("input", p.get("action")))
            out.append({"type": t, "name": p.get("name"), "call_id": p.get("call_id"),
                        "input": (inp if isinstance(inp, str) else json.dumps(inp))[:2000] if inp is not None else None})
    return out


def compactions(rows):
    """The compaction records of a session file: each with its replacement_history (what the session carries forward) and the plain
    text found in it."""
    out = []
    for i, r in enumerate(rows):
        if r.get("type") == "compacted":
            p = r.get("payload") or {}
            out.append({"line": i, "message": p.get("message"), "replacement_history": p.get("replacement_history"),
                        "metadata": p.get("replacement_history_metadata"), "text": memory_text(p)})
    return out


def texts(x):
    """Every text field in a message's content (strings, lists and dicts walked)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [t for v in x for t in texts(v)]
    if isinstance(x, dict):
        if x.get("type") in ("input_text", "output_text", "text", "summary_text") and isinstance(x.get("text"), str):
            return [x["text"]]
        return [t for k, v in x.items() if k in ("content", "summary", "text", "message") for t in texts(v)]
    return []


def memory_text(payload):
    """What a compaction record carries, as text: the retained user messages verbatim, in order (context messages such as
    <environment_context> and developer instructions left out), any assistant message, and a note of any opaque compaction item."""
    parts = []
    for item in payload.get("replacement_history") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "user":
            t = "\n".join(texts(item.get("content")))
            if not t.lstrip().startswith("<"):
                parts.append("[User]: " + t)
        elif item.get("type") == "message" and item.get("role") == "assistant":
            parts.append("[Assistant]: " + "\n".join(texts(item.get("content"))))
        elif item.get("type") in ("compaction", "compaction_summary", "context_compaction"):
            txt = texts(item)
            parts.append("[Summary]: " + "\n".join(txt) if txt else f"[Summary]: <encrypted, {len(item.get('encrypted_content') or '')} chars>")
    return "\n\n".join(parts)


def memory_shape(payload):
    """A structural description of a compaction record's replacement_history: per item its type, role, length and first 60 characters,
    or for an opaque item its encrypted length."""
    out = []
    for item in payload.get("replacement_history") or []:
        if item.get("type") == "message":
            t = "\n".join(texts(item.get("content")))
            out.append({"type": "message", "role": item.get("role"), "chars": len(t), "head": t[:60]})
        else:
            out.append({"type": item.get("type"), "encrypted_chars": len(item.get("encrypted_content") or ""),
                        "plain_text": bool(texts(item))})
    return out


def _cmd(codex_bin, kind, model, cwd, thread_id=None, config=(), extra=()):
    base = [codex_bin, "exec"]
    if kind == "fork":
        base += ["fork"]
    opts = ["--json", "-m", model, "--skip-git-repo-check"]
    for c in BASE_CONFIG + list(config):
        opts += ["-c", c]
    if kind == "new":
        opts += ["-C", str(cwd)]
    return base + opts + list(extra) + ([thread_id] if thread_id else []) + ["-"]


def exec_call(kind, prompt, model, cwd, *, codex_bin, home, thread_id=None, config=(), timeout=900, extra=()):
    """One `codex exec` call: kind "new" (a new session in cwd) or "fork" (a new turn on a fork of thread_id). A fork runs with cwd as
    the process's working directory (`exec fork` takes no -C). Returns the call record; raises CodexError on failure."""
    cmd = _cmd(codex_bin, kind, model, cwd, thread_id, config, extra=extra)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd), env=env(home))
    try:
        out, err = p.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise CodexError("transient", f"timeout after {timeout}s")
    events, tid, text, usage, errs, reconnects = [], None, None, None, [], 0
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(e.get("type"))
        if e.get("type") == "thread.started":
            tid = e.get("thread_id")
        elif e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message":
            text = e["item"].get("text")
        elif e.get("type") == "turn.completed":
            usage = e.get("usage")
        elif e.get("type") == "error" and str(e.get("message", "")).startswith("Reconnecting"):
            reconnects += 1  # Codex retrying a sampling request itself (e.g. after a 429); not a failure
        elif e.get("type") in ("error", "turn.failed"):
            errs.append(json.dumps(e)[:600])
    if p.returncode != 0 or errs or text is None:
        msg = " | ".join(errs) + " | " + err[-600:]
        raise CodexError(classify(msg), f"rc {p.returncode}: {msg}")
    path = rollout_path(tid or thread_id, home)
    rows = read_rollout(path) if path else []
    return {"kind": kind, "thread_id": tid, "parent": thread_id if kind == "fork" else None, "text": text,
            "usage": last_turn_usage(rows) or usage, "usage_cumulative": usage,
            "events": events, "rollout": str(path) if path else None, "rate_limits": last_rate_limits(rows),
            "retries": throttle_retries(err), "reconnect_events": reconnects, "events_raw": out, "tools": tool_items(rows),
            "codex_bin": cmd[0],
            "meta": {**rollout_meta(rows), "requested_model": model, "config": list(config)},
            "t_start": t0, "t_end": time.time(), "stderr_tail": err[-300:]}


class AppServer:
    """A minimal JSON-RPC client for `codex app-server` over stdio, enough for thread/fork (without a turn) and thread/compact/start
    (the manual compact command, the one the interactive client's /compact sends). A reader thread queues every line (select on a
    buffered pipe can stall)."""

    def __init__(self, cwd, *, codex_bin, home, config=()):
        cmd = [codex_bin, "app-server"] + [x for c in BASE_CONFIG + list(config) for x in ("-c", c)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd),
                                  env=env(home), bufsize=1)
        self.q = queue.Queue()
        self.err = []
        threading.Thread(target=self._pump, args=(self.p.stdout, self.q), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.p.stderr, None), daemon=True).start()
        self.n = 0
        self.notes = []
        self.request("initialize", {"clientInfo": {"name": "codex_harness", "version": "0.1"}})
        self.notify("initialized", {})

    def _pump(self, stream, q):
        for line in stream:
            if q is None:
                self.err.append(line)
                del self.err[:-50]
            else:
                q.put(line)
        if q is not None:
            q.put(None)

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _next(self, deadline):
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None
            try:
                line = self.q.get(timeout=left)
            except queue.Empty:
                return None
            if line is None:
                raise CodexError(classify("".join(self.err)), "app-server closed: " + "".join(self.err)[-400:])
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "method" in m and "id" in m:  # a request from the server (an approval, say): declined
                self._send({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "not supported"}})
                continue
            return m

    def request(self, method, params, timeout=120):
        self.n += 1
        rid = self.n
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.time() + timeout
        while True:
            m = self._next(deadline)
            if m is None:
                raise CodexError("transient", f"{method}: no response in {timeout}s")
            if m.get("id") == rid:
                if "error" in m:
                    raise CodexError(classify(json.dumps(m["error"])), f"{method}: {json.dumps(m['error'])[:500]}")
                return m.get("result")
            if "method" in m:
                self.notes.append(m)

    def wait_for(self, pred, timeout=600):
        for m in self.notes:
            if pred(m):
                return m
        deadline = time.time() + timeout
        while True:
            m = self._next(deadline)
            if m is None:
                raise CodexError("transient", f"no matching notification in {timeout}s; seen " +
                                 json.dumps(sorted({x.get('method') for x in self.notes}))[:300])
            if "method" in m:
                self.notes.append(m)
                if pred(m):
                    return m

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()
            self.p.wait()


def _tid(m):
    p = m.get("params") or {}
    return p.get("threadId") or (p.get("thread") or {}).get("id") or (p.get("turn") or {}).get("threadId")


def compact_fork(parent_id, cwd, model, *, codex_bin, home, config=(), timeout=900):
    """Fork a stored thread without a turn (app-server thread/fork) and compact the fork in place with the manual compact command
    (thread/compact/start). The parent is untouched. Returns the fork's id, its compaction record(s) and usage."""
    t0 = time.time()
    a = AppServer(cwd, codex_bin=codex_bin, home=home, config=[f"model={json.dumps(model)}"] + list(config))
    try:
        r = a.request("thread/fork", {"threadId": parent_id, "excludeTurns": True, "model": model}, timeout=120)
        tid = (r.get("thread") or {}).get("id") or r.get("threadId")
        if not tid:
            raise CodexError("fatal", "thread/fork returned no thread id: " + json.dumps(r)[:300])
        a.request("thread/compact/start", {"threadId": tid}, timeout=120)
        a.wait_for(lambda m: m.get("method") in ("thread/compacted", "turn/completed") and _tid(m) in (tid, None), timeout=timeout)
        path = rollout_path(tid, home)
        for _ in range(100):
            if path and compactions(read_rollout(path)):
                break
            time.sleep(0.2)
            path = path or rollout_path(tid, home, wait=0)
    finally:
        a.close()
    rows = read_rollout(path) if path else []
    new = compactions(rows)
    if not new:
        raise CodexError("transient", "compaction produced no compacted record; notes: " + json.dumps([m.get("method") for m in a.notes])[:400])
    model_switch = any("<model_switch>" in t for c in new for i in (c["replacement_history"] or []) if i.get("type") == "message"
                       for t in texts(i.get("content")))
    return {"kind": "compact", "thread_id": tid, "parent": parent_id, "compactions": new, "usage": last_turn_usage(rows),
            "rollout": str(path), "rate_limits": last_rate_limits(rows), "model_switch": model_switch,
            "meta": {**rollout_meta(rows), "requested_model": model, "config": list(config)},
            "notes": sorted({m.get("method") for m in a.notes}), "t_start": t0, "t_end": time.time()}
