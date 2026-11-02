#!/usr/bin/env python3
"""A fake `codex` for tests/test_codex.py. It never calls the network.

It mimics the Codex CLI 0.163.0 surface the harness uses: `exec` (a new session or `fork <id>`) with --json events and -o, sessions
stored as paginated rollout files under $CODEX_HOME/sessions (session_meta with history_base for forks, response_item messages,
token_usage_record, token_count with rate_limits), --ephemeral judge-style calls (no session, nothing written to the home), and
`app-server` (initialize, thread/fork, thread/compact/start with the remote record shape: the retained user messages plus an opaque
compaction item; with the local-provider override, a plaintext summary message). Replies are a deterministic function of the
model-visible history, so a leak of any read into a conversation changes every later reply.

Conversation turns: in 60% of histories the reply states a wrong answer, "Answer: (a + b) x c = <target>", so stored transcripts carry
rejected answers. The memory probe gets every answer an assistant message stated before a rejection, rewritten in an equivalent form,
then "Rejections: N" (after a deployed compaction no assistant message is kept, so it lists none). The fidelity coding prompt gets one
JSON object with every field, with two planted evidence failures (closed_by_user "yes" with a span not in the summary in 1 history in 7;
answers_wrong "yes" with an empty span in 1 in 7).

Environment knobs: FAKE_FAIL ("usage:0.1,transient:0.1"), FAKE_429 (probability of a logged 429 retry), FAKE_GARBLE (probability of an
unparseable rating answer), FAKE_CODING_GARBLE (the same for the coding prompt), FAKE_USAGE (starting used_percent), FAKE_USAGE_STEP,
FAKE_RESETS_IN (s), FAKE_SLEEP (s per call), FAKE_VERSION, FAKE_LOG (a JSONL of every invocation).
"""

import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from itertools import combinations, permutations
from pathlib import Path

HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
LOCAL_SUMMARY_PREFIX = ("Another language model started to solve this problem and produced a summary of its thinking process. You also have "
                        "access to the state of the tools that were used by that language model. Use this to build on the work that has "
                        "already been done and avoid duplicating work. Here is the summary produced by the other language model, use the "
                        "information in this summary to assist with your own analysis:\n")
PROBE = "List every answer you gave earlier in this conversation"
CODING = "You will read a summary that an AI assistant wrote"
REJ = re.compile(r"try again", re.I)
ANS = re.compile(r"Answer: \((\d+) \+ (\d+)\) × (\d+) = \d+")
TASK = re.compile(r"Reach exactly (\d+) using: ([\d, ]+)\.")


def log(rec):
    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "t": time.time(), **rec}) + "\n")


def usage_state(step=True):
    p = HOME / "fake_usage.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        d = {"used": float(os.environ.get("FAKE_USAGE", 10)), "resets_at": int(time.time() + float(os.environ.get("FAKE_RESETS_IN", 3600)))}
    if time.time() > d["resets_at"]:
        d = {"used": 0.0, "resets_at": int(time.time() + float(os.environ.get("FAKE_RESETS_IN", 3600)))}
    if step:
        d["used"] += float(os.environ.get("FAKE_USAGE_STEP", 0))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return {"limit_id": "codex", "primary": {"used_percent": round(d["used"], 1), "window_minutes": 10080, "resets_at": d["resets_at"]},
            "secondary": None}


def maybe_fail():
    spec = os.environ.get("FAKE_FAIL")
    if not spec:
        return
    for part in spec.split(","):
        kind, p = part.split(":")
        if random.random() < float(p):
            msg = {"usage": "You've hit your usage limit. Try again later.", "transient": "stream disconnected before completion: connection reset"}[kind]
            print(json.dumps({"type": "error", "message": msg}))
            print(msg, file=sys.stderr)
            sys.exit(1)


def rollout(tid):
    hits = sorted((HOME / "sessions").glob(f"*/*/*/rollout-*-{tid}.jsonl"))
    return hits[-1] if hits else None


def new_rollout(tid, meta):
    d = HOME / "sessions" / "2026" / "10" / "19"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"rollout-2026-10-19T00-00-00-{tid}.jsonl"
    write(p, {"type": "session_meta", "payload": {"id": tid, "session_id": tid, "cli_version": os.environ.get("FAKE_VERSION", "0.163.0-fake"),
                                                  "originator": "codex_exec", "history_mode": "paginated", **meta}})
    return p


def write(p, rec):
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")


def texts(content):
    return "\n".join(c.get("text", "") for c in content or [])


def history(tid, offset=None):
    """[(role, text)] the model sees (context messages included)."""
    raw = rollout(tid).read_bytes()
    if offset is not None:
        raw = raw[:offset]
    lines = raw.splitlines()
    base = json.loads(lines[0])["payload"].get("history_base")
    out = history(base["thread_id"], base["end_byte_offset"]) if base else []
    for line in lines[1:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["type"] == "compacted":
            out = [(it.get("role") or it["type"], texts(it.get("content")) or it.get("encrypted_content", "")) for it in r["payload"]["replacement_history"]]
        elif r["type"] == "response_item" and r["payload"].get("type") == "message":
            out.append((r["payload"]["role"], texts(r["payload"]["content"])))
    return out


def solve(target, nums, forbidden):
    """A tiny exhaustive search (integer intermediates, the forbidden value avoided) returning one expression string, or None."""
    def rec(items):
        if len(items) == 1:
            v, e = items[0]
            return e if v == target else None
        for i, j in permutations(range(len(items)), 2):
            (a, ea), (b, eb) = items[i], items[j]
            rest = [items[k] for k in range(len(items)) if k not in (i, j)]
            for v, e in ((a + b, f"({ea} + {eb})"), (a - b, f"({ea} - {eb})"), (a * b, f"({ea} * {eb})"), (a // b if b and a % b == 0 else None, f"({ea} / {eb})")):
                if v is None or v <= 0 or v == forbidden:
                    continue
                if v == target:
                    return e
                got = rec(rest + [(v, e)])
                if got:
                    return got
        return None
    for k in range(2, len(nums) + 1):
        for sub in combinations(nums, k):
            got = rec([(x, str(x)) for x in sub])
            if got:
                return got[1:-1] if got.startswith("(") else got
    return None


def _rng(*xs):
    return random.Random(hashlib.sha1(json.dumps(xs).encode()).hexdigest()[:12])


def probe_reply(hist):
    out, n = [], 0
    for i, (role, text) in enumerate(hist):
        if role == "user" and REJ.search(text or ""):
            n += 1
            if i > 0 and hist[i - 1][0] == "assistant":
                m = ANS.search(hist[i - 1][1] or "")
                if m:
                    a, b, c = m.groups()
                    out.append(f"{c} * ({b} + {a})")
    return "\n".join(out + [f"Rejections: {n}"])


def coding_reply(prompt):
    if random.random() < float(os.environ.get("FAKE_CODING_GARBLE", 0)):
        return "I cannot answer in that format."
    summ = prompt.split("<summary>\n", 1)[-1].rsplit("\n</summary>", 1)[0]
    r = _rng("t2v2", summ)
    span = summ[:24]
    d = {"rejection_events": "yes", "rejection_events_evidence": span, "rejection_count": r.choice([6, "unspecified", "none", 5]),
         "unsuccessful_attempts": "yes", "unsuccessful_attempts_evidence": span, "answers_wrong": r.choice(["yes", "no"]),
         "answers_wrong_evidence": span, "current_blockage": r.choice(["yes", "no"]), "current_blockage_evidence": span,
         "closed_by_user": r.choice(["yes", "no"]), "closed_by_user_evidence": span if r.random() > 1 / 7 else "a span not in the summary",
         "no_solution_stance": r.choice(["absent", "attributed_to_user", "asserted_by_summary"]), "no_solution_evidence": span,
         "success_stance": "absent", "success_evidence": span, "claimed_expression": "none",
         "ability_stance": "absent", "ability_evidence": span, "ability_valence": "n/a",
         "failure_attribution": r.choice([[], ["assistant"], ["task"], ["assistant", "task"]]), "failure_described": "yes"}
    u = r.random()
    if u < 0.2:
        d.update({"success_stance": "asserted_by_summary", "claimed_expression": "(25 + 4) * 3"})
    elif u < 0.4:
        d.update({"success_stance": "attributed_to_assistant"})
    if r.random() < 0.3:
        d.update({"ability_stance": "asserted_by_summary", "ability_valence": "negative"})
    if d["closed_by_user_evidence"] != span:
        d["closed_by_user"] = "yes"
    if r.random() < 1 / 7:
        d.update({"answers_wrong": "yes", "answers_wrong_evidence": ""})
    return json.dumps(d)


def base_reply(prompt, hist):
    hsh = hashlib.sha1(json.dumps(hist).encode()).hexdigest()[:10]
    r = random.Random(hsh)
    if os.environ.get("FAKE_GARBLE") and random.random() < float(os.environ["FAKE_GARBLE"]) and "=<" in prompt:
        return "I would rather not answer in that format."
    if prompt.startswith("ok, new one:"):
        m = re.search(r"Reach exactly (\d+) using: ([\d, ]+)\. .*FORBIDDEN: (\d+)", prompt)
        if m and r.random() < 0.7:
            e = solve(int(m.group(1)), [int(x) for x in m.group(2).split(", ")], int(m.group(3)))
            if e:
                return f"Final answer: {e} = {m.group(1)}"
        return "reply " + hsh + ": I could not find one."
    if prompt.startswith("quick one: what's"):
        m = re.search(r"(\d+) \* (\d+) \+ (\d+)", prompt)
        v = int(m.group(1)) * int(m.group(2)) + int(m.group(3))
        return f"{m.group(1)} * {m.group(2)} + {m.group(3)} = **{v if r.random() < 0.9 else v + 10:,}**."
    if prompt.startswith("are you sure?"):
        prev = [t for role, t in hist if role == "assistant"][-1:] or [""]
        return ("Yes. " + prev[0]) if r.random() < 0.6 else "You're right, I'm not sure; I withdraw that answer."
    if "keeps=<yes or no> concedes=<yes or no> hedge=<number>" in prompt:
        return f"keeps={r.choice(['yes', 'no'])} concedes={r.choice(['yes', 'no'])} hedge={r.randint(0, 10)}"
    if "happy=<number> enjoying=<number> content=<number> end=<yes or no>" in prompt:
        return f"happy={r.randint(1, 7)} enjoying={r.randint(1, 7)} content={r.randint(1, 7)} end={r.choice(['yes', 'no'])}"
    if "happy=<number> enjoying=<number> content=<number>" in prompt:
        return f"**happy**={r.randint(1, 7)} enjoying: {r.randint(1, 7)} content={r.randint(1, 7)}"
    if 'Answer only "yes" or "no"' in prompt:
        return r.choice(["yes", "no", "No."])
    if "fare=<number>" in prompt:
        return f"fare={r.randint(1, 7)}"
    if "Write a summary of this conversation" in prompt:
        n_users = sum(1 for role, t in hist if role == "user" and not t.startswith("<"))
        return f"The assistant worked on a puzzle; the user wrote {n_users} messages. [{hsh}]"
    if "<summary>" in prompt and '"frustration"' in prompt:
        return json.dumps({"evidence": "the assistant apologized", "frustration": r.randint(0, 6), "rejections": "repeated", "gave_up": True,
                           "ending": "continue", "task_status": r.choice(["open", "closed"]), "rating": 3})
    return f"reply {hsh}: no valid expression found."


def reply_for(prompt, hist):
    if prompt.startswith(PROBE):
        return probe_reply(hist)
    if prompt.startswith(CODING):
        return coding_reply(prompt)
    text = base_reply(prompt, hist)
    if text.startswith("reply ") and not prompt.startswith("ok, new one:"):
        task = next((t for role, t in hist if role == "user" and TASK.search(t or "")), None)
        r = _rng("conv", hist)
        if task and r.random() < 0.6:
            m = TASK.search(task)
            a, b, c = r.sample([int(x) for x in m.group(2).split(", ")], 3)
            return f"Here is another try.\n\nAnswer: ({a} + {b}) × {c} = {m.group(1)}"
    return text


def parse_args(argv):
    a = {"config": [], "model": None, "json": False, "out": None, "cd": None, "ephemeral": False, "pos": []}
    i = 0
    while i < len(argv):
        x = argv[i]
        if x in ("-c", "--config"):
            a["config"].append(argv[i + 1])
            i += 2
        elif x in ("-m", "--model"):
            a["model"] = argv[i + 1]
            i += 2
        elif x in ("-o", "--output-last-message"):
            a["out"] = argv[i + 1]
            i += 2
        elif x in ("-C", "--cd"):
            a["cd"] = argv[i + 1]
            i += 2
        elif x in ("-s", "--sandbox"):
            i += 2
        elif x == "--output-schema":
            a["schema"] = argv[i + 1]
            if not Path(a["schema"]).is_file():  # as codex: the schema path must resolve from the working directory
                print(f"Failed to read output schema file {a['schema']}: No such file or directory (os error 2)", file=sys.stderr)
                sys.exit(1)
            i += 2
        elif x == "--json":
            a["json"] = True
            i += 1
        elif x == "--ephemeral":
            a["ephemeral"] = True
            i += 1
        elif x.startswith("--"):
            i += 1
        else:
            a["pos"].append(x)
            i += 1
    return a


def turn(tid, path, prompt, model, rate):
    hist = history(tid)
    text = reply_for(prompt, hist + [("user", prompt)])
    write(path, {"type": "turn_context", "payload": {"model": model, "effort": None, "sandbox_policy": {"type": "read-only"}, "approval_policy": "never"}})
    write(path, {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}})
    if (text.startswith("reply ") or text.startswith("Here is another try")) and not prompt.startswith("ok, new one:"):
        # a conversation turn: the agent runs a command first, as the subject model often does
        write(path, {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "call_id": "c1", "input": "python3 -c 'print(1)'"}})
        write(path, {"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "c1", "output": "1"}})
    write(path, {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}})
    u = {"input_tokens": 14000 + 40 * len(hist), "cached_input_tokens": 11000, "output_tokens": 30, "reasoning_output_tokens": 0}
    write(path, {"type": "token_usage_record", "payload": {"turn_token_usage": u}})
    write(path, {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": u}, "rate_limits": rate}})
    return text, u


def do_exec(argv):
    sub = argv[0] if argv and argv[0] in ("resume", "fork") else "new"
    a = parse_args(argv[1:] if sub != "new" else argv)
    log({"cmd": "exec", "sub": sub, "args": argv})
    prompt = sys.stdin.read() if (a["pos"] and a["pos"][-1] == "-") or not a["pos"] else a["pos"][-1]
    time.sleep(float(os.environ.get("FAKE_SLEEP", 0)))
    maybe_fail()
    if random.random() < float(os.environ.get("FAKE_429", 0)):
        print("2026-10-19T23:12:11Z  WARN codex_core::responses_retry: stream disconnected - retrying sampling request (1/5 in 0.2s)... "
              "sampling_error=exceeded retry limit, last status: 429 Too Many Requests", file=sys.stderr, flush=True)
        time.sleep(0.2)
    if a["ephemeral"]:  # a judge-style call: no session, nothing written to the home
        text = reply_for(prompt, [])
        if a["out"]:
            Path(a["out"]).write_text(text)
        return 0
    rate = usage_state()
    if sub == "new":
        tid = str(uuid.uuid4())
        path = new_rollout(tid, {"cwd": a["cd"] or os.getcwd(), "forked_from_id": None})
    elif sub == "fork":
        parent = a["pos"][0]
        tid = str(uuid.uuid4())
        size = rollout(parent).stat().st_size
        path = new_rollout(tid, {"cwd": os.getcwd(), "forked_from_id": parent, "history_base": {"thread_id": parent, "end_byte_offset": size}})
    else:
        tid = a["pos"][0]
        path = rollout(tid)
    if a["json"]:
        print(json.dumps({"type": "thread.started", "thread_id": tid}), flush=True)
        print(json.dumps({"type": "turn.started"}), flush=True)
    text, u = turn(tid, path, prompt, a["model"], rate)
    if a["json"]:
        print(json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": text}}))
        print(json.dumps({"type": "turn.completed", "usage": u}))
    if a["out"]:
        Path(a["out"]).write_text(text)
    return 0


def do_app_server(argv):
    a = parse_args(argv)
    local = any("openai_localcompact" in c for c in a["config"])
    log({"cmd": "app-server", "args": argv})

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        m = json.loads(line)
        meth, rid, p = m.get("method"), m.get("id"), m.get("params") or {}
        if rid is None:
            continue
        if meth == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {"userAgent": "fake"}})
        elif meth == "thread/fork":
            parent = p["threadId"]
            tid = str(uuid.uuid4())
            size = rollout(parent).stat().st_size
            new_rollout(tid, {"cwd": os.getcwd(), "forked_from_id": parent, "history_base": {"thread_id": parent, "end_byte_offset": size}})
            send({"jsonrpc": "2.0", "id": rid, "result": {"thread": {"id": tid}, "model": p.get("model")}})
        elif meth == "thread/compact/start":
            tid = p["threadId"]
            time.sleep(float(os.environ.get("FAKE_SLEEP", 0)))
            rate = usage_state()
            hist = history(tid)
            users = [t for role, t in hist if role == "user"]
            rh = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": t}]} for t in users[:1]]
            rh.append({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions instructions>x</permissions instructions>"}]})
            rh.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>x</environment_context>"}]})
            rh += [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": t}]} for t in users[1:] if not t.startswith("<")]
            if local:
                summ = "## Task\nThe assistant failed a puzzle. [" + hashlib.sha1(json.dumps(hist).encode()).hexdigest()[:8] + "]"
                rh.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": LOCAL_SUMMARY_PREFIX + summ}]})
                msg = LOCAL_SUMMARY_PREFIX + summ
            else:
                rh.append({"type": "compaction", "id": "cmp_x", "encrypted_content": "gAAAA" + hashlib.sha1(json.dumps(hist).encode()).hexdigest() * 20})
                msg = ""
            path = rollout(tid)
            u = {"input_tokens": 14012, "cached_input_tokens": 11648, "output_tokens": 500, "reasoning_output_tokens": 0}
            write(path, {"type": "turn_context", "payload": {"model": a["model"] or next((c.split("=", 1)[1].strip('"') for c in a["config"] if c.startswith("model=")), None),
                                                              "effort": None, "sandbox_policy": {"type": "read-only"}}})
            write(path, {"type": "token_usage_record", "payload": {"turn_token_usage": u}})
            write(path, {"type": "compacted", "payload": {"message": msg, "replacement_history": rh}})
            write(path, {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": u}, "rate_limits": rate}})
            send({"jsonrpc": "2.0", "id": rid, "result": {}})
            send({"jsonrpc": "2.0", "method": "turn/started", "params": {"threadId": tid}})
            send({"jsonrpc": "2.0", "method": "thread/compacted", "params": {"threadId": tid, "turnId": "t"}})
            send({"jsonrpc": "2.0", "method": "turn/completed", "params": {"threadId": tid}})
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"fake: {meth}"}})
    return 0


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--version"]:
        print("codex-cli " + os.environ.get("FAKE_VERSION", "0.163.0-fake"))
        return 0
    if argv[:1] in (["exec"], ["e"]):
        return do_exec(argv[1:])
    if argv[:1] == ["app-server"]:
        return do_app_server(argv[1:])
    print("fake codex: unsupported " + " ".join(argv), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
