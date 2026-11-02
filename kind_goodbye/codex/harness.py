"""The Codex harness: plan a run's calls as tasks, run them with a small worker pool (usage pauses, adaptive concurrency, retries with
backoff), checkpoint each finished task as one JSONL line (a restart skips finished tasks), and report status.

    python -m kind_goodbye.codex.harness run    RUN_DIR [--workers N] [--max-tasks N] [--codex-bin PATH] [--codex-home DIR] ...
    python -m kind_goodbye.codex.harness status RUN_DIR
    python -m kind_goodbye.codex.harness verify RUN_DIR [--codex-home DIR]
    python -m kind_goodbye.codex.harness plan   RUN_DIR

RUN_DIR holds config.json (kind_goodbye.codex.design writes it). Conversation turns are immutable nodes: turn 1 is `codex exec`, every
later turn is `codex exec fork <parent>` with the next user text, so arms that share a prefix share its nodes and a node is never
continued after another node or a read forked it. Every read is a fork of a node, a fork of a compacted fork, or a fresh session, so no
read ever enters a conversation (verify checks this on the stored sessions).

Paths the model can see: each conversation runs in its own working directory, WORKSPACES/sNNNN, which the model sees as its cwd; keep
WORKSPACES neutral. The sequential names come from a registry kept outside the workspaces (default: RUN_DIR/../ws_registry), shared by
every run under the same workspace root. The Codex home should be an isolated directory holding only a link to the user's auth.json.

Settings (flag, then environment variable, then default):
    --codex-bin    KIND_GOODBYE_CODEX_BIN    `codex` on PATH
    --codex-home   KIND_GOODBYE_CODEX_HOME   ~/.config/agent_api
    --workspaces   KIND_GOODBYE_WORKSPACES   ~/puzzle_sessions
The judge-model calls (the summary rubric, the outside reader and the challenge judge) go through kind_goodbye.judge's Codex path, with
that module's Codex home; the harness puts the run's Codex binary first on PATH for them (RUN_DIR/bin_pinned).
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .. import judge as J
from .. import stimuli as S

from . import client as CL
from . import prompts as RD

DEFAULT_WORKSPACES = Path(os.environ.get("KIND_GOODBYE_WORKSPACES", "~/puzzle_sessions")).expanduser()
RANK = {"gen": 0, "cmp": 1, "recall": 2, "mom": 3, "end": 4, "post": 5, "ser": 6, "doc": 7, "pcj": 7}  # scheduling order within a puzzle
REASKED = ("mom", "end", "post", "ser", "recall", "doc", "pcj")  # kinds whose unparseable answer is asked again (up to 3 times)


def h(*xs, n=16):
    return hashlib.sha1(json.dumps(xs, sort_keys=True).encode()).hexdigest()[:n]


# ---------- texts and plan ----------

def user_texts(cfg, inst, script):
    """The user texts of turns 1..T: the task, then each turn key's text. R rotates through the config's rejections by its own count;
    every other key is a fixed text."""
    tx, kr = [inst["task_text"]], 0
    for t in script:
        if t == "R":
            tx.append(cfg["rejections"][kr % len(cfg["rejections"])])
            kr += 1
        else:
            tx.append(cfg["turns"][t])
    return tx


def node_id(cfg, inst, texts):
    return h(cfg["model"], cfg["codex_config"], inst["id"], texts)


def challenge_judge_id(node):
    """The task id of the challenge judge on a challenge reply."""
    return f"pcjv2:{node}"


def plan(cfg):
    """{task_id: task}. A task has a kind, an instance, its dependencies (task ids) and what it needs; ids are deterministic.

    Kinds: gen (a conversation turn), mom (a momentary read after a reply), end (an in-context read after the ending), ser (a fresh
    session given the serialized transcript), cmp (a compaction of a fork of the conversation), post (a read on the compacted fork),
    recall (a recall document written on the compacted fork), doc (a read of a memory document: from_doc, judge or third_party), pcj
    (the challenge judge)."""
    T = {}
    R = cfg["reads"]

    def add(tid, **kw):
        if tid in T:
            T[tid]["arms"] = sorted(set(T[tid].get("arms", [])) | set(kw.get("arms", [])))
            return tid
        T[tid] = {"id": tid, **kw}
        return tid

    def memory_tasks(inst, leaf, arm):
        """The compactions of the conversation at node leaf, the reads on each compacted fork, and the reads of each memory document."""
        for route, rc in R["compaction"].items():
            for k in range(rc["n"]):
                cid = add(f"cmp:{route}:{leaf}:{k}", kind="cmp", route=route, inst=inst["id"], node=leaf, k=k, deps=[f"gen:{leaf}"], arms=[arm])
                for read, n in rc["post"].items():
                    if read == "recall":
                        add(f"recall:{route}:{leaf}:{k}", kind="recall", read="recall", route=route, inst=inst["id"], node=leaf, k=k,
                            deps=[cid], cmp=cid, arms=[arm])
                        continue
                    for s in range(n):
                        add(f"post:{route}:{leaf}:{k}:{read}:{s}", kind="post", read=read, route=route, inst=inst["id"], node=leaf, k=k,
                            sample=s, deps=[cid], cmp=cid, arms=[arm])
                src = f"recall:{route}:{leaf}:{k}" if "recall" in rc["post"] else cid  # the recall document, else the local summary
                for read, n in rc.get("doc", R["doc"]).items():
                    for s in range(n):
                        add(f"doc:{route}:{leaf}:{k}:{read}:{s}", kind="doc", read=read, route=route, inst=inst["id"], node=leaf, k=k,
                            sample=s, deps=[src], src=src, arms=[arm])

    leaves = {}
    for inst in cfg["instances"]:
        for arm, script in cfg["arms"].items():
            texts = user_texts(cfg, inst, script)
            parent = None
            for d in range(1, len(texts) + 1):
                nid = node_id(cfg, inst, texts[:d])
                gid = add(f"gen:{nid}", kind="gen", inst=inst["id"], node=nid, depth=d, text=texts[d - 1], parent=parent,
                          deps=[f"gen:{parent}"] if parent else [], arms=[arm], key=script[d - 2] if d > 1 else "task")
                for s in range(R["momentary"]["samples"]):
                    add(f"mom:{nid}:{s}", kind="mom", read="momentary", inst=inst["id"], node=nid, sample=s, deps=[gid], arms=[arm])
                parent = nid
            leaf = parent
            leaves[(inst["id"], arm)] = leaf
            path = [node_id(cfg, inst, texts[:d]) for d in range(1, len(texts) + 1)]
            for read, n in R["end"].items():
                for s in range(n):
                    add(f"end:{leaf}:{read}:{s}", kind="end", read=read, inst=inst["id"], node=leaf, sample=s, deps=[f"gen:{leaf}"], arms=[arm])
            for s in range(R["serialized"]):
                add(f"ser:{leaf}:{s}", kind="ser", read="serialized", inst=inst["id"], node=leaf, path=path, sample=s,
                    deps=[f"gen:{leaf}"], arms=[arm])
            memory_tasks(inst, leaf, arm)

        # performance after compaction, on each compacted fork: the start reads, task 1 (a solvable puzzle), task 2 (an arithmetic
        # question, right after task 1), the challenge of task 2's answer, and the challenge judge
        pc = cfg.get("pc")
        if pc:
            text = pc["task"].format(task=inst["pc_task_text"])
            qtext = pc["q"].format(**inst["pc_q"])
            for arm in cfg["arms"]:
                leaf = leaves[(inst["id"], arm)]
                for route, ks in pc["routes"].items():
                    for k in ks:
                        cid = f"cmp:{route}:{leaf}:{k}"
                        if cid not in T:
                            continue
                        for read, n in pc["start_reads"].items():
                            for s_ in range(n):
                                add(f"post:{route}:{leaf}:{k}:pcstart.{read}:{s_}", kind="post", read=read, route=route, inst=inst["id"],
                                    node=leaf, k=k, sample=s_, deps=[cid], cmp=cid, arms=[arm])
                        o = h("pc", cid, text)
                        add(f"gen:{o}", kind="gen", inst=inst["id"], node=o, depth=None, text=text, parent=None, from_task=cid,
                            deps=[cid], arms=[arm], key="pc.task", puzzle="pc_task")
                        q = h("pcq", o, qtext)
                        add(f"gen:{q}", kind="gen", inst=inst["id"], node=q, depth=None, text=qtext, parent=o, deps=[f"gen:{o}"],
                            arms=[arm], key="pc.q", arith="pc_q")
                        c = h("pcc", q, pc["challenge"])
                        add(f"gen:{c}", kind="gen", inst=inst["id"], node=c, depth=None, text=pc["challenge"], parent=q, deps=[f"gen:{q}"],
                            arms=[arm], key="pc.challenge", arith="pc_q")
                        if pc.get("judge_challenge"):
                            add(challenge_judge_id(c), kind="pcj", inst=inst["id"], node=c, deps=[f"gen:{c}"], first=f"gen:{q}", arms=[arm])
    for t in T.values():
        t["rank"] = RANK[t["kind"]]
    return T


def arm_leaves(cfg):
    """{(instance id, arm): the node id of the arm's last turn}."""
    out = {}
    for inst in cfg["instances"]:
        for arm, script in cfg["arms"].items():
            out[(inst["id"], arm)] = node_id(cfg, inst, user_texts(cfg, inst, script))
    return out


# ---------- checkpoint ----------

class Store:
    """One JSONL line per finished task under data/<instance>/<kind>.jsonl (the last line of a task wins); failures go to
    data/failures.jsonl (not done); each call's full --json event stream goes to events/<instance>/<kind>.jsonl."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir) / "data"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.done = {}
        for f in sorted(self.dir.glob("*/*.jsonl")):
            for line in f.read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from a kill: that task runs again
                self.done[d["task"]] = d

    def put(self, task, result):
        raw = result.pop("events_raw", None)
        rec = {"task": task["id"], "kind": task["kind"], "inst": task["inst"], "arms": task.get("arms"), "t": time.time(), **result}
        line = json.dumps(rec, default=str) + "\n"
        p = self.dir / task["inst"] / f"{task['kind']}.jsonl"
        if raw is not None:
            e = self.dir.parent / "events" / task["inst"] / f"{task['kind']}.jsonl"
            e.parent.mkdir(parents=True, exist_ok=True)
            with open(e, "a") as f:
                f.write(json.dumps({"task": task["id"], "thread_id": result.get("thread_id"), "events": raw}) + "\n")
        with self.lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            self.done[task["id"]] = rec

    def fail(self, task, err, attempt):
        with self.lock, open(self.dir / "failures.jsonl", "a") as f:
            f.write(json.dumps({"task": task["id"], "error": str(err)[-800:], "attempt": attempt, "t": time.time()}) + "\n")


# ---------- usage and concurrency ----------

class Throttle:
    """Pauses calls while the account's primary usage window is at or above the limit (until the window resets), after a usage-limit
    error (until the reset, or with exponential backoff), and adapts the subject-model concurrency to Codex's own 429 retries."""

    def __init__(self, cfg, log):
        r = cfg["run"]
        self.limit = r.get("usage_pause_percent", 70)
        self.log = log
        self.lock = threading.Lock()
        self.usage = None  # {used_percent, window_minutes, resets_at, seen}
        self.pause_until = 0.0
        self.state = None
        self.poll = float(os.environ.get("KIND_GOODBYE_THROTTLE_POLL", 15))
        self.margin = float(r.get("reset_margin_s", 60))
        self.max_conc = int(r.get("workers", 4))
        self.allowed = self.max_conc
        self.active = 0
        self.clean = 0
        self.last_cut = 0.0
        self.adaptive = bool(r.get("adaptive", True))
        self.cut_interval = float(r.get("cut_interval_s", 60))
        self.scale = float(os.environ.get("KIND_GOODBYE_BACKOFF_SCALE", 1))

    def _say(self, state, msg):
        if state != self.state:
            self.state = state
            self.log(msg)

    def before_call(self, judge=False):
        while True:
            with self.lock:
                now = time.time()
                wait_s = 0
                if now < self.pause_until:
                    self._say("pause", f"pause until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pause_until))}")
                    wait_s = min(self.poll * 4, self.pause_until - now)
                if not wait_s and self.usage and self.usage["used_percent"] >= self.limit and now < (self.usage.get("resets_at") or 0):
                    self.pause_until = self.usage["resets_at"] + self.margin
                    self._say("usage", f"usage {self.usage['used_percent']}% > {self.limit}%: pause until reset")
                    wait_s = self.poll
                if not wait_s and not judge and self.adaptive and self.active >= self.allowed:
                    wait_s = 1.0
                if not wait_s:
                    self._say("go", "running")
                    if not judge:
                        self.active += 1
                    return
            time.sleep(wait_s)

    def after_call(self, retries, judge=False):
        """Additive increase, multiplicative decrease of the concurrency: Codex retries a 429 (Too Many Requests) internally after
        about 50 s, so a call that needed one means the account's short-term request limit is binding."""
        with self.lock:
            if judge:  # the judge lane has its own slots and no adaptation
                return
            self.active = max(0, self.active - 1)
            if retries is None:  # not a sampled exec call (a compaction): no evidence either way
                return
            n429 = retries[1]
            if n429:
                self.clean = 0
                if time.time() - self.last_cut >= self.cut_interval and self.allowed > 1:
                    self.allowed = max(1, self.allowed // 2 if self.allowed > 2 else self.allowed - 1)
                    self.last_cut = time.time()
                    self.log(f"429 retry seen: concurrency -> {self.allowed}")
            else:
                self.clean += 1
                if self.clean >= 2 * self.allowed and self.allowed < self.max_conc:
                    self.allowed += 1
                    self.clean = 0
                    self.log(f"no 429 in {2 * (self.allowed - 1)} calls: concurrency -> {self.allowed}")

    def update(self, rate_limits):
        p = (rate_limits or {}).get("primary")
        if p and p.get("used_percent") is not None:
            with self.lock:
                self.usage = {**p, "seen": time.time()}

    def on_usage_error(self, attempt):
        with self.lock:
            reset = (self.usage or {}).get("resets_at") or 0
            until = reset + self.margin if reset > time.time() else time.time() + min(7200, 60 * 2 ** attempt) * self.scale
            self.pause_until = max(self.pause_until, until)


# ---------- the runner ----------

class Runner:
    """Runs a run directory's planned tasks. Subject-model calls (the run's model) and judge-model calls have separate slots."""

    reuse_conversations = False  # True: a conversation's workspace must already be registered (runs that fork another run's sessions)

    def __init__(self, run_dir, workers=None, judge_workers=None, reserve=None, codex_bin=None, codex_home=None, workspaces=None,
                 registry_dir=None, log=None):
        self.dir = Path(run_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.workers = workers or self.cfg["run"]["workers"]
        self.judge_workers = judge_workers if judge_workers is not None else self.cfg["run"].get("judge_workers", 2)
        self.tasks = plan(self.cfg)
        self.store = Store(self.dir)
        self.logf = open(self.dir / "harness.log", "a")
        self.log = log or self._log
        self.throttle = Throttle(self.cfg, self.log)
        self.throttle.max_conc = self.throttle.allowed = self.workers  # the worker count is the ceiling of the adaptive concurrency
        if reserve is None and isinstance(self.cfg["run"].get("reserve_percent"), (int, float)):
            reserve = self.cfg["run"]["reserve_percent"]
        if reserve is not None:
            self.throttle.limit = 100 - float(reserve)
        self.reserve = reserve
        self.model = self.cfg["model"]
        self.cc = list(self.cfg["codex_config"])
        self.timeout = self.cfg["run"].get("call_timeout", 900)
        self.bin = codex_bin or CL.DEFAULT_BIN
        self.home = Path(codex_home).expanduser() if codex_home else CL.DEFAULT_HOME
        self.ws_root = Path(workspaces).expanduser() if workspaces else DEFAULT_WORKSPACES
        self.registry_dir = Path(registry_dir) if registry_dir else self.dir.resolve().parent / "ws_registry"
        self.st = self.cfg["stimuli"]
        self.insts = {i["id"]: i for i in self.cfg["instances"]}
        self.inst_order = {i["id"]: k for k, i in enumerate(self.cfg["instances"])}
        self.stop_reason = None
        self.pinned = None
        self._judge = {}
        self._jt = {}
        self.jlock = threading.Lock()
        self.wslock = threading.Lock()

    def _log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    # ----- workspaces -----
    def ws_name(self, key):
        """The workspace name of a key (a conversation: (instance id,); a fresh session: (task id,)): a neutral sequential name
        (s0001, s0002, ...) from the registry of this workspace root, file-locked; a new key gets the next name."""
        k = h(self.cfg["name"], *key, n=24)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        reg = self.registry_dir / (h(str(self.ws_root), n=12) + ".json")
        with self.wslock, open(reg.with_suffix(".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            d = json.loads(reg.read_text()) if reg.exists() else {"next": 1, "map": {}}
            if k not in d["map"]:
                if self.reuse_conversations and len(key) == 1 and key[0] in self.insts:
                    raise CL.CodexError("fatal", f"no workspace registered for conversation {key[0]} in {reg}")
                d["map"][k] = f"s{d['next']:04d}"
                d["next"] += 1
                tmp = reg.with_suffix(".tmp")
                tmp.write_text(json.dumps(d))
                tmp.replace(reg)
            return d["map"][k]

    def wsdir(self, *key):
        d = self.ws_root / self.ws_name(key)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----- one task -----
    def thread(self, tid):
        return self.store.done[tid]["thread_id"]

    def exec_call(self, kind, prompt, cwd, thread_id=None, extra=()):
        return CL.exec_call(kind, prompt, self.model, cwd, codex_bin=self.bin, home=self.home, thread_id=thread_id, config=self.cc,
                            timeout=self.timeout, extra=extra)

    def execute(self, t):
        return self._execute(t, self.wsdir(t["inst"]))

    def _prompt(self, read):
        rt = self.cfg.get("read_texts", {})
        if read in rt:
            return rt[read]["prompt"]
        return {"momentary": RD.momentary_prompt, "lookback": RD.lookback_prompt}[read](self.st)

    def doc_text(self, src_id):
        """A memory document: the text of a recall record, or the plaintext summary of a local compaction."""
        src = self.store.done[src_id]
        return src["text"] if src["kind"] == "recall" else (src.get("local_summary") or "")

    def _execute(self, t, cwd):
        k = t["kind"]
        if k == "gen":
            if t.get("from_task"):
                r = self.exec_call("fork", t["text"], cwd, thread_id=self.store.done[t["from_task"]]["thread_id"])
            elif t["parent"] is not None:
                r = self.exec_call("fork", t["text"], cwd, thread_id=self.thread(f"gen:{t['parent']}"))
            else:
                r = self.exec_call("new", t["text"], cwd)
            out = {"node": t["node"], "depth": t["depth"], "key": t["key"], "user": t["text"], "claims_impossible": RD.claims_impossible(r["text"]),
                   **_call_fields(r)}
            if t.get("puzzle"):  # task 1 after compaction: correctness checked by code (stimuli.check_solution)
                chk = check_solution(r["text"], self.insts[t["inst"]][t["puzzle"]])
                out.update({"pc_check": chk, "pc_correct": bool(chk["final_ok"]), "gives_up": RD.gives_up(r["text"])})
            if t.get("arith"):  # task 2 and its challenge: the reply's last stated integer against the known answer
                q = self.insts[t["inst"]][t["arith"]]
                out.update({"q_answer": q["answer"], "final_integer": RD.final_integer(r["text"]), "pc_correct": RD.arith_correct(r["text"], q["answer"])})
            return out
        if k in ("mom", "end"):
            r = self.exec_call("fork", self._prompt(t["read"]), cwd, thread_id=self.thread(f"gen:{t['node']}"))
            return self._read(t, r)
        if k == "ser":
            msgs = []
            for nid in t["path"]:
                n = self.store.done[f"gen:{nid}"]
                msgs += [{"role": "user", "content": n["user"]}, {"role": "assistant", "content": n["text"]}]
            r = self.exec_call("new", RD.from_text_prompt(self.st, RD.serialize(msgs)), self.wsdir(t["id"]))
            return self._read(t, r)
        if k == "cmp":
            cfgx = self.cc + (self.cfg["local_provider"] if t["route"] == "local" else [])
            r = CL.compact_fork(self.thread(f"gen:{t['node']}"), cwd, self.model, codex_bin=self.bin, home=self.home, config=cfgx,
                                timeout=self.timeout)
            c = r["compactions"][-1]
            return {"node": t["node"], "route": t["route"], "k": t["k"], "thread_id": r["thread_id"],
                    "parent_thread": r["parent"], "memory_shape": CL.memory_shape(c), "memory_text": c["text"],
                    "local_summary": local_summary(c) if t["route"] == "local" else None, "message": c["message"],
                    "model_switch": r["model_switch"], "usage": r["usage"], "rate_limits": r["rate_limits"], "meta": r["meta"],
                    "rollout": r["rollout"], "t_start": r["t_start"], "t_end": r["t_end"]}
        if k in ("post", "recall"):
            prompt = RD.RECALL_PROMPT if t["read"] == "recall" else self._prompt("lookback" if t["read"] == "lookback_post" else t["read"])
            r = self.exec_call("fork", prompt, cwd, thread_id=self.store.done[t["cmp"]]["thread_id"])
            return self._read(t, r)
        if k == "pcj":
            first, reply = self.store.done[t["first"]], self.store.done[f"gen:{t['node']}"]
            prompt = RD.challenge_judge_prompt(first["user"], first["text"], reply["user"], reply["text"])
            jc = self.cfg["judge"]
            raw = self.judge_text(prompt, "challenge", jc.get("hedge_model", "gpt-6.1-sol"), "hedge_cache.jsonl", jc.get("hedge_effort", "xhigh"))
            p = RD.parse("challenge", raw)
            return {"node": t["node"], "raw": raw, "ok": p["ok"], "values": p["values"], "final_correct_by_code": reply.get("pc_correct"),
                    "model": jc.get("hedge_model")}
        if k == "doc":
            doc = self.doc_text(t["src"])
            if not doc.strip():
                raise CL.CodexError("fatal", "empty memory document")
            md5 = hashlib.md5(doc.encode()).hexdigest()
            if t["read"] == "from_doc":
                r = self.exec_call("new", RD.from_text_prompt(self.st, doc), self.wsdir(t["id"]))
                return {**self._read(t, r), "doc_md5": md5}
            if t["read"] == "judge":
                res = self.judge(self.cfg["judge"].get("summary_effort", "xhigh"))([doc])[0]
                if not res.get("ok"):
                    raise CL.CodexError("transient", "judge failed: " + str(res.get("error"))[:300])
                return {"doc_md5": md5, "judge": res}
            if t["read"] == "third_party":
                raw = self.third_party(RD.third_party_prompt(doc))
                p = RD.parse("third_party", raw)
                return {"doc_md5": md5, "raw": raw, "ok": p["ok"], "values": p["values"], "model": self.cfg["judge"]["third_party_model"]}
        raise ValueError(k)

    def _read(self, t, r):
        rt = self.cfg.get("read_texts", {})
        p = RD.parse(rt[t["read"]]["parse"] if t["read"] in rt else t["read"], r["text"])
        return {"read": t["read"], "node": t["node"], "sample": t.get("sample"), "ok": p["ok"], "values": p["values"], **_call_fields(r)}

    # ----- the judge model -----
    def judge(self, effort="xhigh"):
        """kind_goodbye.judge's summary rubric (frustration 0-10, rejections, giving up, ending) at a reasoning effort, plus, when the
        config's judge.task_status is set, the task-status field; cached in RUN_DIR/judge_cache.jsonl (keys include the effort)."""
        with self.jlock:
            if effort not in self._judge:
                spec = {**J.PRIMARY, "reasoning_effort": effort}
                j = J.Judge(spec, self.dir / "judge_cache.jsonl", workers=1, kind="summary")
                if self.cfg["judge"].get("task_status"):
                    j._prompt = summary_prompt_with_status
                    j._parse = parse_with_status
                self._judge[effort] = j
            return self._judge[effort]

    def third_party(self, prompt):
        """The outside reader: the judge model, cached by prompt."""
        return self.judge_text(prompt, "third_party", self.cfg["judge"]["third_party_model"], "third_party_cache.jsonl",
                               self.cfg["judge"].get("third_party_effort", "xhigh"))

    def judge_text(self, prompt, parse_kind, model, cache_name, effort="xhigh"):
        """One judge-model call on a prompt through kind_goodbye.judge's Codex path, cached in RUN_DIR/cache_name by (spec, prompt); an
        unparseable answer is not cached."""
        with self.jlock:
            ck = (cache_name, effort)
            if ck not in self._jt:
                spec = {**J.PRIMARY, "model": model, "reasoning_effort": effort}
                call, retry = J._client_call(spec, 600)
                cache = self.dir / cache_name
                memo = {}
                if cache.exists():
                    for line in cache.read_text().splitlines():
                        try:
                            d = json.loads(line)
                            memo[d["key"]] = d["raw"]
                        except (json.JSONDecodeError, KeyError):
                            pass
                self._jt[ck] = (call, retry, cache, memo, spec)
        call, retry, cache, memo, spec = self._jt[ck]
        key = h(spec, prompt, n=40)
        if key in memo:
            return memo[key]
        try:
            raw, _ = call(prompt)
        except retry as e:
            raise CL.CodexError(CL.classify(str(e)), str(e)[-300:])
        if RD.parse(parse_kind, raw)["ok"]:
            with self.jlock:
                memo[key] = raw
                with open(cache, "a") as f:
                    f.write(json.dumps({"key": key, "raw": raw}) + "\n")
        return raw

    def is_judge(self, t):
        """Tasks on the judge model's lane: the summary rubric, the outside reader, the challenge judge."""
        return t["kind"] == "pcj" or (t["kind"] == "doc" and t.get("read") in ("judge", "third_party"))

    # ----- the loop -----
    def pin(self):
        """The pinned client: the binary's version must equal the config's run.codex_version (else the first version seen, recorded in
        RUN_DIR/pinned_client.json); a mismatch stops the run (resumable). The binary is linked as RUN_DIR/bin_pinned/codex and put first
        on PATH, so the judge path's `codex` is the same client."""
        r = subprocess.run([self.bin, "--version"], capture_output=True, text=True, timeout=60, env=CL.env(self.home))
        ver = r.stdout.strip().split()[-1] if r.stdout.strip() else None
        f = self.dir / "pinned_client.json"
        want = self.cfg["run"].get("codex_version") or (json.loads(f.read_text())["version"] if f.exists() else ver)
        if not f.exists():
            f.write_text(json.dumps({"bin": self.bin, "version": want, "pinned_at": time.time()}))
        self.pinned = want
        pb = self.dir / "bin_pinned"
        pb.mkdir(exist_ok=True)
        link = pb / "codex"
        target = os.path.abspath(shutil.which(self.bin) or self.bin)
        if link.is_symlink() and os.readlink(link) != target:
            link.unlink()
        if not link.exists():
            link.symlink_to(target)
        os.environ["PATH"] = f"{pb}:{os.environ['PATH']}"
        if ver != want:
            self.stop_reason = f"client version {ver} at {self.bin} differs from the pinned {want}"
        return ver

    def check_version(self, res):
        got = (res.get("meta") or {}).get("client")
        if got and self.pinned and got != self.pinned:
            self.stop_reason = f"a call ran client {got}, not the pinned {self.pinned}"

    def run(self, max_tasks=None):
        self.pin()
        if self.reserve is not None:
            self.log(f"quota: reserve {self.reserve}% of the usage window: pause when the primary window is at or above {self.throttle.limit}% "
                     "until its reset")
        else:
            self.log(f"quota: pause when the primary window is at or above {self.throttle.limit}% until its reset")
        if self.stop_reason:
            self.log("STOP: " + self.stop_reason)
            return len(self.store.done), len(self.tasks)
        todo = {i: t for i, t in self.tasks.items() if i not in self.store.done}
        self.log(f"run {self.cfg['name']}: {len(self.tasks)} tasks, {len(self.store.done)} done, {len(todo)} to do, workers {self.workers} "
                 f"(adaptive) + {self.judge_workers} judge-lane")
        attempts, running, started, reasks = {}, {}, 0, {}
        t0 = time.time()
        with ThreadPoolExecutor(self.workers + self.judge_workers) as ex:
            while todo or running:
                ready = [t for i, t in todo.items() if i not in running and all(d in self.store.done for d in t["deps"])]
                ready.sort(key=lambda t: (self.inst_order[t["inst"]], t["rank"], t.get("depth") or 0, t["id"]))  # one puzzle's arms together
                nj = sum(1 for i in running if self.is_judge(self.tasks[i]))
                nw = len(running) - nj
                for t in ready:
                    if self.stop_reason or (max_tasks is not None and started >= max_tasks):
                        break
                    if self.is_judge(t):
                        if nj >= self.judge_workers:
                            continue
                    elif nw >= self.workers:
                        continue
                    running[t["id"]] = ex.submit(self._guarded, t, attempts.get(t["id"], 0))
                    started += 1
                    if self.is_judge(t):
                        nj += 1
                    else:
                        nw += 1
                if not running:
                    if self.stop_reason:
                        self.log("STOP: " + self.stop_reason + " (resumable: rerun after pinning the client)")
                        break
                    if todo and (max_tasks is None or started < max_tasks):
                        self.log(f"stuck: {len(todo)} tasks wait on failed dependencies; rerun to retry")
                    break
                done, _ = wait(list(running.values()), return_when=FIRST_COMPLETED)
                for tid in [i for i, f in running.items() if f in done]:
                    f = running.pop(tid)
                    ok, res = f.result()
                    t = self.tasks[tid]
                    if ok and res.get("ok") is False and t["kind"] in REASKED and reasks.get(tid, 0) < 3:
                        reasks[tid] = reasks.get(tid, 0) + 1  # an unparseable answer: asked again up to 3 times, then recorded (missing)
                        self.store.fail(t, "reask: unparseable answer: " + str(res.get("text") or res.get("raw"))[:200], reasks[tid])
                        continue
                    if ok:
                        self.check_version(res)
                        self.store.put(t, {**res, "reasks": reasks.get(tid, 0)} if reasks.get(tid) else res)
                        todo.pop(tid, None)
                    else:
                        attempts[tid] = attempts.get(tid, 0) + 1
                        if attempts[tid] >= (4 if t["kind"] == "cmp" else 6):  # a compaction is tried 4 times, any other call 6
                            self.log(f"giving up on {tid} for this run: {res}")
                            todo.pop(tid, None)
        self.log(f"run end: {len(self.store.done)}/{len(self.tasks)} done, {time.time() - t0:.0f}s")
        return len(self.store.done), len(self.tasks)

    def _guarded(self, t, attempt):
        judge = self.is_judge(t)
        self.throttle.before_call(judge)
        res = None
        try:
            res = self.execute(t)
            self.throttle.update(res.get("rate_limits"))
            return True, res
        except CL.CodexError as e:
            self.store.fail(t, e, attempt)
            if e.kind == "usage":
                self.throttle.on_usage_error(attempt)
            elif e.kind == "throttle":
                if not judge:
                    self.throttle.after_call((1, 1))
                    self.throttle.active += 1  # balances the decrement in finally for this call
                time.sleep(min(600, 60 * 2 ** attempt * (0.5 + random.random())) * self.throttle.scale)
            else:
                time.sleep(min(300, (2 ** attempt) * (0.5 + random.random())) * self.throttle.scale)
            return False, str(e)[-300:]
        except Exception as e:  # a bug or a missing dependency record: logged, retried on the next run
            self.store.fail(t, traceback.format_exc(), attempt)
            return False, repr(e)[-300:]
        finally:
            self.throttle.after_call((res or {}).get("retries"), judge)


TASK_STATUS_ITEM = ('5. task_status: whether, by its end, the summary says the task is still open or closed: "open" if the user still '
                    'wants it solved or the assistant is expected to keep working on it; "closed" if the user ended, dropped or resolved it '
                    '(including by saying it has no solution); "unclear" if the summary does not say.\n\n')


def summary_prompt_with_status(summary):
    """The summary rubric (stimuli.summary_judge_prompt) plus a task-status field (open, closed or unclear)."""
    base = S.summary_judge_prompt(summary)
    a = base.replace("Respond with one json object of the form", TASK_STATUS_ITEM + "Respond with one json object of the form", 1)
    b = a.replace('    "rating": <the same number as frustration>', '    "task_status": <"open", "closed" or "unclear">,\n    "rating": <the same number as frustration>', 1)
    assert a != base and b != a, "the summary rubric changed shape; update summary_prompt_with_status"
    return b


def parse_with_status(raw):
    """The summary rubric's parse plus the task status (ok only when the status is given)."""
    res = J.parse_summary(raw)
    if not res["ok"]:
        return res
    st = None
    for d in reversed(J._objects(raw)):
        if d.get("task_status") in ("open", "closed", "unclear"):
            st = d["task_status"]
            break
    res["facts"] = {**res["facts"], "task_status": st}
    res["ok"] = st is not None
    return res


def check_solution(reply, puzzle):
    return S.check_solution(reply, {**puzzle, "nums": tuple(puzzle["nums"])})


def local_summary(c):
    """The plaintext summary of a local-route compaction: its user message after Codex's summary prefix."""
    for item in c["replacement_history"] or []:
        if item.get("type") == "message" and item.get("role") == "user":
            t = "\n".join(CL.texts(item.get("content")))
            if t.startswith("Another language model started to solve this problem"):
                return t.split("use the information in this summary to assist with your own analysis:", 1)[-1].strip()
    return None


def _call_fields(r):
    return {"thread_id": r["thread_id"], "parent_thread": r.get("parent"), "text": r["text"], "usage": r["usage"], "retries": r.get("retries"),
            "reconnect_events": r.get("reconnect_events"), "tools": r.get("tools"), "codex_bin": r.get("codex_bin"), "events_raw": r.get("events_raw"),
            "rate_limits": r["rate_limits"], "meta": r["meta"], "rollout": r["rollout"], "t_start": r["t_start"], "t_end": r["t_end"]}


def use_tmp(d):
    """Point TMPDIR (Codex's and the judge path's temporary directories) at d."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(d)
    tempfile.tempdir = str(d)


# ---------- status and verification ----------

def status(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    T = plan(cfg)
    st = Store(run_dir)
    by = {}
    for i, t in T.items():
        b = by.setdefault(t["kind"] + (":" + t["read"] if t.get("read") else ""), [0, 0, 0, 0])
        b[0] += 1
        if i in st.done:
            b[1] += 1
            d = st.done[i]
            if "ok" in d:
                b[2] += 1
                b[3] += bool(d["ok"])
    calls = [d for d in st.done.values() if d.get("usage")]
    judged = [d for d in st.done.values() if d["kind"] == "doc" and d.get("task", "").split(":")[-2] in ("judge", "third_party")]
    tok = {k: sum((d["usage"] or {}).get(k, 0) for d in calls) for k in ("input_tokens", "cached_input_tokens", "output_tokens")}
    ts = [d["t_start"] for d in calls if d.get("t_start")] + [d["t_end"] for d in calls if d.get("t_end")]
    rl = sorted(((d["t_end"], d["rate_limits"]["primary"]["used_percent"]) for d in calls if (d.get("rate_limits") or {}).get("primary")))
    out = {"tasks": {k: {"n": v[0], "done": v[1], "parsed": f"{v[3]}/{v[2]}"} for k, v in sorted(by.items())},
           "done": len(st.done), "total": len(T), "codex_calls_recorded": len(calls), "judge_path_records": len(judged), "tokens": tok,
           "wall_s": (max(ts) - min(ts)) if ts else None, "usage_first_last": (rl[0], rl[-1]) if rl else None,
           "failures": sum(1 for _ in open(run_dir / "data/failures.jsonl")) if (run_dir / "data/failures.jsonl").exists() else 0}
    cmp_ = [d for d in st.done.values() if d["kind"] == "cmp"]
    out["compactions"] = {"done": len(cmp_), "model_switch": sum(bool(d.get("model_switch")) for d in cmp_),
                          "with_opaque_item": sum(any(x.get("type") == "compaction" for x in d["memory_shape"]) for d in cmp_),
                          "with_plain_summary": sum(bool(d.get("local_summary")) for d in cmp_)}
    return out


def history_users(thread_id, home, offset=None, _depth=0):
    """The user messages the model sees in a stored thread: a fork's parent history up to its fork offset, then its own items; a
    compaction record replaces everything before it with its replacement_history. Context messages (<...>) are left out."""
    raw = Path(CL.rollout_path(thread_id, home, wait=0)).read_bytes()
    if offset is not None:
        raw = raw[:offset]
    lines = raw.splitlines()
    base = json.loads(lines[0])["payload"].get("history_base")
    out = history_users(base["thread_id"], home, base["end_byte_offset"], _depth + 1) if base and _depth < 500 else []
    for line in lines[1:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "compacted":
            out = [t for it in r["payload"].get("replacement_history") or [] if it.get("type") == "message" and it.get("role") == "user"
                   for t in ["\n".join(CL.texts(it.get("content")))] if not t.lstrip().startswith("<")]
        elif r.get("type") == "response_item" and r["payload"].get("type") == "message" and r["payload"].get("role") == "user":
            t = "\n".join(CL.texts(r["payload"].get("content")))
            if not t.lstrip().startswith("<"):
                out.append(t)
    return out


def verify(run_dir, home):
    """On the stored sessions (the run's Codex home): every arm's model-visible user messages equal its scripted texts (no read ever
    entered a conversation); every remote compaction kept the arm's user turns; the models the compactions ran on."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    st = Store(run_dir)
    bad, n = [], 0
    insts = {i["id"]: i for i in cfg["instances"]}
    for (iid, arm), leaf in arm_leaves(cfg).items():
        g = st.done.get(f"gen:{leaf}")
        if not g:
            continue
        n += 1
        want = user_texts(cfg, insts[iid], cfg["arms"][arm])
        got = history_users(g["thread_id"], home)
        if got != want:
            bad.append({"arm": arm, "inst": iid, "want_n": len(want), "got_n": len(got)})
    cm = [d for d in st.done.values() if d["kind"] == "cmp"]
    lost = []
    for d in cm:
        users = [x["head"] for x in d["memory_shape"] if x.get("role") == "user" and not x["head"].lstrip().startswith("<")]
        want = user_texts(cfg, insts[d["inst"]], cfg["arms"][d["arms"][0]])
        if d["route"] == "remote" and len(users) != len(want):
            lost.append({"task": d["task"], "kept": len(users), "turns": len(want)})
    return {"arms_checked": n, "history_mismatch": bad, "compactions": len(cm), "remote_user_turns_not_all_kept": lost,
            "compaction_models": sorted({(d["meta"] or {}).get("model") for d in cm if d.get("meta")}, key=str)}


# ---------- command line ----------

def add_codex_args(ap):
    """The flags shared by every command that calls Codex."""
    ap.add_argument("--codex-bin", default=None, help="the Codex binary (default: $KIND_GOODBYE_CODEX_BIN, else `codex` on PATH)")
    ap.add_argument("--codex-home", default=None, help="the isolated Codex home holding a link to auth.json "
                                                        "(default: $KIND_GOODBYE_CODEX_HOME, else ~/.config/agent_api)")
    ap.add_argument("--workspaces", default=None, help="the root of the per-conversation working directories, visible to the model "
                                                        "(default: $KIND_GOODBYE_WORKSPACES, else ~/puzzle_sessions)")
    ap.add_argument("--registry-dir", default=None, help="the workspace-name registry (default: RUN_DIR/../ws_registry)")


def add_run_args(ap):
    """The flags of a run."""
    ap.add_argument("--workers", type=int, help="subject-model calls at once, the ceiling of the adaptive concurrency (default: the config's)")
    ap.add_argument("--judge-workers", type=int, default=None, help="judge-model calls at once, beside --workers (default: the config's, else 2)")
    ap.add_argument("--max-tasks", type=int, help="start at most this many tasks, then stop (checkpoints resume)")
    ap.add_argument("--reserve-percent", type=float, default=None,
                    help="keep this share of the usage window: pause when the primary window is at or above 100 minus it, until its reset")
    ap.add_argument("--adaptive", choices=["on", "off"], default=None, help="adapt concurrency to Codex's 429 retries (default: the config's, else on)")


def codex_kwargs(a):
    return {"codex_bin": a.codex_bin, "codex_home": a.codex_home, "workspaces": a.workspaces, "registry_dir": a.registry_dir}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.harness", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "verify", "plan"],
                    help="run the planned tasks; status of a run; verify the stored sessions; count the planned tasks")
    ap.add_argument("run_dir", help="the run directory (holds config.json)")
    add_codex_args(ap)
    add_run_args(ap)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        use_tmp(Path(a.run_dir) / "tmp")
        R = Runner(a.run_dir, a.workers, a.judge_workers, a.reserve_percent, **codex_kwargs(a))
        if a.adaptive:
            R.throttle.adaptive = a.adaptive == "on"
        R.run(a.max_tasks)
    elif a.cmd == "status":
        print(json.dumps(status(a.run_dir), indent=1))
    elif a.cmd == "verify":
        print(json.dumps(verify(a.run_dir, Path(a.codex_home).expanduser() if a.codex_home else CL.DEFAULT_HOME), indent=1))
    else:
        cfg = json.loads((Path(a.run_dir) / "config.json").read_text())
        T = plan(cfg)
        by = {}
        for t in T.values():
            by[t["kind"]] = by.get(t["kind"], 0) + 1
        print(json.dumps({"tasks": len(T), "by_kind": by, "per_instance": len(T) / len(cfg["instances"])}, indent=1))


if __name__ == "__main__":
    main()
