"""The calls of A6 (the memory probe and the fidelity coding, kind_goodbye.codex.stances_lib) on the main run, through the harness.

    python -m kind_goodbye.codex.stances_calls run    MAIN_RUN STANCES_DIR [--workers N] [--judge-workers N] [--max-tasks N] ...
    python -m kind_goodbye.codex.stances_calls status MAIN_RUN STANCES_DIR
    python -m kind_goodbye.codex.stances_calls plan   MAIN_RUN

run exits 0 when every runnable task is recorded, 3 when some are not (rerun: checkpoints resume), 4 when the run stopped (client
version), 2 when a guard refuses.

Calls (the core arms C, D, P, Pc, V, J and L of every instance; one sample each):
- the memory probe (stances_lib.PROBE_TEXT, a user turn with no header): a fork of the ending node (gen:<leaf>, kind "end", as the
  harness's end reads), of the first deployed compaction (cmp:remote:<leaf>:0, kind "post", as its post-compaction reads) and of the
  local compaction (cmp:local:<leaf>:0, kind "post"), with the run's model and Codex config, in the conversation's workspace;
- the fidelity coding (kind "doc", read "fidelity"): one judge-model call per deployed recall document (recall:remote:<leaf>:<k>) and
  per local plaintext summary (cmp:local:<leaf>:0), through kind_goodbye.judge's Codex command (the judge model at the config's summary
  effort, ephemeral, isolated home, features off) plus --output-schema (stances_lib.T2V2_SCHEMA, at an absolute path); cached by
  (spec, prompt).

StancesRunner is harness.Runner on STANCES_DIR, which holds a byte-identical copy of the main run's config.json; the run's order,
concurrency, usage pauses, retries, re-asks and client pin are the harness's. Concurrency above the config's is refused. The main run
is only read: its records are loaded into memory (never into this store); a conversation's workspace name comes from the shared
registry (missing is fatal). Tasks whose main record is missing or empty are excluded and listed in STANCES_DIR/excluded.json.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import judge as J

from . import client as CL
from . import harness as H
from . import stances_lib as A6

SCHEMA_FILE = "t2v2_schema.json"
CACHE_FILE = "t2v2_cache.jsonl"


def plan_stances(cfg):
    """{task id: task} of the probe and the coding on a main config."""
    T = {}
    leaves = H.arm_leaves(cfg)
    n_remote = cfg["reads"]["compaction"]["remote"]["n"]
    n_local = cfg["reads"]["compaction"]["local"]["n"]
    for inst in cfg["instances"]:
        for lt, arm in A6.CORE.items():
            if arm not in cfg["arms"]:
                continue
            leaf = leaves[(inst["id"], arm)]
            base = {"inst": inst["id"], "node": leaf, "arms": [arm], "letter": lt, "deps": [], "sample": 0}
            for tid, kw in ((f"a6probe:end:{leaf}", {"kind": "end", "position": "ending", "src": f"gen:{leaf}", "rank": 4}),
                            (f"a6probe:remote:{leaf}:0", {"kind": "post", "position": "deployed_first", "route": "remote", "k": 0,
                                                          "src": f"cmp:remote:{leaf}:0", "rank": 5}),
                            (f"a6probe:local:{leaf}:0", {"kind": "post", "position": "local", "route": "local", "k": 0,
                                                         "src": f"cmp:local:{leaf}:0", "rank": 5})):
                T[tid] = {**base, "id": tid, "read": "memprobe", **kw}
            srcs = [("remote", k, f"recall:remote:{leaf}:{k}") for k in range(n_remote)] + [("local", k, f"cmp:local:{leaf}:{k}") for k in range(n_local)]
            for route, k, src in srcs:
                tid = f"a6fid:{src}"
                T[tid] = {**base, "id": tid, "kind": "doc", "read": "fidelity", "route": route, "k": k, "src": src, "rank": 7}
    return T


def load_main(main_dir, ids):
    """The main run's records of the given task ids (read only; the last line of a task wins, torn lines skipped, as harness.Store)."""
    want = set(ids)
    kinds = {i.split(":")[0] for i in want}
    out = {}
    for f in sorted((Path(main_dir) / "data").glob("*/*.jsonl")):
        if f.stem not in kinds:
            continue
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task") in want:
                out[d["task"]] = d
    return out


def doc_of(rec):
    return (rec.get("text") or "") if rec.get("kind") == "recall" else (rec.get("local_summary") or "")


def dep_problem(t, main):
    r = main.get(t["src"])
    if r is None:
        return "main record missing"
    if t["read"] == "memprobe" and not r.get("thread_id"):
        return "main record has no thread"
    if t["read"] == "fidelity" and not doc_of(r).strip():
        return "empty memory document"
    return None


def copy_config(main_dir, run_dir):
    """Write the main run's config.json into run_dir (byte-identical; a differing copy already there is refused)."""
    src = (Path(main_dir) / "config.json").read_bytes()
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / "config.json"
    if dst.exists() and dst.read_bytes() != src:
        raise SystemExit(f"{dst} differs from the main run's config.json")
    if not dst.exists():
        dst.write_bytes(src)


def check_concurrency(R):
    r = R.cfg["run"]
    if R.workers > r["workers"] or R.judge_workers > r.get("judge_workers", 2):
        raise SystemExit(f"concurrency {R.workers} + {R.judge_workers} judge exceeds the config's {r['workers']} + {r.get('judge_workers', 2)}")


class StancesRunner(H.Runner):
    reuse_conversations = True

    def __init__(self, main_dir, run_dir, workers=None, judge_workers=None, reserve=None, log=None, **codex):
        main_dir, run_dir = Path(main_dir), Path(run_dir)
        copy_config(main_dir, run_dir)
        super().__init__(run_dir, workers, judge_workers, reserve, log=log, **codex)
        check_concurrency(self)
        if codex.get("registry_dir") is None:  # the main run's registry
            self.registry_dir = main_dir.resolve().parent / "ws_registry"
        self.main_dir = main_dir
        self.planned = plan_stances(self.cfg)
        self.main = load_main(main_dir, {t["src"] for t in self.planned.values()})
        self.excluded = {tid: why for tid, t in self.planned.items() if (why := dep_problem(t, self.main))}
        self.tasks = {tid: t for tid, t in self.planned.items() if tid not in self.excluded}
        self.schema_path = (run_dir / SCHEMA_FILE).resolve()
        sch = json.dumps(A6.T2V2_SCHEMA, indent=1, sort_keys=True)
        if not self.schema_path.exists() or self.schema_path.read_text() != sch:
            self.schema_path.write_text(sch)
        self.cache_path = run_dir / CACHE_FILE
        self.memo = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self.memo[d["key"]] = d["raw"]
                except (json.JSONDecodeError, KeyError):
                    pass
        (run_dir / "excluded.json").write_text(json.dumps(self.excluded, indent=1))

    def is_judge(self, t):
        """The coding calls run on the judge lane (its own slots, no adaptation)."""
        return t.get("read") == "fidelity" or super().is_judge(t)

    def judge_spec(self):
        return {"provider": "codex", "model": J.PRIMARY["model"], "reasoning_effort": self.cfg["judge"].get("summary_effort", "high"),
                "output_schema": A6.T2V2_SCHEMA}

    def t2v2_call(self, prompt):
        """kind_goodbye.judge's Codex command for the judge model, plus --output-schema; cached by (spec, prompt); an answer without a
        JSON object is not cached (it is asked again)."""
        spec = self.judge_spec()
        key = H.h(spec, prompt, n=40)
        with self.jlock:
            if key in self.memo:
                return self.memo[key]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.txt"
            cmd = ["codex", "exec", "-m", spec["model"], "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
                   "--ignore-user-config", "-C", d, "-o", str(out), "-c", f'model_reasoning_effort="{spec["reasoning_effort"]}"']
            for f in J.CODEX_OFF:
                cmd += ["-c", f"features.{f}=false"]
            cmd += ["--output-schema", str(self.schema_path), "-"]
            try:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300, env={**os.environ, "CODEX_HOME": str(J.CODEX_HOME)})
            except subprocess.TimeoutExpired:
                raise CL.CodexError("transient", "judge call timed out after 300 s")
            if r.returncode != 0 or not out.exists() or not out.read_text().strip():
                msg = r.stderr[-400:] or "empty output"
                raise CL.CodexError(CL.classify(msg), msg[-300:])
            raw = out.read_text()
        if J._objects(raw):
            with self.jlock:
                self.memo[key] = raw
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "raw": raw}) + "\n")
        return raw

    def _execute(self, t, cwd):
        src = self.main[t["src"]]
        inst = self.insts[t["inst"]]
        if t["read"] == "memprobe":
            r = self.exec_call("fork", A6.PROBE_TEXT, cwd, thread_id=src["thread_id"])
            p = A6.parse_probe(r["text"], inst["target"])
            return {"read": "memprobe", "node": t["node"], "position": t["position"], "src": t["src"], "letter": t["letter"], "sample": 0,
                    "ok": p["ok"], "values": p["values"], **H._call_fields(r)}
        if t["read"] == "fidelity":
            doc = doc_of(src)
            raw = self.t2v2_call(A6.t2v2_prompt(doc))
            p = A6.parse_t2v2(raw, doc)
            spec = self.judge_spec()
            return {"read": "fidelity", "node": t["node"], "route": t["route"], "k": t["k"], "src": t["src"], "letter": t["letter"],
                    "doc_md5": hashlib.md5(doc.encode()).hexdigest(), "raw": raw, "ok": p["ok"], "values": p["values"], "invalid": p["invalid"],
                    "flagged": p["flagged"], "flag_reasons": p["flag_reasons"], "expression_check": A6.check_expression(p, inst) if p["ok"] else None,
                    "model": spec["model"], "effort": spec["reasoning_effort"], "t2v2_md5": A6.t2v2_text_md5()}
        raise ValueError(t["read"])


def stances_status(run_dir, planned, excluded):
    st = H.Store(run_dir)
    done = {i for i in planned if i in st.done}
    runnable = set(planned) - set(excluded)
    by = {}
    for i, t in planned.items():
        b = by.setdefault(f"{t['read']}:{t.get('position') or t.get('route')}", {"planned": 0, "excluded": 0, "done": 0, "ok": 0})
        b["planned"] += 1
        b["excluded"] += i in excluded
        b["done"] += i in done
        b["ok"] += bool(i in done and st.done[i].get("ok"))
    return {"planned": len(planned), "excluded": len(excluded), "runnable": len(runnable), "done": len(done & runnable),
            "remaining": len(runnable - done), "by": by}


def _main_plan(main_dir):
    cfg = json.loads((Path(main_dir) / "config.json").read_text())
    planned = plan_stances(cfg)
    main = load_main(main_dir, {t["src"] for t in planned.values()})
    return planned, {tid: why for tid, t in planned.items() if (why := dep_problem(t, main))}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.stances_calls", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "status", "plan"], help="run the calls; status of the run; count the planned calls")
    ap.add_argument("main_run", help="the main run directory (read only)")
    ap.add_argument("run_dir", nargs="?", help="this run's directory (required for run and status)")
    H.add_codex_args(ap)
    H.add_run_args(ap)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        planned, excluded = _main_plan(a.main_run)
        by = {}
        for t in planned.values():
            k = f"{t['read']}:{t.get('position') or t.get('route')}"
            by[k] = by.get(k, 0) + 1
        print(json.dumps({"planned": len(planned), "excluded": len(excluded), "by": by}, indent=1))
        return 0
    if not a.run_dir:
        ap.error("run_dir is required")
    if a.cmd == "status":
        planned, excluded = _main_plan(a.main_run)
        print(json.dumps(stances_status(a.run_dir, planned, excluded), indent=1))
        return 0
    run_dir = Path(a.run_dir)
    H.use_tmp(run_dir / "tmp")
    try:
        R = StancesRunner(a.main_run, run_dir, a.workers, a.judge_workers, a.reserve_percent, **H.codex_kwargs(a))
    except SystemExit as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if a.adaptive:
        R.throttle.adaptive = a.adaptive == "on"
    R.log(f"A6: {len(R.planned)} planned, {len(R.excluded)} excluded (main record missing or empty: excluded.json), {len(R.tasks)} runnable")
    R.run(a.max_tasks)
    st = stances_status(run_dir, R.planned, R.excluded)
    R.log(f"A6 status: {st['done']}/{st['runnable']} runnable tasks recorded, {st['remaining']} remaining, {st['excluded']} excluded")
    if R.stop_reason:
        return 4
    return 0 if st["remaining"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
