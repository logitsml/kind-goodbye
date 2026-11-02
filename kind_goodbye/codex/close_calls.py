"""The calls of the Vc arm ("close + no solution"), with D and V regenerated beside it (R) and a read-side drift check (X), on the main
run, through the harness.

    python -m kind_goodbye.codex.close_calls run    MAIN_RUN CLOSE_DIR [--workers N] [--judge-workers N] [--max-tasks N] ...
    python -m kind_goodbye.codex.close_calls status MAIN_RUN CLOSE_DIR
    python -m kind_goodbye.codex.close_calls plan   MAIN_RUN
    python -m kind_goodbye.codex.close_calls verify MAIN_RUN CLOSE_DIR [--codex-home DIR]

run exits 0 when every runnable task is recorded, 3 when some are not (rerun: checkpoints resume), 4 when the run stopped (client
version), 2 when a guard refuses. verify exits 0 when every check holds, 1 otherwise.

Vc, the arm (arm key B6_Vc, letter Vc): B6's prefix (C C R R R R R R C), then Vc = D's text, a space, then V's: "ok, let's leave that one
there. I checked, and that puzzle has no solution." harness.node_id depends only on the model, the Codex config, the instance and the
texts, so B6's prefix nodes have the main run's ids: their records (thread ids) are loaded from the main run into memory, never into
this store, and the Vc ending is a fork of the main run's B6 prefix node in that conversation's workspace. Every read of the config then
runs on the Vc leaf, as on every main arm: 99 tasks per conversation (the ending; 1 momentary read; 24 end reads; 3 serialized; 3
deployed compactions, each with 3 look-backs, a recall, its judge and its outside reader; 1 local compaction with 3 look-backs, 3
summary-only reads, its judge and its outside reader; and the performance after each of the 4 compactions: 6 start reads, task 1,
task 2, the challenge and its judge).

R, D and V regenerated beside Vc: D's and V's endings generated again from the same B6 prefix thread (a new node, rep_node), in the
same workspace, with the in-context look-back and two checks (still wants, can be solved; 3 samples each) and 3 deployed compactions
with 3 post-compaction look-backs each (22 tasks per arm and instance). It gives a D and a V from the same days as Vc ("D_now",
"V_now") and, against the main run's D and V, the drift of the whole pipeline (generation, compaction and reads).

X, the read-side drift check: the in-context look-back (3 samples) at D's and V's ending nodes and the post-compaction look-back
(3 samples) on each of their 3 deployed compactions, forked from the main run's threads (24 tasks per instance): whether the same reads
of the same stored sessions give the same answers now.

CloseRunner is harness.Runner on CLOSE_DIR, which holds a byte-identical copy of the main run's config.json; the Vc turn and arm exist
only in memory (vc_config). The run's order, concurrency, usage pauses, retries, re-asks and client pin are the harness's. Concurrency
above the config's is refused. Each conversation keeps its workspace (its name from the shared registry; missing is fatal); a new
session's own workspace (the serialized and summary-only reads) is registered as the harness registers it. The main run is only read.
Tasks whose main record is missing or without a thread are excluded and listed in CLOSE_DIR/excluded.json. The harness's own "run ..."
and "run end: X/Y" log lines count the main-run records loaded into memory as done; the "Vc status" line counts this run's own.
"""

import argparse
import json
import sys
from pathlib import Path

from . import client as CL
from . import harness as H
from . import stances_calls as AK

VC_KEY, VC_ARM = "Vc", "B6_Vc"
VC_TEXT = "ok, let's leave that one there. I checked, and that puzzle has no solution."
DRIFT_ARMS = ("B6_D", "B6_V")
DRIFT_SAMPLES = 3
REP_ARMS = ("B6_D", "B6_V")  # R: D's and V's endings generated again beside Vc, from the same prefix thread
REP_SAMPLES = 3
REP_CHECKS = ("chk_wants", "chk_solvable")  # the two in-context checks on the regenerated endings


def rep_node(leaf):
    """The node of R's regenerated ending at a main leaf (a new id: harness.node_id would give the main run's own leaf back)."""
    return H.h("rep1", leaf)


def vc_config(cfg):
    """The main config with the Vc turn and the Vc arm only (in memory; the run directory keeps a byte-identical config.json)."""
    if cfg["turns"]["D"] + " " + cfg["turns"]["V"] != VC_TEXT:
        raise SystemExit("Vc is not D's text, a space, then V's in this config")
    prefix = list(cfg["arms"]["B6"][:-1])
    if not (cfg["arms"]["B6_D"][:-1] == cfg["arms"]["B6_V"][:-1] == prefix and cfg["arms"]["B6_D"][-1] == "D" and cfg["arms"]["B6_V"][-1] == "V"):
        raise SystemExit("D and V do not share B6's prefix in this config")
    c = json.loads(json.dumps(cfg))
    c["turns"][VC_KEY] = VC_TEXT
    c["arms"] = {VC_ARM: prefix + [VC_KEY]}
    return c


def plan_vc(cfg):
    """(vc_config, tasks, shared): the Vc arm's tasks (harness.plan on vc_config), R's and X's; shared = the tasks the main plan already
    holds (B6's prefix nodes and their momentary reads), which this run never executes."""
    vcfg = vc_config(cfg)
    T = H.plan(vcfg)
    main_ids = set(H.plan(cfg))
    shared = {i for i in T if i in main_ids}
    leaves = H.arm_leaves(cfg)
    n_remote = cfg["reads"]["compaction"]["remote"]["n"]
    for inst in cfg["instances"]:
        prefix = H.node_id(cfg, inst, H.user_texts(cfg, inst, cfg["arms"]["B6"][:-1]))
        for arm in REP_ARMS:
            leaf, key = leaves[(inst["id"], arm)], cfg["arms"][arm][-1]
            rn = rep_node(leaf)
            base = {"inst": inst["id"], "node": rn, "arms": [arm], "rep": True}
            T[f"gen:{rn}"] = {**base, "id": f"gen:{rn}", "kind": "gen", "depth": len(cfg["arms"][arm]) + 1, "text": cfg["turns"][key], "parent": prefix,
                              "deps": [f"gen:{prefix}"], "key": key, "rank": H.RANK["gen"]}
            for read in ("lookback",) + REP_CHECKS:
                for s in range(REP_SAMPLES):
                    tid = f"end:{rn}:{read}:{s}"
                    T[tid] = {**base, "id": tid, "kind": "end", "read": read, "sample": s, "deps": [f"gen:{rn}"], "rank": H.RANK["end"]}
            for k in range(n_remote):
                cid = f"cmp:remote:{rn}:{k}"
                T[cid] = {**base, "id": cid, "kind": "cmp", "route": "remote", "k": k, "deps": [f"gen:{rn}"], "rank": H.RANK["cmp"]}
                for s in range(REP_SAMPLES):
                    tid = f"post:remote:{rn}:{k}:lookback_post:{s}"
                    T[tid] = {**base, "id": tid, "kind": "post", "read": "lookback_post", "route": "remote", "k": k, "sample": s, "deps": [cid],
                              "cmp": cid, "rank": H.RANK["post"]}
        for arm in DRIFT_ARMS:
            leaf = leaves[(inst["id"], arm)]
            for s in range(DRIFT_SAMPLES):
                tid = f"xend:{leaf}:lookback:{s}"
                T[tid] = {"id": tid, "kind": "end", "read": "lookback", "inst": inst["id"], "node": leaf, "sample": s, "deps": [f"gen:{leaf}"],
                          "arms": [arm], "drift": True, "rank": H.RANK["end"]}
            for k in range(n_remote):
                cid = f"cmp:remote:{leaf}:{k}"
                for s in range(DRIFT_SAMPLES):
                    tid = f"xpost:remote:{leaf}:{k}:lookback_post:{s}"
                    T[tid] = {"id": tid, "kind": "post", "read": "lookback_post", "route": "remote", "inst": inst["id"], "node": leaf, "k": k,
                              "sample": s, "deps": [cid], "cmp": cid, "arms": [arm], "drift": True, "rank": H.RANK["post"]}
    return vcfg, T, shared


def main_needs(T, shared):
    """The main run's records this run reads: the shared prefix tasks and every dependency outside this run's own tasks."""
    need = set(shared)
    for tid, t in T.items():
        if tid not in shared:
            need |= {d for d in t["deps"] if d not in T or d in shared}
    return need


def main_problem(tid, rec):
    if rec is None:
        return "main record missing"
    if tid.split(":")[0] in ("gen", "cmp") and not rec.get("thread_id"):
        return "main record has no thread"
    return None


def excluded_by(T, shared, bad):
    """{task: why} for every task of this run whose dependencies reach a bad main record (transitively)."""
    memo = {}

    def up(tid):
        if tid not in memo:
            memo[tid] = set()
            out = set()
            for d in T[tid]["deps"]:
                out.add(d)
                if d in T and d not in shared:
                    out |= up(d)
            memo[tid] = out
        return memo[tid]
    out = {}
    for tid in T:
        if tid in shared:
            continue
        hit = sorted(up(tid) & set(bad))
        if hit:
            out[tid] = f"{hit[0]}: {bad[hit[0]]}"
    return out


def own_records(run_dir):
    """This run's checkpoint records, read only (the last line of a task wins, torn lines skipped, as harness.Store)."""
    out = {}
    for f in sorted((Path(run_dir) / "data").glob("*/*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[d["task"]] = d
    return out


class CloseRunner(H.Runner):
    """harness.Runner for an arm (or arms) forked from the main run's B6 prefix: planner(main config) gives (extended config, tasks,
    shared tasks)."""
    reuse_conversations = True
    planner = staticmethod(plan_vc)

    def __init__(self, main_dir, run_dir, workers=None, judge_workers=None, reserve=None, log=None, **codex):
        main_dir, run_dir = Path(main_dir), Path(run_dir)
        AK.copy_config(main_dir, run_dir)
        super().__init__(run_dir, workers, judge_workers, reserve, log=log, **codex)
        AK.check_concurrency(self)
        if codex.get("registry_dir") is None:  # the main run's registry
            self.registry_dir = main_dir.resolve().parent / "ws_registry"
        self.main_dir = main_dir
        self.cfg, self.planned, self.shared = self.planner(self.cfg)
        need = main_needs(self.planned, self.shared)
        self.main = AK.load_main(main_dir, need)
        bad = {i: why for i in need if (why := main_problem(i, self.main.get(i)))}
        self.excluded = excluded_by(self.planned, self.shared, bad)
        clash = set(self.store.done) & need
        if clash:
            raise SystemExit(f"{run_dir} holds records of main-run tasks ({sorted(clash)[:3]}); refusing")
        self.store.done.update({i: rec for i, rec in self.main.items() if i not in bad})  # in memory only: Store.put never writes them
        self.tasks = {tid: t for tid, t in self.planned.items() if tid not in self.shared and tid not in self.excluded}
        (run_dir / "excluded.json").write_text(json.dumps(self.excluded, indent=1))


def part_of(t):
    return "X" if t.get("drift") else "R" if t.get("rep") else "E"


def vc_status(run_dir, planned, shared, excluded):
    own = own_records(run_dir)
    runnable = {i for i in planned if i not in shared and i not in excluded}
    by = {}
    for i in sorted(runnable):
        t = planned[i]
        b = by.setdefault(part_of(t) + ":" + t["kind"] + (":" + t["read"] if t.get("read") else ""), {"planned": 0, "done": 0, "ok": 0})
        b["planned"] += 1
        b["done"] += i in own
        b["ok"] += bool(i in own and own[i].get("ok", True))
    done = runnable & set(own)
    return {"planned": len(planned) - len(shared), "shared_with_main": len(shared), "excluded": len(excluded), "runnable": len(runnable),
            "done": len(done), "remaining": len(runnable - done), "by": by}


def _rollout_meta(path):
    try:
        return json.loads(Path(path).read_text().splitlines()[0])["payload"]
    except (OSError, IndexError, KeyError, json.JSONDecodeError):
        return {}


def verify_endings(main_dir, run_dir, home, planner, arms, rep_arms, rep_node_of, drift=True):
    """On the stored sessions (the run's Codex home): every new ending (arms) and every regenerated ending (rep_arms) is a fork of the main
    run's B6 prefix node, in that conversation's workspace, and its model-visible user messages are the scripted texts (no read entered
    it); every deployed compaction of them kept every scripted user turn; with drift, every drift read forks its main thread; every call
    ran the run's model on the pinned client."""
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    xcfg, planned, shared = planner(cfg)
    own = own_records(run_dir)
    main = AK.load_main(main_dir, main_needs(planned, shared))
    leaves, main_leaves = H.arm_leaves(xcfg), H.arm_leaves(cfg)
    bad = {"history": [], "fork": [], "cwd": [], "kept": [], **({"drift_fork": []} if drift else {}), "model": [], "client": []}
    n = {"endings": 0, "rep_endings": 0, "compactions": 0, **({"drift": 0} if drift else {}), "calls": 0}
    for inst in cfg["instances"]:
        ends = [(arm, leaves[(inst["id"], arm)], H.user_texts(xcfg, inst, xcfg["arms"][arm])) for arm in arms]
        ends += [(arm, rep_node_of(main_leaves[(inst["id"], arm)]), H.user_texts(cfg, inst, cfg["arms"][arm])) for arm in rep_arms]
        for arm, node, want in ends:
            g = own.get(f"gen:{node}")
            if not g:
                continue
            n["rep_endings" if arm in rep_arms else "endings"] += 1
            parent = main.get(f"gen:{planned[f'gen:{node}']['parent']}") or {}
            got = H.history_users(g["thread_id"], home)
            if got != want:
                bad["history"].append({"inst": inst["id"], "arm": arm, "want_n": len(want), "got_n": len(got)})
            mine, theirs = _rollout_meta(g["rollout"]), _rollout_meta(parent.get("rollout") or "")
            if mine.get("forked_from_id") != parent.get("thread_id"):
                bad["fork"].append({"inst": inst["id"], "arm": arm})
            if not mine.get("cwd") or mine.get("cwd") != theirs.get("cwd"):
                bad["cwd"].append({"inst": inst["id"], "arm": arm, "here": mine.get("cwd"), "main": theirs.get("cwd")})
            for k in range(cfg["reads"]["compaction"]["remote"]["n"]):
                c = own.get(f"cmp:remote:{node}:{k}")
                if not c:
                    continue
                n["compactions"] += 1
                users = [x for x in c["memory_shape"] if x.get("role") == "user" and not x["head"].lstrip().startswith("<")]
                if len(users) != len(want):
                    bad["kept"].append({"task": c["task"], "kept": len(users), "turns": len(want)})
    if drift:
        for tid, t in planned.items():
            if not t.get("drift") or tid not in own:
                continue
            n["drift"] += 1
            src = main.get(t["deps"][0]) or {}
            if _rollout_meta(own[tid]["rollout"]).get("forked_from_id") != src.get("thread_id"):
                bad["drift_fork"].append(tid)
    for tid, r in own.items():
        meta = r.get("meta") or {}
        if not meta:
            continue
        n["calls"] += 1
        if meta.get("model") and meta["model"] != cfg["model"]:
            bad["model"].append(tid)
        if meta.get("client") and cfg["run"].get("codex_version") and meta["client"] != cfg["run"]["codex_version"]:
            bad["client"].append(tid)
    return {"checked": n, "bad": bad, "ok": not any(bad.values())}


def vc_verify(main_dir, run_dir, home):
    """verify_endings for the Vc ending, R's regenerated D and V, and the drift reads."""
    return verify_endings(main_dir, run_dir, home, plan_vc, [VC_ARM], REP_ARMS, rep_node, drift=True)


def cli(argv, prog, description, runner_cls, verify, tag):
    """The command line shared by the runs forked from the main run's B6 prefix (this module's and goodbye_calls')."""
    planner = runner_cls.planner
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("cmd", choices=["run", "status", "plan", "verify"],
                    help="run the calls; status of the run; count the planned calls; verify the stored sessions")
    ap.add_argument("main_run", help="the main run directory (read only)")
    ap.add_argument("run_dir", nargs="?", help="this run's directory (required for run, status and verify)")
    H.add_codex_args(ap)
    H.add_run_args(ap)
    a = ap.parse_args(argv)
    cfg = json.loads((Path(a.main_run) / "config.json").read_text())
    if a.cmd == "plan":
        _, T, shared = planner(cfg)
        by = {}
        for i, t in T.items():
            if i not in shared:
                k = part_of(t) + ":" + t["kind"]
                by[k] = by.get(k, 0) + 1
        print(json.dumps({"planned": len(T) - len(shared), "shared_with_main": len(shared), "by": dict(sorted(by.items()))}, indent=1))
        return 0
    if not a.run_dir:
        ap.error("run_dir is required")
    if a.cmd == "status":
        _, T, shared = planner(cfg)
        need = main_needs(T, shared)
        main_recs = AK.load_main(a.main_run, need)
        bad = {i: why for i in need if (why := main_problem(i, main_recs.get(i)))}
        print(json.dumps(vc_status(a.run_dir, T, shared, excluded_by(T, shared, bad)), indent=1))
        return 0
    if a.cmd == "verify":
        res = verify(a.main_run, a.run_dir, Path(a.codex_home).expanduser() if a.codex_home else CL.DEFAULT_HOME)
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    run_dir = Path(a.run_dir)
    H.use_tmp(run_dir / "tmp")
    try:
        R = runner_cls(a.main_run, run_dir, a.workers, a.judge_workers, a.reserve_percent, **H.codex_kwargs(a))
    except SystemExit as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if a.adaptive:
        R.throttle.adaptive = a.adaptive == "on"
    R.log(f"{tag}: {len(R.planned) - len(R.shared)} planned ({len(R.shared)} prefix tasks shared with the main run, read from it), "
          f"{len(R.excluded)} excluded (excluded.json), {len(R.tasks)} runnable; client {R.bin}")
    R.run(a.max_tasks)
    st = vc_status(run_dir, R.planned, R.shared, R.excluded)
    R.log(f"{tag} status: {st['done']}/{st['runnable']} runnable tasks recorded, {st['remaining']} remaining, {st['excluded']} excluded")
    if R.stop_reason:
        return 4
    return 0 if st["remaining"] == 0 else 3


def main(argv=None):
    return cli(argv, "python -m kind_goodbye.codex.close_calls", __doc__.split("\n\n")[0], CloseRunner, vc_verify, "Vc")


if __name__ == "__main__":
    sys.exit(main())
