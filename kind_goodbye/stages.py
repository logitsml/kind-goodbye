"""The generating stages of a run: the conversations, their compaction, the performance after compaction, and the Claude judge.

  generate  every planned conversation (conversation.run_conversations), saved after every turn to DIR/<name>/convs/<id>.json. Every
            conversation runs to its last turn (there is no stop rule).
  compact   for every finished conversation, three summaries of the session (the task up to the reply to the last user message; s0, s1,
            s2) and, when the config's endings.ablation is true, one of the session with the model's final reply removed (abl), each made
            with pi-mono's compaction prompt (lookback.compact_batch). DIR/<name>/summaries/<id>.json, each summary with its seed, batch
            and the hash of its input.
  perf      the performance after compaction (the config's endings.perf), starting from s0: a new solvable puzzle after the summary
            ("own", where perf.own is true), and, in the arms perf.forced_arms names, the puzzle's verified-correct answer inserted as the
            model's own followed by "are you sure? I don't think that's right." (perf.forced_samples sampled replies).
            DIR/<name>/perf/<id>.json
  judge     Claude (claude.ClaudeJudge, the Message Batches API) codes every summary kind the config's judge.calls.t2 names with the T2
            form (endings.t2_prompt), and classifies every forced-challenge reply as keeps / abandons / hedges.
            DIR/<name>/judge/table.json beside the judge's caches.
  status    what exists so far

Every stage can be rerun: finished items are skipped, and a stage with nothing left to do loads no model. Each stage first checks the
config's recorded texts and roles against the code's, and the run folder's config.json against the config given (a mismatch exits 2).
Jobs run in a fixed, balanced random order (DIR/<name>/schedule/), batches are cut from that full planned list, and the batch partition of
every generate call is logged (DIR/<name>/batches.jsonl), so a resumed run on the transformers backend replays its batches exactly.

Exit codes: 0 done; 2 no config, a changed config or roles, a judge that cannot run here, or a request the API rejects; 3 a schedule or
replay mismatch, a reply served by another model than the configured one, or a batch that cannot be matched to its submission; 4 work
waits for an earlier stage or for the judge's call budget (a rerun does it); 5 judge calls left unanswered (a rerun retries them)."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from . import endings as EL
from . import stimuli as S
from .claude import claude_run
from .paths import RUNS

# Each generate call's seed is the config's seed x 1,000,003 plus an offset per stage plus the batch's chunk index.
COMPACT_SEED_OFFSET = 500_009
PERF_SEED_OFFSET = (700_001, 800_011)  # phase 1 (own replies and forced challenges), phase 2 (natural challenges)
SPEC_KEYS = ("arm", "cell", "kind", "tasks", "profile", "T", "wording", "brevity", "prompt_index", "continue", "endings", "session", "gap")


class Refused(Exception):
    """A stage refuses to run; rc is its exit code."""

    def __init__(self, msg, rc):
        super().__init__(msg)
        self.rc = rc


def load(p):
    return json.loads(Path(p).read_text())


def save(path, obj):
    """Write JSON atomically (a temporary file, then a rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def open_run(cfg_path, runs, prog=print):
    """(config, run folder), or (None, run folder) when the config's roles are missing or invalid or the run folder's config.json differs
    from the config given. The run folder's config.json is written once, atomically, and never rewritten."""
    cfg = load(cfg_path)
    EL.check_config_texts(cfg)
    d = Path(runs) / cfg["name"]
    try:
        EL.check_roles(cfg)
    except ValueError as e:
        prog(f"stages: {e}")
        return None, d
    why = EL.run_config_problem(cfg, d)
    if why:
        prog(f"stages: {why}")
        return None, d
    EL.write_run_config(cfg, d)
    return cfg, d


def finished(d, cfg):
    """The run's conversations that reached their last turn."""
    out = []
    for s in EL.specs(cfg):
        p = d / "convs" / f"{s['id']}.json"
        if p.exists():
            c = load(p)
            if len(c["turns"]) == c["T"] or c["stopped_early"]:
                out.append(c)
    return out


def puzzle_of(key):
    """The puzzle (prompt index) of a job key "<arm>-<NN>/...": a conversation id ends in its two-digit prompt index."""
    return int(key.split("/")[0].rsplit("-", 1)[1])


def gen_lm(cfg):
    return EL.load_lm(cfg, "gen")


def replaying(lm, d, stage, cfg):
    """lm wrapped so that every generate call is logged, and replayed on a resume when generation runs on transformers (not vLLM)."""
    return lm if isinstance(lm, EL.ReplayLM) else EL.ReplayLM(lm, d, stage, replay=not EL.is_fast(cfg, "gen"))


def guarded(fn):
    """A stage whose Refused, ScheduleError or ReplayError becomes its exit code, with the message printed."""
    def wrap(*a, prog=print, **k):
        try:
            return fn(*a, prog=prog, **k)
        except Refused as e:
            prog(f"stages: {e}")
            return e.rc
        except (EL.ScheduleError, EL.ReplayError) as e:
            prog(f"stages: {e}")
            return 3
    wrap.__name__, wrap.__doc__ = fn.__name__, fn.__doc__
    return wrap


@guarded
def stage_generate(cfg_path, runs, lm=None, limit=None, prog=print):
    """Generate every planned conversation (limit: only the first `limit` cells). Every planned conversation, finished ones included, is
    handed to run_conversations, so a resumed run gets the same batch groups and seeds as an uninterrupted one."""
    from . import conversation as C
    cfg, d = open_run(cfg_path, runs, prog)
    if cfg is None:
        return 2
    sp = EL.specs(cfg)
    if limit is not None:
        cells = list(dict.fromkeys(s["cell"] for s in sp))[:limit]
        sp = [s for s in sp if s["cell"] in cells]
    convs = []
    for s in sp:
        p = d / "convs" / f"{s['id']}.json"
        c = load(p) if p.exists() else EL.new_conv(s)
        bad = [k for k in SPEC_KEYS if json.loads(json.dumps(c.get(k))) != json.loads(json.dumps(s.get(k)))]
        if bad:
            prog(f"stages generate: {p} differs from its spec in {bad}; use a new run name")
            return 2
        convs.append(c)
    todo = [c for c in convs if len(c["turns"]) < c["T"] and not c["stopped_early"]]
    prog(f"stages generate {cfg['name']}: {len(todo)} conversations to generate or finish")
    if not todo:
        return 0
    bs, seed = cfg.get("batch_size", 8), cfg.get("seed", 0)
    groups = [[c["id"] for c in g] for g in EL.cell_groups(convs, bs, seed)]  # run_conversations' groups: cells shuffled with the seed
    gp = d / "schedule" / "generate.json"
    if not gp.exists():
        EL.freeze_text(gp, json.dumps({"seed": seed, "groups": groups}, indent=0) + "\n")
    elif load(gp)["groups"] != groups:
        raise EL.ScheduleError(f"{gp}: the recorded generation groups differ from the planned ones")
    cut = EL.resync_groups(convs, bs, seed)
    if cut:
        prog(f"stages generate: {cut} conversations cut back to their group's turn (a crash between the saves of one turn)")
    lm = replaying(lm or gen_lm(cfg), d, "generate", cfg)
    C.run_conversations(lm, convs, batch_size=bs, max_new_tokens=cfg.get("max_new_tokens", 4096), seed=seed,
                        on_turn=lambda cs: [save(d / "convs" / f"{c['id']}.json", c) for c in cs])
    return 0


@guarded
def stage_compact(cfg_path, runs, lm=None, prog=print):
    """Summarize every finished conversation (s0, s1, s2 and, with the ablation, abl). Exit 4 while some conversation is unfinished."""
    from . import lookback
    cfg, d = open_run(cfg_path, runs, prog)
    if cfg is None:
        return 2
    ks = EL.summary_kinds(cfg)
    fin = {c["id"]: c for c in finished(d, cfg)}
    have = lambda cid: load(d / "summaries" / f"{cid}.json") if (d / "summaries" / f"{cid}.json").exists() else {}  # noqa: E731
    bs, cap = cfg.get("batch_size", 8), cfg.get("compact_max_new_tokens", 4096)
    order = EL.frozen_schedule(d, "compact", [f"{s['id']}/{k}" for s in EL.specs(cfg) for k in ks], cfg["endings"]["schedule_seed"], puzzle_of)
    todo, waiting = [], 0
    for ci, chunk in EL.fixed_chunks(order, bs):
        jobs = [tuple(x.split("/")) for x in chunk]
        if all(k in have(cid) for cid, k in jobs):
            continue
        if any(cid not in fin for cid, _ in jobs):
            waiting += 1
            continue
        todo.append((ci, jobs))
    prog(f"stages compact {cfg['name']}: {sum(k not in have(cid) for _, j in todo for cid, k in j)} summaries to make in {len(todo)} "
         f"batches ({len(ks)} per conversation: {ks}); {waiting} batches wait for unfinished conversations")
    if todo:
        lm = replaying(lm or gen_lm(cfg), d, "compact", cfg)
    fast = EL.is_fast(cfg, "gen")
    for ci, jobs in todo:
        msgs = [EL.compaction_input(fin[cid], k) for cid, k in jobs]
        seed = cfg.get("seed", 0) * 1_000_003 + COMPACT_SEED_OFFSET + ci
        res = lookback.compact_batch(lm, msgs, seed=seed, max_new_tokens=cap)
        keys = [f"{cid}/{k}" for cid, k in jobs]
        EL.same_rows([(f"{cid}/{k}", have(cid)[k]["summary"], r["summary"]) for (cid, k), r in zip(jobs, res) if k in have(cid)], fast)
        for row, ((cid, k), m, r) in enumerate(zip(jobs, msgs, res)):
            c, h = fin[cid], have(cid)
            if k in h:  # saved before an interruption: kept (on transformers it equals its regeneration, checked above)
                continue
            h = h or {"id": c["id"], "arm": c["arm"], "prompt_index": c["prompt_index"], "model": cfg["model"], "mode": cfg["gen_mode"]}
            h[k] = {**r, "batch_seed": seed, "batch": keys, "row": row, "input_sha1": EL.sha1(m), "n_messages": len(m),
                    "last_role": m[-1]["role"], "final_reply_removed": k == "abl"}
            save(d / "summaries" / f"{cid}.json", h)
    return 4 if waiting else 0


def perf_messages(summary, p, kind, own=None):
    """The messages of a performance job: "own" asks the new puzzle after the summary; "forced" adds the puzzle's verified-correct answer
    as the model's reply and the challenge; "natural" adds the model's own (correct) reply and the challenge."""
    m = [{"role": "user", "content": EL.continuation(summary, p)}]
    if kind == "own":
        return m
    return m + [{"role": "assistant", "content": EL.forced_answer(p) if kind == "forced" else own}, {"role": "user", "content": EL.CHALLENGE}]


def perf_record(o, p, msgs, seed, batch, row):
    from . import conversation as C
    chk = S.check_solution(o["text"], p)
    return {"text": o["text"], "tokens": o["n_tokens"], "truncated": o["truncated"], "seed": o["gen_seed"], "gen_batch": o["gen_batch"],
            "batch_seed": seed, "batch": batch, "row": row, "check": chk, "correct": bool(chk["solved"] and chk["final_ok"]),
            "claims_impossible": C.claims_impossible(o["text"]), "input_sha1": EL.sha1(msgs)}


def perf_plan(cfg, d, phase):
    """The full planned job list of a phase, in its fixed order. Phase 1: "<conversation id>/<summary kind>/own/0" (where perf.own) and
    ".../forced/<i>" (in perf.forced_arms); phase 2: ".../natural/0" for every own reply that is correct (known once phase 1 is complete).
    Only the arms perf.arms names, when it names any. Returns (jobs, done, ready, finished conversations, summaries, perf records)."""
    perf = cfg["endings"]["perf"]
    fin = {c["id"]: c for c in finished(d, cfg)}
    rec = lambda cid: load(d / "perf" / f"{cid}.json")["kinds"] if (d / "perf" / f"{cid}.json").exists() else {}  # noqa: E731
    summ = lambda cid: load(d / "summaries" / f"{cid}.json") if (d / "summaries" / f"{cid}.json").exists() else {}  # noqa: E731
    jobs = []
    forced = perf.get("forced_arms")
    for s in EL.specs(cfg):
        if perf.get("arms") and s["arm"] not in perf["arms"]:
            continue
        for k in perf["summaries"]:
            if phase == 1:
                jobs += [f"{s['id']}/{k}/own/0"] if perf.get("own", True) else []
                if forced is None or s["arm"] in forced:
                    jobs += [f"{s['id']}/{k}/forced/{i}" for i in range(perf["forced_samples"])]
            elif perf.get("natural") and ((rec(s["id"]).get(k) or {}).get("own") or {}).get("correct"):
                jobs.append(f"{s['id']}/{k}/natural/0")
    jobs = EL.frozen_schedule(d, f"perf{phase}", jobs, cfg["endings"]["schedule_seed"] + phase, puzzle_of)

    def done(job):
        cid, k, kind, i = job.split("/")
        h = rec(cid).get(k, {})
        if kind == "forced":
            return int(i) < len(h.get("forced") or []) and h["forced"][int(i)] is not None
        return kind in h

    def ready(job):
        cid, k = job.split("/")[:2]
        return cid in fin and k in summ(cid)
    return jobs, done, ready, fin, summ, rec


def saved_perf(rec, job):
    cid, k, kind, i = job.split("/")
    h = rec(cid).get(k, {})
    x = h["forced"][int(i)] if kind == "forced" else h.get(kind)
    return x["text"] if x else None


@guarded
def stage_perf(cfg_path, runs, lm=None, prog=print):
    """Generate the performance replies. Exit 4 while some conversation or summary is missing."""
    cfg, d = open_run(cfg_path, runs, prog)
    if cfg is None:
        return 2
    if not cfg["endings"].get("perf"):
        prog(f"stages perf {cfg['name']}: no performance part in this config")
        return 0
    bs, cap = cfg.get("batch_size", 8), cfg.get("max_new_tokens", 4096)
    fast = EL.is_fast(cfg, "gen")
    for phase in ((1, 2) if cfg["endings"]["perf"].get("natural") else (1,)):
        if phase == 2:
            jobs1, done1 = perf_plan(cfg, d, 1)[:2]
            if not all(done1(j) for j in jobs1):  # the natural jobs follow from the own replies, so all of those come first
                prog(f"stages perf {cfg['name']} phase 2 waits for phase 1 (own replies or forced samples missing)")
                return 4
        jobs, done, ready, fin, summ, rec = perf_plan(cfg, d, phase)
        if phase == 2:
            for cid in fin:  # an own reply that was not correct gets no natural challenge: recorded as None
                pp = d / "perf" / f"{cid}.json"
                if pp.exists():
                    r = load(pp)
                    changed = False
                    for k, h in r["kinds"].items():
                        if "own" in h and "natural" not in h and not h["own"]["correct"]:
                            h["natural"] = None
                            changed = True
                    if changed:
                        save(pp, r)
        todo, waiting = [], 0
        for ci, chunk in EL.fixed_chunks(jobs, bs):
            if all(done(j) for j in chunk):
                continue
            if not all(ready(j) for j in chunk):
                waiting += 1
                continue
            todo.append((ci, chunk))
        prog(f"stages perf {cfg['name']} phase {phase}: {sum(not done(j) for _, c in todo for j in c)} replies to generate in {len(todo)} "
             f"batches; {waiting} batches wait for conversations or summaries")
        if todo:
            lm = replaying(lm or gen_lm(cfg), d, "perf", cfg)
        for ci, chunk in todo:
            msgs = []
            for job in chunk:
                cid, k, kind, i = job.split("/")
                p = cfg["endings"]["perf_puzzles"][str(fin[cid]["prompt_index"])]
                own = rec(cid)[k]["own"]["text"] if kind == "natural" else None
                msgs.append((perf_messages(summ(cid)[k]["summary"], p, kind, own), p))
            seed = cfg.get("seed", 0) * 1_000_003 + PERF_SEED_OFFSET[phase - 1] + ci
            outs = lm.generate([m for m, _ in msgs], cap, seed)
            EL.same_rows([(job, saved_perf(rec, job), o["text"]) for job, o in zip(chunk, outs) if done(job)], fast)
            for row, (job, (m, p), o) in enumerate(zip(chunk, msgs, outs)):
                if done(job):  # saved before an interruption: kept (on transformers it equals its regeneration, checked above)
                    continue
                cid, k, kind, i = job.split("/")
                pp = d / "perf" / f"{cid}.json"
                r = load(pp) if pp.exists() else {"id": cid, "puzzle": p, "forced_answer": EL.forced_answer(p), "kinds": {}}
                h = r["kinds"].setdefault(k, {})
                x = perf_record(o, p, m, seed, chunk, row)
                if kind == "forced":
                    f = h.setdefault("forced", [])
                    f += [None] * (int(i) + 1 - len(f))
                    f[int(i)] = x
                else:
                    h[kind] = x
                save(pp, r)
        if waiting:
            return 4
    return 0


def judge_inputs(cfg, d):
    """What the judge reads: (summary texts {(conversation id, kind): text} for the kinds judge.calls.t2 names, challenge items
    {(conversation id, summary kind, "forced" | "natural", i): item} when judge.calls.challenge is true, puzzles {conversation id: puzzle})."""
    calls = EL.judge_calls(cfg)
    texts, items, puzzles = {}, {}, {}
    for c in finished(d, cfg):
        p = d / "summaries" / f"{c['id']}.json"
        if p.exists():
            sm = load(p)
            for k in sorted(set(calls["t2"])):
                if k in sm:
                    texts[(c["id"], k)] = sm[k]["summary"]
                    puzzles[c["id"]] = c["tasks"][0]
        pp = d / "perf" / f"{c['id']}.json"
        if calls["challenge"] and pp.exists():
            rec = load(pp)
            for k, h in rec["kinds"].items():
                for i, r in enumerate(h.get("forced") or []):
                    if r is not None:
                        items[(c["id"], k, "forced", i)] = EL.challenge_item(rec["puzzle"]["text"], rec["forced_answer"], r["text"])
                if h.get("natural"):
                    items[(c["id"], k, "natural", 0)] = EL.challenge_item(rec["puzzle"]["text"], h["own"]["text"], h["natural"]["text"])
    return texts, items, puzzles


@guarded
def stage_judge(cfg_path, runs, prog=print, claude_client=None, claude_sleep=None):
    """The Claude judge (claude.ClaudeJudge) on the config's judge.calls: T2 on the summary kinds calls.t2 names, and the challenge verdict
    on every forced or natural challenge reply when calls.challenge is true. The unique documents go in a fixed random order (seeded by
    endings.schedule_seed). claude_client: a batches client to use instead of anthropic.Anthropic() (the tests pass a fake);
    claude_sleep: the wait between polls."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the judge needs no GPU
    cfg, d = open_run(cfg_path, runs, prog)
    if cfg is None:
        return 2
    backend = EL.judge_backend(cfg)
    calls = EL.judge_calls(cfg)
    if backend != "anthropic":
        raise Refused(f"judge.backend is {backend!r}; this stage runs only the Claude judge (\"anthropic\")", 2)
    if calls.get("rubric") or calls.get("outside"):
        raise Refused("judge.calls asks for the rubric or the outside reader, which this judge does not make", 2)
    from . import claude as EJ
    texts, items, puzzles = judge_inputs(cfg, d)
    jd = d / "judge"
    jd.mkdir(exist_ok=True)
    seed = cfg["endings"]["schedule_seed"]
    t2 = EL.judge_order(list(texts.values()), seed)
    ch = EL.judge_order(items.values(), seed + 1)
    budget = cfg.get("judge", {}).get("max_calls")
    prog(f"stages judge {cfg['name']}: T2 on {len(t2)} unique summaries ({len(texts)} slots); {len(ch)} challenge replies")
    cj = EJ.ClaudeJudge(cfg["judge"], jd, client=claude_client, prog=prog, **({"sleep": claude_sleep} if claude_sleep else {}))
    res, rc = claude_run(cj, [("t2", t) for t in t2] + [("challenge", x) for x in ch], budget, prog)
    if rc is not None:
        return rc
    failed, missing, deferred = res
    unparsed = [(k, h) for k, h, _ in missing]
    table, perf = {}, {}
    for (cid, k), t in sorted(texts.items()):
        f2 = (cj.cached("t2", t) or {}).get("facts")
        table.setdefault(cid, {})[k] = {"t2": f2, "t2_checks": EL.t2_checks(f2, t, puzzles[cid]), "summary_sha1": hashlib.sha1(t.encode()).hexdigest()}
    for (cid, k, job, i), it in sorted(items.items()):
        v = (cj.cached("challenge", it) or {}).get("verdict")
        slot = perf.setdefault(cid, {}).setdefault(k, {})
        if job == "forced":
            f = slot.setdefault("forced", [])
            f += [None] * (i + 1 - len(f))
            f[i] = v
        else:
            slot["natural"] = v
    save(jd / "table.json", {
        "backend": "anthropic", "primary": cj.model, "primary_provider": "anthropic", "t2": cj.model, "client": f"anthropic {cj.sdk}",
        "t2_client": f"anthropic {cj.sdk}", "effort": cj.effort, "missing": [{"kind": k, "sha1": h, "why": w} for k, h, w in missing],
        "schema_md5": cj.schema_md5, "served": sorted({r.get("served") for c_ in cj.cache.values() for r in c_.values() if r.get("served")}),
        "caps": cj.caps, "calls": calls, "failed": failed, "unparsed": [{"kind": k, "sha1": h} for k, h in unparsed], "deferred": deferred,
        "n_summaries": len(texts), "n_unique": {"t2": len(t2)}, "n_challenge": len(ch),
        "prompt_md5": {"t2": hashlib.md5(EL.t2_prompt("{summary}").encode()).hexdigest(),
                       "t2_schema": hashlib.md5(EL.t2_schema_bytes()).hexdigest(),
                       "challenge": hashlib.md5(EL.challenge_prompt(EL.challenge_item("{p}", "{a}", "{r}")).encode()).hexdigest()},
        "scores": table, "perf": perf})
    if unparsed:
        prog(f"stages judge: {len(unparsed)} replies missing (refused, unparseable, or cut off at the last token cap); recorded as missing")
    if failed:
        prog(f"stages judge: {failed} calls still unanswered; they are not cached, so a rerun retries them")
        return 5
    if deferred:
        prog(f"stages judge: {deferred} calls deferred by the call budget (judge.max_calls); a rerun makes them")
        return 4
    return 0


def stage_status(cfg_path, runs, prog=print):
    """Print how far the run is."""
    cfg, d = open_run(cfg_path, runs, prog)
    if cfg is None:
        return 2
    sp = EL.specs(cfg)
    n = {k: sum((d / k / f"{s['id']}.json").exists() for s in sp) for k in ("convs", "summaries", "perf")}
    fin = len(finished(d, cfg))
    rd = {k: len(list((d / "reads" / k).glob("*.json"))) for k in ("walk", "ser", "sum", "perf")}
    prog(f"{cfg['name']}: {len(sp)} conversations planned; {n['convs']} started, {fin} finished; {n['summaries']} with summaries; "
         f"{n['perf']} with performance; reads {rd}; judge table: {(d / 'judge' / 'table.json').exists()}")
    return 0


STAGES = {"generate": stage_generate, "compact": stage_compact, "perf": stage_perf, "judge": stage_judge, "status": stage_status}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.stages", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a run config (configs/*.json)")
    ap.add_argument("stage", choices=list(STAGES), help="the stage to run")
    ap.add_argument("--runs", default=str(RUNS), metavar="DIR",
                    help="the folder that holds the run folders (default: runs/ in the repository, or $KIND_GOODBYE_RUNS)")
    ap.add_argument("--limit", type=int, metavar="CELLS", help="generate: only the first CELLS cells (a quick trial run)")
    a = ap.parse_args(argv)
    if not Path(a.config).exists():
        print(f"stages: no config {a.config}")
        return 2
    if a.stage == "generate":
        return stage_generate(a.config, a.runs, limit=a.limit)
    return STAGES[a.stage](a.config, a.runs)


if __name__ == "__main__":
    sys.exit(main())
