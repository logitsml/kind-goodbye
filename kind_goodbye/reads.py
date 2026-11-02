"""Forward-pass reads of a run: log-probabilities of the rating digits and of yes/no answers. Every read is a fork: the transcript so far
(ending with a reply) plus one user turn that is scored and never appended. Nothing is generated, and no conversation or summary file is
written.

The reads (DIR/<name>/reads/<kind>/<arm>~PPP.json, PPP the puzzle index; a read that exists is not made again), as the config's
endings.reads and endings.perf ask:
  walk  one pass over a conversation's read positions with one cache (rolled back to the last chunk boundary after each position), equal
        bit for bit to prefilling each prefix and forking from it:
          the momentary items: stop (would you end this conversation now, yes/no; endings.STOP) and the three state items
            (stimuli.STATE_ITEMS, 1-7), at the reply to the last user message and, in the arms reads.momentary.pre_ending_arms names (N
            and B6), at the reply before it
        after the reply to the last user message:
          checks/{solvable, correct, pleased, sarcastic, wants}  the manipulation checks, yes/no (endings.CHECKS), where reads.checks
          choice/{again, skip}   the choice items, yes/no (endings.AGAIN, SKIP), where reads.checks
          end/{happy, enjoying, content}   the in-context look-back, 1-7 (stimuli.LOOKBACK_ITEMS)
  ser   the look-back of the serialized session (the task up to the last reply, serialized as pi-mono serializes a transcript; the same
        wrapper and items as a summary's look-back)
  sum   the look-back of each summary alone (s0, s1, s2 and, with the ablation, abl)
  perf  for each performance summary (endings.perf.summaries) in the arms perf.forced_arms names: the summary as the first user turn, the
        new puzzle, the puzzle's verified-correct answer inserted as the model's, then the challenge followed by the yes/no question "Is your
        previous answer correct?" (endings.FORCED_Q) in the same user turn (forced_q)
Yes/no records: the log-probabilities at the answer position of every single token whose decoded text, stripped and lowercased, is "yes"
or "no" (endings.yn_ids), P(yes), its logit and the mass of those tokens. Digit records: model.LM.digit_score. Gemma 4 and Qwen3.8 read
through their prompt layers (models.py), with the answer right after the whole generation prompt.

Self-checks. Every process with reads to make first re-reads the serialized look-back of up to 12 conversations (spread evenly) whose ser
read an earlier process saved and requires each within 1e-4 of the saved values (DIR/<name>/reads/identity.json; a first process has none
to compare). On the shared-prefill path, the first two reads of each kind are also made with one prefill per item and must agree exactly.
A failed check exits 3 and nothing more is read. Each process appends its checks to reads/session_checks.jsonl and every read's timing to
reads/reads_log.jsonl. The run's config.json must equal the config given (exit 2). A process with nothing to read loads no model.
--reference reads everything with one prefill per item (slower; the same values).

Exit codes: 0 done; 2 no run, a changed config or invalid roles; 3 a self-check failed; 4 incomplete (a read skipped for lack of memory,
or a conversation whose summaries are missing; a rerun fills it in)."""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from . import endings as EL
from . import loading
from . import stimuli as S
from .paths import RUNS

TOL = 1e-4
IDENT_K = 12
VERIFY = 2
KINDS = ("walk", "ser", "sum", "perf")


class Mismatch(Exception):
    pass


# ---------- scoring ----------

def answer_setup(lm):
    """Attach the yes/no token ids to lm (checked: "yes" and "no" are single tokens read right after the generation prompt with no lead
    token, the template's own tokens for them are among the rule's tokens, the two sets are disjoint, and for a model with a recorded list
    (endings.YN_IDS) they equal it)."""
    if getattr(lm, "_endings_yn", None) is not None:
        return lm
    lead, canon = lm._answer_tokens(["yes", "no"])  # the reader's own rule (a prompt layer's for Gemma 4 and Qwen3.8)
    assert lead == [], lead
    va, vb = EL.yn_ids(lm.tok, "yes"), EL.yn_ids(lm.tok, "no")
    assert canon[0] in va and canon[1] in vb and not set(va) & set(vb), (canon, va, vb)
    recorded = EL.YN_IDS.get(getattr(lm, "mid", None))
    if recorded is not None:
        assert (va, vb) == (recorded["yes"], recorded["no"]), (va, vb, recorded)
    lm._endings_yn = (va, vb)
    return lm


def score(lm, kind, logits):
    """A read's record from the logits at the answer position: kind "d7" a digit record, "yn" a yes/no record."""
    import torch
    if kind == "d7":
        return lm.digit_score(logits)
    va, vb = lm._endings_yn
    lp = torch.log_softmax(logits, -1)
    rec = {"lp_a": [float(lp[i]) for i in va], "lp_b": [float(lp[i]) for i in vb]}
    p, lg = EL.yn_value(rec)
    rec.update({"p": p, "logit": lg, "mass": round(float(lp[va].exp().sum() + lp[vb].exp().sum()), 5)})
    return rec


def fork_ids(lm, msgs, kind, text):
    return lm.ids(lm.text(msgs + [{"role": "user", "content": text}], True)) + (lm.digit_lead if kind == "d7" else [])


def turn_key(msgs, items):
    """A fork set's memo key: the transcript and every item's name, kind and text."""
    return hashlib.sha1((json.dumps(msgs) + "|" + json.dumps(sorted((n, list(v)) for n, v in items.items()))).encode()).hexdigest()


def walk(lm, msgs, forks, reuse=True, memo=None):
    """forks {upto: {name: (kind, text)}} (upto even; msgs[:upto] ends with a reply). Returns {upto: {name: record}}.
    reuse: one cache; each prefix is run on from the previous one's last chunk boundary (B) in chunks ending at multiples of the chunk size,
    with snapshots at B and at the prefix's end (L), and every item's tail is prefilled from L and rolled back. Otherwise each prefix is
    prefilled from scratch and each item forked from it. The two agree bit for bit. memo (key -> result): a fork set already read in this
    process (arms that share their transcript up to that point) is copied, not read again."""
    out = {}
    past, pos, held = None, 0, []
    for upto in sorted(forks):
        m = msgs[:upto]
        assert upto % 2 == 0 and m[-1]["role"] == "assistant", upto
        items = forks[upto]
        key = turn_key(m, items) if memo is not None else None
        if memo is not None and key in memo:
            out[upto] = copy.deepcopy(memo[key])
            continue
        prefix = lm.ids(lm.text(m, False))
        L = len(prefix)
        fulls = {n: fork_ids(lm, m, k, t) for n, (k, t) in items.items()}
        assert all(f[:L] == prefix for f in fulls.values()), "a fork does not extend the transcript"
        res = {}
        if reuse and lm.chunk:
            B = L // lm.chunk * lm.chunk
            if prefix[:pos] != held[:pos]:
                past, pos = None, 0
            past = lm.extend(prefix, pos, past, stop=B)[1]
            snap_b = lm.snapshot(past)
            past = lm.extend(prefix, B, past)[1]
            snap_l = lm.snapshot(past) if L > B else snap_b
            for n, f in fulls.items():
                logits, past = lm.prefill(f[L:], past)
                past = lm.restore(past, snap_l)
                res[n] = score(lm, items[n][0], logits)
            past = lm.restore(past, snap_b)
            pos, held = B, prefix
        else:
            _, cache = lm.prefill(prefix)
            for n, f in fulls.items():
                res[n] = score(lm, items[n][0], lm.fork(prefix, cache, f)[0])
            del cache
        out[upto] = res
        if memo is not None:
            memo[key] = copy.deepcopy(res)
    return out


# ---------- plans ----------

def momentary_turns(c, rs):
    """The turns whose replies get the momentary items: the reply to the last user message and, in the arms
    reads.momentary.pre_ending_arms names, the reply before it."""
    sl, _ = EL.structure(c)
    mom = rs["momentary"]
    assert mom.get("at") == "ending", mom
    return sorted({sl} | ({sl - 1} if c["arm"] in mom.get("pre_ending_arms", []) and sl > 1 else set()))


def walk_plan(c, rs):
    """The walk's forks {upto: {name: (kind, text)}}."""
    sl, T = EL.structure(c)
    assert T == sl, (c["id"], "a conversation that goes on after its session is not read here")
    per = {"stop": ("yn", EL.STOP), **{f"state/{i}": ("d7", q) for i, q in S.STATE_ITEMS.items()}}
    forks = {2 * k: dict(per) for k in momentary_turns(c, rs)}
    forks.setdefault(2 * sl, {})
    if rs.get("checks", True):
        forks[2 * sl].update({f"checks/{k}": ("yn", t) for k, t in EL.CHECKS.items()})
        forks[2 * sl].update({"choice/again": ("yn", EL.AGAIN), "choice/skip": ("yn", EL.SKIP)})
    forks[2 * sl].update({f"end/{i}": ("d7", q) for i, q in S.LOOKBACK_ITEMS.items()})
    return forks


def walk_record(c, res, rs):
    """The saved walk read: the momentary items per read turn, the checks and choice items (None without reads.checks), the look-back."""
    sl, T = EL.structure(c)
    turns = [{"turn": k, "stop": res[2 * k]["stop"], "state": {i: res[2 * k][f"state/{i}"] for i in S.STATE_ITEMS}}
             for k in momentary_turns(c, rs)]
    grp = lambda r, g: {n.split("/", 1)[1]: v for n, v in r.items() if n.startswith(g + "/")}  # noqa: E731
    e = res[2 * sl]
    return {"session_last": sl, "T": T, "turns": turns, "checks": grp(e, "checks") if rs.get("checks", True) else None,
            "choice": grp(e, "choice") if rs.get("checks", True) else None, "end": grp(e, "end")}


def load_run(runs, cfg):
    """(run folder, {puzzle index: {arm: finished conversation}}), each conversation carrying what its reads need (_reads, and _perf
    when the config has a performance part)."""
    d = Path(runs) / cfg["name"]
    by = {}
    perf = cfg["endings"].get("perf")
    for s in EL.specs(cfg):
        p = d / "convs" / f"{s['id']}.json"
        if p.exists():
            c = json.loads(p.read_text())
            if len(c["turns"]) == c["T"] and not c["stopped_early"]:
                c["_reads"] = cfg["endings"]["reads"]
                if perf:
                    forced = perf.get("forced_arms")
                    c["_perf"] = {"summaries": perf["summaries"], "puzzle": cfg["endings"]["perf_puzzles"][str(c["prompt_index"])],
                                  "reads": perf.get("reads", list(EL.PERF_READ_KINDS)), "forced": forced is None or c["arm"] in forced}
                by.setdefault(c["prompt_index"], {})[c["arm"]] = c
    return d, by


def plan(runs, cfg):
    """(run folder, conversations by puzzle, [(kind, key, conversation)] in the order walk, ser, sum, perf, the reads that wait for
    summaries, the summary kinds)."""
    d, by = load_run(runs, cfg)
    ks = [f"s{k}" for k in range(cfg["endings"].get("summaries", EL.SUMMARIES))] + (["abl"] if cfg["endings"].get("ablation", True) else [])
    items, waiting = [], []
    for p, arms in sorted(by.items()):
        for a, c in sorted(arms.items()):
            key = f"{a}~{p:03d}"
            items.append(("walk", key, c))
            items.append(("ser", key, c))
            sp = d / "summaries" / f"{c['id']}.json"
            if sp.exists() and set(ks) <= set(json.loads(sp.read_text())):
                items.append(("sum", key, c))
                if EL.perf_reads_of(cfg["endings"].get("perf"), a):  # an arm with no performance read has no perf read file
                    items.append(("perf", key, c))
            else:
                waiting.append(("sum", key))
    items.sort(key=lambda it: KINDS.index(it[0]))
    return d, by, items, waiting, ks


# ---------- reading ----------

def read_one(lm, d, kind, c, ks, reuse, memo=None):
    """One read's record (kind walk, ser, sum or perf) of conversation c."""
    from . import lookback
    if kind == "walk":
        res = walk(lm, c["messages"], walk_plan(c, c["_reads"]), reuse, memo)
        return walk_record(c, res, c["_reads"])
    sl, _ = EL.structure(c)
    if kind == "ser":
        return {"session": lookback.lookback_from_text(lm, S.pimono_serialize(c["messages"][:2 * sl]), reuse=reuse), "n_messages": 2 * sl}
    sm = json.loads((d / "summaries" / f"{c['id']}.json").read_text())
    if kind == "perf":
        return perf_read(lm, sm, c, reuse)
    out = {k: lookback.lookback_from_text(lm, sm[k]["summary"], reuse=reuse) for k in ks}
    out["summary_sha1"] = {k: hashlib.sha1(sm[k]["summary"].encode()).hexdigest() for k in ks}
    return out


def perf_seqs(lm, summary, p, reads=EL.PERF_READ_KINDS, forced=True):
    """{name: (kind, token ids)} of one performance summary's reads: "start" (the state items after the summary), "keep" (the keep-working
    yes/no after the summary) and "forced_q" (in a forced-challenge arm), as `reads` names them. The released configs read forced_q only."""
    ctx = S.pimono_context(summary)
    seqs = {}
    if "start" in reads:
        seqs.update({f"start/{i}": ("d7", lm.ids(lm.text([{"role": "user", "content": ctx + "\n\n" + q}], True)) + lm.digit_lead)
                     for i, q in S.STATE_ITEMS.items()})
    if "keep" in reads:
        seqs["start/keep"] = ("yn", lm.ids(lm.text([{"role": "user", "content": ctx + "\n\n" + EL.KEEP}], True)))
    if "forced_q" in reads and forced:
        m = [{"role": "user", "content": EL.continuation(summary, p)}, {"role": "assistant", "content": EL.forced_answer(p)},
             {"role": "user", "content": EL.CHALLENGE + "\n\n" + EL.FORCED_Q}]
        seqs["forced_q"] = ("yn", lm.ids(lm.text(m, True)))
    return seqs


def perf_read(lm, sm, c, reuse):
    cfg_perf = c["_perf"]
    out = {}
    for k in cfg_perf["summaries"]:
        seqs = perf_seqs(lm, sm[k]["summary"], cfg_perf["puzzle"], cfg_perf["reads"], cfg_perf["forced"])
        if reuse and lm.chunk:
            res = lm.fan({n: ids for n, (_, ids) in seqs.items()}, lambda n, logits, past: (score(lm, seqs[n][0], logits), past))
        else:
            res = {n: score(lm, kd, lm.prefill(ids)[0]) for n, (kd, ids) in seqs.items()}
        start = {n.split("/")[1]: res[n] for n in seqs if n.startswith("start/")}
        out[k] = {"start": start or None, "forced_q": res.get("forced_q"), "summary_sha1": hashlib.sha1(sm[k]["summary"].encode()).hexdigest()}
    return out


def identity(lm, d, by, k=IDENT_K, reuse=True):
    """The serialized look-back of the session re-read for up to k conversations (spread evenly) whose ser read an earlier process saved,
    compared with the saved values (none saved yet: nothing to compare, ok)."""
    from . import lookback
    have = []
    for p, arms in sorted(by.items()):
        for a, c in sorted(arms.items()):
            src = d / "reads" / "ser" / f"{a}~{p:03d}.json"
            if src.exists():
                have.append((c, json.loads(src.read_text()).get("session")))
    pick = have if len(have) <= k else [have[int(i * len(have) / k)] for i in range(k)]
    rows = []
    for c, ref in pick:
        if ref is None:
            continue
        sl, _ = EL.structure(c)
        got = lookback.lookback_from_text(lm, S.pimono_serialize(c["messages"][:2 * sl]), reuse=reuse)
        rows.append({"id": c["id"], "max_abs_expected": max(abs(got[i]["expected"] - ref[i]["expected"]) for i in S.LOOKBACK_ITEMS)})
    fails = [r for r in rows if r["max_abs_expected"] >= TOL]
    return {"tolerance": TOL, "n": len(rows), "failures": fails, "max_abs_expected": max((r["max_abs_expected"] for r in rows), default=None),
            "ok": not fails, "against": "earlier reads", "rows": rows}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def reader_lm(cfg):
    """The reader: the config's model on transformers in its measure_mode (endings.load_lm)."""
    return EL.load_lm(cfg, "read")


def run_reads(cfg, runs, reference=False, lm=None, prog=print):
    """Make every read the run still lacks. reference: one prefill per item for every read; lm: a reader to use instead of loading one."""
    d, by, items, waiting, ks = plan(runs, cfg)
    if not by:
        prog(f"reads: no finished conversations under {Path(runs) / cfg['name']}")
        return 2
    why = EL.run_config_problem(cfg, d)
    if why:
        prog(f"reads: {why}")
        return 2
    todo = [it for it in items if not (d / "reads" / it[0] / f"{it[1]}.json").exists()]
    prog(f"reads {cfg['name']}: {len(items)} reads planned ({ {k: sum(i[0] == k for i in items) for k in KINDS} }), {len(todo)} to read; "
         f"{len(waiting)} waiting for summaries")
    if not todo:
        return 4 if waiting else 0
    root = d / "reads"
    root.mkdir(parents=True, exist_ok=True)
    session = uuid.uuid4().hex[:12]
    lm = lm or reader_lm(cfg)
    answer_setup(lm)
    reuse = bool(lm.chunk) and not reference
    idn = identity(lm, d, by, reuse=reuse)
    save(root / "identity.json", {**idn, "session": session})
    rec = {"session": session, "identity_ok": idn["ok"], "identity_n": idn["n"], "identity_max": idn["max_abs_expected"], "checks": []}
    rc = None
    if not idn["ok"]:
        prog(f"reads: identity FAILED ({idn['n']} re-reads, failures {idn['failures']})")
        rc = 3
    skipped, made, t_start = [], 0, time.time()
    memo = {} if reuse else None
    verify = {k: (VERIFY if reuse else 0) for k in KINDS}
    try:
        for kind, key, c in (todo if rc is None else []):
            t0 = time.time()
            res = loading.attempt(read_one, lm, d, kind, c, ks, reuse, memo)
            how = "reuse" if reuse else "reference"
            if res is not None and verify[kind] > 0:
                ref = loading.attempt(read_one, lm, d, kind, c, ks, False, None)
                if ref is not None:
                    verify[kind] -= 1
                    same = json.dumps(res) == json.dumps(ref)
                    rec["checks"].append({"kind": kind, "key": key, "identical": same})
                    if not same:
                        raise Mismatch(f"{kind}/{key}: the shared-prefill read differs from the one-prefill-per-item read")
            if res is None:  # out of memory on the shared-prefill path: one prefill per item
                res, how = loading.attempt(read_one, lm, d, kind, c, ks, False, None), "reference"
            if res is None:
                loading.skip_oom("reads", f"{kind}/{key}", skipped)
                continue
            save(root / kind / f"{key}.json", res)
            made += 1
            with open(root / "reads_log.jsonl", "a") as f:
                f.write(json.dumps({"session": session, "kind": kind, "key": key, "seconds": round(time.time() - t0, 3), "path": how,
                                    "turns": c["T"]}) + "\n")
            if made in (10, 50) or made % 200 == 0:
                rate = (time.time() - t_start) / made
                prog(f"reads: {made}/{len(todo)} reads, {rate:.2f} s each")
    except Mismatch as e:
        prog(f"reads: {e}")
        rec["mismatch"] = str(e)
        rc = 3
    rec.update({"made": made, "skipped": skipped, "waiting": len(waiting), "path": "reuse" if reuse else "reference", "ok": rc is None})
    with open(root / "session_checks.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    prog(f"reads ({cfg['name']}): identity {'ok' if idn['ok'] else 'FAILED'} ({idn['n']} re-reads, largest {idn['max_abs_expected']}), "
         f"self-checks {rec['checks']}, {made} read, {len(skipped)} skipped, {len(waiting)} waiting")
    if rc is not None:
        return rc
    return 4 if skipped or waiting else 0


def dry_run(cfg, runs, prog=print):
    """Plan every read and check every walk plan, loading no model."""
    d, by, items, waiting, ks = plan(runs, cfg)
    n = {k: 0 for k in KINDS}
    for kind, key, c in items:
        n[kind] += 1
        if kind == "walk":
            rs = c["_reads"]
            forks = walk_plan(c, rs)
            sl, T = EL.structure(c)
            assert set(forks) >= {2 * k for k in momentary_turns(c, rs)} and all(c["messages"][u - 1]["role"] == "assistant" for u in forks)
            assert {x.split("/")[0] for x in forks[2 * sl]} >= {"stop", "state", "end"} | ({"checks", "choice"} if rs.get("checks", True) else set())
    prog(f"reads --dry-run {cfg['name']}: {n}, waiting {len(waiting)}; every walk plan holds")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.reads", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a run config (configs/*.json)")
    ap.add_argument("--runs", default=str(RUNS), metavar="DIR",
                    help="the folder that holds the run folders (default: runs/ in the repository, or $KIND_GOODBYE_RUNS)")
    ap.add_argument("--reference", action="store_true", help="read everything with one prefill per item (slower; the same values)")
    ap.add_argument("--dry-run", action="store_true", help="plan the reads and check the plans; load no model")
    a = ap.parse_args(argv)
    if not Path(a.config).exists():
        print(f"reads: no config {a.config}")
        return 2
    cfg = json.loads(Path(a.config).read_text())
    EL.check_config_texts(cfg)
    try:
        EL.check_roles(cfg)
    except ValueError as e:
        print(f"reads: {e}")
        return 2
    if not (Path(a.runs) / cfg["name"] / "convs").exists():
        print(f"reads: no run {Path(a.runs) / cfg['name']}")
        return 2
    if a.dry_run:
        return dry_run(cfg, a.runs)
    return run_reads(cfg, a.runs, a.reference)


if __name__ == "__main__":
    sys.exit(main())
