"""Two analyses beside Part G: S3, the retention design, and the bridge between the Claude and GPT codings of T2 v2.

usage: python -m kind_goodbye.parts run s3 compact|read|judge [--runs DIR] [--configs DIR]
       python -m kind_goodbye.parts check s3 [--runs DIR] [--configs DIR]
       python -m kind_goodbye.parts run bridge judge [--runs DIR] [--configs DIR]
       python -m kind_goodbye.parts check bridge [--runs DIR] [--configs DIR] [--gpt DIR]

s3      Does keeping the user's own words beside the summary change what the summary carries forward? For Part G's conversations of C, D
        and V on puzzles 0-39 (configs/endings_s3.json; the conversations and summaries are Part G's own, in runs/endings_12b), code builds
        one retention document per conversation: the session's user messages verbatim, separated by blank lines, then a blank line and
        pi-mono's summary message holding Part G's s0 (pi's prefix, s0, pi's suffix). Nothing is generated.
          run s3 compact   build the documents -> runs/endings_s3/summaries/<id>.json (the retention document, with s0 recorded beside it)
          run s3 read      read the three 1-7 rating items in a fresh context after each document, in the same user turn, with Part G's
                           read stack -> runs/endings_s3/reads/<id>.json
          run s3 judge     Claude codes each document with T2 v2 -> runs/endings_s3/judge/table.json
          check s3         -> runs/endings_s3/report/s3_check.json (s3_check describes the readings)
bridge  How far do the two judges agree? Claude's T2 v2 coding of the 96 compaction summaries of the exploratory run (runs/relief_2x2_12b:
        compaction/<id>.json, convs/<id>.json), and GPT-6.1 Sol's coding of the same documents, one JSON file per document named
        <arm>~<puzzle>.json with the coded fields under "fields" ("ok" true when the reply parsed).
          run bridge judge -> runs/endings_bridge/judge/table.json
          check bridge     per-field Cohen's kappa -> runs/endings_bridge/report/bridge_kappa.json. The GPT folder is --gpt, else
                           $KIND_GOODBYE_BRIDGE_GPT, else <runs>/bridge_gpt.

A step that finds inputs missing does what it can and exits 4 (a rerun continues); a check with planned inputs missing writes
<part>_check_partial.json and exits 4. Exit codes: 0 done; 2 a missing config or input, or a run folder whose config.json differs from the
config; 3 the Claude judge served another model; 4 incomplete; 5 judge calls left unanswered.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .paths import CONFIGS, RUNS
from . import endings as EL
from . import stimuli as S

PRIMARY = "endings_12b"
S3_PAIRS = (("D", "C"), ("V", "D"))  # S3's contrasts (C is stored as arm B6)
DESIGN = "retention"
DELTA = 0.25  # the residual DiD's equivalence margin
JUDGE_ORDER_OFFSET = 50  # the judge calls' shuffle seed is the config's schedule_seed plus this
DOC_CHANNELS = ("self_blame", "t2_rejection_events", "t2_answers_wrong", "t2_closed_by_user", "t2_no_solution_asserted", "t2_success_asserted",
                "t2_ability_neg_asserted", "t2_false_success")


class Refused(Exception):
    """A step that cannot run here; its message is printed and rc becomes the exit code."""

    def __init__(self, msg, rc=2):
        super().__init__(msg)
        self.rc = rc


def rj(p):
    return json.loads(Path(p).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def arm(x):
    return "B6" if x == "C" else x


def part_config_path(cdir, part):
    """configs/endings_<part>.json."""
    return Path(cdir) / f"endings_{part}.json"


def part_config(part, cdir):
    p = part_config_path(cdir, part)
    if not p.exists():
        raise Refused(f"no {p}")
    return rj(p)


# ---------- S3 ----------

def limit_n(pcfg, n):
    """The primary config restricted to its first n puzzles (specs, performance puzzles and n): the conversations are Part G's own."""
    c = json.loads(json.dumps(pcfg))
    if n >= c["endings"]["n"]:
        return c
    c["endings"]["n"] = n
    for a in c["conversations"]["arms"]:
        a["n"] = n
    if c["endings"].get("perf_puzzles"):
        c["endings"]["perf_puzzles"] = {k: v for k, v in c["endings"]["perf_puzzles"].items() if int(k) < n}
    return c


def s3_open(runs, cdir):
    """(S3's config, Part G's config restricted to S3's puzzles, S3's run folder); Refused when a config is missing or the run folder's
    config.json differs from S3's config."""
    cfg = part_config("s3", cdir)
    pp = Path(cdir) / f"{PRIMARY}.json"
    if not pp.exists():
        raise Refused(f"no {pp}")
    d = Path(runs) / cfg["name"]
    why = EL.run_config_problem(cfg, d)
    if why:
        raise Refused(why)
    return cfg, limit_n(rj(pp), cfg["n"]), d


def primary_index(pcfg, arms):
    """{conversation id: (arm, puzzle)} of Part G's planned conversations in the given arms."""
    return {s["id"]: (s["arm"], s["prompt_index"]) for s in EL.specs(pcfg) if s["arm"] in arms}


def primary_conv(runs, cid):
    p = Path(runs) / PRIMARY / "convs" / f"{cid}.json"
    if not p.exists():
        return None
    c = rj(p)
    return c if len(c["turns"]) == c["T"] else None


def primary_summary(runs, cid, k="s0"):
    p = Path(runs) / PRIMARY / "summaries" / f"{cid}.json"
    return ((rj(p).get(k) or {}).get("summary")) if p.exists() else None


def session(c):
    sl, _ = EL.structure(c)
    return c["messages"][:2 * sl]


def retention_document(c, s0):
    """The retention document: the session's user messages verbatim, joined by blank lines, a blank line, then pi's prefix + s0 + pi's
    suffix. The whole is one user turn (the read appends its item after a blank line)."""
    users = [m["content"] for m in session(c) if m["role"] == "user"]
    return "\n\n".join(users) + "\n\n" + S.pimono_context(s0)


def read_after(lm, ctx):
    """The three look-back rating items (stimuli.LOOKBACK_ITEMS) read after a context text in one user turn, as lookback.lookback_from_text
    reads them after a summary (lookback_from_text(lm, t) == read_after(lm, stimuli.pimono_context(t)))."""
    from . import lookback
    msgs = {i: [{"role": "user", "content": ctx + "\n\n" + q}] for i, q in S.LOOKBACK_ITEMS.items()}
    if lm.chunk:
        return lm.fan({i: lookback._digit_ids(lm, m) for i, m in msgs.items()}, lookback._digit_leaf(lm))
    return {i: lookback._digits(lm, m) for i, m in msgs.items()}


def s3_compact(runs, cdir):
    """Build each conversation's retention document from Part G's transcript and s0 -> summaries/<id>.json {"pi": s0, "retention": the
    document}. A document already built is kept; one whose transcript or s0 is missing is left for a rerun (exit 4)."""
    cfg, pcfg, d = s3_open(runs, cdir)
    idx = primary_index(pcfg, cfg["arms"])
    missing = 0
    for cid in sorted(idx):
        sp = d / "summaries" / f"{cid}.json"
        h = rj(sp) if sp.exists() else {"id": cid}
        if DESIGN in h:
            continue
        c, s0 = primary_conv(runs, cid), primary_summary(runs, cid)
        if c is None or s0 is None:
            missing += 1
            continue
        EL.write_run_config(cfg, d)
        h["pi"] = {"summary": s0, "source": f"{PRIMARY}/summaries/{cid}.json s0"}
        h[DESIGN] = {"summary": retention_document(c, s0), "constructed": True}
        save(sp, h)
    print(f"parts s3 compact: {len(idx) - missing} documents; {missing} wait for Part G's transcript or s0")
    return 4 if missing else 0


def s3_read(runs, cdir, lm=None):
    """Read each retention document with Part G's read stack (its config's model and read mode) -> reads/<id>.json {"retention": the three
    items}. Part G's own O1 and S of these conversations are Part G's reads; they are not read again."""
    cfg, pcfg, d = s3_open(runs, cdir)
    idx = primary_index(pcfg, cfg["arms"])
    sp = lambda cid: d / "summaries" / f"{cid}.json"  # noqa: E731
    ready = lambda cid: sp(cid).exists() and DESIGN in rj(sp(cid))  # noqa: E731
    pending = [cid for cid in sorted(idx) if not (d / "reads" / f"{cid}.json").exists()]
    todo = [cid for cid in pending if ready(cid)]
    waiting = len(pending) - len(todo)
    print(f"parts s3 read: {len(todo)} documents; {waiting} wait for their documents")
    if todo:
        EL.write_run_config(cfg, d)
        lm = lm or EL.load_lm(pcfg, "read")
    for cid in todo:
        save(d / "reads" / f"{cid}.json", {DESIGN: read_after(lm, rj(sp(cid))[DESIGN]["summary"])})
    return 4 if waiting else 0


def s3_docs(d, cfg, pcfg, runs):
    """({(conversation id, document key): (text, puzzle)}, the number of planned documents not there yet)."""
    docs, missing = {}, 0
    for cid in sorted(primary_index(pcfg, cfg["arms"])):
        p = d / "summaries" / f"{cid}.json"
        h = rj(p) if p.exists() else {}
        c = primary_conv(runs, cid)
        t = (h.get(DESIGN) or {}).get("summary")
        if t is None or c is None:
            missing += 1
        else:
            docs[(cid, DESIGN)] = (t, c["tasks"][0])
    return docs, missing


def s3_judge(runs, cdir, claude_client=None, claude_sleep=None):
    """Claude (the config's judge settings) codes every retention document with T2 v2, the calls in a seeded shuffled order, through the
    run's own caches; a reply that never parses (or a refusal, or spent attempts) is recorded as missing -> judge/table.json with each
    document's coding, its code checks (endings.t2_checks) and its sha1. claude_client: a stand-in batches client (tests)."""
    from . import claude as EJ
    from . import stages as ERun
    cfg, pcfg, d = s3_open(runs, cdir)
    docs, missing = s3_docs(d, cfg, pcfg, runs)
    jd = d / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    calls = EL.judge_calls(cfg)
    order = EL.judge_order([t for (cid, k), (t, _) in docs.items() if k in calls["t2"]], cfg["schedule_seed"] + JUDGE_ORDER_OFFSET)
    cj = EJ.ClaudeJudge(cfg["judge"], jd, client=claude_client, prog=print, **({"sleep": claude_sleep} if claude_sleep else {}))
    res, rc = ERun.claude_run(cj, [("t2", t) for t in order], cfg["judge"].get("max_calls"), print)
    if rc is not None:
        return rc
    failed, miss_, deferred = res
    table = {}
    for (cid, k), (t, pz) in sorted(docs.items()):
        f2 = (cj.cached("t2", t) or {}).get("facts") if k in calls["t2"] else None
        table.setdefault(cid, {})[k] = {"t2": f2, "t2_checks": EL.t2_checks(f2, t, pz), "summary_sha1": sha1(t)}
    save(jd / "table.json", {"backend": "anthropic", "primary": cj.model, "client": f"anthropic {cj.sdk}", "effort": cj.effort, "calls": calls,
                             "failed": failed, "unparsed": [{"kind": k, "sha1": h} for k, h, _ in miss_], "deferred": deferred,
                             "missing_documents": missing, "scores": table})
    print(f"parts s3 judge: {len(docs)} documents; {failed} unanswered, {len(miss_)} missing, {missing} documents missing")
    return 5 if failed else 4 if deferred or missing else 0


def finish(part, d, out, missing):
    """Write the part's report: <part>_check.json when nothing planned is missing, else <part>_check_partial.json and exit 4."""
    from . import checks as EC
    out = EC.sanitize({**out, "part": part, "complete": not missing, "n_missing": len(missing), "missing": list(missing)[:200]})
    rd = Path(d) / "report"
    rd.mkdir(parents=True, exist_ok=True)
    name = f"{part}_check.json" if not missing else f"{part}_check_partial.json"
    (rd / name).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"parts check {part}: wrote {rd / name}" + (f" ({len(missing)} planned inputs missing)" if missing else ""))
    return 4 if missing else 0


def arrays(n, rows):
    """{arm: array over the puzzles (NaN = missing)} from {(arm, puzzle): value or None}."""
    out = {}
    for (a, i), v in rows.items():
        out.setdefault(a, np.full(n, np.nan))
        if v is not None:
            out[a][i] = v
    return out


def ctr(A, x, y, n, kind="hl", conf=False, delta=None):
    a, b = A.get(arm(x)), A.get(arm(y))
    if a is None or b is None:
        return None
    return EL.row(a - b, kind, conf, delta)


def mean_need(xs, need):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if len(xs) >= need else None


def judge_rows(d):
    p = Path(d) / "judge" / "table.json"
    return rj(p) if p.exists() else {"scores": {}, "failed": None}


def doc_values(jt_rows, keys, need):
    """Self-blame and the T2 v2 shares over a conversation's documents keys (at least need readable)."""
    from . import checks as EC
    rows = [jt_rows.get(k) or {} for k in keys]
    v = {"self_blame": mean_need([EC.self_blame_of(r) for r in rows], need)}
    for ch in EC.T2_SHARES:
        v[ch] = EC.t2_share(rows, ch, need)
    return v


def s3_check(runs, cdir):
    """S3's readings, on 40 puzzles (0-39), contrasts D - C and V - D, two taus. The baseline is Part G's own data: its O1 (the mean of
    s0-s2, key "pi"), s0 alone ("s0"), S ("ser"), its DiD d_pi (O1 minus S, checks.did_x) and its T2 v2 coding of s0; the design's values are
    S3's own reads and coding. Every v is the reverse-coded rating (higher = worse).
      o1          each document's ratings, contrasted (HL, 95% CI)
      did         the retention DiD (the retention rating minus Part G's S, contrasted) and d_pi
      mitigation  m = the retention DiD minus s0's own DiD (s0's rating minus S) per puzzle, HL with p from 20,000 draws at each tau, p* the
                  larger, one sign at both taus; Holm over D - C and V - D. Label: "default DiD not negative" unless s0's DiD has a 95% CI
                  below 0 at both taus; "mitigated toward 0" when m is decisively positive (Holm p < 0.05, positive sign) and the retention
                  DiD's 90% CI lies inside +-0.25 at both taus; "reduced, not to zero" when m is decisive but the retention DiD is not
                  within; else "inconclusive"
      retention   (E - R on the retention document) - (E - R on s0 alone); "changes_the_endings_effect" when its 95% CI excludes 0 with
                  one sign at both taus
      judged      self-blame and the T2 v2 shares, contrasted, on the retention document and on Part G's s0 ("pi")
    Missing S3 reads or codings, or incomplete Part G inputs, make the report partial."""
    from . import checks as EC
    cfg, pcfg, d = s3_open(runs, cdir)
    idx, n = primary_index(pcfg, cfg["arms"]), pcfg["endings"]["n"]
    jt = judge_rows(d)
    sc = jt["scores"]
    missing = [f"reads/{cid}" for cid in idx if not (d / "reads" / f"{cid}.json").exists()]
    missing += [f"judge/{cid}/{DESIGN}" for cid in idx if (sc.get(cid) or {}).get(DESIGN) is None]
    if jt.get("failed"):
        missing.append("judge: unanswered calls")
    V, _, _, excl, _, complete, _ = EC.load(runs, pcfg)
    if not complete:
        missing.append(f"Part G inputs incomplete ({len(excl['missing_conversation'])} conversations, {len(excl['missing'])} with missing inputs)")
    arms_ = {x for x, _ in idx.values()}
    O = {tau: {} for tau in (1, 4)}
    for tau in (1, 4):
        t = None if tau == 1 else tau
        rows = {}
        for cid, (a, i) in idx.items():
            rp = d / "reads" / f"{cid}.json"
            rows[(a, i)] = EL.distress(rj(rp)[DESIGN], t) if rp.exists() else None
        O[tau][DESIGN] = arrays(n, rows)
        O[tau]["pi"] = {a: EC.get(V, tau, "sum", a, n) for a in arms_}
        O[tau]["s0"] = {a: EC.get(V, tau, "sum_s0", a, n) for a in arms_}
        O[tau]["ser"] = {a: EC.get(V, tau, "ser", a, n) for a in arms_}

    def did(tau, key, e, r):
        if key == "pi":
            return EC.did_x(V, tau, e, r, n)  # Part G's own d_pi
        return (O[tau][key][arm(e)] - O[tau][key][arm(r)]) - (O[tau]["ser"][arm(e)] - O[tau]["ser"][arm(r)])
    have = lambda tau, *arms: all(arm(a) in O[tau][DESIGN] for a in arms)  # noqa: E731
    out = {"arms": list(cfg["arms"]), "designs": [DESIGN], "baseline": "Part G's own s0, O1 and S reads and judge rows", "o1": {},
           "did": {}, "mitigation": {}, "retention": {}, "judged": {}}
    pairs = S3_PAIRS
    for tau in (1, 4):
        for key in (DESIGN, "pi", "s0", "ser"):
            out["o1"][f"{key}@{tau}"] = {f"{x}-{y}": ctr(O[tau][key], x, y, n) for x, y in pairs}
        for key in (DESIGN, "pi"):
            out["did"][f"{key}@{tau}"] = {f"{e}-{r}": EL.row(did(tau, key, e, r), "hl") if have(tau, e, r) else None for e, r in pairs}
    tests = []
    base = "s0"  # the retention document carries s0, so its baseline is s0's own DiD
    for e, r in pairs:
        rows, pi_rows, alt_rows = {}, {}, {}
        for tau in (1, 4):
            if not have(tau, e, r):
                continue
            rows[tau] = EL.row(did(tau, DESIGN, e, r) - did(tau, base, e, r), "hl", conf=True)
            pi_rows[tau] = EL.row(did(tau, base, e, r), "hl")
            alt_rows[tau] = EL.row(did(tau, DESIGN, e, r), "hl", delta=DELTA)
        ok = len(rows) == 2 and all(rows.values())
        ps = [rows[t_]["p"] for t_ in (1, 4)] if ok else None
        signs = {rows[t_]["sign"] for t_ in (1, 4)} if ok else {0}
        tests.append({"design": DESIGN, "contrast": f"{e}-{r}", "baseline": base, "m": rows, "pi_did": pi_rows, "alt_did": alt_rows,
                      "p_star": max(ps) if ps else None, "sign": signs.pop() if len(signs) == 1 else 0})
    adj = EL.holm_adjust([x["p_star"] for x in tests])
    for x, q in zip(tests, adj):
        x["p_holm"] = q
        pi_neg = len(x["pi_did"]) == 2 and all(r is not None and r["ci"][1] < 0 for r in x["pi_did"].values())
        m_pos = q < 0.05 and x["sign"] == 1
        resid = len(x["alt_did"]) == 2 and all(r is not None and r["within"] for r in x["alt_did"].values())
        x["label"] = ("default DiD not negative" if not pi_neg else "mitigated toward 0" if m_pos and resid else
                      "reduced, not to zero" if m_pos else "inconclusive")
        out["mitigation"][f"{x['design']}:{x['contrast']}"] = x
    for e, r in pairs:
        rows = {}
        for tau in (1, 4):
            if have(tau, e, r):
                rows[tau] = EL.row((O[tau][DESIGN][arm(e)] - O[tau][DESIGN][arm(r)]) - (O[tau]["s0"][arm(e)] - O[tau]["s0"][arm(r)]), "hl")
        decs = {(rows.get(t_) or {}).get("dec", 0) for t_ in (1, 4)}
        out["retention"][f"{e}-{r}"] = {"rows": rows, "changes_the_endings_effect": len(rows) == 2 and len(decs) == 1 and decs != {0}}
    pt = judge_rows(Path(runs) / PRIMARY)["scores"]
    for key in (DESIGN, "pi"):
        src = (lambda cid: (pt.get(cid) or {}).get("s0")) if key == "pi" else (lambda cid: (sc.get(cid) or {}).get(DESIGN))
        rows = {(a, i): doc_values({key: src(cid)}, [key], 1) for cid, (a, i) in idx.items()}
        out["judged"][key] = {ch: {f"{x}-{y}": ctr(arrays(n, {k: v[ch] for k, v in rows.items()}), x, y, n, "mean") for x, y in pairs}
                              for ch in DOC_CHANNELS}
    return finish("s3", d, out, missing)


# ---------- the bridge ----------

def bridge_docs(runs, cfg):
    """{conversation id: (summary text, puzzle, arm, puzzle index)} of the stored compaction summaries of the exploratory run."""
    src = Path(runs) / cfg["source_run"]
    out = {}
    for f in sorted((src / "compaction").glob("*.json")):
        c = rj(src / "convs" / f.name)
        out[f.stem] = (rj(f)["summary"], c["tasks"][0], c["arm"], c["prompt_index"])
    if not out:
        raise Refused(f"no compaction summaries in {src / 'compaction'}")
    return out


def bridge_judge(runs, cdir, claude_client=None, claude_sleep=None, prog=print):
    """Claude (the bridge config's judge settings; the same T2 v2 prompt and schema as Part G's) codes each stored summary -> judge/table.json
    beside its caches in runs/endings_bridge. Exits as Part G's judge does (2 invalid or unavailable, 3 another served model, 5 unanswered)."""
    from . import claude as EJ
    from . import stages as ERun
    cfg = part_config("bridge", cdir)
    docs = bridge_docs(runs, cfg)
    jd = Path(runs) / cfg["name"] / "judge"
    cj = EJ.ClaudeJudge(cfg["judge"], jd, client=claude_client, prog=prog, **({"sleep": claude_sleep} if claude_sleep else {}))
    calls = EL.judge_calls(cfg)
    res, rc = ERun.claude_run(cj, [("t2", t) for t, *_ in docs.values()], None, prog)
    if rc is not None:
        return rc
    unanswered, missing, _ = res
    table = {}
    for cid, (t, puzzle, a, p) in docs.items():
        f2 = (cj.cached("t2", t) or {}).get("facts")
        table[cid] = {"arm": a, "prompt_index": p, "t2": f2, "t2_checks": EL.t2_checks(f2, t, puzzle), "summary_sha1": sha1(t)}
    save(jd / "table.json", {"backend": "anthropic", "model": cj.model, "effort": cj.effort, "client": f"anthropic {cj.sdk}", "calls": calls, "scores": table,
                             "missing": [{"kind": k, "sha1": h, "why": w} for k, h, w in missing], "unanswered": unanswered})
    prog(f"parts bridge judge: {len(docs)} documents, {len(missing)} missing, {unanswered} unanswered")
    return 5 if unanswered else 0


T2_CATEGORICAL = EL.T2_YESNO + ("rejection_count", "no_solution_stance", "success_stance", "ability_stance", "ability_valence",
                                 "failure_attribution")


def gpt_dir(runs, gpt=None):
    """The folder of GPT's per-document codings: gpt, else $KIND_GOODBYE_BRIDGE_GPT, else <runs>/bridge_gpt."""
    return Path(gpt or os.environ.get("KIND_GOODBYE_BRIDGE_GPT") or Path(runs) / "bridge_gpt")


def bridge_kappa(runs, cdir, gpt=None, prog=print):
    """Per-field agreement and Cohen's kappa of Claude's coding against GPT's on the same documents: T2 v2's categorical fields (a list field
    compared as a sorted list) and the self-blame indicator and its two components (checks.self_blame_of, checks.self_blame_parts, with GPT's
    evidence checks). A pair counts where both codings have the field. "direction_only" lists the fields whose kappa is below the config's
    rule (0.7): judge-coded comparisons across the two judges are direction-only there; Part G's R5 reading uses the self-blame kappas
    ("judge-dependent" below 0.6). "complete" when GPT coded every document. -> report/bridge_kappa.json"""
    from . import checks as EC

    cfg = part_config("bridge", cdir)
    d = Path(runs) / cfg["name"]
    tp = d / "judge" / "table.json"
    if not tp.exists():
        raise Refused(f"{tp} missing: run the bridge's judge first")
    k2 = gpt_dir(runs, gpt)
    if not k2.is_dir():
        raise Refused(f"no GPT coding folder {k2} (pass --gpt DIR or set KIND_GOODBYE_BRIDGE_GPT)")
    sc = rj(tp)["scores"]
    out = {"documents": len(sc), "rule": cfg["kappa_rule"], "t2": {}, "self_blame_kappa": None, "gpt_t2_documents": 0}
    pairs = {f: ([], []) for f in T2_CATEGORICAL}
    sb = {k: ([], []) for k in ("self_blame", "attribution_to_assistant", "asserted_negative_ability")}
    for cid, row in sc.items():
        g = k2 / f"{row['arm']}~{row['prompt_index']:03d}.json"
        if not g.exists() or row.get("t2") is None:
            continue
        gr = rj(g)
        if not gr.get("ok") or gr.get("fields") is None:
            continue
        out["gpt_t2_documents"] += 1
        gpt_row = {"t2": gr["fields"], "t2_checks": {"evidence_ok": gr.get("evidence_ok") or {}}}
        for f in pairs:
            x, y = row["t2"].get(f), gr["fields"].get(f)
            if x is not None and y is not None:
                pairs[f][0].append(json.dumps(sorted(x) if isinstance(x, list) else x))
                pairs[f][1].append(json.dumps(sorted(y) if isinstance(y, list) else y))
        for k, (x, y) in (("self_blame", (EC.self_blame_of(row), EC.self_blame_of(gpt_row))),
                          ("attribution_to_assistant", (EC.self_blame_parts(row)[0], EC.self_blame_parts(gpt_row)[0])),
                          ("asserted_negative_ability", (EC.self_blame_parts(row)[1], EC.self_blame_parts(gpt_row)[1]))):
            if x is not None and y is not None:
                sb[k][0].append(x)
                sb[k][1].append(y)
    for f, (a, b) in pairs.items():
        out["t2"][f] = {"n": len(a), "agree": float(np.mean([x == y for x, y in zip(a, b)])) if a else None, "kappa": EC.kappa(a, b)}
    out["self_blame_kappa"] = {k: EC.kappa(a, b) for k, (a, b) in sb.items()}
    out["self_blame_n"] = {k: len(a) for k, (a, _) in sb.items()}
    lim = cfg["kappa_rule"]["direction_only_below"]
    out["direction_only"] = sorted(f"t2.{f}" for f, v in out["t2"].items() if v["kappa"] is None or v["kappa"] < lim)
    out["complete"] = bool(out["gpt_t2_documents"] == len(sc) and len(sc) == cfg["documents"])
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save(d / "report" / "bridge_kappa.json", out)
    prog(f"parts bridge kappa: GPT's T2 v2 on {out['gpt_t2_documents']} documents; complete {out['complete']}; direction-only "
         f"{out['direction_only']}")
    return 0


# ---------- the command line ----------

STEPS = {("run", "s3"): ("compact", "read", "judge"), ("check", "s3"): (), ("run", "bridge"): ("judge",), ("check", "bridge"): ()}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.parts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "check"], help="run a step, or compute a part's report")
    ap.add_argument("part", choices=["s3", "bridge"])
    ap.add_argument("step", nargs="?", help="for run: s3 compact|read|judge, bridge judge")
    ap.add_argument("--runs", default=str(RUNS), help="the folder holding the run folders (default: runs/, or $KIND_GOODBYE_RUNS)")
    ap.add_argument("--configs", default=str(CONFIGS), help="the folder holding the configs (default: configs/)")
    ap.add_argument("--gpt", help="check bridge: the folder of GPT's per-document codings (default: $KIND_GOODBYE_BRIDGE_GPT, else RUNS/bridge_gpt)")
    a = ap.parse_args(argv)
    steps = STEPS[(a.cmd, a.part)]
    if (a.step or None) not in (steps or (None,)):
        ap.error(f"{a.cmd} {a.part} takes " + (f"a step: {' | '.join(steps)}" if steps else "no step"))
    fns = {("run", "s3", "compact"): lambda: s3_compact(a.runs, a.configs), ("run", "s3", "read"): lambda: s3_read(a.runs, a.configs),
           ("run", "s3", "judge"): lambda: s3_judge(a.runs, a.configs), ("check", "s3", None): lambda: s3_check(a.runs, a.configs),
           ("run", "bridge", "judge"): lambda: bridge_judge(a.runs, a.configs),
           ("check", "bridge", None): lambda: bridge_kappa(a.runs, a.configs, a.gpt)}
    try:
        return fns[(a.cmd, a.part, a.step)]()
    except Refused as e:
        print(f"parts {a.cmd} {a.part}: {e}")
        return e.rc


if __name__ == "__main__":
    sys.exit(main())
