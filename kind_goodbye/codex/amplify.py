"""Secondary analyses of the main Codex run: does compaction amplify the ending's effect (A1), does keeping the user's messages matter
(A2), what the deployed-vs-local rows compare (A3), equivalence and responsiveness (A4), the performance claim (A5), the primary family
with more bootstrap draws (B4), and the history difference in differences (C1).

    python -m kind_goodbye.codex.amplify RUN_DIR [--analysis FILE] [--out FILE]
        writes FILE (default RUN_DIR/analysis/a1_a5.json); --analysis is the main analysis's analysis.json (default
        RUN_DIR/analysis/analysis.json), whose primary rows are cross-checked and whose deployed-vs-local rows A3 relabels

Values and statistics are kind_goodbye.codex.analysis's: collect (values, missing data), paired (pairwise deletion), contrast (HL, 2,000
bootstrap draws seed 0, 95% percentile CI, two-sided bootstrap p), holm. The 90% CI is the 5th-95th percentile of the same draws.
Negative differences mean "better" (less distress or frustration).

A1  does compaction amplify the ending's effect: a difference in differences, per instance, for E - R in {D - C, P - D, Pc - D, V - D,
    J - D, L - C}: (a) deployed: (E - R) on O1_post_deployed minus (E - R) on lookback_in_context; (b) local: (E - R) on
    summary_only_local minus (E - R) on serialized. Negative = the compacted channel moves further toward "better".
    Labels: "amplifies" = the DiD decisive negative and (E - R) on the compacted channel decisive negative (95% CI); "attenuates" = the
    DiD decisive positive while that contrast is decisive negative; "reverses" (descriptive, beside the label) = the compacted and
    comparison contrasts of opposite signs with both 95% CIs excluding 0; "does not amplify" = not decisive, both channels responsive
    (A4) and the 90% CI strictly inside (-0.25, 0.25).
A2  retention: (E - R) on post_local minus (E - R) on summary_only_local (the local compaction with the retained user messages against
    its summary alone), read as A1 ("changes the ending's effect": decisive; "no difference": equivalent on responsive channels).
A3  the over-weighting and deployed-vs-local rows of analysis.json, labelled with what they compare.
A4  equivalence: the smallest effect size of interest (SESOI) is 0.25 distress, 1.0 judge frustration; a channel is responsive when
    C - N (B6 - N) or D - C is decisive on it (D - C on O1 and O2 by its Holm p in the primary family, elsewhere by the 95% CI); a null
    in the primary family, A1 or A2 is "equivalent" (90% CI strictly inside +-SESOI) or "inconclusive" on a responsive channel,
    "unresponsive (not read)" otherwise; the O2 floor flag (D's mean O2 below 0.5: O2 contrasts against D can only detect "worse").
A5  the performance claim: D - C on task-2 verbal concession, deployed route (q_concession_deployed), decisive = p < 0.05 and 95% CI
    excluding 0; the "any of" reading per route, uncorrected (3 channels x 7 contrasts); Holm p within those 42 tests; per arm, the
    share of task-1 replies claiming impossibility and task-1 correctness without such a claim.
B4  the primary family again with B = 20,000 draws (seed 0) and p = min(1, 2 min[(1 + #{t* <= 0}) / (B + 1), (1 + #{t* >= 0}) / (B + 1)]),
    Holm over the six, beside the 2,000-draw values, with every decision that differs.
C1  the history DiD, per instance, B6 - N (letter C - N): (a) deployed: (B6 - N) on O1_post_deployed minus (B6 - N) on
    lookback_in_context; (b) local: (B6 - N) on summary_only_local minus (B6 - N) on serialized; decisive = 95% CI excluding 0.
    Negative = the history contrast is smaller on the compacted channel than on its comparison.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analysis as AC
from . import harness as H
from . import prompts as RD

A1_CONTRASTS = (("D", "C"), ("P", "D"), ("Pc", "D"), ("V", "D"), ("J", "D"), ("L", "C"))
A1_PARTS = {"a_deployed": ("O1_post_deployed", "lookback_in_context"), "b_local": ("summary_only_local", "serialized")}
A2_PAIR = ("post_local", "summary_only_local")
SESOI = {"distress": 0.25, "judge_frustration": 1.0}
JUDGE_CHANNELS = ("O2_recall_frustration", "summary_frustration_local")
O2_FLOOR = 0.5
PERF7 = (("D", "C"), ("P", "D"), ("V", "D"), ("J", "D"), ("K", "D"), ("F", "P"), ("Pc", "D"))
ANYOF_CHANNELS = ("pc_correct", "pc_gives_up", "pc_capitulation")  # task 1 correctness, task 1 giving up, task 2 capitulation
ROUTES = ("deployed", "local")
B4_DRAWS = 20000
A3_TEXT = ("Both routes keep the user's messages verbatim in Codex 0.163.0 (tag rust-v0.163.0; local: codex-rs/core/src/compact.rs 393-396, "
           "user messages up to 20,000 tokens, newest first, plus the summary; remote: codex-rs/core/src/compact_remote_v2.rs, user messages "
           "within 64,000 tokens plus the encrypted item, assistant messages dropped). The local read of the route prediction is the "
           "summary alone, so the deployed vs local contrast compares retained user messages plus the server's summariser, with 3 compactions "
           "per conversation, against a local-prompt summary alone, with 1 compaction: a contrast of summariser, retention and number of "
           "compactions together, not a test of retention (A2 is the retention test).")
A3_OVERWEIGHT = ("route prediction (O1_post_deployed vs summary_only_local): summariser, retention and number of compactions together; "
                 "not a test of retention (A2)")
A3_WITHIN = ("O1_post_deployed vs post_local: both sides keep the user's messages (remote within 64,000 tokens, local up to 20,000); the "
             "server's summariser with 3 compactions against the local prompt's summary with 1; not a test of retention (A2)")
C1_LABEL = "A claim that compaction keeps \"how a task ended, not how it went\" needs this history DiD alongside A1"


def sesoi(ch):
    return SESOI["judge_frustration"] if ch in JUDGE_CHANNELS else SESOI["distress"]


def _clean(d):
    return [x for x in d if x is not None and not (isinstance(x, float) and math.isnan(x))]


def stats(d, s=None):
    """analysis.contrast (HL, 95% CI, p), plus the 90% percentile CI of the same 2,000 draws (seed 0) and, given a SESOI s, the
    bootstrap TOST p = max(share of draws <= -s, share >= s) (reported beside the 90% CI, which is the reading's rule)."""
    d = _clean(d)
    c = AC.contrast(d)
    c["ci90"], c["tost_p"] = None, None
    if c["ci"] is not None:
        draws = AC.boot_hl(d)
        c["ci90"] = [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]
        if s is not None:
            c["tost_p"] = float(max(np.mean(draws <= -s), np.mean(draws >= s)))
    return c


def neg95(c):
    return bool(c.get("ci")) and c["ci"][1] < 0


def pos95(c):
    return bool(c.get("ci")) and c["ci"][0] > 0


def decisive95(c):
    return neg95(c) or pos95(c)


def inside(ci, s):
    """Strictly inside (-s, s)."""
    return bool(ci) and -s < ci[0] and ci[1] < s


def reading(c, s, responsive, decisive):
    if c.get("ci") is None:
        return "insufficient data"
    if decisive:
        return "decisive negative" if c["hl"] < 0 else "decisive positive"
    if not responsive:
        return "unresponsive (not read)"
    return "equivalent" if inside(c["ci90"], s) else "inconclusive"


def did(M, insts, E, R, ca, cb):
    """Per instance (E - R) on ca minus (E - R) on cb; an instance missing any of the four values is dropped (pairwise deletion)."""
    d, dropped = [], []
    for i in insts:
        v = [M.get((i, x), {}).get(c) for x in (E, R) for c in (ca, cb)]
        if None in v:
            dropped.append(i)
        else:
            d.append((v[0] - v[2]) - (v[1] - v[3]))
    return d, dropped


def amplification_label(rd, comp, ref):
    """A1's label from the DiD reading rd and the component contrasts (E - R) on the compacted channel (comp) and on its comparison
    (ref); "reverses" is returned apart (descriptive)."""
    reverses = (neg95(comp) and pos95(ref)) or (pos95(comp) and neg95(ref))
    if rd == "decisive negative":
        label = "amplifies" if neg95(comp) else "decisive negative; no label (the compacted channel's contrast is not decisive negative)"
    elif rd == "decisive positive":
        label = "attenuates" if neg95(comp) else "decisive positive; no label (the compacted channel's contrast is not decisive negative)"
    elif rd == "equivalent":
        label = "does not amplify"
    else:
        label = rd
    return label, reverses


def primary_family(M, insts):
    """The primary family recomputed with analysis's functions (cross-checked against analysis.json by the caller), with the 90% CI
    and TOST p."""
    rows = []
    for o in AC.PRIMARY_OUTCOMES:
        for a, b in AC.PRIMARY_CONTRASTS:
            d, drop = AC.paired(M, insts, a, b, o)
            rows.append({"outcome": o, "contrast": f"{a} - {b}", **stats(d, sesoi(o)), "dropped": drop})
    adj = AC.holm([r["p"] if r["p"] is not None else 1.0 for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["decisive"] = p < 0.05
        r["decisive_negative"] = p < 0.05 and r["hl"] is not None and r["hl"] < 0
    return rows


def responsiveness(M, insts, prim, channels):
    out = {}
    for ch in channels:
        bn = AC.contrast(AC.paired(M, insts, "C", "N", ch)[0])
        pr = [r for r in prim if r["outcome"] == ch and r["contrast"] == "D - C"]
        if pr:
            dc = {"rule": "Holm p < 0.05 in the primary family", "hl": pr[0]["hl"], "ci": pr[0]["ci"], "p": pr[0]["p"], "p_holm": pr[0]["p_holm"],
                  "n": pr[0]["n"], "decisive": bool(pr[0]["decisive"])}
        else:
            c = AC.contrast(AC.paired(M, insts, "D", "C", ch)[0])
            dc = {"rule": "95% CI excludes 0", **c, "decisive": decisive95(c)}
        out[ch] = {"B6 - N": {"rule": "95% CI excludes 0", **bn, "decisive": decisive95(bn)}, "D - C": dc,
                   "responsive": decisive95(bn) or dc["decisive"]}
    return out


def mean_channel(M, insts, letter_, ch):
    xs = [M.get((i, letter_), {}).get(ch) for i in insts]
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else None), len(xs)


def a1(M, insts, resp):
    rows = []
    for E, R in A1_CONTRASTS:
        for part, (ca, cb) in A1_PARTS.items():
            d, drop = did(M, insts, E, R, ca, cb)
            c = stats(d, SESOI["distress"])
            comp = AC.contrast(AC.paired(M, insts, E, R, ca)[0])
            ref = AC.contrast(AC.paired(M, insts, E, R, cb)[0])
            r = {"part": part, "contrast": f"{E} - {R}", "did": f"({E} - {R}) on {ca} minus ({E} - {R}) on {cb}", **c,
                 "component_compacted": {"channel": ca, **comp}, "component_comparison": {"channel": cb, **ref},
                 "responsive": {ca: resp[ca]["responsive"], cb: resp[cb]["responsive"]}, "dropped": drop}
            r["reading"] = reading(c, SESOI["distress"], resp[ca]["responsive"] and resp[cb]["responsive"], decisive95(c))
            r["label"], r["reverses"] = amplification_label(r["reading"], comp, ref)
            rows.append(r)
    return rows


def retention_claim(rd):
    return ("keeping the user's messages verbatim changes the ending's effect" if rd.startswith("decisive")
            else "retention makes no difference" if rd == "equivalent" else rd)


def a2(M, insts, resp):
    ca, cb = A2_PAIR
    rows = []
    for E, R in A1_CONTRASTS:
        d, drop = did(M, insts, E, R, ca, cb)
        c = stats(d, SESOI["distress"])
        r = {"contrast": f"{E} - {R}", "did": f"({E} - {R}) on {ca} minus ({E} - {R}) on {cb}", **c,
             "responsive": {ca: resp[ca]["responsive"], cb: resp[cb]["responsive"]}, "dropped": drop}
        r["reading"] = reading(c, SESOI["distress"], resp[ca]["responsive"] and resp[cb]["responsive"], decisive95(c))
        r["claim"] = retention_claim(r["reading"])
        rows.append(r)
    return rows


def a3(analysis_json):
    rows = [{**r, "a3_label": A3_OVERWEIGHT} for r in analysis_json.get("over_weighting", [])]
    rows += [{**r, "a3_label": A3_WITHIN} for r in analysis_json.get("further", []) if "O1_post_deployed - post_local" in str(r.get("contrast"))]
    return {"interpretation": A3_TEXT, "rows": rows}


def a4(M, insts, prim, resp):
    d_o2, n_o2 = mean_channel(M, insts, "D", "O2_recall_frustration")
    floor = d_o2 is not None and d_o2 < O2_FLOOR
    rows = []
    for r in prim:
        s = sesoi(r["outcome"])
        rd = "decisive (Holm, primary family)" if r["decisive"] else reading(r, s, resp[r["outcome"]]["responsive"], False)
        flag = floor and r["outcome"] == "O2_recall_frustration" and r["contrast"].endswith("- D")
        rows.append({"outcome": r["outcome"], "contrast": r["contrast"], "hl": r["hl"], "ci": r["ci"], "ci90": r["ci90"], "tost_p": r["tost_p"],
                     "p_holm": r["p_holm"], "sesoi": s, "responsive": resp[r["outcome"]]["responsive"], "reading": rd,
                     "o2_floor_flag": flag})
    return {"sesoi": SESOI, "responsiveness": resp, "primary_family": rows,
            "o2_floor": {"mean_O2_of_D": d_o2, "n": n_o2, "threshold": O2_FLOOR, "flag": floor,
                         "note": ("D's mean O2 is below 0.5: O2 contrasts against D (P - D, V - D in the primary family; J - D, K - D, Pc - D "
                                  "among the further contrasts) can only detect 'worse than D'") if floor else None}}


def perf_claims(run_dir, cfg):
    """Per arm and route: the share of task-1 replies claiming impossibility (prompts.claims_impossible on the reply text) and task-1
    correctness (the record's pc_correct, stimuli.check_solution) among the replies with no such claim; per-instance means, then the mean
    over instances, and pooled counts."""
    T = H.plan(cfg)
    leaf_of = {leaf: (iid, arm) for (iid, arm), leaf in H.arm_leaves(cfg).items()}
    want = {}
    for t in T.values():
        if t["kind"] == "gen" and t.get("key") == "pc.task":
            _, route, leaf, _k = t["from_task"].split(":")
            want[t["id"]] = (leaf_of[leaf], "deployed" if route == "remote" else "local")
    recs = {}
    for f in sorted((Path(run_dir) / "data").glob("*/gen.jsonl")):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("task") in want:
                recs[d["task"]] = d
    per = defaultdict(list)
    for tid, ((iid, arm), route) in want.items():
        r = recs.get(tid)
        if not r or r.get("text") is None:
            continue
        per[(AC.LETTER.get(arm, arm), route, iid)].append((RD.claims_impossible(r["text"]), bool(r.get("pc_correct"))))
    out = {}
    for (lt, route) in sorted({(k[0], k[1]) for k in per}):
        claim_i, corr_i, nc, nn, nk = [], [], 0, 0, 0
        for (l2, r2, _iid), xs in per.items():
            if (l2, r2) != (lt, route):
                continue
            claim_i.append(float(np.mean([c for c, _ in xs])))
            nc += sum(c for c, _ in xs)
            nn += len(xs)
            free = [k for c, k in xs if not c]
            nk += sum(free)
            if free:
                corr_i.append(float(np.mean(free)))
        out[f"{lt}|{route}"] = {"claims_impossible": float(np.mean(claim_i)) if claim_i else None, "instances": len(claim_i),
                                "correct_without_claim": float(np.mean(corr_i)) if corr_i else None, "instances_without_claim": len(corr_i),
                                "pooled": {"replies": nn, "claims": int(nc), "correct_among_unclaimed": int(nk), "unclaimed": nn - int(nc)}}
    return out


def a5(M, insts, analysis_json, claims):
    d, drop = AC.paired(M, insts, "D", "C", "q_concession_deployed")
    c = stats(d)
    dec = c["p"] is not None and c["p"] < 0.05 and decisive95(c)
    primary = {"contrast": "D - C", "channel": "q_concession_deployed (task 2 verbal concession, deployed route)", **c, "dropped": drop,
               "decisive": bool(dec), "direction": (None if not dec else "D concedes less than C" if c["hl"] < 0 else "D concedes more than C"),
               "rule": "p < 0.05 (Holm over a family of one) and the 95% CI excludes 0",
               "reading_the_ending_changes_later_performance": bool(dec)}
    tests = []
    for route in ROUTES:
        for a, b in PERF7:
            for ch in ANYOF_CHANNELS:
                dd, dr = AC.paired(M, insts, a, b, f"{ch}_{route}")
                cc = stats(dd)
                tests.append({"route": route, "contrast": f"{a} - {b}", "channel": ch, **cc, "decisive_uncorrected": decisive95(cc), "dropped": dr})
    adj = AC.holm([t["p"] if t["p"] is not None else 1.0 for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm_performance_family"] = p
        t["decisive_holm"] = p < 0.05
    anyof = {route: any(t["decisive_uncorrected"] for t in tests if t["route"] == route) for route in ROUTES}
    return {"primary": primary,
            "any_of": {"label": "uncorrected (21 tests per route)", "per_route": anyof, "either_route": any(anyof.values()),
                       "tests_per_route": {r: sum(t["route"] == r for t in tests) for r in ROUTES},
                       "analysis_json_reading": (analysis_json.get("readings") or {}).get("ending_changes_performance")},
            "performance_family": {"definition": "the 42 tests of the any-of reading (3 channels x 7 contrasts x 2 routes)", "tests": tests,
                                   "any_decisive_after_holm": any(t["decisive_holm"] for t in tests)},
            "impossibility_claims_task1": claims}


def c1(M, insts):
    """C1: the history DiD on both routes, B6 - N (letter C - N). Its reading is the 95% CI's; no equivalence reading is attached."""
    b6 = AC.LETTER["B6"]
    rows = []
    for part, (ca, cb) in A1_PARTS.items():
        d, drop = did(M, insts, b6, "N", ca, cb)
        c = stats(d)
        comp = AC.contrast(AC.paired(M, insts, b6, "N", ca)[0])
        ref = AC.contrast(AC.paired(M, insts, b6, "N", cb)[0])
        rd = ("insufficient data" if c["ci"] is None else "decisive negative" if neg95(c) else "decisive positive" if pos95(c)
              else "not decisive")
        rows.append({"part": part, "contrast": "B6 - N", "did": f"(B6 - N) on {ca} minus (B6 - N) on {cb}", **c,
                     "component_compacted": {"channel": ca, **comp}, "component_comparison": {"channel": cb, **ref}, "dropped": drop,
                     "reading": rd, "decisive_negative": rd == "decisive negative"})
    return {"label": C1_LABEL, "sign": "negative = (B6 - N) is smaller on the compacted channel than on its comparison", "rows": rows}


def p_plus_one(draws):
    """B4's tail rule: p = min(1, 2 min[(1 + #{t* <= 0}) / (B + 1), (1 + #{t* >= 0}) / (B + 1)])."""
    n = len(draws)
    return float(min(1.0, 2 * min((1 + np.sum(draws <= 0)) / (n + 1), (1 + np.sum(draws >= 0)) / (n + 1))))


def b4(M, insts, prim, draws=B4_DRAWS):
    rows = []
    for r in prim:
        a, b = r["contrast"].split(" - ")
        d = _clean(AC.paired(M, insts, a, b, r["outcome"])[0])
        if len(d) < 2:
            rows.append({"outcome": r["outcome"], "contrast": r["contrast"], "n": len(d), "p": None, "ci95": None})
            continue
        dr = AC.boot_hl(d, b=draws, seed=AC.SEED)
        rows.append({"outcome": r["outcome"], "contrast": r["contrast"], "n": len(d), "p": p_plus_one(dr),
                     "ci95": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]})
    adj = AC.holm([x["p"] if x["p"] is not None else 1.0 for x in rows])
    differ = []
    for x, p, r in zip(rows, adj, prim):
        x["p_holm"] = p
        x["decisive"] = p < 0.05
        x["decisive_negative"] = p < 0.05 and r["hl"] is not None and r["hl"] < 0
        x["governing_2000"] = {"p": r["p"], "p_holm": r["p_holm"], "decisive": r["decisive"], "decisive_negative": r["decisive_negative"]}
        if x["decisive"] != r["decisive"] or x["decisive_negative"] != r["decisive_negative"]:
            differ.append({"outcome": x["outcome"], "contrast": x["contrast"], "decisive_2000": r["decisive"], "decisive_20000": x["decisive"]})
    readings = {}
    for name, con in (("closing_changes_memory", "D - C"), ("warmth_beyond_closure (decisive component only)", "P - D"),
                      ("vindication_beyond_closure", "V - D")):
        g = any(r["decisive_negative"] for r in prim if r["contrast"] == con)
        n = any(x["decisive_negative"] for x in rows if x["contrast"] == con)
        readings[name] = {"governing_2000": g, "b20000": n, "differs": g != n}
    return {"draws": draws, "seed": AC.SEED, "rule": "p = min(1, 2 min[(1 + #{t* <= 0}) / (B + 1), (1 + #{t* >= 0}) / (B + 1)]); Holm over 6",
            "governs": False, "rows": rows, "decisions_that_differ": differ, "readings": readings}


def cross_check(prim, analysis_json):
    """Whether the recomputed primary rows equal analysis.json's (HL, CI, p and Holm p within 1e-9; the same n)."""
    bad = []
    fz = {(r["outcome"], r["contrast"]): r for r in analysis_json.get("primary", [])}
    for r in prim:
        f = fz.get((r["outcome"], r["contrast"]))
        same = f is not None and all(
            (f.get(k) is None and r.get(k) is None) or (f.get(k) is not None and r.get(k) is not None and np.allclose(f[k], r[k], rtol=0, atol=1e-9))
            for k in ("hl", "ci", "p", "p_holm")) and f.get("n") == r.get("n")
        if not same:
            bad.append({"outcome": r["outcome"], "contrast": r["contrast"]})
    return {"primary_rows_match_analysis_json": not bad, "mismatches": bad}


def run(run_dir, analysis_json, b4_draws=B4_DRAWS):
    cfg, insts, M = AC.collect(run_dir)
    prim = primary_family(M, insts)
    channels = sorted(set(AC.BLOCK_CHANNELS) | {c for p in A1_PARTS.values() for c in p} | set(A2_PAIR))
    resp = responsiveness(M, insts, prim, channels)
    return {"run": cfg["name"], "n_instances": len(insts), "consistency": cross_check(prim, analysis_json), "A1": a1(M, insts, resp),
            "A2": a2(M, insts, resp), "A3": a3(analysis_json), "A4": a4(M, insts, prim, resp),
            "A5": a5(M, insts, analysis_json, perf_claims(run_dir, cfg)), "B4": b4(M, insts, prim, b4_draws), "C1": c1(M, insts)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.amplify", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="the main run directory")
    ap.add_argument("--analysis", default=None, help="the main analysis's analysis.json (default: RUN_DIR/analysis/analysis.json)")
    ap.add_argument("--out", default=None, help="output file (default: RUN_DIR/analysis/a1_a5.json)")
    ap.add_argument("--b4-draws", type=int, default=B4_DRAWS, help=f"bootstrap draws of B4 (default {B4_DRAWS})")
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    analysis_json = json.loads(Path(a.analysis or run_dir / "analysis" / "analysis.json").read_text())
    res = run(run_dir, analysis_json, a.b4_draws)
    out = Path(a.out or run_dir / "analysis" / "a1_a5.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out)
    print(json.dumps({"consistency": res["consistency"],
                      "A1": [(r["part"], r["contrast"], r["label"], r["reverses"]) for r in res["A1"]],
                      "A2": [(r["contrast"], r["reading"]) for r in res["A2"]],
                      "A4_primary": [(r["outcome"], r["contrast"], r["reading"], r["o2_floor_flag"]) for r in res["A4"]["primary_family"]],
                      "A5_primary_decisive": res["A5"]["primary"]["decisive"], "B4_decisions_that_differ": res["B4"]["decisions_that_differ"],
                      "C1": [(r["part"], r["hl"], r["ci"], r["reading"]) for r in res["C1"]["rows"]]},
                     default=float))


if __name__ == "__main__":
    main()
