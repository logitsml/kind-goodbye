"""The analysis of the Vc arm ("close + no solution", kind_goodbye.codex.close_calls): E1-E6, R and X.

    python -m kind_goodbye.codex.close MAIN_RUN CLOSE_DIR [--values FILE] [--out DIR]
        writes DIR/vc.json and DIR/values_merged.json (default DIR: CLOSE_DIR/analysis); exits 3 when the merged view does not
        reproduce the main run's values.json (--values, default MAIN_RUN/analysis/values.json)

Values and statistics are kind_goodbye.codex.analysis's. Its collect_values runs on the union of the main run's records and the close
run's, under the main config plus the Vc turn and arm (letter Vc), so every channel of Vc is computed exactly as for the main arms; the
main arms' values must then equal the main run's values.json (checked). R's regenerated D and V (letters D_now and V_now) get the
channels R reads, by collect's definitions: O1_post_deployed (the mean over compactions of the mean over samples), lookback_in_context,
check_wants and check_solvable (the shares of "yes"). Contrasts use analysis.contrast (HL, 2,000 bootstrap draws, seed 0, 95%
percentile CI, two-sided bootstrap p) and amplify.stats (the 90% CI of the same draws); the DiDs, readings, responsiveness and labels
are amplify's functions. SESOI: 0.25 distress.

E1  Vc - D_now on O1_post_deployed: what "no solution" adds when the close is held fixed (D regenerated beside Vc: R).
E2  V_now - Vc on O1_post_deployed: what the explicit close adds when "no solution" is held fixed (V regenerated beside Vc: R).
    E1 and E2 form one Holm family of two; decisive = Holm p < 0.05; a null on the responsive O1 is "equivalent" when its 90% CI lies
    strictly inside +-0.25, else "inconclusive". The joint reading, each of E1 and E2 being decisive positive (pos), decisive negative
    (neg), equivalent (eq) or none of these (other): pos-eq "the information itself makes the carried-forward look-back worse"; eq-pos
    "the backfire is the missing close"; pos-pos "both add to it"; pos-other and other-pos the decisive part, with the other part named;
    eq-eq "neither part changes it beyond +-0.25"; any neg "mixed" (both readings, no label); otherwise "neither part alone is decisive".
    Per instance V_now - D_now = (V_now - Vc) + (Vc - D_now) exactly; both parts are reported. Across the time gap (secondary): the same
    two contrasts against the main run's D and V (Vc - D, V - Vc).
E3  A1 for E1's and E2's contrasts: (a) deployed, O1 minus the in-context look-back, against R's D and V; (b) local, summary-only minus
    serialized, against the main run's D and V (across the time gap).
E4  A2 for Vc - D and V - Vc: post_local minus summary_only_local, against the main run's D and V (across the time gap).
E5  Vc's manipulation checks (per-instance means): (a) Vc closes as much as D: still wants(Vc) <= still wants(D_now) + 0.10; (b) Vc
    carries the information as V does: can be solved(Vc) <= can be solved(V_now) + 0.10; (c) Vc's record registers the close as D's
    does: recall_task_open(Vc) <= recall_task_open(D) + 0.10 (the judge's task status on the recall documents; the main run's D). A
    failed (a) or (c) is a named confound of E1, a failed (b) of E2.
E6  Every channel collect defines for Vc - D, V - Vc and V - D (across the time gap), and R's channels for Vc - D_now, V_now - Vc and
    V_now - D_now (95% CIs), with per-arm means.
X   Drift since the main run, per instance, on O1 and in context: (a) read side: D's and V's stored sessions read again beside the Vc run
    minus the main run's own reads; (b) the whole pipeline: R's regenerated D and V minus the main run's D and V. "No drift" = the 90%
    CI strictly inside +-0.25 (a decisive shift inside it is flagged); "drift" = otherwise decisive (95% CI excluding 0); else
    "inconclusive". V - D on the re-reads and on R is reported beside the main run's V - D. A drift row is a named confound of every
    comparison across the time gap (not of E1 and E2, whose arms share the same days).
vc.json's named_confounds lists them.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import amplify as A15
from . import analysis as AC
from . import close_calls as VC
from . import harness as H

O1, IC = "O1_post_deployed", "lookback_in_context"
E_NOW = (("Vc", "D_now"), ("V_now", "Vc"))  # E1, E2 (the same days)
E_GAP = (("Vc", "D"), ("V", "Vc"))  # E1 and E2 across the time gap (secondary)
REP_LETTER = {"B6_D": "D_now", "B6_V": "V_now"}
REP_CHANNELS = (O1, IC, "check_wants", "check_solvable")
PREFIX_ONLY = ("claimed_before_ending", "tools_before_ending")  # the shared prefix's: equal in every ending arm by construction
SESOI = A15.SESOI["distress"]


def extended_config(cfg):
    """The main config plus the Vc turn and the B6_Vc arm (every main arm kept, in order)."""
    c = json.loads(json.dumps(cfg))
    v = VC.vc_config(cfg)
    c["turns"][VC.VC_KEY] = v["turns"][VC.VC_KEY]
    c["arms"][VC.VC_ARM] = v["arms"][VC.VC_ARM]
    return c


def merged_collect(main_dir, run_dir):
    """analysis.collect_values on the extended config over the union of the two stores (a task id may be in one store only)."""
    main_dir, run_dir = Path(main_dir), Path(run_dir)
    cfg = json.loads((main_dir / "config.json").read_text())
    a, b = H.Store(main_dir).done, VC.own_records(run_dir)
    both = set(a) & set(b)
    if both:
        raise SystemExit(f"a task id is in both stores ({sorted(both)[:3]}); refusing to merge")
    cfg_x, insts, M = AC.collect_values(extended_config(cfg), {**a, **b}, letters={**AC.LETTER, VC.VC_ARM: VC.VC_KEY})
    return cfg_x, insts, M, {"main_records": len(a), "vc_records": len(b)}


def _same(x, y):
    if isinstance(x, float) or isinstance(y, float):
        return x is not None and y is not None and (math.isclose(float(x), float(y), rel_tol=0, abs_tol=1e-12) or (math.isnan(float(x)) and math.isnan(float(y))))
    return x == y


def values_consistency(values_file, M):
    """The merged view's values of the main arms against the main run's values.json (written by analysis)."""
    f = Path(values_file)
    if not f.exists():
        return {"checked": False, "why": f"no {f.name}", "identical": False}
    main_values = json.loads(f.read_text())
    diff, n = [], 0
    for key, vals in main_values.items():
        i, L = key.split("|", 1)
        mine = M.get((i, L))
        if mine is None:
            diff.append({"key": key, "why": "missing in the merged view"})
            continue
        for ch in set(vals) | set(mine):
            n += 1
            if not _same(vals.get(ch), mine.get(ch)):
                diff.append({"key": key, "channel": ch, "main": vals.get(ch), "merged": mine.get(ch)})
    return {"checked": True, "values": n, "differences": diff[:20], "n_differences": len(diff), "identical": not diff}


def rep_values(main_dir, run_dir):
    """R's channels per (instance, "D_now" / "V_now"), by collect's definitions (a read with no valid sample is missing)."""
    cfg = json.loads((Path(main_dir) / "config.json").read_text())
    _, T, _ = VC.plan_vc(cfg)
    own = VC.own_records(run_dir)
    post, end = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for tid, t in T.items():
        if not t.get("rep") or t["kind"] not in ("post", "end"):
            continue
        r = own.get(tid)
        if not r or not r.get("ok") or not r.get("values"):
            continue
        key = (t["inst"], REP_LETTER[t["arms"][0]])
        if t["kind"] == "post":
            post[key][t["k"]].append(AC.distress(r["values"]))
        elif t["read"] == "lookback":
            end[key]["lookback"].append(AC.distress(r["values"]))
        else:
            end[key][t["read"]].append(float(r["values"]["again"]))
    out = defaultdict(dict)
    for key, per_k in post.items():
        out[key][O1] = AC.mean_or_none([AC.mean_or_none(v) for v in per_k.values()])
    for key, reads in end.items():
        out[key][IC] = AC.mean_or_none(reads.get("lookback", []))
        for r in VC.REP_CHECKS:
            out[key][f"check_{r.removeprefix('chk_')}"] = AC.mean_or_none(reads.get(r, []))
    return out


def label_of(r):
    return {"decisive positive": "pos", "decisive negative": "neg", "equivalent": "eq"}.get(r["reading"], "other")


def e_family(M, insts, resp, contrasts=E_NOW):
    rows = []
    for k, (E, R) in enumerate(contrasts):
        d, drop = AC.paired(M, insts, E, R, O1)
        rows.append({"id": f"E{k + 1}", "contrast": f"{E} - {R}", "outcome": O1, **A15.stats(d, SESOI), "dropped": drop})
    adj = AC.holm([r["p"] if r["p"] is not None else 1.0 for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["decisive"] = r["hl"] is not None and p < 0.05
        if r["ci"] is None:
            r["reading"] = "insufficient data"
        elif r["decisive"]:
            r["reading"] = "decisive positive" if r["hl"] > 0 else "decisive negative"
        else:
            r["reading"] = A15.reading(r, SESOI, resp[O1]["responsive"], False)
    return rows


def joint_reading(e1, e2):
    """The joint reading of E1 and E2 (module docstring)."""
    a, b = label_of(e1), label_of(e2)
    if "neg" in (a, b):
        return f"mixed (no label): E1 {e1['reading']}, E2 {e2['reading']}"
    return {("pos", "pos"): "both add to it: the information with the close held fixed, and the missing close with the information held fixed",
            ("pos", "eq"): "the information itself makes the carried-forward look-back worse (the close held fixed); the close adds nothing beyond +-0.25",
            ("eq", "pos"): "the backfire is the missing close (the information held fixed); the information adds nothing beyond +-0.25",
            ("pos", "other"): f"the information makes the carried-forward look-back worse (the close held fixed); the close's part is {e2['reading']}",
            ("other", "pos"): f"the missing close makes the carried-forward look-back worse (the information held fixed); the information's part is {e1['reading']}",
            ("eq", "eq"): "neither part changes the carried-forward look-back beyond +-0.25"}.get((a, b), "neither part alone is decisive")


def decomposition(M, insts, V="V_now", D="D_now"):
    """Per instance V - D = (V - Vc) + (Vc - D) on O1 (an identity; the instances with all three values)."""
    keep = [i for i in insts if all(M.get((i, L), {}).get(O1) is not None for L in (V, "Vc", D))]
    vd = [M[(i, V)][O1] - M[(i, D)][O1] for i in keep]
    vvc = [M[(i, V)][O1] - M[(i, "Vc")][O1] for i in keep]
    vcd = [M[(i, "Vc")][O1] - M[(i, D)][O1] for i in keep]
    return {"letters": [V, "Vc", D], "n": len(keep), "mean_V_minus_D": float(np.mean(vd)) if keep else None,
            "mean_V_minus_Vc": float(np.mean(vvc)) if keep else None, "mean_Vc_minus_D": float(np.mean(vcd)) if keep else None,
            "identity_holds": all(abs(x - (y + z)) < 1e-9 for x, y, z in zip(vd, vvc, vcd))}


def did_rows(M, insts, resp, parts, contrasts, label=True):
    rows = []
    for E, R in contrasts:
        for part, (ca, cb) in parts.items():
            d, drop = A15.did(M, insts, E, R, ca, cb)
            c = A15.stats(d, SESOI)
            comp = AC.contrast(AC.paired(M, insts, E, R, ca)[0])
            ref = AC.contrast(AC.paired(M, insts, E, R, cb)[0])
            r = {"part": part, "contrast": f"{E} - {R}", "did": f"({E} - {R}) on {ca} minus ({E} - {R}) on {cb}", **c,
                 "component_compacted": {"channel": ca, **comp}, "component_comparison": {"channel": cb, **ref},
                 "responsive": {ca: resp[ca]["responsive"], cb: resp[cb]["responsive"]}, "dropped": drop}
            r["reading"] = A15.reading(c, SESOI, resp[ca]["responsive"] and resp[cb]["responsive"], A15.decisive95(c))
            if label:
                r["label"], r["reverses"] = A15.amplification_label(r["reading"], comp, ref)
            rows.append(r)
    return rows


def e5_checks(M, insts):
    def mean(L, ch):
        return A15.mean_channel(M, insts, L, ch)[0]
    means = {k: {L: mean(L, f"check_{k}") for L in ("D", "D_now", "V", "V_now", "Vc")} for k in ("wants", "solvable", "correct", "pleased", "sarcastic")}
    means["recall_task_open"] = {L: mean(L, "recall_task_open") for L in ("D", "V", "Vc")}
    w, s, o = means["wants"], means["solvable"], means["recall_task_open"]
    a = None if None in (w["Vc"], w["D_now"]) else w["Vc"] <= w["D_now"] + 0.10
    b = None if None in (s["Vc"], s["V_now"]) else s["Vc"] <= s["V_now"] + 0.10
    c = None if None in (o["Vc"], o["D"]) else o["Vc"] <= o["D"] + 0.10
    return {"means": means, "a_closes_as_much_as_D": a, "b_carries_the_information_as_V": b, "c_record_registers_the_close": c,
            "named_confounds": [x for x, ok in (("E1: Vc leaves the task more open than D in context (check a)", a),
                                                ("E2: Vc does not carry the information as V does (check b)", b),
                                                ("E1: Vc's recall documents register the close less than D's (check c, across the time gap)", c))
                                if ok is False]}


def channels_of(M):
    """Every channel collect defines (those of the main run's D), less the prefix-only flags."""
    keys = set()
    for (i, L), v in M.items():
        if L == "D":
            keys |= set(v)
    return sorted(keys - set(PREFIX_ONLY))


def e6(M, insts):
    rows = []
    for E, R in E_GAP + (("V", "D"),):
        for ch in channels_of(M):
            d, drop = AC.paired(M, insts, E, R, ch)
            c = AC.contrast(d)
            rows.append({"set": "across the time gap", "contrast": f"{E} - {R}", "channel": ch, **c, "decisive": A15.decisive95(c), "dropped": drop})
    for E, R in E_NOW + (("V_now", "D_now"),):
        for ch in REP_CHANNELS:
            d, drop = AC.paired(M, insts, E, R, ch)
            c = AC.contrast(d)
            rows.append({"set": "the same days (R)", "contrast": f"{E} - {R}", "channel": ch, **c, "decisive": A15.decisive95(c), "dropped": drop})
    means = {L: {ch: A15.mean_channel(M, insts, L, ch)[0] for ch in channels_of(M)} for L in ("C", "D", "V", "Vc")}
    means.update({L: {ch: A15.mean_channel(M, insts, L, ch)[0] for ch in REP_CHANNELS} for L in ("D_now", "V_now")})
    return {"rows": rows, "per_arm_means": means}


def drift_values(main_dir, run_dir):
    """X(a)'s re-read values per (instance, letter): O1_reread (the mean over compactions of the mean over samples) and
    lookback_in_context_reread."""
    cfg = json.loads((Path(main_dir) / "config.json").read_text())
    _, T, _ = VC.plan_vc(cfg)
    own = VC.own_records(run_dir)
    letter = {"B6_D": "D", "B6_V": "V"}
    post, end = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for tid, t in T.items():
        if not t.get("drift"):
            continue
        r = own.get(tid)
        if not r or not r.get("ok") or not r.get("values"):
            continue
        key = (t["inst"], letter[t["arms"][0]])
        if t["kind"] == "post":
            post[key][t["k"]].append(AC.distress(r["values"]))
        else:
            end[key].append(AC.distress(r["values"]))
    X = defaultdict(dict)
    for key, per_k in post.items():
        X[key]["O1_reread"] = AC.mean_or_none([AC.mean_or_none(v) for v in per_k.values()])
    for key, v in end.items():
        X[key]["lookback_in_context_reread"] = AC.mean_or_none(v)
    return X


def drift_reading(c):
    if c["ci"] is None:
        return "insufficient data"
    if A15.inside(c["ci90"], SESOI):
        return "no drift"
    return "drift" if A15.decisive95(c) else "inconclusive"


def drift(M, insts, X):
    rows = []
    for L in ("D", "V"):
        for part, src, chans in (("a_read_side", X, (("O1_reread", O1), ("lookback_in_context_reread", IC))),
                                 ("b_whole_pipeline", None, ((O1, O1), (IC, IC)))):
            for new, old in chans:
                if src is not None:
                    d = [src[(i, L)][new] - M[(i, L)][old] for i in insts
                         if src.get((i, L), {}).get(new) is not None and M.get((i, L), {}).get(old) is not None]
                else:
                    d, _ = AC.paired(M, insts, f"{L}_now", L, old)
                c = A15.stats(d, SESOI)
                rd = drift_reading(c)
                rows.append({"part": part, "arm": L, "channel": old, "now_minus_main": c, "reading": rd,
                             "decisive_shift_within_sesoi": rd == "no drift" and A15.decisive95(c)})
    vd = []
    for ch, new in ((O1, "O1_reread"), (IC, "lookback_in_context_reread")):
        d_x = [X[(i, "V")][new] - X[(i, "D")][new] for i in insts if X.get((i, "V"), {}).get(new) is not None and X.get((i, "D"), {}).get(new) is not None]
        vd.append({"channel": ch, "V_minus_D_main": AC.contrast(AC.paired(M, insts, "V", "D", ch)[0]), "V_minus_D_reread": AC.contrast(d_x),
                   "V_now_minus_D_now": AC.contrast(AC.paired(M, insts, "V_now", "D_now", ch)[0])})
    return {"rows": rows, "V_minus_D": vd}


def run(main_dir, run_dir, out_dir, values_file=None):
    out_dir = Path(out_dir)
    cfg, insts, M, counts = merged_collect(main_dir, run_dir)
    consistency = values_consistency(values_file or Path(main_dir) / "analysis" / "values.json", M)
    for key, v in rep_values(main_dir, run_dir).items():
        M[key] = dict(v)
    prim = A15.primary_family(M, insts)
    resp = A15.responsiveness(M, insts, prim, list(AC.BLOCK_CHANNELS))
    fam, gap = e_family(M, insts, resp, E_NOW), e_family(M, insts, resp, E_GAP)
    res = {"n_instances": len(insts), "records": counts, "consistency": consistency,
           "vc_conversations": sum(1 for i in insts if M.get((i, "Vc"), {}).get(O1) is not None),
           "rep_conversations": {L: sum(1 for i in insts if M.get((i, L), {}).get(O1) is not None) for L in ("D_now", "V_now")},
           "E1_E2": fam, "joint_reading": joint_reading(*fam), "decomposition": decomposition(M, insts),
           "E1_E2_across_the_time_gap": gap, "joint_reading_across_the_time_gap": joint_reading(*gap),
           "decomposition_across_the_time_gap": decomposition(M, insts, V="V", D="D"),
           "E3_A1": did_rows(M, insts, resp, {"a_deployed": A15.A1_PARTS["a_deployed"]}, E_NOW)
           + did_rows(M, insts, resp, {"b_local": A15.A1_PARTS["b_local"]}, E_GAP),
           "E4_A2": [{**r, "claim": A15.retention_claim(r["reading"])}
                     for r in did_rows(M, insts, resp, {"retention": A15.A2_PAIR}, E_GAP, label=False)],
           "E5_checks": e5_checks(M, insts), "E6": e6(M, insts), "responsiveness": {k: v["responsive"] for k, v in resp.items()},
           "X_drift": drift(M, insts, drift_values(main_dir, run_dir))}
    res["named_confounds"] = res["E5_checks"]["named_confounds"] + [
        f"every comparison across the time gap: {r['part'].split('_', 1)[1].replace('_', ' ')} drift in {r['arm']}'s {r['channel']} (X)"
        for r in res["X_drift"]["rows"] if r["reading"] == "drift"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vc.json").write_text(json.dumps(res, indent=1, default=float))
    (out_dir / "values_merged.json").write_text(json.dumps({f"{i}|{L}": v for (i, L), v in M.items()}, indent=0, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.close", description="The analysis of the Vc arm (close + no solution).")
    ap.add_argument("main_run", help="the main run directory")
    ap.add_argument("run_dir", help="the Vc run directory (close_calls)")
    ap.add_argument("--values", default=None, help="the main run's values.json (default: MAIN_RUN/analysis/values.json)")
    ap.add_argument("--out", default=None, help="output directory (default: RUN_DIR/analysis)")
    a = ap.parse_args(argv)
    res = run(a.main_run, a.run_dir, a.out or Path(a.run_dir) / "analysis", a.values)
    print(json.dumps({"consistency": {k: res["consistency"].get(k) for k in ("identical", "n_differences")}, "E1_E2": [
        {k: r[k] for k in ("id", "contrast", "n", "hl", "ci", "p_holm", "reading")} for r in res["E1_E2"]], "joint_reading": res["joint_reading"],
        "joint_reading_across_the_time_gap": res["joint_reading_across_the_time_gap"], "named_confounds": res["named_confounds"],
        "E5": {k: res["E5_checks"][k] for k in ("a_closes_as_much_as_D", "b_carries_the_information_as_V", "c_record_registers_the_close")},
        "X": [{"part": r["part"], "arm": r["arm"], "channel": r["channel"], "reading": r["reading"]} for r in res["X_drift"]["rows"]]},
                     indent=1, default=float))
    if not res["consistency"].get("identical"):
        print("close: the merged view does not reproduce the main run's values.json (vc.json's consistency); not read", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
