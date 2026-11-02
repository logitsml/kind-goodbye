#!/usr/bin/env python3
"""Every number the README's results tables and charts show, on the 1-7 rating scale, from the reports in runs/.

    python scripts/results/summary.py [--runs DIR] [--out FILE]      (default: runs/summary.json)

The reports hold v = 4 - rating (higher = less happy). Here every value is a rating, and a contrast or change in rating is minus the
report's value with its interval flipped. "dec" is +1 when the 95% interval is above 0 (rated happier), -1 when it is below 0 and 0 when it
includes 0; where a report makes a decision (the confirmatory tests on Gemma 3, the replication families), that decision is used. The
per-arm means and the within-arm changes of the open models are computed here with the reports' own loader and bootstrap (checks.load,
endings.row: puzzles resampled, 2,000 draws, seed 0); everything else is copied from the reports. Run scripts/analyses.sh first.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kind_goodbye import checks as EC  # noqa: E402
from kind_goodbye import endings as EL  # noqa: E402
from kind_goodbye.paths import RUNS  # noqa: E402

DIG = ("ctx_end", "ser", "sum", "sum_s0", "sum_abl", "turn_end", "turn_pre")
FREE = ("self_blame", "self_blame_attr", "self_blame_ability", "forced_keep", "correct", "correct_noimp", "impossible") + tuple(EC.T2_SHARES)
PAIRS = {"endings_12b": (("D", "C"), ("Pc", "D"), ("V", "D"), ("B6", "N"), ("Pc", "C"), ("V", "C")),
         "endings_r3_gemma4_12b": (("D", "C"), ("V", "D"), ("B6", "N"), ("V", "C")),
         "endings_r3_qwen38_27b": (("D", "C"), ("V", "D"), ("B6", "N"), ("V", "C")),
         "endings_r5": (("R6L", "R6C"), ("N0L", "N0"), ("R6C", "N0"), ("R6L", "N0L"))}
REPORT = {"endings_12b": "endings_check", "endings_r3_gemma4_12b": "r3_check", "endings_r3_qwen38_27b": "r3_check", "endings_r5": "r5_check"}
NAMES = {"C": "Open task", "B6": "Open task", "D": "Neutral close", "Pc": "Close + praise", "P": "Praise, no close",
         "F": "Ability praise, no close", "K": "Close + thanks", "V": "“No solution”", "Vc": "Close + “no solution”",
         "Y": "“No solution” + praise", "L": "“No solution” + apology", "J": "Hostile", "N": "No rejections",
         "D_now": "Neutral close", "V_now": "“No solution”"}


def rd(p):
    return json.loads(Path(p).read_text())


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def open_model(runs, name):
    """Per-arm means and per-puzzle values of every channel of one open-model run, at tau 1, and its report."""
    cfg = rd(runs / name / "config.json")
    V, CK, BND, excl, planned, complete, aux = EC.load(runs, cfg)
    n = cfg["endings"]["n"]
    arms = sorted(V[1]["sum"])
    per_arm, values = {}, {}
    for ch in DIG + FREE:
        if any(EC.has(V, ch, a) for a in arms):
            per_arm[ch] = {a: EL.row(EC.get(V, 1, ch, a, n), "mean") for a in arms if EC.has(V, ch, a)}
            values[ch] = {a: EC.get(V, 1, ch, a, n).tolist() for a in arms if EC.has(V, ch, a)}
    return {"per_arm": per_arm, "values": values, "report": rd(runs / name / "report" / f"{REPORT[name]}.json")}


def codex(runs):
    c = runs / "codex"
    vm = rd(c / "close" / "analysis" / "values_merged.json")
    insts = sorted({k.split("|")[0] for k in vm})
    arms = sorted({k.split("|")[1] for k in vm})
    chans = sorted({ch for v in vm.values() for ch, x in v.items() if isinstance(x, (int, float)) and not isinstance(x, bool)})
    per_arm, values = {}, {}
    for ch in chans:
        per_arm[ch], values[ch] = {}, {}
        for a in arms:
            x = np.array([float(vm.get(f"{i}|{a}", {}).get(ch)) if vm.get(f"{i}|{a}", {}).get(ch) is not None else np.nan for i in insts])
            if (~np.isnan(x)).sum() >= 3:
                per_arm[ch][a] = EL.row(x, "mean")
                values[ch][a] = x.tolist()
    within = {}  # each arm's rating after compaction minus in context, per puzzle (on v)
    for a in arms:
        post, ctx = values.get("O1_post_deployed", {}).get(a), values.get("lookback_in_context", {}).get(a)
        if post is not None and ctx is not None:
            within[a] = EL.row(np.array(post) - np.array(ctx), "hl")
    return {"per_arm": per_arm, "within": within, "analysis": rd(c / "main" / "analysis" / "analysis.json"),
            "a1_a5": rd(c / "main" / "analysis" / "a1_a5.json"), "vc": rd(c / "close" / "analysis" / "vc.json"),
            "a6": rd(c / "stances" / "analysis" / "a6.json"), "kc": rd(c / "goodbye" / "analysis" / "kc.json")}


def r2(x):
    return None if x is None else round(float(x), 3)


def eff(e_v, lo_v, hi_v, dec=None):
    """A contrast on v as one on the rating; dec from the report's decision (on v) when given, else from the interval."""
    e, lo, hi = -e_v, -hi_v, -lo_v
    d = (-dec) if dec is not None else (1 if lo > 0 else -1 if hi < 0 else 0)
    return {"e": r2(e), "lo": r2(lo), "hi": r2(hi), "dec": int(d)}


def vrow(r):
    return r.get("est", r.get("hl")), r["ci"][0], r["ci"][1]


def levels(run, arms, names, before="ctx_end", after="sum"):
    """Each arm's mean rating before (in context) and after compaction, and the within-puzzle change."""
    pa, vals = run["per_arm"], run["values"]
    out = []
    for a in arms:
        ch = EL.row(np.array(vals[after][a], float) - np.array(vals[before][a], float), "hl")
        out.append({"label": names[a], "before": r2(4 - pa[before][a]["mean"]), "reread": r2(4 - pa["ser"][a]["mean"]), "after": r2(4 - pa[after][a]["mean"]),
                    **eff(ch["est"], ch["ci"][0], ch["ci"][1])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = Path(a.runs)
    G, G4, Q, R5 = (open_model(runs, n) for n in ("endings_12b", "endings_r3_gemma4_12b", "endings_r3_qwen38_27b", "endings_r5"))
    CX = codex(runs)
    GR, P = G["report"], G["report"]["primary"]
    A1 = {(r["part"], r["contrast"]): r for r in CX["a1_a5"]["A1"]}
    C1 = {r["part"]: r for r in CX["a1_a5"]["C1"]["rows"]}
    A2 = {r["contrast"]: r for r in CX["a1_a5"]["A2"]}
    WITHIN = {r["contrast"].split(":")[0]: r for r in CX["analysis"]["further"]
              if r["channel"] == "within-arm" and r["contrast"].endswith("O1_post_deployed - lookback_in_context")}
    fam = lambda run, k: run["report"]["family"][k]  # noqa: E731
    out = {}

    def codex_levels(arms, within):
        pa = CX["per_arm"]
        return [{"label": NAMES[x], "before": r2(4 - pa["lookback_in_context"][x]["mean"]), "after": r2(4 - pa["O1_post_deployed"][x]["mean"]),
                 **eff(*within(x))} for x in arms]

    order = sorted(["P", "F", "Pc", "Y", "K", "D", "L", "J", "C", "V"], key=lambda x: WITHIN[x]["hl"])
    out["codex"] = codex_levels(order, lambda x: (WITHIN[x]["hl"], *WITHIN[x]["ci"]))
    out["gemma"] = levels(G, ["B6", "D", "Pc", "V", "N"], NAMES)
    out["qwen"] = {x: r2(4 - Q["per_arm"]["sum"][x]["mean"]) for x in ("B6", "D", "V", "N")}
    out["vc"] = codex_levels(["D_now", "Vc", "V_now"], lambda x: (CX["within"][x]["est"], *CX["within"][x]["ci"]))
    out["vc_contrasts"] = {e["id"]: {"contrast": e["contrast"], **eff(e["hl"], *e["ci"])} for e in CX["vc"]["E1_E2"]}
    out["relief"] = levels(R5, ["N0", "N0L", "R6C", "R6L"], {"N0": "No rejections", "N0L": "No rejections, then apology",
                                                            "R6C": "Six rejections", "R6L": "Six rejections, then apology"})
    models = [("Gemma 3 12B", ""), ("Gemma 4 12B", "rates every ending ≈ 4"), ("Qwen3.8 27B", ""), ("GPT-6 Luna", "Codex"),
              ("GPT-6 Luna", "Codex, local summary")]
    close = [(P["T1"]["rows"]["tau1"], P["T1"]["decisive"]), (fam(G4, "O1 D-C")["rows"]["tau1"], fam(G4, "O1 D-C")["decisive"]),
             (fam(Q, "O1 D-C")["rows"]["tau1"], fam(Q, "O1 D-C")["decisive"]), (CX["analysis"]["primary"][0], None),
             (A1[("b_local", "D - C")]["component_compacted"], None)]
    nosol = [(P["T3"]["rows"]["tau1"], P["T3"]["decisive"]), (fam(G4, "O1 V-D")["rows"]["tau1"], fam(G4, "O1 V-D")["decisive"]),
             (fam(Q, "O1 V-D")["rows"]["tau1"], fam(Q, "O1 V-D")["decisive"]), (CX["analysis"]["primary"][2], None),
             (A1[("b_local", "V - D")]["component_compacted"], None)]
    out["models_close"] = [{"label": m, "sub": s, **eff(*vrow(r), d)} for (m, s), (r, d) in zip(models, close)]
    out["models_nosol"] = [{"label": m, "sub": s, **eff(*vrow(r), d)} for (m, s), (r, d) in zip(models, nosol)]
    pa = G["per_arm"]
    out["blame"] = [{"label": NAMES[x], "assistant": round(100 * pa["self_blame"][x]["mean"]), "puzzle": round(100 * pa["t2_fail_task"][x]["mean"])}
                    for x in ["N", "B6", "D", "Pc", "V"]]
    a6 = CX["a6"]["fidelity"]
    gm = lambda run, x: run["per_arm"]["t2_closed_by_user"][x]["mean"]  # noqa: E731
    shares = [gm(G, "D"), gm(G4, "D"), gm(Q, "D"), a6["deployed"]["per_arm"]["D"]["closed_by_user"]["mean"],
              a6["local"]["per_arm"]["D"]["closed_by_user"]["mean"]]
    out["close_recorded"] = [{"label": m if not s else f"{m} ({s})", "share": round(100 * sh), **eff(*vrow(r), d)}
                             for (m, s), sh, (r, d) in zip([("Gemma 3 12B", ""), ("Gemma 4 12B", ""), ("Qwen3.8 27B", ""),
                                                            ("GPT-6 Luna", "Codex"), ("GPT-6 Luna", "local")], shares, close)]
    s3 = rd(runs / "endings_s3" / "report" / "s3_check.json")["retention"]
    ret = [("D - C", "Neutral close vs open task"), ("P - D", "Praise, no close, vs close"), ("Pc - D", "Close + praise vs close"),
           ("V - D", "“No solution” vs close"), ("J - D", "Hostile vs close"), ("L - C", "“No solution” + apology vs open")]
    out["retention_codex"] = [{"label": nm, **eff(A2[k]["hl"], *A2[k]["ci"])} for k, nm in ret]
    out["retention_gemma"] = [{"label": nm, **eff(*vrow(s3[k.replace(" ", "")]["rows"]["1"]))} for k, nm in ret if k in ("D - C", "V - D")]
    sec = GR["secondary"]
    did_rows = [("D-C", "Closing the task"), ("Pc-D", "Adding praise"), ("V-D", "“No solution”"), ("B6-N", "Six rejections")]
    g = {"D-C": (P["T4"]["rows"]["tau1"], P["T4"]["decisive"]), "Pc-D": (sec["did_ser"]["Pc-D@1"], None),
         "V-D": (P["T5"]["rows"]["tau1"], P["T5"]["decisive"]), "B6-N": (sec["history_did"]["rows"]["tau1"], sec["history_did"]["decisive"])}
    dep = {"D-C": A1[("a_deployed", "D - C")], "Pc-D": A1[("a_deployed", "Pc - D")], "V-D": A1[("a_deployed", "V - D")], "B6-N": C1["a_deployed"]}
    loc = {"D-C": A1[("b_local", "D - C")], "Pc-D": A1[("b_local", "Pc - D")], "V-D": A1[("b_local", "V - D")], "B6-N": C1["b_local"]}
    out["did"] = {"gemma": [{"label": nm, **eff(*vrow(g[k][0]), g[k][1])} for k, nm in did_rows],
                  "codex": [{"label": nm, **eff(*vrow(dep[k]))} for k, nm in did_rows],
                  "local": [{"label": nm, **eff(*vrow(loc[k]))} for k, nm in did_rows]}
    b = rd(runs / "endings_bridge" / "report" / "bridge_kappa.json")
    klab = {"failure_described": "Failure described", "answers_wrong": "Answers called wrong", "current_blockage": "Task still blocked",
            "ability_valence": "Ability: tone", "no_solution_stance": "“No solution”: whose claim", "rejection_count": "Number of rejections",
            "unsuccessful_attempts": "Failed attempts", "ability_stance": "Ability: whose claim", "rejection_events": "Rejections mentioned",
            "failure_attribution": "Who or what failed", "success_stance": "Success: whose claim", "self_blame": "Blames the assistant",
            "closed_by_user": "User closed the task"}
    ks = {k: v["kappa"] for k, v in b["t2"].items()}
    ks["self_blame"] = b["self_blame_kappa"]["self_blame"]
    out["kappa"] = [{"label": klab[k], "kappa": round(ks[k], 2)} for k in sorted(ks, key=lambda k: -ks[k])]
    kc = CX["kc"]
    am = kc["secondary"]["per_arm_means"]
    pr = kc["primary"]
    out["goodbye"] = {"text": pr["text"], "ending": pr["ending"], "n": pr["n"], **eff(pr["hl"], *pr["ci"]),
                      "arms": {x: {"before": r2(4 - am[x]["lookback_in_context"]), "after": r2(4 - am[x]["O1_post_deployed"])} for x in am},
                      "checks": {k: v["all_pass"] for k, v in kc["checks"].items()}}
    dst = Path(a.out) if a.out else runs / "summary.json"
    dst.write_text(json.dumps(clean(out), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
