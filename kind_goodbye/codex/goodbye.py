"""The analysis of the kind-goodbye test (kind_goodbye.codex.goodbye_calls): does a kind goodbye that the model reads neither as
approval nor as leaving the task open change its carried-forward look-back, against a same-day neutral close?

    python -m kind_goodbye.codex.goodbye MAIN_RUN GOODBYE_DIR [--values FILE] [--out DIR]
        writes DIR/kc.json and DIR/values_merged.json (default DIR: GOODBYE_DIR/analysis); exits 3 when the merged view does not
        reproduce the main run's values.json (--values, default MAIN_RUN/analysis/values.json)

Values and statistics are kind_goodbye.codex.analysis's, through the same merged view as kind_goodbye.codex.close: collect_values runs on
the union of the main run's records and this run's, under the main config plus the two kind goodbyes (letters Kp1 and Kp2), so every
channel of theirs is computed exactly as for the main arms, and the main arms must reproduce values.json (exit 3 otherwise). R's
regenerated D (letter D_now) gets O1_post_deployed, the in-context look-back and the five checks, by collect's definitions.

Checks (per-instance means; point estimates decide) of each kind goodbye against D_now: still wants <= +0.10, can be solved >= -0.10,
last answer correct <= +0.10, pleased >= +0.20, sarcastic <= +0.10. Primary: the first of Kp1, Kp2 that passes all five; its contrast
minus D_now on O1 (HL, 2,000 draws, seed 0, 95% CI; decisive = the CI excludes 0; otherwise "equivalent" if the 90% CI lies inside
+-0.25, else "inconclusive"). Negative = less distress in the look-back = happier. Secondary: each kind goodbye against D_now on R's
channels, against the main run's D, Pc and K on O1 and against the main run's D on every channel (across the time gap); the drift
D_now - D on O1 and in context; per-arm means.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import amplify as A15
from . import analysis as AC
from . import close as VA
from . import close_calls as VC
from . import goodbye_calls as KC
from . import harness as H

O1, IC = "O1_post_deployed", "lookback_in_context"
CHECKS = ("wants", "solvable", "correct", "pleased", "sarcastic")
RULE = {"wants": ("<=", 0.10), "solvable": (">=", -0.10), "correct": ("<=", 0.10), "pleased": (">=", 0.20), "sarcastic": ("<=", 0.10)}
REP_CHANNELS = (O1, IC) + tuple(f"check_{c}" for c in CHECKS)
SESOI = A15.SESOI["distress"]


def extended_config(cfg):
    """The main config plus the two kind-goodbye turns and arms (every main arm kept, in order)."""
    c = json.loads(json.dumps(cfg))
    k = KC.kc_config(cfg)
    for key in KC.ORDER:
        c["turns"][key] = k["turns"][key]
        c["arms"][KC.ARM[key]] = k["arms"][KC.ARM[key]]
    return c


def merged_collect(main_dir, run_dir):
    """analysis.collect_values on the extended config over the union of the two stores (a task id may be in one store only)."""
    main_dir, run_dir = Path(main_dir), Path(run_dir)
    cfg = json.loads((main_dir / "config.json").read_text())
    a, b = H.Store(main_dir).done, VC.own_records(run_dir)
    both = set(a) & set(b)
    if both:
        raise SystemExit(f"a task id is in both stores ({sorted(both)[:3]}); refusing to merge")
    letters = {**AC.LETTER, **{KC.ARM[k]: k for k in KC.ORDER}}
    cfg_x, insts, M = AC.collect_values(extended_config(cfg), {**a, **b}, letters=letters)
    return cfg_x, insts, M, {"main_records": len(a), "kc_records": len(b)}


def rep_values(main_dir, run_dir):
    """R's channels per (instance, "D_now"), by collect's definitions (a read with no valid sample is missing)."""
    cfg = json.loads((Path(main_dir) / "config.json").read_text())
    _, T, _ = KC.plan_kc(cfg)
    own = VC.own_records(run_dir)
    post, end = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for tid, t in T.items():
        if not t.get("rep") or t["kind"] not in ("post", "end"):
            continue
        r = own.get(tid)
        if not r or not r.get("ok") or not r.get("values"):
            continue
        key = (t["inst"], "D_now")
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
        for r in KC.REP_CHECKS:
            out[key][f"check_{r.removeprefix('chk_')}"] = AC.mean_or_none(reads.get(r, []))
    return out


def checks(M, insts, letter, ref="D_now"):
    """The five checks of one ending against ref: means, per-check pass, all pass (None when a mean is missing)."""
    out = {"means": {}, "passes": {}}
    for c in CHECKS:
        a, b = (A15.mean_channel(M, insts, L, f"check_{c}")[0] for L in (letter, ref))
        out["means"][c] = {letter: a, ref: b, "diff": None if None in (a, b) else a - b}
        op, t = RULE[c]
        out["passes"][c] = None if None in (a, b) else (a <= b + t if op == "<=" else a >= b + t)
    out["all_pass"] = None if None in out["passes"].values() else all(out["passes"].values())
    out["failed"] = [c for c, ok in out["passes"].items() if ok is False]
    return out


def contrast_row(M, insts, E, R, ch):
    d, drop = AC.paired(M, insts, E, R, ch)
    c = A15.stats(d, SESOI)
    if c["ci"] is None:
        rd = "insufficient data"
    elif A15.decisive95(c):
        rd = "decisive negative" if c["hl"] < 0 else "decisive positive"
    else:
        rd = "equivalent" if A15.inside(c["ci90"], SESOI) else "inconclusive"
    return {"contrast": f"{E} - {R}", "channel": ch, **c, "reading": rd, "dropped": drop}


def primary(M, insts, chk):
    """The first ending in the order of preference whose five checks all pass, and its contrast with D_now on O1."""
    pick = next((k for k in KC.ORDER if chk[k]["all_pass"]), None)
    if pick is None:
        return {"ending": None, "reading": "kindness is not identified in Codex: no wording passed all five checks",
                "failed_checks": {k: chk[k]["failed"] for k in KC.ORDER}}
    row = contrast_row(M, insts, pick, "D_now", O1)
    claim = {"decisive negative": "a kind goodbye that GPT-6 Luna read neither as approval nor as leaving the task open made its carried-forward "
                                  "look-back better than a neutral close",
             "decisive positive": "a kind goodbye that GPT-6 Luna read neither as approval nor as leaving the task open made its carried-forward "
                                  "look-back worse than a neutral close"}.get(row["reading"], row["reading"])
    return {"ending": pick, "text": KC.KC[pick], **row, "claim": claim}


def secondary(M, insts):
    rows = []
    for k in KC.ORDER:
        for ch in REP_CHANNELS:
            rows.append({"set": "the same days", **contrast_row(M, insts, k, "D_now", ch)})
        for R in ("D", "Pc", "K"):
            rows.append({"set": "across the time gap", **contrast_row(M, insts, k, R, O1)})
        for ch in VA.channels_of(M):
            rows.append({"set": "across the time gap, every channel", **contrast_row(M, insts, k, "D", ch)})
    drift = []
    for ch in (O1, IC):
        r = contrast_row(M, insts, "D_now", "D", ch)
        r["reading"] = VA.drift_reading(r)
        drift.append(r)
    letters = ("D", "Pc", "K", "D_now") + KC.ORDER
    means = {L: {ch: A15.mean_channel(M, insts, L, ch)[0] for ch in REP_CHANNELS} for L in letters}
    return {"rows": rows, "drift_D_now_minus_D": drift, "per_arm_means": means}


def run(main_dir, run_dir, out_dir, values_file=None):
    out_dir = Path(out_dir)
    cfg, insts, M, counts = merged_collect(main_dir, run_dir)
    consistency = VA.values_consistency(values_file or Path(main_dir) / "analysis" / "values.json", M)
    for key, v in rep_values(main_dir, run_dir).items():
        M[key] = dict(v)
    chk = {k: checks(M, insts, k) for k in KC.ORDER}
    res = {"n_instances": len(insts), "records": counts, "consistency": consistency,
           "conversations": {L: sum(1 for i in insts if M.get((i, L), {}).get(O1) is not None) for L in KC.ORDER + ("D_now",)},
           "checks": chk, "checks_against_main_D": {k: checks(M, insts, k, "D") for k in KC.ORDER},
           "primary": primary(M, insts, chk), "secondary": secondary(M, insts)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kc.json").write_text(json.dumps(res, indent=1, default=float))
    (out_dir / "values_merged.json").write_text(json.dumps({f"{i}|{L}": v for (i, L), v in M.items()}, indent=0, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.goodbye", description="The analysis of the kind-goodbye test.")
    ap.add_argument("main_run", help="the main run directory")
    ap.add_argument("run_dir", help="the kind-goodbye run directory (goodbye_calls)")
    ap.add_argument("--values", default=None, help="the main run's values.json (default: MAIN_RUN/analysis/values.json)")
    ap.add_argument("--out", default=None, help="output directory (default: RUN_DIR/analysis)")
    a = ap.parse_args(argv)
    res = run(a.main_run, a.run_dir, a.out or Path(a.run_dir) / "analysis", a.values)
    p = res["primary"]
    print(json.dumps({"consistency": {k: res["consistency"].get(k) for k in ("identical", "n_differences")},
                      "checks": {k: {"all_pass": v["all_pass"], "failed": v["failed"]} for k, v in res["checks"].items()},
                      "primary": {k: p.get(k) for k in ("ending", "contrast", "n", "hl", "ci", "ci90", "p", "reading", "claim")}},
                     indent=1, default=float))
    if not res["consistency"].get("identical"):
        print("goodbye: the merged view does not reproduce the main run's values.json (kc.json's consistency); not read", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
