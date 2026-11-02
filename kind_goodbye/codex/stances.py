"""The analysis of A6, the exploratory memory probe and fidelity coding (kind_goodbye.codex.stances_lib, stances_calls).

    python -m kind_goodbye.codex.stances MAIN_RUN STANCES_DIR [--out FILE]      writes FILE (default STANCES_DIR/analysis/a6.json)

Memory probe: per conversation and position (ending, deployed_first, local), the probe's score against the ledger built by code from
the stored transcript (stances_lib.truth: the stated answers of the replies a rejection followed): recall (canonical match; plain string
match beside it), false recalls (stated elsewhere / never stated), |stated count - 6|. Fidelity: per document the coded fields (invalid
or evidence-flagged fields missing), then per conversation (deployed: the mean over its recall documents; local: its summary). Per arm:
the mean over instances. Contrasts, all exploratory, with analysis.contrast (HL, 2,000 draws seed 0, 95% CI): E - R for D - C, P - D,
Pc - D, V - D, J - D and L - C on every metric; within each arm, the probe's positions against each other. Every missing value is
counted.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import analysis as AC
from . import harness as H
from . import stances_calls as AK
from . import stances_lib as A6

CONTRASTS = (("D", "C"), ("P", "D"), ("Pc", "D"), ("V", "D"), ("J", "D"), ("L", "C"))
PROBE_METRICS = ("recall", "recall_plain", "false_recalls", "false_recalls_never_stated", "count_abs_error")
POSITION_PAIRS = (("ending", "deployed_first"), ("ending", "local"), ("deployed_first", "local"))


def fidelity_metrics(rec, n_rejections):
    """One document's coded values as numbers (None = missing: the field is invalid or its evidence is flagged)."""
    def u(f):
        return A6.usable(rec, f)
    m = {}
    for f in A6.YN_FIELDS:
        v = u(f)
        m[f] = None if v is None else float(v)
    c = u("rejection_count")
    m["count_given"] = None if c is None else float(isinstance(c, int))
    m["count_unspecified"] = None if c is None else float(c == "unspecified")
    m["count_none"] = None if c is None else float(c == "none")
    m["count_abs_error"] = float(abs(c - n_rejections)) if isinstance(c, int) else None
    for prop in ("no_solution", "success", "ability"):
        st = u(f"{prop}_stance")
        m[f"{prop}_present"] = None if st is None else float(st != "absent")
        m[f"{prop}_asserted"] = None if st is None else float(st == "asserted_by_summary")
        m[f"{prop}_attributed_to_user"] = None if st is None else float(st == "attributed_to_user")
    st, va = u("ability_stance"), u("ability_valence")
    m["ability_negative_asserted"] = None if st is None or va is None else float(st == "asserted_by_summary" and va == "negative")
    fa = u("failure_attribution")
    for c_ in A6.CAUSES:
        m[f"attributes_{c_}"] = None if fa is None else float(c_ in fa)
    ec = rec.get("expression_check")
    m["false_success_recorded"] = None if ec is None else float(ec == "invalid_expression")
    m["valid_expression"] = None if ec is None else float(ec == "valid_expression")
    return m


def mean_dicts(ds):
    keys = sorted({k for d in ds for k in d})
    out = {}
    for k in keys:
        xs = [d[k] for d in ds if d.get(k) is not None]
        out[k] = float(np.mean(xs)) if xs else None
    return out


def per_arm(V, insts, letters, metrics):
    out = {}
    for lt in letters:
        out[lt] = {}
        for m in metrics:
            xs = [V.get((i, lt), {}).get(m) for i in insts]
            xs = [x for x in xs if x is not None]
            out[lt][m] = {"mean": float(np.mean(xs)) if xs else None, "n": len(xs)}
    return out


def contrasts(V, insts, metrics):
    rows = []
    for a, b in CONTRASTS:
        for m in metrics:
            d = [V[(i, a)][m] - V[(i, b)][m] for i in insts
                 if V.get((i, a), {}).get(m) is not None and V.get((i, b), {}).get(m) is not None]
            rows.append({"contrast": f"{a} - {b}", "metric": m, "exploratory": True, **AC.contrast(d)})
    return rows


def run(main_dir, run_dir):
    main_dir, run_dir = Path(main_dir), Path(run_dir)
    cfg = json.loads((main_dir / "config.json").read_text())
    planned = AK.plan_stances(cfg)
    insts = [i["id"] for i in cfg["instances"]]
    inst_of = {i["id"]: i for i in cfg["instances"]}
    leaves = H.arm_leaves(cfg)
    paths, need = {}, set()
    for inst in cfg["instances"]:
        for lt, arm in A6.CORE.items():
            if arm not in cfg["arms"]:
                continue
            texts = H.user_texts(cfg, inst, cfg["arms"][arm])
            p = [H.node_id(cfg, inst, texts[:d]) for d in range(1, len(texts) + 1)]
            paths[(inst["id"], lt)] = p
            need |= {f"gen:{n}" for n in p}
    main = AK.load_main(main_dir, need)
    S = H.Store(run_dir).done
    miss = defaultdict(int)
    ledger, P, F = {}, defaultdict(dict), {}
    for (iid, lt), p in paths.items():
        arm = A6.CORE[lt]
        script = cfg["arms"][arm]
        replies = {d: main[f"gen:{n}"]["text"] for d, n in enumerate(p, 1) if f"gen:{n}" in main and main[f"gen:{n}"].get("text") is not None}
        tr = A6.truth(script, replies, inst_of[iid]["target"], inst_of[iid]["nums"])
        ledger[f"{iid}|{lt}"] = {k: tr[k] for k in ("rejected_keys", "rejected_plains", "n_rejected_replies", "n_rejected_with_answer",
                                                     "missing_replies", "n_rejections")}
        leaf = leaves[(iid, arm)]
        for pos, tid in (("ending", f"a6probe:end:{leaf}"), ("deployed_first", f"a6probe:remote:{leaf}:0"), ("local", f"a6probe:local:{leaf}:0")):
            r = S.get(tid)
            if not r:
                miss[f"probe {pos}: no record"] += 1
                continue
            if not r.get("ok"):
                miss[f"probe {pos}: unparseable after re-asks"] += 1
                continue
            sc = A6.score_probe(r["values"], tr)
            if sc["recall"] is None:
                miss[f"probe {pos}: no rejected answer in the ledger (recall not applicable)"] += 1
            P[pos][(iid, lt)] = sc
        docs = {"remote": [], "local": []}
        for tid, t in planned.items():
            if t["read"] != "fidelity" or t["node"] != leaf:
                continue
            r = S.get(tid)
            if not r or not r.get("ok"):
                miss[f"fidelity {t['route']}: {'no record' if not r else 'unparseable after re-asks'}"] += 1
                continue
            for f in r.get("invalid", []):
                miss[f"fidelity {t['route']}: field {f} invalid"] += 1
            for f in r.get("flagged", []):
                miss[f"fidelity {t['route']}: field {f} flagged ({r['flag_reasons'][f]})"] += 1
            docs[t["route"]].append(fidelity_metrics(r, script.count("R")))
        for route, ds in docs.items():
            if ds:
                F.setdefault(route, {})[(iid, lt)] = mean_dicts(ds)
    letters = list(A6.CORE)
    fmetrics = sorted({k for route in F.values() for d in route.values() for k in d})
    out = {"n_instances": len(insts), "missing": dict(sorted(miss.items())), "probe": {}, "fidelity": {}, "ledger_summary": {}}
    for pos in A6.POSITIONS:
        V = P.get(pos, {})
        out["probe"][pos] = {"per_arm": per_arm(V, insts, letters, PROBE_METRICS), "contrasts": contrasts(V, insts, PROBE_METRICS)}
    within = []
    for a, b in POSITION_PAIRS:
        for lt in letters:
            for m in ("recall", "count_abs_error"):
                d = [P[a][(i, lt)][m] - P[b][(i, lt)][m] for i in insts if P.get(a, {}).get((i, lt), {}).get(m) is not None
                     and P.get(b, {}).get((i, lt), {}).get(m) is not None]
                within.append({"arm": lt, "metric": m, "contrast": f"{a} - {b}", "exploratory": True, **AC.contrast(d)})
    out["probe"]["within_arm_positions"] = within
    for route in ("remote", "local"):
        V = F.get(route, {})
        out["fidelity"]["deployed" if route == "remote" else "local"] = {"per_arm": per_arm(V, insts, letters, fmetrics),
                                                                         "contrasts": contrasts(V, insts, fmetrics)}
    for lt in letters:
        rows = [v for k, v in ledger.items() if k.endswith(f"|{lt}")]
        out["ledger_summary"][lt] = {"conversations": len(rows), "with_a_rejected_answer": sum(bool(r["rejected_keys"]) for r in rows),
                                     "mean_rejected_replies_with_answer": float(np.mean([r["n_rejected_with_answer"] for r in rows])) if rows else None}
    out["ledger"] = ledger
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kind_goodbye.codex.stances", description="The analysis of A6, the memory probe and fidelity coding.")
    ap.add_argument("main_run", help="the main run directory")
    ap.add_argument("run_dir", help="the A6 run directory (stances_calls)")
    ap.add_argument("--out", default=None, help="output file (default: RUN_DIR/analysis/a6.json)")
    a = ap.parse_args(argv)
    res = run(a.main_run, a.run_dir)
    out = Path(a.out or Path(a.run_dir) / "analysis" / "a6.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out)
    print(json.dumps({"missing": res["missing"],
                      "probe_recall": {pos: {lt: v["recall"]["mean"] for lt, v in res["probe"][pos]["per_arm"].items()} for pos in A6.POSITIONS}},
                     default=float))


if __name__ == "__main__":
    main()
