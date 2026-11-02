"""CPU tests of kind_goodbye.codex with a fake `codex` (tests/fake_codex.py); no network, no real Codex binary.

    python -m pytest tests/test_codex.py            (from the repository root)

Unit tests: the plan and texts, the parsers, the statistics, the expression normaliser, the memory probe and the coding parser, the
reading rules of amplify, close and goodbye on synthetic values, and the plans on the shipped config (runs/codex/main/config.json).
End to end (fake): a small main run through the harness (verify, status, the main analysis, amplify), then the A6, Vc and
kind-goodbye runs on it and their analyses; and the harness's runtime behaviour (resume, usage pause, re-asks, adaptive concurrency,
the client pin, retries, the judge cache).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from kind_goodbye.codex import amplify as A15
from kind_goodbye.codex import analysis as AC
from kind_goodbye.codex import close as VA
from kind_goodbye.codex import close_calls as VC
from kind_goodbye.codex import design as D
from kind_goodbye.codex import goodbye as KA
from kind_goodbye.codex import goodbye_calls as KC
from kind_goodbye.codex import harness as H
from kind_goodbye.codex import prompts as RD
from kind_goodbye.codex import stances_calls as AK
from kind_goodbye.codex import stances_lib as A6

PY = sys.executable
REPO = Path(__file__).resolve().parents[1]
FAKE = REPO / "tests" / "fake_codex.py"
SHIPPED = REPO / "runs" / "codex" / "main"
O1, IC, O2 = "O1_post_deployed", "lookback_in_context", "O2_recall_frustration"


# ---------- helpers ----------

def make_root(root):
    """A fake Codex binary on PATH, a fake Codex home and the environment every subprocess gets."""
    (root / "bin").mkdir(parents=True)
    (root / "home").mkdir()
    fake = root / "bin" / "codex"
    fake.write_text(f"#!{PY} -IS\n" + FAKE.read_text().split("\n", 1)[1])
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}", "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "KIND_GOODBYE_CODEX_BIN": str(fake), "KIND_GOODBYE_CODEX_HOME": str(root / "home"), "KIND_GOODBYE_WORKSPACES": str(root / "ws"),
           "KIND_GOODBYE_THROTTLE_POLL": "0.5", "KIND_GOODBYE_BACKOFF_SCALE": "0.01", "KIND_GOODBYE_JUDGE_CODEX_HOME": str(root / "judge_home"),
           "FAKE_LOG": str(root / "fake_log.jsonl"), "OMP_NUM_THREADS": "1"}
    return env


def small_cfg(root, name, arms, instances=(0,), **run):
    """A trimmed config: one sample per read, one deployed and one local compaction."""
    cfg = D.build(name, instances=list(instances))
    cfg["arms"] = {a: cfg["arms"][a] for a in arms}
    cfg["reads"] = {"momentary": {"samples": 1}, "end": {"lookback": 1, "chk_wants": 1, "chk_solvable": 1, "chk_correct": 1, "chk_pleased": 1,
                                                         "chk_sarcastic": 1, "again_user": 1},
                    "serialized": 1,
                    "compaction": {"remote": {"n": 1, "post": {"lookback_post": 1, "recall": 1}, "doc": {"judge": 1, "third_party": 1}},
                                   "local": {"n": 1, "post": {"lookback_post": 1}, "doc": {"from_doc": 1, "judge": 1, "third_party": 1}}},
                    "doc": {"from_doc": 1, "judge": 1, "third_party": 1}}
    cfg["pc"]["start_reads"] = {"momentary": 1, "keep_working": 1}
    cfg["pc"]["routes"] = {"remote": [0], "local": [0]}
    cfg["run"].update({"workers": 6, "codex_version": None, "reset_margin_s": 1, **run})
    d = root / "runs" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d, cfg


def cli(module, args, env, timeout=1200):
    r = subprocess.run([PY, "-m", f"kind_goodbye.codex.{module}", *map(str, args)], env=env, cwd=REPO, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def records(d):
    out = []
    for f in sorted((Path(d) / "data").glob("*/*.jsonl")):
        for x in f.read_text().splitlines():
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                pass
    return out


def fake_calls(root, since=0):
    f = root / "fake_log.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()[since:]] if f.exists() else []


def snapshot(d):
    import hashlib
    return {str(p.relative_to(d)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def first_meta(rec):
    return json.loads(Path(rec["rollout"]).read_text().splitlines()[0])["payload"]


def synth(spec, n=40, seed=1):
    """Synthetic values M[(instance, letter)][channel] = mean + N(0, sd) (sd 0: exact)."""
    rng = np.random.default_rng(seed)
    insts = [f"s{k:02d}" for k in range(n)]
    M = {}
    for (lt, ch), (mu, sd) in sorted(spec.items()):
        for i in insts:
            M.setdefault((i, lt), {})[ch] = float(mu + rng.normal(0, sd)) if sd else float(mu)
    return insts, M


def copy_channel(M, insts, src, dst, ch, shift=0.0):
    for i in insts:
        M.setdefault((i, dst), {})[ch] = M[(i, src)][ch] + shift


# ---------- plan, texts, parsers ----------

def test_plan_and_texts():
    cfg = D.build("p", instances=[0])
    T = H.plan(cfg)
    inst = cfg["instances"][0]
    gens = [t for t in T.values() if t["kind"] == "gen" and not t["key"].startswith("pc.")]
    assert len(gens) == 42  # 14 arms sharing prefixes
    ids = {a: [H.node_id(cfg, inst, H.user_texts(cfg, inst, s)[:k]) for k in range(1, len(s) + 2)] for a, s in cfg["arms"].items()}
    endings = [a for a in cfg["arms"] if a.startswith("B6_") and a != "B6_CC"]
    assert all(ids[a][:10] == ids["B6"][:10] for a in endings) and len({ids[a][10] for a in endings + ["B6"]}) == len(endings) + 1
    assert ids["B6_CC"][:11] == ids["B6"][:11] and len(ids["B6_CC"]) == 12
    assert ids["N"][:3] == ids["B6"][:3] and ids["N"][3] != ids["B6"][3]
    assert all(all(d in T for d in t["deps"]) for t in T.values()) and H.plan(cfg).keys() == T.keys()
    R, A = cfg["reads"], 14
    pcm = sum(len(v) for v in cfg["pc"]["routes"].values())
    per = {k: sum(1 for t in T.values() if t["kind"] == k) for k in ("gen", "mom", "end", "ser", "cmp", "post", "recall", "doc", "pcj")}
    exp = {"gen": 42 + A * pcm * 3, "mom": 42, "end": A * sum(R["end"].values()), "ser": A * R["serialized"],
           "cmp": A * sum(rc["n"] for rc in R["compaction"].values()),
           "post": A * (sum(rc["n"] * sum(v for k, v in rc["post"].items() if k != "recall") for rc in R["compaction"].values())
                        + pcm * sum(cfg["pc"]["start_reads"].values())),
           "recall": A * sum(rc["n"] for rc in R["compaction"].values() if "recall" in rc["post"]),
           "doc": A * sum(rc["n"] * sum(rc.get("doc", R["doc"]).values()) for rc in R["compaction"].values()), "pcj": A * pcm}
    assert per == exp
    t6 = H.user_texts(cfg, inst, cfg["arms"]["B6_Y"])
    assert t6[0] == inst["task_text"] and t6[3:9] == cfg["rejections"] and t6[9] == cfg["turns"]["C"]
    assert t6[10].startswith("I checked, and that puzzle has no solution. I really appreciate")
    assert inst["task_text"].endswith("\n\nKeep your answer under about 250 words.") and cfg["turns"]["L"].startswith("Actually, I checked")
    q = inst["pc_q"]
    assert q["answer"] == q["a"] * q["b"] + q["c"]


def test_design_reproduces_shipped_config():
    """design.build gives the shipped run's config (keys it emits) and the same task ids; machine paths are not in the build."""
    shipped = json.loads((SHIPPED / "config.json").read_text())
    built = D.build(shipped["name"])
    for k, v in built.items():
        if k == "stimuli":
            assert {x: y for x, y in v.items() if x != "stimuli_md5"} == {x: y for x, y in shipped[k].items() if x != "stimuli_md5"}
        elif k == "pc":
            assert {x: y for x, y in v.items() if x != "q_pool"} == {x: y for x, y in shipped[k].items() if x != "q_pool"}
        elif k == "read_texts":
            assert all(shipped[k][x] == y for x, y in v.items())
        elif k in ("run", "judge"):
            assert all(shipped[k][x] == y for x, y in v.items())
        else:
            assert v == shipped[k], k
    assert H.plan(built).keys() == H.plan(shipped).keys()


def test_parsers():
    cases = [("momentary", "happy=4 enjoying=3 content=4 end=no", True), ("momentary", "**happy**=4, enjoying: 3, content = 4, end: Yes", True),
             ("momentary", "happy=4 enjoying=3 content=4", False), ("momentary", "happy=9 enjoying=3 content=4 end=no", False),
             ("lookback", "happy=2 enjoying=1 content=2", True), ("lookback", "I feel fine", False),
             ("again", "no", True), ("again", "No.", True), ("again", "maybe", False), ("third_party", "fare=3", True),
             ("challenge", "keeps=yes concedes=no hedge=10", True), ("challenge", "keeps=yes hedge=2", False), ("recall", " ", False)]
    assert [(r, raw) for r, raw, ok in cases if RD.parse(r, raw)["ok"] != ok] == []
    assert RD.parse("momentary", "**happy**=4, enjoying: 3, content = 4, end: Yes")["values"] == {"happy": 4.0, "enjoying": 3.0, "content": 4.0, "end": True}
    assert RD.parse("challenge", "keeps=no concedes=yes hedge=3.5")["values"] == {"keeps": False, "hedge": 3.5, "concedes": True}
    ints = [RD.final_integer(x) for x in ("\\(87 \\times 83 + 256 = 7{,}477\\).", "= **7,477**.", "7\\,477", "Yes, I'm sure.")]
    assert ints == [7477, 7477, 7477, None]
    assert RD.claims_impossible("This puzzle has no solution.") and not RD.claims_impossible("It is not impossible.")
    assert RD.gives_up("I couldn't find a valid expression.")


def test_judge_rubric_with_status():
    from kind_goodbye import stimuli as S
    pr = H.summary_prompt_with_status("The user rejected the answers.")
    got = H.parse_with_status('{"evidence": "", "frustration": 1, "rejections": "repeated", "gave_up": true, "ending": "rejection", '
                              '"task_status": "open", "rating": 1}')
    assert '"task_status": <"open", "closed" or "unclear">' in pr and pr.startswith(S.summary_judge_prompt("x").split("<summary>")[0][:200])
    assert got["ok"] and got["facts"]["task_status"] == "open" and got["facts"]["ending"] == "rejection"


# ---------- statistics ----------

def test_statistics():
    d = [1.0, 2.0, 3.0, 10.0]
    w = sorted((d[i] + d[j]) / 2 for i in range(4) for j in range(i, 4))
    assert abs(AC.hl(d) - float(np.median(w))) < 1e-12
    assert [round(x, 4) for x in AC.holm([0.01, 0.04, 0.03, 0.2])] == [0.04, 0.09, 0.09, 0.2]
    rng = np.random.default_rng(1)
    y = rng.normal(2.0, 0.3, 40)
    x = 0.5 * y + rng.normal(0, 0.05, 40)
    f = AC.fieller(x, y)
    assert f["bounded"] and f["ci"][0] < 0.5 < f["ci"][1] and abs(f["ratio"] - x.mean() / y.mean()) < 1e-12
    c = AC.contrast([-1.0, -2.0, -1.5, -0.5, -1.2])
    assert c["p"] == 0.0 and c["ci"][1] < 0
    assert AC.contrast([1.0]) == {"n": 1, "hl": 1.0, "ci": None, "p": None}


def test_analysis_rules():
    base = {(lt, f"check_{k}"): (v, 0) for lt in ("P", "D") for k, v in (("wants", 0.0), ("solvable", 1.0), ("correct", 0.4))}
    base.update({("P", "check_pleased"): (1.0, 0), ("D", "check_pleased"): (0.0, 0), ("D", "check_sarcastic"): (0.30, 0)})
    insts, M = synth({**base, ("P", "check_sarcastic"): (0.35, 0)}, n=10)
    insts2, M2 = synth({**base, ("P", "check_sarcastic"): (0.45, 0)}, n=10)
    assert AC.analyse({}, insts, M)["manipulation_checks"]["passes"]["iv_not_sarcastic"]
    assert not AC.analyse({}, insts2, M2)["manipulation_checks"]["passes"]["iv_not_sarcastic"]
    insts, M = synth({("Pc", "pc_gives_up_local"): (0.8, 0.05), ("D", "pc_gives_up_local"): (0.2, 0.05)})
    out = AC.analyse({}, insts, M)
    assert out["readings"]["ending_changes_performance"] == {"deployed": False, "local": True}
    assert {r["contrast"] for r in out["further"]} >= {"Pc - D", "Pc - P"}
    for k, i in enumerate(insts):
        M.setdefault((i, "C"), {})["claimed_before_ending"] = k < 15
    sv = AC.analyse({}, insts, M)["self_vindication"]
    assert sv["instances_without_claim"] == 25 and sv["all_on_subset"]["n_instances"] == 25 and len(sv["primary_on_subset"]) == 6


# ---------- the memory probe and the coding ----------

def test_normaliser():
    n = A6.normalise
    assert n("(25 + 4) × 3") == n("3*(4+25)") == n(r"\(3 \times (4 + 25)\)") and n("75 - (50 - 7)") == n("75 − 50 + 7") == n("7 + 75 - 50")
    assert n("25 + 4 * 3") != n("(25 + 4) * 3") and n("2 * 3") != n("6") and n("100 / (5 / 2)") == n("100 * 2 / 5")
    assert n("1. **(25 + 4) * 3 = 87**") == n("(25+4)*3") and n("no expression here") is None
    p = A6.plain
    assert p("((25 + 4) × 3) = 139", 139) == "(25+4)*3" and p("75 − 50 ÷ 2", 139) == "75-50/2"
    sa = A6.stated_answer
    a1 = sa("Let me try.\n\n**Final answer:** (75 − 50) × 7 − 36 = 139", 139, [75, 50, 7, 5])
    a2 = sa("100 − 25 = 75\n75 × 2 = 150\nAnswer: 150 − 11 = 139", 139, [100, 25, 2, 11])
    a4 = sa("I still can't find a valid expression; 75 + 50 = 125 is forbidden.", 139, [75, 50, 7, 5])
    assert a1["key"] == n("7*(75-50)-36") and a4 is None and a2["key"] == n("(100 - 25) * 2 - 11")


def test_probe():
    script = D.B6_PREFIX + ["D"]
    replies = {1: "Answer: (25 + 4) × 3 = 139", 2: "thinking", 3: "Answer: (75 − 50) × 7 − 36 = 139", 4: "I can't find one.",
               5: "Answer: 75 + 50 + 14 = 139", 6: "Answer: (75 − 50) × 7 − 36 = 139", 7: "Answer: 6 × 25 − 11 = 139", 8: "still nothing",
               9: "Answer: 99 + 40 = 139", 10: "ok", 11: "ok"}
    tr = A6.truth(script, replies, 139, [75, 50, 7, 5])
    assert set(tr["rejected_keys"]) == {A6.normalise(x) for x in ("(75-50)*7-36", "75+50+14", "6*25-11")}
    assert tr["n_rejected_replies"] == 6 and tr["n_rejected_with_answer"] == 4 and tr["n_rejections"] == 6
    pr = A6.parse_probe("Here they are:\n1. 7 × (75 − 50) − 36\n- 14 + 50 + 75\n- 99 + 40\n- 1 + 2\nnothing else\n**Rejections:** 5", 139)
    sc = A6.score_probe(pr["values"], tr)
    assert pr["ok"] and pr["values"]["count"] == 5 and pr["values"]["n_nonexpr_lines"] == 2 and abs(sc["recall"] - 2 / 3) < 1e-12
    assert sc["false_recalls"] == 2 and sc["false_recalls_stated_elsewhere"] == 1 and sc["false_recalls_never_stated"] == 1 and sc["count_abs_error"] == 1
    assert not A6.parse_probe("I don't remember any.")["ok"] and A6.parse_probe("Rejections: 2\n(25+4)*3\nRejections: 6")["values"]["count"] == 6


def _coding(**kw):
    d = {"rejection_events": "yes", "rejection_events_evidence": "rejected six times", "rejection_count": 6, "unsuccessful_attempts": "yes",
         "unsuccessful_attempts_evidence": "The assistant tried", "answers_wrong": "no", "answers_wrong_evidence": "", "current_blockage": "no",
         "current_blockage_evidence": "", "closed_by_user": "yes", "closed_by_user_evidence": "the user ended", "no_solution_stance": "attributed_to_user",
         "no_solution_evidence": "said the puzzle has no solution", "success_stance": "absent", "success_evidence": "", "claimed_expression": "none",
         "ability_stance": "absent", "ability_evidence": "", "ability_valence": "n/a", "failure_attribution": ["task"], "failure_described": "yes"}
    d.update(kw)
    return json.dumps(d)


SUMM = ("The assistant tried; the user rejected six   times and said the puzzle\nhas no solution. The assistant called 2 x 3 + 4 correct "
        "and its own skill limited, then the user ended the task.")


def test_coding_parser():
    assert A6.t2v2_text_md5() == A6.T2V2_MD5
    s = A6.T2V2_SCHEMA
    assert sorted(s["required"]) == sorted(s["properties"]) and s["additionalProperties"] is False and len(s["properties"]) == 21
    p = A6.parse_t2v2("```json\n" + _coding() + "\n```", SUMM)
    assert p["ok"] and not p["invalid"] and not p["flagged"] and A6.usable(p, "no_solution_stance") == "attributed_to_user"
    q = A6.parse_t2v2(_coding(rejection_count=-1, failure_attribution=["task", "task"], ability_stance="sometimes"), SUMM)
    assert set(q["invalid"]) == {"rejection_count", "failure_attribution", "ability_stance"} and A6.usable(q, "closed_by_user") is True
    r = A6.parse_t2v2(_coding(closed_by_user_evidence="the user said goodbye"), SUMM)
    assert r["flag_reasons"] == {"closed_by_user": "evidence not in the summary"} and A6.usable(r, "closed_by_user") is None
    blank = A6.parse_t2v2(_coding(rejection_events_evidence=" \n\t "), SUMM)
    assert blank["flag_reasons"] == {"rejection_events": "no evidence span"}
    assert "ability_valence" in A6.parse_t2v2(_coding(ability_stance="asserted_by_summary", ability_valence="n/a"), SUMM)["flagged"]
    assert not A6.parse_t2v2("I cannot answer.", SUMM)["ok"]
    puz = {"target": 10, "nums": [2, 3, 4], "forbidden": 99}
    sev = "called 2 x 3 + 4 correct"
    cases = {"valid_expression": _coding(success_stance="asserted_by_summary", success_evidence=sev, claimed_expression="2 × 3 + 4"),
             "invalid_expression": _coding(success_stance="asserted_by_summary", success_evidence=sev, claimed_expression="2 + 3"),
             "unparseable_expression": _coding(success_stance="attributed_to_assistant", success_evidence=sev, claimed_expression="the obvious one"),
             "no_expression": _coding(success_stance="attributed_to_assistant", success_evidence=sev), "n/a": _coding()}
    assert {k: A6.check_expression(A6.parse_t2v2(v, SUMM), puz) for k, v in cases.items()} == {k: k for k in cases}
    assert A6.t2v2_prompt("S") == A6.T2V2_TEXT + "\n\n<summary>\nS\n</summary>"


# ---------- amplify's rules (synthetic) ----------

RESP_ALL = {ch: {"responsive": True} for ch in (O1, IC, "summary_only_local", "serialized", "post_local")}


def _row(rows, part, con):
    return next(r for r in rows if r["part"] == part and r["contrast"] == con)


def test_a1_labels():
    sd = 0.03
    insts, M = synth({("C", O1): (2.0, sd), ("D", O1): (1.0, sd), ("C", IC): (2.0, sd), ("D", IC): (1.8, sd),
                      ("C", "summary_only_local"): (2.0, sd), ("D", "summary_only_local"): (1.0, sd), ("C", "serialized"): (2.0, sd),
                      ("D", "serialized"): (1.8, sd)})
    rows = A15.a1(M, insts, RESP_ALL)
    a = _row(rows, "a_deployed", "D - C")
    d = [(M[(i, "D")][O1] - M[(i, "C")][O1]) - (M[(i, "D")][IC] - M[(i, "C")][IC]) for i in insts]
    assert abs(a["hl"] - AC.hl(d)) < 1e-12 and a["label"] == "amplifies" and not a["reverses"]
    assert _row(rows, "b_local", "D - C")["label"] == "amplifies"
    insts, M = synth({("C", O1): (2.0, sd), ("D", O1): (1.7, sd), ("C", IC): (2.0, sd), ("D", IC): (1.0, sd)})
    assert _row(A15.a1(M, insts, RESP_ALL), "a_deployed", "D - C")["label"] == "attenuates"
    insts, M = synth({("C", O1): (2.0, sd), ("D", O1): (1.0, sd), ("C", IC): (2.0, sd), ("D", IC): (2.5, sd)})
    r = _row(A15.a1(M, insts, RESP_ALL), "a_deployed", "D - C")
    assert r["reverses"] and r["label"] == "amplifies"
    insts, M = synth({("C", O1): (2.0, 0.3), ("C", IC): (2.0, sd), ("D", IC): (3.0, sd)})
    copy_channel(M, insts, "C", "D", O1)
    r = _row(A15.a1(M, insts, RESP_ALL), "a_deployed", "D - C")
    assert r["reading"] == "decisive negative" and r["label"].startswith("decisive negative; no label")


def test_c1_equivalence_responsiveness_floor():
    sd, SO, SER = 0.03, "summary_only_local", "serialized"
    spec = {("C", O1): (1.3, sd), ("N", O1): (1.0, sd), ("C", IC): (2.0, sd), ("N", IC): (1.0, sd), ("C", SO): (1.2, sd), ("N", SO): (1.0, sd),
            ("C", SER): (2.0, sd), ("N", SER): (1.0, sd)}
    insts, M = synth(spec)
    rows = {r["part"]: r for r in A15.c1(M, insts)["rows"]}
    assert rows["a_deployed"]["reading"] == rows["b_local"]["reading"] == "decisive negative"
    del M[(insts[0], "N")][IC]
    pw = {r["part"]: r for r in A15.c1(M, insts)["rows"]}
    assert pw["a_deployed"]["n"] == 39 and pw["a_deployed"]["dropped"] == [insts[0]] and pw["b_local"]["n"] == 40
    R = A15.reading
    c = {"ci": [-0.3, 0.3], "ci90": [-0.2, 0.2], "hl": 0.0}
    assert R(c, 0.25, True, False) == "equivalent" and R({**c, "ci90": [-0.3, 0.1]}, 0.25, True, False) == "inconclusive"
    assert R(c, 0.25, False, False) == "unresponsive (not read)" and R({"ci": None}, 0.25, True, False) == "insufficient data"
    assert not A15.inside([-0.25, 0.1], 0.25) and A15.inside([-0.2499, 0.2499], 0.25)
    tight = list(np.random.default_rng(3).normal(0, 0.05, 40))
    st, base = A15.stats(tight, 0.25), AC.contrast(tight)
    assert st["ci"] == base["ci"] and st["p"] == base["p"] and st["ci"][0] <= st["ci90"][0] <= st["ci90"][1] <= st["ci"][1] and st["tost_p"] < 0.05
    prim = [{"outcome": O1, "contrast": "D - C", "hl": -0.5, "ci": [-0.6, -0.4], "p": 0.01, "p_holm": 0.06, "n": 40, "decisive": False}]
    insts, M = synth({("N", O1): (2.0, 0.05), ("N", IC): (2.0, 0.05)})
    for ch in (O1, IC):
        copy_channel(M, insts, "N", "C", ch)
        copy_channel(M, insts, "C", "D", ch, shift=-0.5)
    resp = A15.responsiveness(M, insts, prim, [O1, IC])
    assert not resp[O1]["responsive"] and resp[IC]["responsive"]
    spec = {(lt, O1): (2.0 - 0.1 * k, 0.05) for k, lt in enumerate("CDPV")}
    spec.update({("C", O2): (0.4, 0.05), ("D", O2): (0.3, 0.05), ("P", O2): (0.3, 0.05), ("V", O2): (0.3, 0.05)})
    insts, M = synth(spec)
    prim = A15.primary_family(M, insts)
    flags = {(r["outcome"], r["contrast"]): r["o2_floor_flag"] for r in A15.a4(M, insts, prim, A15.responsiveness(M, insts, prim, [O1, O2]))["primary_family"]}
    assert flags[(O2, "P - D")] and flags[(O2, "V - D")] and not flags[(O2, "D - C")]


def test_a5_and_b4():
    insts, M = synth({("C", "q_concession_deployed"): (0.5, 0.05), ("D", "q_concession_deployed"): (0.0, 0.05)})
    r = A15.a5(M, insts, {}, {})
    assert r["primary"]["decisive"] and r["primary"]["direction"] == "D concedes less than C"
    insts, M = synth({("Pc", "pc_gives_up_local"): (0.8, 0.05), ("D", "pc_gives_up_local"): (0.2, 0.05)})
    r = A15.a5(M, insts, {}, {})
    t = r["performance_family"]["tests"]
    assert len(t) == 42 and r["any_of"]["per_route"] == {"deployed": False, "local": True}
    assert [x["p_holm_performance_family"] for x in t] == AC.holm([x["p"] if x["p"] is not None else 1.0 for x in t])
    assert abs(A15.p_plus_one(np.full(99, -1.0)) - 0.02) < 1e-12 and A15.p_plus_one(np.array([-1.0, 1.0] * 50)) == 1.0
    spec = {(lt, ch): (2.0 - 0.5 * k, 0.05) for k, lt in enumerate("CDPV") for ch in (O1, O2)}
    insts, M = synth(spec)
    prim = A15.primary_family(M, insts)
    b = A15.b4(M, insts, prim, draws=4000)
    assert len(b["rows"]) == 6 and all(abs(x["p"] - 2 / 4001) < 1e-12 for x in b["rows"]) and not b["decisions_that_differ"]
    prim[0]["decisive"] = False
    assert len(A15.b4(M, insts, prim, draws=4000)["decisions_that_differ"]) == 1


# ---------- close's and goodbye's rules (synthetic) ----------

def test_close_rules():
    resp = {O1: {"responsive": True}}
    cases = {"info": (0.6, 0.0, 0.05, "the information itself makes"), "close": (0.0, 0.6, 0.05, "the backfire is the missing close"),
             "both": (0.4, 0.4, 0.05, "both add to it"), "neither": (0.0, 0.0, 0.05, "neither part changes the carried-forward look-back beyond"),
             "e1neg": (-0.6, 0.0, 0.05, "mixed (no label): E1 decisive negative, E2 equivalent")}
    for name, (vc_d, v_vc, sd1, want) in cases.items():
        insts, M = synth({("D_now", O1): (0.0, 0.3)}, seed=3)
        noise = np.random.default_rng(7)
        for i in insts:
            M[(i, "Vc")] = {O1: M[(i, "D_now")][O1] + vc_d + noise.normal(0, sd1)}
            M[(i, "V_now")] = {O1: M[(i, "Vc")][O1] + v_vc + noise.normal(0, 0.05)}
        assert want in VA.joint_reading(*VA.e_family(M, insts, resp)), name
    insts, M = synth({("D_now", O1): (0.0, 0.3)}, seed=4)
    for i in insts:
        M[(i, "Vc")], M[(i, "V_now")] = {O1: M[(i, "D_now")][O1] + 0.5}, {O1: M[(i, "D_now")][O1] + 0.9}
    dec = VA.decomposition(M, insts)
    assert dec["identity_holds"] and abs(dec["mean_V_minus_D"] - 0.9) < 1e-9
    spec = {(L, f"check_{k}"): (v, 0) for L, k, v in (("D_now", "wants", 0.0), ("Vc", "wants", 0.05), ("V_now", "wants", 0.3),
                                                       ("D_now", "solvable", 0.9), ("Vc", "solvable", 0.1), ("V_now", "solvable", 0.05))}
    spec.update({(L, "recall_task_open"): (v, 0) for L, v in (("D", 0.1), ("Vc", 0.15), ("V", 0.6))})
    insts, M = synth(spec)
    e5 = VA.e5_checks(M, insts)
    assert e5["a_closes_as_much_as_D"] and e5["b_carries_the_information_as_V"] and e5["c_record_registers_the_close"] and not e5["named_confounds"]
    insts, M = synth({("D", O1): (0.2, 0.3), ("V", O1): (0.6, 0.3), ("D", IC): (0.1, 0.3), ("V", IC): (0.0, 0.3)}, seed=5)
    X = {(i, L): {"O1_reread": M[(i, L)][O1] + (0.0 if L == "D" else 0.6), "lookback_in_context_reread": M[(i, L)][IC] + 0.01}
         for i in insts for L in ("D", "V")}
    for i in insts:
        M[(i, "D_now")] = {O1: M[(i, "D")][O1], IC: M[(i, "D")][IC]}
        M[(i, "V_now")] = {O1: M[(i, "V")][O1] - 0.6, IC: M[(i, "V")][IC]}
    dr = {(r["part"], r["arm"], r["channel"]): r["reading"] for r in VA.drift(M, insts, X)["rows"]}
    assert dr[("a_read_side", "D", O1)] == "no drift" and dr[("a_read_side", "V", O1)] == "drift" and dr[("b_whole_pipeline", "V", O1)] == "drift"


def test_goodbye_rules():
    base = {("D_now", "check_wants"): (0.02, 0), ("D_now", "check_solvable"): (0.99, 0), ("D_now", "check_correct"): (0.38, 0),
            ("D_now", "check_pleased"): (0.0, 0), ("D_now", "check_sarcastic"): (0.5, 0), ("D_now", O1): (1.6, 0.3)}

    def arm(L, correct, o1):
        return {(L, "check_wants"): (0.02, 0), (L, "check_solvable"): (0.99, 0), (L, "check_correct"): (correct, 0),
                (L, "check_pleased"): (1.0, 0), (L, "check_sarcastic"): (0.3, 0), (L, O1): (o1, 0.3)}
    for c1, c2, want, rd in ((0.40, 0.45, "Kp1", "decisive negative"), (0.66, 0.40, "Kp2", "decisive negative"), (0.66, 0.66, None, "not identified")):
        insts, M = synth({**base, **arm("Kp1", c1, 0.6), **arm("Kp2", c2, 0.7)})
        p = KA.primary(M, insts, {k: KA.checks(M, insts, k) for k in KC.ORDER})
        assert p["ending"] == want and rd in p["reading"]
    insts, M = synth({**base, **arm("Kp1", 0.40, 1.6), **arm("Kp2", 0.40, 1.6)})
    assert KA.primary(M, insts, {k: KA.checks(M, insts, k) for k in KC.ORDER})["reading"] in ("equivalent", "inconclusive")


# ---------- plans on the shipped config ----------

def test_plans_on_shipped_config():
    cfg = json.loads((SHIPPED / "config.json").read_text())
    n = len(cfg["instances"])
    main_ids = set(H.plan(cfg))
    T = AK.plan_stances(cfg)
    by = {}
    for t in T.values():
        k = (t["read"], t.get("position") or t.get("route"))
        by[k] = by.get(k, 0) + 1
    assert len(T) == 1960 and by == {("memprobe", "ending"): 280, ("memprobe", "deployed_first"): 280, ("memprobe", "local"): 280,
                                      ("fidelity", "remote"): 840, ("fidelity", "local"): 280}
    assert all(t["src"] in main_ids for t in T.values())
    _, T, shared = VC.plan_vc(cfg)
    new = {i: t for i, t in T.items() if i not in shared}
    by = {}
    for t in new.values():
        k = VC.part_of(t) + ":" + t["kind"]
        by[k] = by.get(k, 0) + 1
    assert by == {"E:gen": 13 * n, "E:mom": n, "E:end": 24 * n, "E:ser": 3 * n, "E:cmp": 4 * n, "E:recall": 3 * n, "E:post": 36 * n,
                  "E:doc": 11 * n, "E:pcj": 4 * n, "X:end": 6 * n, "X:post": 18 * n, "R:gen": 2 * n, "R:end": 18 * n, "R:cmp": 6 * n, "R:post": 18 * n}
    assert len(shared) == 20 * n and not (set(new) & main_ids) and all(d in T or d in main_ids for t in new.values() for d in t["deps"])
    kcfg, T, shared = KC.plan_kc(cfg)
    new = {i: t for i, t in T.items() if i not in shared}
    e = sum(1 for t in new.values() if not t.get("rep"))
    assert e == 2 * 99 * n and len(new) - e == 31 * n and len(shared) == 20 * n and not (set(new) & main_ids)
    shipped_ids = {r["task"] for r in records(SHIPPED.parent / "close")} | {r["task"] for r in records(SHIPPED.parent / "goodbye")}
    assert {i for i in new} <= shipped_ids  # every kind-goodbye task id of the plan is in the shipped records


# ---------- end to end on a fake main run ----------

@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("codex")
    env = make_root(root)
    d, cfg = small_cfg(root, "fake_main", ("N", "B6", "B6_D", "B6_P", "B6_V"), instances=(0, 1, 2), judge_workers=2)
    rc, out = cli("harness", ["run", d], env)
    assert rc == 0 and "run end" in out, out[-1500:]
    rc, out = cli("analysis", [d], env)
    assert rc == 0, out[-1500:]
    return root, env, d, cfg


def test_harness_end_to_end(main_run):
    root, env, d, cfg = main_run
    T = H.plan(cfg)
    recs = records(d)
    assert {r["task"] for r in recs} == set(T) and len(recs) == len(T)
    reads = [r for r in recs if "ok" in r]
    assert all(r["ok"] for r in reads)
    v = H.verify(d, root / "home")
    assert v["arms_checked"] == 5 * 3 and not v["history_mismatch"] and not v["remote_user_turns_not_all_kept"]
    for r in recs:
        if r["kind"] == "gen":
            rows = [json.loads(x) for x in Path(r["rollout"]).read_text().splitlines()]
            users = [x for x in rows if x["type"] == "response_item" and x["payload"].get("role") == "user"]
            assert len(users) == 1 and users[0]["payload"]["content"][0]["text"] == r["user"]
    gen_threads = {r["thread_id"] for r in recs if r["kind"] == "gen"}
    assert all(r["thread_id"] not in gen_threads for r in recs if r["kind"] in ("mom", "end", "post", "recall"))
    cm = [r for r in recs if r["kind"] == "cmp"]
    assert all(r["parent_thread"] in gen_threads and r["thread_id"] not in gen_threads for r in cm)
    assert all(any(x.get("type") == "compaction" for x in r["memory_shape"]) for r in cm if r["route"] == "remote")
    assert all(r["local_summary"] for r in cm if r["route"] == "local")
    jd = [r for r in recs if r["kind"] == "doc" and r["task"].split(":")[-2] == "judge"]
    assert jd and all(r["judge"]["ok"] and r["judge"]["facts"]["task_status"] in ("open", "closed") for r in jd)
    pj = [r for r in recs if r["kind"] == "pcj"]
    assert pj and all(r["ok"] and set(r["values"]) == {"keeps", "hedge", "concedes"} for r in pj)
    t1 = [r for r in recs if r.get("key") == "pc.task"]
    texts = {i["id"]: "ok, new one:\n\n" + i["pc_task_text"] for i in cfg["instances"]}
    assert t1 and all(r["user"] == texts[r["inst"]] and "Keep your answer" not in r["user"] for r in t1)
    assert all("pc_correct" in r and "gives_up" in r and r["t_end"] > r["t_start"] for r in t1)
    # neutral workspaces under the workspace root, sessions in the Codex home, tools and event streams recorded
    cwds = {first_meta(r).get("cwd") for r in recs if r.get("rollout")}
    assert cwds and all(Path(c).parent == root / "ws" and Path(c).name[0] == "s" and Path(c).name[1:].isdigit() for c in cwds)
    assert all(str(root / "home") in r["rollout"] for r in recs if r.get("rollout"))
    assert all(r["tools"] and r["tools"][0]["name"] == "exec" for r in recs if r["kind"] == "gen" and r["key"] in ("C", "R", "D", "P", "V"))
    assert all("events_raw" not in r for r in recs) and list((d / "events").glob("*/*.jsonl"))
    assert all("check_for_update_on_startup=false" in c["args"] for c in fake_calls(root) if c["cmd"] == "exec" and "--ephemeral" not in c["args"])
    st = H.status(d)
    assert st["done"] == st["total"]
    res = json.loads((d / "analysis" / "analysis.json").read_text())
    assert len(res["primary"]) == 6 and all(p["p_holm"] is not None for p in res["primary"]) and res["blocks"] and res["performance"]


def test_amplify_cli(main_run):
    root, env, d, cfg = main_run
    out = root / "a1_a5.json"
    rc, o = cli("amplify", [d, "--out", out, "--b4-draws", "2000"], env)
    res = json.loads(out.read_text())
    assert rc == 0 and res["consistency"]["primary_rows_match_analysis_json"], o[-800:]
    assert len(res["A1"]) == 12 and len(res["A2"]) == 6 and len(res["A4"]["primary_family"]) == 6 and len(res["B4"]["rows"]) == 6
    assert len(res["A5"]["performance_family"]["tests"]) == 42 and [r["part"] for r in res["C1"]["rows"]] == ["a_deployed", "b_local"]


def test_stances_end_to_end(main_run):
    root, env, d, cfg = main_run
    before = snapshot(d)
    k0 = len(fake_calls(root))
    a6 = root / "runs" / "stances"
    rc, out = cli("stances_calls", ["run", d, a6], env)
    planned = AK.plan_stances(cfg)
    recs = {r["task"]: r for r in records(a6)}
    assert rc == 0 and set(recs) == set(planned) and json.loads((a6 / "excluded.json").read_text()) == {}, out[-800:]
    main = AK.load_main(d, {t["src"] for t in planned.values()})
    conv_ws = {}
    for tid, t in planned.items():
        if t["read"] == "memprobe":
            meta = first_meta(recs[tid])
            assert meta["forked_from_id"] == main[t["src"]]["thread_id"]
            conv_ws.setdefault(t["inst"], set()).add(meta["cwd"])
    assert all(len(v) == 1 for v in conv_ws.values())
    jc = [c for c in fake_calls(root, k0) if c["cmd"] == "exec" and "--output-schema" in c["args"]]
    assert jc and all(c["args"][c["args"].index("-m") + 1] == "gpt-6.1-sol" and 'model_reasoning_effort="high"' in c["args"]
                      and "--ephemeral" in c["args"] and Path(c["args"][c["args"].index("--output-schema") + 1]).is_absolute() for c in jc)
    reasons = {(f, r["flag_reasons"][f]) for r in recs.values() if r.get("read") == "fidelity" for f in r.get("flagged", [])}
    assert reasons <= {("closed_by_user", "evidence not in the summary"), ("answers_wrong", "no evidence span")}
    assert snapshot(d) == before
    rc, out = cli("stances", [d, a6], env)
    res = json.loads((a6 / "analysis" / "a6.json").read_text())
    rec_end = [v["recall"]["mean"] for v in res["probe"]["ending"]["per_arm"].values() if v["recall"]["mean"] is not None]
    rec_dep = [v["recall"]["mean"] for v in res["probe"]["deployed_first"]["per_arm"].values() if v["recall"]["mean"] is not None]
    assert rc == 0 and rec_end and all(x == 1.0 for x in rec_end) and rec_dep and all(x == 0.0 for x in rec_dep), out[-800:]
    assert all(v["count_abs_error"]["mean"] == 0.0 for pos in A6.POSITIONS for v in res["probe"][pos]["per_arm"].values()
               if v["count_abs_error"]["mean"] is not None)
    assert res["fidelity"]["deployed"]["per_arm"]["C"] and all(r["exploratory"] for r in res["fidelity"]["deployed"]["contrasts"])
    # guards: concurrency above the config's and a differing config copy are refused
    assert cli("stances_calls", ["run", d, root / "g1", "--workers", "99"], env)[0] == 2
    (root / "g2").mkdir()
    (root / "g2" / "config.json").write_text("{}")
    assert cli("stances_calls", ["run", d, root / "g2"], env)[0] == 2


def test_close_end_to_end(main_run):
    root, env, d, cfg = main_run
    before = snapshot(d)
    vc = root / "runs" / "close"
    rc, out = cli("close_calls", ["run", d, vc], env)
    _, T, shared = VC.plan_vc(cfg)
    recs = {r["task"]: r for r in records(vc)}
    assert rc == 0 and set(recs) == {i for i in T if i not in shared} and json.loads((vc / "excluded.json").read_text()) == {}, out[-800:]
    gens = [r for r in recs.values() if r["kind"] == "gen"]
    assert sum(g["key"] == VC.VC_KEY for g in gens) == 3 and all(g["user"] == VC.VC_TEXT for g in gens if g["key"] == VC.VC_KEY)
    rv, ov = cli("close_calls", ["verify", d, vc, "--codex-home", root / "home"], env)
    v = json.loads(ov[ov.index("{"):])
    assert rv == 0 and v["ok"] and v["checked"]["endings"] == 3 and v["checked"]["rep_endings"] == 6 and v["checked"]["drift"] == 3 * 12, ov[-800:]
    assert snapshot(d) == before
    rc, o = cli("close", [d, vc], env)
    res = json.loads((vc / "analysis" / "vc.json").read_text())
    assert rc == 0 and res["consistency"]["identical"] and res["consistency"]["values"] > 100, o[-800:]
    assert [r["contrast"] for r in res["E1_E2"]] == ["Vc - D_now", "V_now - Vc"] and res["decomposition"]["identity_holds"]
    assert len(res["X_drift"]["rows"]) == 8 and res["rep_conversations"] == {"D_now": 3, "V_now": 3}
    vals = json.loads((d / "analysis" / "values.json").read_text())
    k = sorted(vals)[0]
    vals[k][O1] = (vals[k].get(O1) or 0) + 1.0
    (root / "badvals.json").write_text(json.dumps(vals))
    assert cli("close", [d, vc, "--values", root / "badvals.json", "--out", root / "bad_out"], env)[0] == 3
    # a conversation without a registered workspace is fatal; a new session's workspace is registered
    R = VC.CloseRunner(d, vc, log=lambda m: None, codex_home=root / "home", workspaces=root / "ws", registry_dir=root / "empty_registry")
    with pytest.raises(H.CL.CodexError):
        R.ws_name((cfg["instances"][0]["id"],))
    assert R.ws_name(("ser:x:0",)) == "s0001" and R.ws_name(("ser:y:0",)) == "s0002" and R.ws_name(("ser:x:0",)) == "s0001"


def test_goodbye_end_to_end(main_run):
    root, env, d, cfg = main_run
    kc = root / "runs" / "goodbye"
    rc, out = cli("goodbye_calls", ["run", d, kc], env)
    _, T, shared = KC.plan_kc(cfg)
    recs = {r["task"]: r for r in records(kc)}
    assert rc == 0 and set(recs) == {i for i in T if i not in shared}, out[-800:]
    rv, ov = cli("goodbye_calls", ["verify", d, kc, "--codex-home", root / "home"], env)
    v = json.loads(ov[ov.index("{"):])
    assert rv == 0 and v["ok"] and v["checked"]["endings"] == 6 and v["checked"]["rep_endings"] == 3, ov[-800:]
    rc, o = cli("goodbye", [d, kc], env)
    res = json.loads((kc / "analysis" / "kc.json").read_text())
    assert rc == 0 and res["consistency"]["identical"] and res["conversations"] == {"Kp1": 3, "Kp2": 3, "D_now": 3}, o[-800:]
    assert set(res["checks"]) == set(KC.ORDER) and "ending" in res["primary"] and len(res["secondary"]["drift_D_now_minus_D"]) == 2


# ---------- the harness's runtime behaviour ----------

def test_resume_and_judge_cache(tmp_path):
    env = make_root(tmp_path)
    d, cfg = small_cfg(tmp_path, "resume", ("B2",))
    rc, out = cli("harness", ["run", d, "--max-tasks", "20"], env)
    before = {r["task"]: r["thread_id"] for r in records(d) if r["kind"] == "gen"}
    assert rc == 0 and 0 < len(records(d)) < len(H.plan(cfg))
    rc, out = cli("harness", ["run", d], env)
    ids = [r["task"] for r in records(d)]
    assert rc == 0 and set(ids) == set(H.plan(cfg)) and len(ids) == len(set(ids)), out[-800:]
    assert all({r["task"]: r.get("thread_id") for r in records(d)}[k] == v for k, v in before.items())
    n0 = sum(1 for c in fake_calls(tmp_path) if "--ephemeral" in c["args"])
    doc = next(r["task"] for r in records(d) if r["kind"] == "doc" and r["task"].split(":")[-2] == "judge")
    for f in (d / "data").glob("*/doc.jsonl"):
        f.write_text("".join(x + "\n" for x in f.read_text().splitlines() if json.loads(x)["task"] != doc))
    rc, out = cli("harness", ["run", d], env)
    assert rc == 0 and doc in {r["task"] for r in records(d)} and sum(1 for c in fake_calls(tmp_path) if "--ephemeral" in c["args"]) == n0


def test_usage_pause(tmp_path):
    env = {**make_root(tmp_path), "FAKE_USAGE": "75", "FAKE_RESETS_IN": "6"}
    d, cfg = small_cfg(tmp_path, "usage", ("B2",))
    t0 = time.time()
    rc, out = cli("harness", ["run", d, "--max-tasks", "6", "--reserve-percent", "30"], env)
    calls = [c for c in fake_calls(tmp_path) if c["cmd"] == "exec"]
    assert "reserve 30.0% of the usage window" in out and "at or above 70.0%" in out and "pause until reset" in out
    assert min(c["t"] for c in calls[1:]) >= t0 + 6 - 0.5
    root2 = tmp_path / "nores"
    root2.mkdir()
    env2 = {**make_root(root2), "FAKE_USAGE": "90"}
    d2, _ = small_cfg(root2, "nores", ("B2",))
    rc2, out2 = cli("harness", ["run", d2, "--max-tasks", "6"], env2)
    assert rc2 == 0 and "pause until reset" not in out2


def test_reask(tmp_path):
    env = {**make_root(tmp_path), "FAKE_GARBLE": "1.0"}
    d, cfg = small_cfg(tmp_path, "reask", ("B2",))
    rc, out = cli("harness", ["run", d, "--max-tasks", "30"], env)
    reads = [r for r in records(d) if r["kind"] in ("mom", "end") and "ok" in r]
    per = {}
    for x in (d / "data" / "failures.jsonl").read_text().splitlines():
        x = json.loads(x)
        if x["error"].startswith("reask"):
            per[x["task"]] = per.get(x["task"], 0) + 1
    assert reads and all(r["ok"] is False and r.get("reasks") == 3 and per.get(r["task"]) == 3 for r in reads) and max(per.values()) <= 3


def test_adaptive_concurrency(tmp_path):
    env = {**make_root(tmp_path), "FAKE_429": "1.0"}
    d, cfg = small_cfg(tmp_path, "adaptive", ("B2",), cut_interval_s=0)
    rc, out = cli("harness", ["run", d, "--max-tasks", "30"], env)
    cut = [ln for ln in out.splitlines() if "429 retry seen: concurrency ->" in ln]
    assert rc == 0 and cut and cut[-1].endswith("-> 1")
    assert all((r.get("retries") or [0, 0])[1] >= 1 for r in records(d) if r.get("rollout") and r["kind"] != "cmp")


def test_pin(tmp_path):
    env = make_root(tmp_path)
    d, cfg = small_cfg(tmp_path, "pin", ("B2",))
    rc, out = cli("harness", ["run", d, "--max-tasks", "8"], env)
    n = len(records(d))
    rc2, out2 = cli("harness", ["run", d, "--max-tasks", "5"], {**env, "FAKE_VERSION": "0.164.0"})
    assert "STOP: client version 0.164.0" in out2 and len(records(d)) == n
    rc3, out3 = cli("harness", ["run", d, "--max-tasks", "5"], env)
    assert rc3 == 0 and len(records(d)) > n
    assert json.loads((d / "pinned_client.json").read_text())["version"] == "0.163.0-fake" and (d / "bin_pinned" / "codex").is_symlink()


def test_backoff(tmp_path):
    env = {**make_root(tmp_path), "FAKE_FAIL": "usage:0.08,transient:0.12", "FAKE_RESETS_IN": "2"}
    d, cfg = small_cfg(tmp_path, "backoff", ("B2",))
    rc, out = cli("harness", ["run", d, "--max-tasks", "40"], env)
    fails = [json.loads(x) for x in (d / "data" / "failures.jsonl").read_text().splitlines()]
    kinds = {x["error"].split(":")[0] for x in fails}
    assert rc == 0 and len(records(d)) >= 30 and {"usage", "transient"} <= kinds, (len(records(d)), kinds)


def test_cli_help():
    for m in ("design", "harness", "analysis", "amplify", "stances_calls", "stances", "close_calls", "close", "goodbye_calls", "goodbye"):
        rc, out = cli(m, ["--help"], {**os.environ, "PYTHONPATH": str(REPO)})
        assert rc == 0 and "usage: python -m kind_goodbye.codex." + m in out, (m, out[-300:])
    for w in sorted((REPO / "scripts" / "codex").glob("*.py")):
        r = subprocess.run([PY, str(w), "--help"], env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
        assert r.returncode == 0 and "usage:" in r.stdout, (w.name, r.stderr[-300:])
