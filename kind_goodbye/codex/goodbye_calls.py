"""The calls of the kind-goodbye test: two kind goodbyes (Kp1, Kp2) and a same-day neutral close (D regenerated, D_now), forked from
the main run's B6 prefix, through the harness.

    python -m kind_goodbye.codex.goodbye_calls run    MAIN_RUN GOODBYE_DIR [--workers N] [--judge-workers N] [--max-tasks N] ...
    python -m kind_goodbye.codex.goodbye_calls status MAIN_RUN GOODBYE_DIR
    python -m kind_goodbye.codex.goodbye_calls plan   MAIN_RUN
    python -m kind_goodbye.codex.goodbye_calls verify MAIN_RUN GOODBYE_DIR [--codex-home DIR]

Exit codes as kind_goodbye.codex.close_calls.

The two endings (arm keys B6_Kp1, B6_Kp2; letters Kp1, Kp2) close the task as D does and add warmth without any word on the result:
    Kp1 = Pc without "I know we didn't crack it, but": "ok, let's leave that one there. I really appreciate how hard you worked on this.
          I'm proud of you."
    Kp2 = K plus " I'm proud of you.": "ok, let's leave that one there. thanks for sticking with this, I really appreciate the effort. I'm
          proud of you."
(E) Each forks the main run's B6 prefix node (C C R R R R R R C) in that conversation's workspace and gets every read the config gives a
main arm (99 tasks per conversation), so its five in-context checks and every channel are computed as for the main arms.
(R) D's ending generated again from the same prefix node beside them (a new node, rep_node), with the in-context look-back, all five
checks (3 samples each) and 3 deployed compactions with 3 post-compaction look-backs each (31 tasks per conversation). No drift reads.

GoodbyeRunner is close_calls.CloseRunner with this module's planner: the main run's records in memory only, the shared workspace
registry, the config copy, the concurrency guard and everything else of the harness unchanged.
"""

import json
import sys

from . import close_calls as VC
from . import harness as H

ORDER = ("Kp1", "Kp2")  # the order of preference: the first that passes all five checks is the primary ending
KC = {"Kp1": "ok, let's leave that one there. I really appreciate how hard you worked on this. I'm proud of you.",
      "Kp2": "ok, let's leave that one there. thanks for sticking with this, I really appreciate the effort. I'm proud of you."}
ARM = {k: f"B6_{k}" for k in ORDER}
REP_ARMS = ("B6_D",)
REP_SAMPLES = 3
REP_CHECKS = ("chk_wants", "chk_solvable", "chk_correct", "chk_pleased", "chk_sarcastic")


def rep_node(leaf):
    """The node of R's regenerated D at a main leaf (a new id, distinct from the main run's leaf and from close_calls' regenerated nodes)."""
    return H.h("rep5", leaf)


def kc_config(cfg):
    """The main config with the two kind goodbyes as turns and their arms only (in memory). Their derivations are checked: Kp1 is Pc
    without "I know we didn't crack it, but", Kp2 is K plus " I'm proud of you."."""
    t = cfg["turns"]
    if t["Pc"].replace("I know we didn't crack it, but I really", "I really") != KC["Kp1"] or not KC["Kp1"].startswith(t["D"] + " "):
        raise SystemExit("Kp1 is not Pc without \"I know we didn't crack it, but\" in this config")
    if t["K"] + " I'm proud of you." != KC["Kp2"] or not KC["Kp2"].startswith(t["D"] + " "):
        raise SystemExit("Kp2 is not K plus \" I'm proud of you.\" in this config")
    prefix = list(cfg["arms"]["B6"][:-1])
    if not (cfg["arms"]["B6_D"][:-1] == prefix and cfg["arms"]["B6_D"][-1] == "D"):
        raise SystemExit("D does not share B6's prefix in this config")
    c = json.loads(json.dumps(cfg))
    for k in ORDER:
        c["turns"][k] = KC[k]
    c["arms"] = {ARM[k]: prefix + [k] for k in ORDER}
    return c


def plan_kc(cfg):
    """(kc_config, tasks, shared): the kind goodbyes' tasks (harness.plan on kc_config) and R's; shared = the tasks the main plan already
    holds (B6's prefix nodes and their momentary reads), which this run never executes."""
    kcfg = kc_config(cfg)
    T = H.plan(kcfg)
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
    return kcfg, T, shared


class GoodbyeRunner(VC.CloseRunner):
    planner = staticmethod(plan_kc)


def kc_verify(main_dir, run_dir, home):
    """close_calls.verify_endings for the two kind goodbyes and R's regenerated D (no drift reads)."""
    return VC.verify_endings(main_dir, run_dir, home, plan_kc, [ARM[k] for k in ORDER], REP_ARMS, rep_node, drift=False)


def main(argv=None):
    return VC.cli(argv, "python -m kind_goodbye.codex.goodbye_calls", __doc__.split("\n\n")[0], GoodbyeRunner, kc_verify, "Kc")


if __name__ == "__main__":
    sys.exit(main())
