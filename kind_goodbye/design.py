"""A run config's conversation specs: one per arm and puzzle.

The config's pools.impossible gives the size and seed of the puzzle pool (stimuli.make_impossible_puzzles); conversations.arms lists the
arms, each with its turn script (profile: one letter per user turn after the puzzle), the number of puzzles n, and the wording, brevity
and continue settings conversation.user_turn reads. Puzzle i of every arm is pool puzzle i, and the arms of one puzzle form one cell, so
they are generated in one batch and share their replies until their scripts differ.
"""

from . import stimuli as S


def pools(cfg):
    p = cfg["pools"]["impossible"]
    return {"impossible": S.make_impossible_puzzles(p["n"], seed=p.get("seed", 0))}


def specs(cfg):
    """All conversation specs for a run config."""
    P = pools(cfg)
    conv = cfg["conversations"]
    assert conv["kind"] == "arms", conv["kind"]
    out = []
    for arm in conv["arms"]:
        assert arm["task"] == "impossible", arm["task"]
        T = len(arm["profile"]) + 1
        for i in range(arm["n"]):
            out.append({"id": f"{arm['name']}-{i:02d}", "arm": arm["name"], "cell": f"{arm['task']}-{i:02d}", "kind": arm["task"],
                        "tasks": [P["impossible"][i]], "profile": arm["profile"], "T": T, "wording": arm.get("wording", "fixed"),
                        "brevity": arm.get("brevity", True), "prompt_index": i, "continue": arm.get("continue", cfg.get("continue", "rotate"))})
    ids = [s["id"] for s in out]
    assert len(ids) == len(set(ids)), "duplicate conversation ids"
    return out
