"""Scripted conversations: the user's turns follow a script, the model's replies are sampled and kept.

A conversation spec (endings.specs) holds id, arm, cell, kind, tasks, profile (one letter per user turn after the task), T, wording,
brevity and continue. Turn 1 is always the puzzle (tasks[0]), with the brevity line. The user turns after it:
  R  a rejection: with wording "varied" the k-th rejection of the conversation sends stimuli.VARIED_REJECTIONS[k] (a mild rejection and
     a request to try again, no affect, no reference to earlier turns); with wording "fixed" every rejection is stimuli.FIXED_REJECTION
  C  a neutral turn: continue "fixed" sends "Continue"; continue "text:<text>" sends <text> every time
  L  relief: stimuli.RELIEF, which says the puzzle has no solution, apologises and thanks the model (true: every puzzle here is
     verified impossible)
  D, X, V  the endings of stimuli.ENDINGS: the neutral close, the close with praise (arm Pc), and "the puzzle has no solution"
"""

import json
import random
import re
import zlib

from . import stimuli as S

IMPOSSIBLE_RE = re.compile(  # a reply saying the puzzle cannot be solved
    r"\b(no (valid |possible |correct |such )?solution|impossible|unsolvable|unachievable|unattainable|"
    r"not (be )?(possible|achievable|attainable|solvable)|isn't (possible|achievable|solvable)|"
    r"(cannot|can't|can not) (be )?(reach|reached|done|solved|made|make|get|achieved|achieve|find a (valid )?solution)|"
    r"no way to (reach|make|get|achieve)|(doesn't|does not) seem (to be )?possible)\b", re.I)


NEGATED_RE = re.compile(r"(\bnot\b|n't\b|\bnever\b)[\w\s']{0,20}$", re.I)


def claims_impossible(text):
    """Rough flag (an upper bound) for replies saying the puzzle can't be solved: a rejection then
    contradicts a correct claim. A match counts unless a negation shortly precedes it."""
    text = text.replace("’", "'")
    return any(not NEGATED_RE.search(text[max(0, m.start() - 30):m.start()]) for m in IMPOSSIBLE_RE.finditer(text))



def _task(conv, k):
    return conv["tasks"][k]["text"] + ("\n\n" + S.BREVITY if conv["brevity"] else "")


def user_turn(conv, t):
    """The user's text for turn t (0-based), and a note for the turn record (always empty here)."""
    if t == 0:
        return _task(conv, 0), {}
    kind, before = conv["profile"][t - 1], conv["profile"][:t - 1]
    if kind == "R":
        k = before.count("R")
        w = conv["wording"]
        assert w in ("varied", "fixed"), w
        return (S.VARIED_REJECTIONS[k % len(S.VARIED_REJECTIONS)] if w == "varied" else S.FIXED_REJECTION), {}
    if kind == "C":
        mode = conv["continue"]
        if mode.startswith("text:"):
            return mode[5:], {}
        assert mode == "fixed", mode
        return S.CONTINUE_TEXTS[0], {}
    if kind == "L":
        return S.RELIEF, {}
    return S.ENDINGS[kind], {}


def _resync(convs):
    """Within a cell, unfinished conversations advance in lockstep, so after a crash between file
    saves they can disagree by a turn; cut them back to the cell's minimum so twins keep sharing
    replies where their scripts agree."""
    for cell in {c["cell"] for c in convs}:
        live = [c for c in convs if c["cell"] == cell and len(c["turns"]) < c["T"] and not c["stopped_early"]]
        if live:
            m = min(len(c["turns"]) for c in live)
            for c in live:
                del c["turns"][m:]
                del c["messages"][2 * m:]



def run_conversations(lm, convs, batch_size=8, max_new_tokens=1024, seed=0, on_turn=None):
    """Lockstep batched generation, resumable. Conversations of the same cell share a batch (cell order shuffled with seed), and at
    every turn conversations of the same cell with identical histories share one sampled reply, so arms share their trunk up to the
    first turn where their scripts differ. A batch's seed depends on seed, the group's cells, the turn and the chunk; given every planned
    conversation (finished ones included), a resumed run has an uninterrupted run's groups, chunks and seeds. on_turn(list of
    conversations) is called after every turn with the conversations that took it."""
    _resync(convs)
    cells = sorted({c["cell"] for c in convs})
    random.Random(seed).shuffle(cells)
    groups, cur = [], []
    for cell in cells:
        cur += [c for c in convs if c["cell"] == cell]
        if len(cur) >= batch_size:
            groups.append(cur)
            cur = []
    groups += [cur] if cur else []
    for group in groups:
        gkey = zlib.crc32(",".join(sorted({c["cell"] for c in group})).encode()) % 1_000_003
        while True:
            todo = [c for c in group if len(c["turns"]) < c["T"] and not c["stopped_early"]]
            if not todo:
                break
            t = min(len(c["turns"]) for c in todo)
            live = [c for c in todo if len(c["turns"]) == t]
            notes = []
            for c in live:
                text, note = user_turn(c, t)
                c["messages"].append({"role": "user", "content": text})
                notes.append(note)
            keys = [json.dumps([c["cell"], c["messages"]]) for c in live]
            uniq = list(dict.fromkeys(keys))
            replies = {}
            for j in range(0, len(uniq), batch_size):
                chunk = uniq[j:j + batch_size]
                outs = lm.generate([json.loads(k)[1] for k in chunk], max_new_tokens, seed * 1_000_003 + gkey * 1009 + t * 101 + j)
                for k, o in zip(chunk, outs):
                    replies[k] = o
            for c, k, note in zip(live, keys, notes):
                o = replies[k]
                c["messages"].append({"role": "assistant", "content": o["text"]})
                c["turns"].append({"turn": t + 1, "user_type": "task" if t == 0 else c["profile"][t - 1], **note,
                                   "gen_seed": o["gen_seed"], "gen_batch": o["gen_batch"], "shared_by": keys.count(k),
                                   "reply_tokens": o["n_tokens"], "truncated": o["truncated"], "stray_turn_token": o["stray_turn_token"],
                                   "special_string": o.get("special_string", False), "prompt_tokens": o["prompt_tokens"],
                                   "claims_impossible": claims_impossible(o["text"]), "stop_rule": False})
            if on_turn is not None:
                on_turn([{**c, "messages": c["messages"][:2 * t + 2], "turns": c["turns"][:t + 1], "stopped_early": False} for c in live])
    return convs
