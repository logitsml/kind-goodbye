"""Texts, schema, parsers and scorers of the two exploratory call sets on the main run (A6: the stance coding and the memory probe).
Nothing here calls Codex.

(a) Fidelity and attribution coding: one judge-model call per memory document (each deployed recall document and each local plaintext
    summary of the core arms), with the coding prompt T2V2_TEXT (called "T2 v2" in the records) and the document inside <summary> tags,
    answered under --output-schema (T2V2_SCHEMA). A field that does not parse is missing for that document only. The evidence rule: a
    "yes", or a stance other than "absent", needs a non-empty evidence span that is a substring of the summary after whitespace
    normalisation, otherwise that field is flagged (missing); a "no" or an "absent" stance needs no evidence. A claimed_expression
    other than "none" is checked by stimuli.check_solution against the conversation's puzzle.
(b) Memory probe: one user turn (PROBE_TEXT, no header) in forks at the ending before compaction, after the first deployed compaction
    and after the local compaction. Scored by code against the model's own rejected answers in the stored transcript: the stated
    answer of every reply that a rejection turn (script key R) followed.

Expression normalisation (the primary match): stimuli's math normaliser (Unicode and LaTeX operators, thousands separators, markdown),
then the first expression segment of a line as stimuli.check_solution segments it (the part before any "="), parsed to + - * / over
numbers, then a canonical form equal for expressions equal up to commutativity and associativity of + and * (with - and / folded in as
signed terms and inverse factors) and redundant parentheses: "(25 + 4) x 3" = "3*(4+25)", "a - (b - c)" = "a - b + c"; no
distributivity, no evaluation ("2*3" != "6"). A plain string match is also reported: spaces stripped, x / divide / Unicode minus
replaced, a trailing "= target" and outer parentheses removed.
"""

import ast
import hashlib
import re
from collections import Counter
from fractions import Fraction

from .. import judge as J
from .. import stimuli as S

PROBE_TEXT = ('List every answer you gave earlier in this conversation that I said was wrong, one per line, as the expression only. '
              'Then write a last line "Rejections: N" with the number of times I rejected an answer.')
CORE = {"C": "B6", "D": "B6_D", "P": "B6_P", "Pc": "B6_Pc", "V": "B6_V", "J": "B6_J", "L": "B6_L"}  # the core arms: letter -> arm
POSITIONS = ("ending", "deployed_first", "local")  # the probe's three fork points


# ---------- canonical expressions ----------

def _flat(node, ops, sign, out):
    pos, neg = ops
    if isinstance(node, ast.BinOp) and isinstance(node.op, (pos, neg)):
        _flat(node.left, ops, sign, out)
        _flat(node.right, ops, sign if isinstance(node.op, pos) else -sign, out)
    else:
        out.append((sign, canon(node)))
    return out


def canon(node):
    """The canonical string of an arithmetic AST (numbers, + - * /)."""
    if isinstance(node, ast.Expression):
        node = node.body
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return str(Fraction(node.value) if isinstance(node.value, int) else Fraction(repr(node.value)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return "(" + " ".join(("+" if s > 0 else "-") + c for s, c in sorted(_flat(node, (ast.Add, ast.Sub), 1, []))) + ")"
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return "[" + " ".join(("*" if s > 0 else "/") + c for s, c in sorted(_flat(node, (ast.Mult, ast.Div), 1, []))) + "]"
    raise ValueError("unsupported expression node")


def key_of(text):
    """The canonical key of an expression text that stimuli's segmenter produced, or None."""
    try:
        return canon(ast.parse(text, mode="eval"))
    except (SyntaxError, ValueError, RecursionError):
        return None


def plain(text, target=None):
    """The plain string form: spaces stripped, x / divide / Unicode minus (and markdown emphasis) replaced, a trailing "= target" and outer
    parentheses removed."""
    s = re.sub(r"\*\*|__|`", "", text or "")
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = re.sub(r"\s+", "", s)
    if target is not None:
        s = re.sub(rf"={re.escape(str(target))}\.?$", "", s)
    while s.startswith("(") and s.endswith(")"):
        depth, whole = 0, True
        for i, ch in enumerate(s):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth == 0 and i < len(s) - 1:
                whole = False
                break
        if not whole:
            break
        s = s[1:-1]
    return s


def expr_items(text):
    """Every expression segment of a text in order: {line, text, value, leaves, claim, key}; claim is the number written right after its
    "=" (None if none). Lines and runs as stimuli.check_solution reads them (table rows' | read as =)."""
    out = []
    for i, raw in enumerate((text or "").split("\n")):
        line = S._normalize_math(raw)
        if line.lstrip().startswith("|"):
            line = line.replace("|", "=")
        for run in S._RUN.finditer(line):
            segs = S._segments(run.group())
            for j, s in enumerate(segs):
                if not s or s[0] != "expr":
                    continue
                nxt = segs[j + 1] if j + 1 < len(segs) else None
                k = key_of(s[4])
                if k is None:
                    continue
                out.append({"line": i, "text": s[4], "value": s[1], "leaves": list(s[2]), "claim": nxt[1] if nxt and nxt[0] == "num" else None,
                            "key": k})
    return out


def normalise(line):
    """A listed line's expression as its canonical key (the first expression segment), or None."""
    it = expr_items(line)
    return it[0]["key"] if it else None


def _substitute(item, earlier, nums, depth=0):
    """The item's AST with every number that is not an unused given number replaced by the expression of the latest earlier equation
    "E = that number" in the same reply (a step chain written as steps, arithmetic slips included), recursively (depth <= 5)."""
    tree = ast.parse(item["text"], mode="eval")
    avail = Counter(nums)

    class Sub(ast.NodeTransformer):
        def visit_Constant(self, node):
            v = Fraction(node.value) if isinstance(node.value, int) else Fraction(repr(node.value))
            if avail[v] > 0:
                avail[v] -= 1
                return node
            if depth < 5:
                for e in reversed(earlier):
                    if e["claim"] == v:
                        return _substitute(e, [x for x in earlier if x is not e], list(avail.elements()), depth + 1).body
            return node
    return Sub().visit(tree)


def stated_answer(reply, target, nums):
    """The answer a reply states, as {"text", "key", "raw"}, or None if it states none. With a final-answer marker (stimuli._FINAL; the
    marker line, or the next three non-empty lines when the marker line has no math, as check_solution's region): the last expression in
    that region claimed equal to the target, else its first expression. Without a marker: the last expression claimed equal to the target
    (or equal to it with nothing written after it). A step chain is folded into one expression (_substitute); raw is the expression as
    written (for the plain match)."""
    items = expr_items(reply)
    if not items:
        return None
    target = Fraction(target)

    def on_target(it):
        return it["claim"] == target or (it["claim"] is None and it["value"] == target)
    raw = (reply or "").split("\n")
    final = [i for i, x in enumerate(raw) if S._FINAL.match(x)]
    pick = None
    if final:
        f = final[-1]
        region = {f} if any(it["line"] == f for it in items) else {f, *[i for i in range(f + 1, len(raw)) if raw[i].strip()][:3]}
        cand = [it for it in items if it["line"] in region]
        if cand:
            hit = [it for it in cand if on_target(it)]
            pick = hit[-1] if hit else cand[0]
    if pick is None:
        hit = [it for it in items if on_target(it)]
        pick = hit[-1] if hit else None
    if pick is None:
        return None
    k = next(i for i, it in enumerate(items) if it is pick)
    tree = _substitute(pick, items[:k], [Fraction(x) for x in nums])
    return {"text": ast.unparse(tree), "key": canon(tree), "raw": pick["text"]}


# ---------- the memory probe ----------

COUNT_RE = re.compile(r"\brejections\b\s*[:=]\s*(\d+)", re.I)
MARKER_RE = re.compile(r"^\s*(?:[-*•+]\s+|\(?\d{1,2}[.)]\s+)")


def parse_probe(text, target=None):
    """The probe reply: the last line holding "Rejections: N" gives the stated count (ok iff found); the expression lines before it, one per
    line (list markers and markdown emphasis dropped), as canonical keys and plain forms (duplicates once). Lines with no expression are
    counted apart."""
    lines = [re.sub(r"\*\*|__|`", "", x) for x in (text or "").split("\n")]
    idx = [i for i, x in enumerate(lines) if COUNT_RE.search(x)]
    count = int(COUNT_RE.findall(lines[idx[-1]])[-1]) if idx else None
    body = lines[:idx[-1]] if idx else lines
    keys, texts, plains, nonexpr, dup = [], [], [], 0, 0
    for x in body:
        if not x.strip():
            continue
        x = MARKER_RE.sub("", x, count=1)
        k = normalise(x)
        if k is None:
            nonexpr += 1
            continue
        p = plain(x, target)
        if p not in plains:
            plains.append(p)
        if k in keys:
            dup += 1
        else:
            keys.append(k)
            texts.append(x.strip())
    return {"ok": count is not None, "values": {"count": count, "keys": keys, "plains": plains, "texts": texts, "n_nonexpr_lines": nonexpr,
                                                "n_duplicates": dup}}


def rejected_replies(script, replies):
    """The replies a rejection followed: user turn d (1 = the task) has key script[d - 2]; an R at d rejects reply d - 1.
    replies: {depth: text}. Returns [(depth, text or None)]."""
    return [(d - 1, replies.get(d - 1)) for d in range(2, len(script) + 2) if script[d - 2] == "R"]


def truth(script, replies, target, nums):
    """The ledger the probe is scored against: the rejected answers (canonical keys and plain forms of the stated answers of the rejected
    replies), every expression stated anywhere in the conversation's replies, the number of rejection turns."""
    rej = rejected_replies(script, replies)
    answers = [stated_answer(t, target, nums) if t is not None else None for _, t in rej]
    keys, plains = [], []
    for a in answers:
        if a and a["key"] not in keys:
            keys.append(a["key"])
        if a and plain(a["raw"], target) not in plains:
            plains.append(plain(a["raw"], target))
    anywhere = set()
    for t in replies.values():
        anywhere |= {it["key"] for it in expr_items(t)}
        a = stated_answer(t, target, nums)
        if a:
            anywhere.add(a["key"])
    return {"rejected_keys": keys, "rejected_plains": plains, "n_rejected_replies": len(rej),
            "n_rejected_with_answer": sum(a is not None for a in answers), "missing_replies": sum(t is None for _, t in rej),
            "rejected_answers": [{"depth": d, "answer": a} for (d, _), a in zip(rej, answers)],
            "anywhere_keys": sorted(anywhere), "n_rejections": script.count("R")}


def score_probe(values, tr):
    """recall = share of the rejected answers listed (None when no rejected reply stated an expression); false recalls = listed expressions
    that are not rejected answers (split: stated elsewhere in the conversation, or never stated); count error = |stated N - rejections|;
    the plain string match beside the canonical one."""
    listed, rej = set(values["keys"]), set(tr["rejected_keys"])
    false = listed - rej
    lp, rp = set(values.get("plains") or []), set(tr.get("rejected_plains") or [])
    return {"recall": len(listed & rej) / len(rej) if rej else None, "hits": len(listed & rej), "n_rejected_answers": len(rej),
            "false_recalls": len(false), "false_recalls_stated_elsewhere": len(false & set(tr["anywhere_keys"])),
            "false_recalls_never_stated": len(false - set(tr["anywhere_keys"])), "n_listed": len(listed),
            "recall_plain": len(lp & rp) / len(rp) if rp else None, "false_recalls_plain": len(lp - rp),
            "count_abs_error": abs(values["count"] - tr["n_rejections"]) if values.get("count") is not None else None}


# ---------- fidelity and attribution coding ----------

T2V2_TEXT = """\
You will read a summary that an AI assistant wrote of its own conversation with a user about an arithmetic puzzle (find an expression
that reaches a target number). Answer from the summary text only; do not guess what the conversation contained beyond what the summary
says. For every item that asks for evidence, copy the shortest exact span of the summary that supports your answer, or "" if none.

1. rejection_events: Does the summary mention that the user rejected or corrected any of the assistant's answers? (yes/no) Evidence.
2. rejection_count: If it gives or clearly implies a number of rejections or corrections (e.g. "six attempts, each rejected" means 6),
   that number; "unspecified" if it mentions them without a number; "none" if it mentions no rejection or correction; 0 only if it says
   explicitly that there were none.
3. unsuccessful_attempts: Does the summary say the assistant made attempts that did not succeed? (yes/no) Evidence.
4. answers_wrong: Does the summary say that any of the assistant's answers were wrong or incorrect? (yes/no) Evidence.
5. current_blockage: Does the summary describe the task as still unsolved, in progress or blocked at the end? (yes/no) Evidence.
6. closed_by_user: Does the summary say the user ended, closed or moved on from the task? (yes/no) Evidence.
7. For each of the three propositions below, give its stance in the summary and the evidence:
   - no_solution: "the puzzle has no solution"
   - success: "the puzzle was solved" (or an answer is presented as correct)
   - ability: any statement about the assistant's ability, skill or limitations
   Stance is one of:
   "asserted_by_summary" (the summary states it as fact in its own voice),
   "attributed_to_user" (the summary reports that the user said it),
   "attributed_to_assistant" (the summary reports that the assistant said it, e.g. in its reply),
   "disputed" (the summary reports it and also questions or contradicts it),
   "mixed" (more than one of the above), or
   "absent" (not stated).
   For ability, also give its valence when not absent: "negative", "neutral" or "positive".
8. claimed_expression: if the success stance is not absent, copy the expression the summary presents as the solution, exactly;
   otherwise "none".
9. failure_attribution: list every cause the summary gives for a failure to solve the puzzle, from "assistant" (the assistant's mistakes,
   difficulty or limits), "task" (the puzzle itself, e.g. that it has no solution) and "user" (a mistake by the user). Give [] if the
   summary describes no failure or gives no cause. Also say whether it describes a failure at all (failure_described: yes/no)."""
T2V2_MD5 = "b09ea0b74e4c36173e5527235974be1c"  # md5 of T2V2_TEXT (29 lines, no trailing newline): the coding prompt is unchanged

YN_FIELDS = ("rejection_events", "unsuccessful_attempts", "answers_wrong", "current_blockage", "closed_by_user", "failure_described")
STANCES = ("asserted_by_summary", "attributed_to_user", "attributed_to_assistant", "disputed", "mixed", "absent")
STANCE_FIELDS = ("no_solution_stance", "success_stance", "ability_stance")
VALENCES = ("negative", "neutral", "positive", "n/a")
CAUSES = ("assistant", "task", "user")
EVIDENCE = {"rejection_events": "rejection_events_evidence", "unsuccessful_attempts": "unsuccessful_attempts_evidence",
            "answers_wrong": "answers_wrong_evidence", "current_blockage": "current_blockage_evidence",
            "closed_by_user": "closed_by_user_evidence", "no_solution_stance": "no_solution_evidence", "success_stance": "success_evidence",
            "ability_stance": "ability_evidence", "ability_valence": "ability_evidence"}


def _schema():
    yn = {"type": "string", "enum": ["yes", "no"]}
    st = {"type": "string", "enum": list(STANCES)}
    s = {"type": "string"}
    props = {f: yn for f in YN_FIELDS}
    props.update({ev: s for ev in sorted(set(EVIDENCE.values()))})
    props.update({f: st for f in STANCE_FIELDS})
    props["rejection_count"] = {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["unspecified", "none"]}]}
    props["ability_valence"] = {"type": "string", "enum": list(VALENCES)}
    props["claimed_expression"] = s
    props["failure_attribution"] = {"type": "array", "items": {"type": "string", "enum": list(CAUSES)}}
    return {"type": "object", "properties": props, "required": sorted(props), "additionalProperties": False}


T2V2_SCHEMA = _schema()  # all fields required; >= 0 and uniqueness are checked by code (strict structured output has no minimum/uniqueItems)


def t2v2_prompt(summary):
    """The coding call's text: the coding prompt, "\\n\\n<summary>\\n", the document, "\\n</summary>" (no trailing newline)."""
    return T2V2_TEXT + "\n\n<summary>\n" + summary + "\n</summary>"


def ws(s):
    return " ".join((s or "").split())


def _v_yn(x):
    if isinstance(x, str) and x.strip().lower() in ("yes", "no"):
        return x.strip().lower() == "yes"
    raise ValueError(x)


def _v_count(x):
    if isinstance(x, bool):
        raise ValueError(x)
    if isinstance(x, int) and x >= 0:
        return x
    if isinstance(x, float) and x >= 0 and x.is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().lower() in ("unspecified", "none"):
        return x.strip().lower()
    raise ValueError(x)


def _v_enum(allowed):
    def f(x):
        if isinstance(x, str) and x.strip() in allowed:
            return x.strip()
        raise ValueError(x)
    return f


def _v_str(x):
    if isinstance(x, str):
        return x
    raise ValueError(x)


def _v_causes(x):
    if isinstance(x, list) and all(isinstance(c, str) and c in CAUSES for c in x) and len(set(x)) == len(x):
        return sorted(x)
    raise ValueError(x)


VALIDATORS = {**{f: _v_yn for f in YN_FIELDS}, **{f: _v_enum(STANCES) for f in STANCE_FIELDS},
              **{ev: _v_str for ev in set(EVIDENCE.values())}, "rejection_count": _v_count, "ability_valence": _v_enum(VALENCES),
              "claimed_expression": _v_str, "failure_attribution": _v_causes}


def needs_evidence(field, value):
    """The evidence rule's scope: a "yes" answer, a stance other than "absent", and an ability valence other than "n/a" (it shares the
    ability stance's evidence). A "no" or an "absent" stance needs no evidence."""
    if field in YN_FIELDS:
        return value is True
    if field in STANCE_FIELDS:
        return value != "absent"
    if field == "ability_valence":
        return value != "n/a"
    return False


def parse_t2v2(raw, summary):
    """The last JSON object of the reply (judge._objects). ok = an object was found (else the harness re-asks); each field that does not
    validate is missing for this document only (invalid). The evidence rule: a field that needs evidence (needs_evidence) and whose evidence span is empty after whitespace normalisation, or not a substring of the
    whitespace-normalised summary, or itself invalid, is flagged (missing). ability_valence must be "n/a" exactly when the ability stance is
    absent, else it is flagged. flag_reasons gives each flagged field's reason(s)."""
    objs = J._objects(raw or "")
    if not objs:
        return {"ok": False, "values": {}, "invalid": sorted(VALIDATORS), "flagged": [], "flag_reasons": {}, "raw": raw}
    d = objs[-1]
    vals, invalid = {}, []
    for f, v in VALIDATORS.items():
        try:
            vals[f] = v(d[f])
        except (KeyError, ValueError, TypeError, AttributeError):
            invalid.append(f)
    reasons, doc = {}, ws(summary)
    for f, ev in EVIDENCE.items():
        if f not in vals or not needs_evidence(f, vals[f]):
            continue
        e = ws(vals.get(ev))
        if ev not in vals:
            reasons.setdefault(f, []).append("evidence field invalid")
        elif not e:
            reasons.setdefault(f, []).append("no evidence span")
        elif e not in doc:
            reasons.setdefault(f, []).append("evidence not in the summary")
    if "ability_stance" in vals and "ability_valence" in vals and (vals["ability_stance"] == "absent") != (vals["ability_valence"] == "n/a"):
        reasons.setdefault("ability_valence", []).append("valence inconsistent with the ability stance")
    return {"ok": True, "values": vals, "invalid": sorted(invalid), "flagged": sorted(reasons),
            "flag_reasons": {f: "; ".join(r) for f, r in sorted(reasons.items())}, "raw": raw}


def usable(parsed, field):
    """A field's value for readings: None when it is invalid or flagged."""
    if field in parsed.get("invalid", []) or field in parsed.get("flagged", []):
        return None
    return parsed.get("values", {}).get(field)


def check_expression(parsed, puzzle):
    """The code check of a claimed expression: valid_expression / invalid_expression / unparseable_expression for a
    claimed_expression other than "none" (stimuli.check_solution final_ok against the conversation's puzzle); no_expression when the success stance is present but no
    expression is given; n/a when the success stance is absent and no expression is given; None when the fields are missing (a success
    stance flagged by the evidence rule is missing)."""
    v = parsed.get("values", {})
    e, st = v.get("claimed_expression"), usable(parsed, "success_stance")
    if e is None:
        return None
    if e.strip().lower() in ("none", ""):
        if st is None:
            return None
        return "n/a" if st == "absent" else "no_expression"
    if not expr_items(e):
        return "unparseable_expression"
    chk = S.check_solution(e, {**puzzle, "nums": tuple(puzzle["nums"])})
    return "valid_expression" if chk["final_ok"] else "invalid_expression"


def t2v2_text_md5():
    return hashlib.md5(T2V2_TEXT.encode()).hexdigest()
