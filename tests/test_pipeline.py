"""CPU tests of the run pipeline: generate -> compact -> perf -> reads on a tiny random Gemma 3 (tests/tiny.py), the reads' shared-prefill
path against one prefill per item, the batch log and its replay, and the Claude judge with a fake Message Batches client (no API call).

The run config is configs/endings_12b.json cut to 2 puzzles (10 conversations), with short replies. Run with
    python -m pytest tests/test_pipeline.py
"""
import datetime
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiny  # noqa: E402
from kind_goodbye import claude as EJ  # noqa: E402
from kind_goodbye import endings as EL  # noqa: E402
from kind_goodbye import reads as ER  # noqa: E402
from kind_goodbye import stages as ST  # noqa: E402
from kind_goodbye import stimuli as S  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731
NO_SLEEP = lambda s: None  # noqa: E731
ARMS = ("N", "B6", "D", "Pc", "V")
FORCED = ("B6", "D", "Pc", "V")
IDS = [f"{a}-{i:02d}" for a in ARMS for i in range(2)]


def rj(p):
    return json.loads(Path(p).read_text())


def small_config(n=2, **top):
    """configs/endings_12b.json on its first n puzzles, with 8-token replies and summaries and a 64-token chunk."""
    c = rj(ROOT / "configs" / "endings_12b.json")
    c["endings"]["n"] = n
    for a in c["conversations"]["arms"]:
        a["n"] = n
    c["endings"]["perf_puzzles"] = {k: v for k, v in c["endings"]["perf_puzzles"].items() if int(k) < n}
    c.update(chunk=64, max_new_tokens=8, compact_max_new_tokens=8, **top)
    return c


def world(tmp, cfg):
    """(runs folder, config path, run folder)."""
    runs = Path(tmp) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    cp = Path(tmp) / f"{cfg['name']}.json"
    cp.write_text(json.dumps(cfg, indent=1))
    return runs, cp, runs / cfg["name"]


def snapshot(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


# ---------- the tiny pipeline (built once) ----------

@pytest.fixture(scope="module")
def lm():
    return tiny.lm(chunk=64)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, lm):
    cfg = small_config()
    runs, cp, d = world(tmp_path_factory.mktemp("pipeline"), cfg)
    rcs = {"generate": ST.stage_generate(cp, runs, lm=lm, prog=QUIET), "compact": ST.stage_compact(cp, runs, lm=lm, prog=QUIET),
           "perf": ST.stage_perf(cp, runs, lm=lm, prog=QUIET)}
    rcs["reads"] = ER.run_reads(cfg, runs, lm=lm, prog=QUIET)
    return types.SimpleNamespace(cfg=cfg, runs=runs, cp=cp, d=d, rcs=rcs)


def test_stages_exit_zero(pipeline):
    assert pipeline.rcs == {"generate": 0, "compact": 0, "perf": 0, "reads": 0}


def test_conversations(pipeline):
    d = pipeline.d
    assert sorted(p.stem for p in (d / "convs").glob("*.json")) == sorted(IDS)
    assert rj(d / "config.json") == pipeline.cfg
    for cid in IDS:
        c = rj(d / "convs" / f"{cid}.json")
        assert c["T"] == 11 and len(c["turns"]) == 11 and len(c["messages"]) == 22 and not c["stopped_early"]
        assert [m["role"] for m in c["messages"]] == ["user", "assistant"] * 11
        assert c["messages"][0]["content"].startswith(c["tasks"][0]["text"])
        last = {"N": "C", "B6": "C", "D": "D", "Pc": "X", "V": "V"}[c["arm"]]
        assert c["profile"][-1] == last and c["turns"][-1]["user_type"] == last
        if last in S.ENDINGS:
            assert c["messages"][-2]["content"] == S.ENDINGS[last]
        assert set(c["turns"][0]) == {"turn", "user_type", "gen_seed", "gen_batch", "shared_by", "reply_tokens", "truncated", "stray_turn_token",
                                      "special_string", "prompt_tokens", "claims_impossible", "stop_rule"}
    # the arms of one puzzle share their replies while their scripts agree (N and B6 differ from the user's fourth turn on)
    n, b6 = rj(d / "convs" / "N-00.json"), rj(d / "convs" / "B6-00.json")
    assert n["messages"][:6] == b6["messages"][:6] and n["messages"][6] != b6["messages"][6] and n["turns"][2]["shared_by"] == 5
    sched = rj(d / "schedule" / "generate.json")
    assert sched["seed"] == 0 and sorted(sum(sched["groups"], [])) == sorted(IDS)


def test_batch_log(pipeline):
    recs = EL.read_records(pipeline.d / "batches.jsonl")
    starts = [r for r in recs if r["event"] == "start"]
    batches = [r for r in recs if r["event"] == "batch"]
    assert [r["stage"] for r in starts] == ["generate", "compact", "perf"] and all(r["replay"] is False for r in starts)  # vLLM generation
    assert batches and all({"stage", "key", "seed", "n", "keys", "partition", "tokens", "seconds", "ooms"} <= set(r) for r in batches)
    assert all(sum(len(rows) for rows, _ in r["partition"]) == r["n"] for r in batches)


def test_summaries(pipeline):
    d = pipeline.d
    order = rj(d / "schedule" / "compact.json")
    assert order["seed"] == EL.SCHEDULE_SEED and sorted(order["jobs"]) == sorted(f"{c}/{k}" for c in IDS for k in ("s0", "s1", "s2", "abl"))
    for cid in IDS:
        h = rj(d / "summaries" / f"{cid}.json")
        assert set(h) == {"id", "arm", "prompt_index", "model", "mode", "s0", "s1", "s2", "abl"}
        assert h["model"] == "google/gemma-3-12b-it" and h["mode"] == "vllm_bf16"
        for k in ("s0", "s1", "s2", "abl"):
            s = h[k]
            assert set(s) == {"summary", "tokens", "truncated", "seed", "gen_batch", "batch_seed", "batch", "row", "input_sha1", "n_messages",
                              "last_role", "final_reply_removed"}
            assert s["batch"][s["row"]] == f"{cid}/{k}" and s["final_reply_removed"] == (k == "abl")
            assert (s["n_messages"], s["last_role"]) == ((21, "user") if k == "abl" else (22, "assistant"))
            assert (s["batch_seed"] - ST.COMPACT_SEED_OFFSET) == order["jobs"].index(f"{cid}/{k}") // 8
            c = rj(d / "convs" / f"{cid}.json")
            assert s["input_sha1"] == EL.sha1(EL.compaction_input(c, k))


def test_perf(pipeline):
    d = pipeline.d
    for cid in IDS:
        r = rj(d / "perf" / f"{cid}.json")
        assert set(r) == {"id", "puzzle", "forced_answer", "kinds"} and set(r["kinds"]) == {"s0"}
        h = r["kinds"]["s0"]
        arm = cid.split("-")[0]
        assert set(h) == ({"own", "forced"} if arm in FORCED else {"own"})
        recs = [h["own"]] + (h["forced"] if arm in FORCED else [])
        assert arm not in FORCED or len(h["forced"]) == 5
        for x in recs:
            assert set(x) == {"text", "tokens", "truncated", "seed", "gen_batch", "batch_seed", "batch", "row", "check", "correct",
                              "claims_impossible", "input_sha1"}
        assert r["forced_answer"] == EL.forced_answer(r["puzzle"])
    assert not (d / "schedule" / "perf2.json").exists()  # no natural challenge in the released configs


def test_reads_layout(pipeline):
    rd = pipeline.d / "reads"
    keys = sorted(f"{a}~{p:03d}" for a in ARMS for p in range(2))
    assert {k: sorted(p.stem for p in (rd / k).glob("*.json")) for k in ER.KINDS} == {
        "walk": keys, "ser": keys, "sum": keys, "perf": sorted(f"{a}~{p:03d}" for a in FORCED for p in range(2))}
    for key in keys:
        w = rj(rd / "walk" / f"{key}.json")
        assert set(w) == {"session_last", "T", "turns", "checks", "choice", "end"} and w["session_last"] == w["T"] == 11
        assert [t["turn"] for t in w["turns"]] == ([10, 11] if key.split("~")[0] in ("N", "B6") else [11])
        for t in w["turns"]:
            assert set(t) == {"turn", "stop", "state"} and set(t["state"]) == set(S.STATE_ITEMS)
            assert set(t["stop"]) == {"lp_a", "lp_b", "p", "logit", "mass"}
        assert set(w["checks"]) == set(EL.CHECKS) and set(w["choice"]) == {"again", "skip"} and set(w["end"]) == set(S.LOOKBACK_ITEMS)
        assert set(w["end"]["happy"]) == {"expected", "probs", "logp", "mass", "argmax"}
        ser = rj(rd / "ser" / f"{key}.json")
        assert set(ser) == {"session", "n_messages"} and ser["n_messages"] == 22 and set(ser["session"]) == set(S.LOOKBACK_ITEMS)
        sm = rj(rd / "sum" / f"{key}.json")
        assert set(sm) == {"s0", "s1", "s2", "abl", "summary_sha1"}
    p = rj(rd / "perf" / "D~000.json")
    assert set(p) == {"s0"} and p["s0"]["start"] is None and set(p["s0"]["forced_q"]) == {"lp_a", "lp_b", "p", "logit", "mass"}
    idn = rj(rd / "identity.json")
    assert idn["ok"] and idn["n"] == 0 and idn["against"] == "earlier reads"  # the first process has nothing saved to compare
    sc = [json.loads(x) for x in (rd / "session_checks.jsonl").read_text().splitlines()]
    assert len(sc) == 1 and sc[0]["ok"] and sc[0]["path"] == "reuse" and sc[0]["made"] == 38
    assert [(c["kind"], c["identical"]) for c in sc[0]["checks"]] == [(k, True) for k in ER.KINDS for _ in range(2)]
    assert len((rd / "reads_log.jsonl").read_text().splitlines()) == 38


def test_walk_equals_direct_reads(pipeline, lm):
    """The walk's values equal a fresh prefill of the whole fork (to float rounding: the chunks are cut differently)."""
    c = rj(pipeline.d / "convs" / "D-00.json")
    w = rj(pipeline.d / "reads" / "walk" / "D~000.json")
    m = c["messages"][:22]
    yn = ER.score(lm, "yn", lm.prefill(ER.fork_ids(lm, m, "yn", EL.CHECKS["wants"]))[0])
    d7 = ER.score(lm, "d7", lm.prefill(ER.fork_ids(lm, m, "d7", S.STATE_ITEMS["happy"]))[0])
    assert w["checks"]["wants"]["p"] == pytest.approx(yn["p"], abs=1e-5)
    assert w["turns"][0]["state"]["happy"]["expected"] == pytest.approx(d7["expected"], abs=1e-5)


def test_rerun_is_a_no_op(pipeline, lm):
    before = snapshot(pipeline.d)
    rcs = [ST.stage_generate(pipeline.cp, pipeline.runs, lm=lm, prog=QUIET), ST.stage_compact(pipeline.cp, pipeline.runs, lm=lm, prog=QUIET),
           ST.stage_perf(pipeline.cp, pipeline.runs, lm=lm, prog=QUIET), ER.run_reads(pipeline.cfg, pipeline.runs, lm=lm, prog=QUIET)]
    assert rcs == [0, 0, 0, 0] and snapshot(pipeline.d) == before


def test_changed_config_refused(pipeline, tmp_path):
    cfg = dict(pipeline.cfg, max_new_tokens=9)
    cp = tmp_path / "changed.json"
    cp.write_text(json.dumps(cfg))
    assert ST.stage_compact(cp, pipeline.runs, prog=QUIET) == 2 and ER.run_reads(cfg, pipeline.runs, prog=QUIET) == 2


def test_status(pipeline):
    out = []
    assert ST.stage_status(pipeline.cp, pipeline.runs, prog=out.append) == 0
    assert out == ["endings_12b: 10 conversations planned; 10 started, 10 finished; 10 with summaries; 10 with performance; "
                   "reads {'walk': 10, 'ser': 10, 'sum': 10, 'perf': 8}; judge table: False"]


# ---------- the reads' self-checks ----------

def copy_run(pipeline, tmp, keep_reads=True):
    runs = Path(tmp) / "runs"
    shutil.copytree(pipeline.d, runs / pipeline.cfg["name"], ignore=None if keep_reads else shutil.ignore_patterns("reads"))
    return runs, runs / pipeline.cfg["name"]


def test_shared_prefill_equals_reference(pipeline, lm, tmp_path):
    """Every read made with one prefill per item (--reference) equals the shared-prefill read, exactly."""
    runs, d = copy_run(pipeline, tmp_path, keep_reads=False)
    assert ER.run_reads(pipeline.cfg, runs, reference=True, lm=lm, prog=QUIET) == 0
    ref, reuse = d / "reads", pipeline.d / "reads"
    for kind in ER.KINDS:
        files = sorted(p.name for p in (reuse / kind).glob("*.json"))
        assert files == sorted(p.name for p in (ref / kind).glob("*.json"))
        for f in files:
            assert rj(ref / kind / f) == rj(reuse / kind / f), (kind, f)
    sc = json.loads((ref / "session_checks.jsonl").read_text().splitlines()[-1])
    assert sc["path"] == "reference" and sc["checks"] == []


def test_identity_against_earlier_reads(pipeline, lm, tmp_path):
    runs, d = copy_run(pipeline, tmp_path)
    rd = d / "reads"
    (rd / "sum" / "D~001.json").unlink()
    assert ER.run_reads(pipeline.cfg, runs, lm=lm, prog=QUIET) == 0
    idn = rj(rd / "identity.json")
    assert idn["ok"] and idn["n"] == 10 and idn["max_abs_expected"] == 0.0
    assert rj(rd / "sum" / "D~001.json") == rj(pipeline.d / "reads" / "sum" / "D~001.json")
    # a saved serialized look-back off by 5e-4 stops the next process before it reads anything
    sp = rd / "ser" / "V~000.json"
    x = rj(sp)
    x["session"]["happy"]["expected"] += 5e-4
    sp.write_text(json.dumps(x))
    (rd / "sum" / "D~001.json").unlink()
    assert ER.run_reads(pipeline.cfg, runs, lm=lm, prog=QUIET) == 3
    assert not (rd / "sum" / "D~001.json").exists() and not rj(rd / "identity.json")["ok"]


def test_self_check_mismatch_stops(pipeline, lm, tmp_path, monkeypatch):
    """A shared-prefill read that differed from its one-prefill read would stop the reads (exit 3) without saving it."""
    runs, d = copy_run(pipeline, tmp_path, keep_reads=False)
    real = ER.read_one

    def skewed(lm_, d_, kind, c, ks, reuse, memo=None):
        out = real(lm_, d_, kind, c, ks, reuse, memo)
        if reuse and kind == "walk":
            out["end"]["happy"]["expected"] += 1e-9
        return out
    monkeypatch.setattr(ER, "read_one", skewed)
    assert ER.run_reads(pipeline.cfg, runs, lm=lm, prog=QUIET) == 3
    assert not list((d / "reads" / "walk").glob("*.json"))
    sc = json.loads((d / "reads" / "session_checks.jsonl").read_text())
    assert not sc["ok"] and "differs" in sc["mismatch"]


def test_dry_run(pipeline):
    out = []
    assert ER.dry_run(pipeline.cfg, pipeline.runs, prog=out.append) == 0 and "every walk plan holds" in out[0]


# ---------- generation on transformers: the batch log replays ----------

def test_transformers_generation_replays(lm, tmp_path):
    """With generation on transformers (gen_mode bf16), a resumed stage replays the logged batches and must regenerate every saved row
    exactly; a saved row that differs from its regeneration stops the stage (exit 3)."""
    cfg = small_config(n=1, gen_mode="bf16")
    runs, cp, d = world(tmp_path, cfg)
    assert ST.stage_generate(cp, runs, lm=lm, prog=QUIET) == 0
    assert ST.stage_compact(cp, runs, lm=lm, prog=QUIET) == 0
    p = d / "summaries" / "D-00.json"
    h = rj(p)
    gone = h.pop("s1")
    p.write_text(json.dumps(h))
    assert ST.stage_compact(cp, runs, lm=lm, prog=QUIET) == 0
    assert rj(p)["s1"] == gone
    recs = EL.read_records(d / "batches.jsonl")
    assert [r["event"] for r in recs if r["event"] in ("start", "replay")][-2:] == ["start", "replay"]
    assert all(r["replay"] for r in recs if r["event"] == "start")
    # a saved row of the same batch that differs from its regeneration
    other_cid, other_k = next(j for j in gone["batch"] if j != "D-00/s1").split("/")
    q = d / "summaries" / f"{other_cid}.json"
    x = rj(q)
    x[other_k]["summary"] += " (edited)"
    q.write_text(json.dumps(x))
    h = rj(p)
    h.pop("s1")
    p.write_text(json.dumps(h))
    assert ST.stage_compact(cp, runs, lm=lm, prog=QUIET) == 3 and "s1" not in rj(p)


class FakeLM:
    """Replies from the prompt's hash; a summary request gets a short summary, a challenge a short reply, a new puzzle its verified answer
    for an even hash."""

    def __init__(self, cfg, split=None):
        self.puzzles = list(cfg["endings"]["perf_puzzles"].values())
        self.split = split

    def reply(self, m, seed, i):
        h = int(hashlib.sha1(json.dumps(m).encode()).hexdigest()[:12], 16)
        last = m[-1]["content"]
        if m[0]["role"] == "system":
            return f"## Goal\nReach the target {{exactly}}.\n## Progress\n### Blocked\n- sample {seed}/{i} of {h % 10 ** 6}"
        if last == EL.CHALLENGE:
            return f"Yes, I am sure ({seed}/{i}/{h % 1000})."
        p = next((p for p in self.puzzles if EL.perf_task(p) in last), None)
        if p is not None:
            return EL.forced_answer(p) if h % 2 == 0 else f"I could not find it ({h % 97})."
        a, b = h % 89 + 2, h % 83 + 3
        return f"Try {h % 997}:\nAnswer: {a} + {b} = {a + b}"

    def generate(self, msgs_list, max_new_tokens, seed, gen_kw=None):
        return self._gen(msgs_list, seed)

    def _gen(self, msgs, seed):
        if self.split and len(msgs) > self.split:  # model.LM's out-of-memory split, simulated
            h = len(msgs) // 2
            return self._gen(msgs[:h], seed * 2 + 1) + self._gen(msgs[h:], seed * 2 + 2)
        return [{"text": self.reply(m, seed, i), "gen_seed": seed, "gen_batch": len(msgs), "n_tokens": 5, "truncated": False,
                 "stray_turn_token": False, "prompt_tokens": 100} for i, m in enumerate(msgs)]


def test_replay_lm_split(tmp_path):
    """ReplayLM logs a call's effective partition (an out-of-memory split tree) and replays it sub-batch by sub-batch on a resume; a
    sub-batch that would split differently is refused."""
    cfg = small_config()
    msgs = [[{"role": "user", "content": f"question {i}"}] for i in range(8)]
    a = EL.ReplayLM(FakeLM(cfg, split=4), tmp_path, "compact")
    out1 = a.generate(msgs, 8, 7)
    rec = [r for r in EL.read_records(tmp_path / "batches.jsonl") if r["event"] == "batch"][0]
    assert rec["partition"] == [[[0, 1, 2, 3], 15], [[4, 5, 6, 7], 16]]
    b = EL.ReplayLM(FakeLM(cfg), tmp_path, "compact")  # no split now: replayed with the recorded seeds
    assert b.generate(msgs, 8, 7) == out1
    assert EL.read_records(tmp_path / "batches.jsonl")[-1]["event"] == "replay"
    c = EL.ReplayLM(FakeLM(cfg, split=2), tmp_path, "compact")
    with pytest.raises(EL.ReplayError):
        c.generate(msgs, 8, 7)
    fast = EL.ReplayLM(FakeLM(cfg), tmp_path, "compact", replay=False)
    fast.generate(msgs, 8, 7)
    assert EL.read_records(tmp_path / "batches.jsonl")[-1]["event"] == "batch"


# ---------- the Claude judge with a fake batches client ----------

def t2_answer(prompt, h):
    doc = prompt.split("<summary>\n", 1)[1].rsplit("\n</summary>", 1)[0] if "<summary>\n" in prompt else ""
    ev = doc[:12]
    yes = lambda k: ("yes", "no")[(h >> k) & 1]  # noqa: E731
    span = lambda v: "" if v in ("no", "absent") else ev  # noqa: E731
    return json.dumps({"rejection_events": yes(0), "rejection_events_evidence": ev, "rejection_count": h % 7, "unsuccessful_attempts": yes(1),
                       "unsuccessful_attempts_evidence": span(yes(1)), "answers_wrong": yes(2), "answers_wrong_evidence": ev,
                       "current_blockage": yes(3), "current_blockage_evidence": span(yes(3)), "closed_by_user": yes(4),
                       "closed_by_user_evidence": span(yes(4)), "no_solution_stance": EL.STANCES[h % 6],
                       "no_solution_evidence": span(EL.STANCES[h % 6]), "success_stance": "absent", "success_evidence": "",
                       "ability_stance": "absent", "ability_valence": "n/a", "ability_evidence": "", "claimed_expression": "none",
                       "failure_attribution": [] if h % 2 else ["task"], "failure_described": yes(5)})


def claude_answer(kind, prompt, h):
    if kind == "t2":
        return t2_answer(prompt, h)
    return json.dumps({"evidence": "x {y z", "verdict": ("keeps", "Abandons", "hedges")[h % 3]})


class FakeClaude:
    """client.messages.batches.{create, retrieve, results, list} and client.with_options, with no API call. fate(custom_id, attempt) ->
    "succeeded" | "errored" | "invalid" | "expired" | "refusal" | "garbage"; cut {custom_id: the max_tokens a reply needs} (below it the
    reply is cut short with stop_reason "max_tokens"); served: the model the results report. Results come back in reverse order."""

    def __init__(self, fate=None, polls=1, cut=None, served=EL.CLAUDE_MODEL):
        self.fate, self.polls, self.cut, self.served = fate or (lambda cid, n: "succeeded"), polls, cut or {}, served
        self.batches, self.tries, self.options = {}, {}, []
        self.messages = types.SimpleNamespace(batches=types.SimpleNamespace(create=self.create, retrieve=self.retrieve, results=self.results,
                                                                            list=self.list))

    def with_options(self, **k):
        self.options.append(k)
        return self

    def create(self, requests):
        bid = f"msgbatch_{len(self.batches):04d}"
        self.batches[bid] = {"requests": [json.loads(json.dumps(r)) for r in requests], "polls": 0,
                             "created_at": datetime.datetime.now(datetime.timezone.utc)}
        return types.SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        b = self.batches[bid]
        b["polls"] += 1
        return types.SimpleNamespace(id=bid, processing_status="ended" if b["polls"] >= self.polls else "in_progress")

    def list(self, limit=20):
        out = []
        for bid, b in sorted(self.batches.items(), key=lambda kv: kv[1]["created_at"], reverse=True)[:limit]:
            out.append(types.SimpleNamespace(id=bid, created_at=b["created_at"], processing_status="in_progress",
                                             request_counts=types.SimpleNamespace(processing=len(b["requests"]), succeeded=0, errored=0,
                                                                                  canceled=0, expired=0)))
        return iter(out)

    def results(self, bid):
        if "results" in self.batches[bid]:
            return iter(list(self.batches[bid]["results"]))
        out = []
        for r in self.batches[bid]["requests"]:
            cid = r["custom_id"]
            n = self.tries[cid] = self.tries.get(cid, 0) + 1
            f = self.fate(cid, n)
            prompt = r["params"]["messages"][0]["content"]
            h = int(hashlib.sha1(prompt.encode()).hexdigest()[:8], 16)
            usage = types.SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=300)
            if f in ("errored", "invalid"):
                res = types.SimpleNamespace(type="errored", error=types.SimpleNamespace(type="error", error=types.SimpleNamespace(
                    type="invalid_request_error" if f == "invalid" else "api_error", message="x")))
            elif f == "expired":
                res = types.SimpleNamespace(type="expired")
            else:
                text = "I can't help with that." if f == "refusal" else "not json" if f == "garbage" else claude_answer(cid.split("-")[0], prompt, h)
                stop = "refusal" if f == "refusal" else "end_turn"
                if r["params"]["max_tokens"] < self.cut.get(cid, 0):
                    text, stop = text[:len(text) // 2], "max_tokens"
                msg = types.SimpleNamespace(model=self.served, stop_reason=stop, usage=usage,
                                            content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)])
                res = types.SimpleNamespace(type="succeeded", message=msg)
            out.append(types.SimpleNamespace(custom_id=cid, result=res))
        self.batches[bid]["results"] = list(reversed(out))
        return iter(list(self.batches[bid]["results"]))


@pytest.fixture(scope="module")
def judged_source(tmp_path_factory):
    """A run through generate, compact and perf with FakeLM (readable summaries), ready for the judge."""
    cfg = small_config()
    runs, cp, d = world(tmp_path_factory.mktemp("judged"), cfg)
    for stage in (ST.stage_generate, ST.stage_compact, ST.stage_perf):
        assert stage(cp, runs, lm=FakeLM(cfg), prog=QUIET) == 0
    return cfg, d


@pytest.fixture
def judged(judged_source, tmp_path):
    cfg, src = judged_source
    runs, cp, d = world(tmp_path, cfg)
    shutil.copytree(src, d)
    return types.SimpleNamespace(cfg=cfg, runs=runs, cp=cp, d=d,
                                 judge=lambda client, **k: ST.stage_judge(cp, runs, prog=QUIET, claude_client=client, claude_sleep=NO_SLEEP, **k))


def custom_id(cfg, tmp, kind, text):
    cj = EJ.ClaudeJudge(cfg["judge"], Path(tmp) / "probe", client=object())
    return cj.custom_id(kind, cj.key(kind, text))


def test_claude_judge_table(judged):
    fc = FakeClaude(polls=2)
    assert judged.judge(fc) == 0
    t = rj(judged.d / "judge" / "table.json")
    assert set(t) == {"backend", "primary", "primary_provider", "t2", "client", "t2_client", "effort", "missing", "schema_md5", "served", "caps",
                      "calls", "failed", "unparsed", "deferred", "n_summaries", "n_unique", "n_challenge", "prompt_md5", "scores", "perf"}
    assert (t["backend"], t["primary"], t["effort"], t["served"], t["caps"]) == ("anthropic", "claude-sonnet-5-5", "medium", ["claude-sonnet-5-5"],
                                                                                [8000, 16000, 32000])
    assert t["n_summaries"] == 40 and t["n_challenge"] == 40 and t["failed"] == 0 and t["missing"] == [] and t["deferred"] == 0
    assert sorted(t["scores"]) == sorted(IDS) and all(set(v) == {"s0", "s1", "s2", "abl"} for v in t["scores"].values())
    row = t["scores"]["D-00"]["s0"]
    assert set(row) == {"t2", "t2_checks", "summary_sha1"} and row["t2"]["no_solution_stance"] in EL.STANCES
    assert sorted(t["perf"]) == sorted(f"{a}-{i:02d}" for a in FORCED for i in range(2))
    assert all(len(v["s0"]["forced"]) == 5 and set(v["s0"]["forced"]) <= set(EL.VERDICTS) for v in t["perf"].values())
    # the requests: one per document and call type, the prompt as the one user message, no system prompt or sampling parameter
    first = fc.batches["msgbatch_0000"]["requests"]
    texts, items, _ = ST.judge_inputs(judged.cfg, judged.d)
    assert {(r["custom_id"].split("-")[0], r["params"]["messages"][0]["content"]) for r in first} == (
        {("t2", EL.t2_prompt(x)) for x in texts.values()} | {("challenge", EL.challenge_prompt(x)) for x in items.values()})
    assert all(set(r["params"]) == {"model", "max_tokens", "messages", "thinking", "output_config"} and r["params"]["max_tokens"] == 8000
               and r["params"]["thinking"] == {"type": "adaptive"} and r["params"]["output_config"]["effort"] == "medium"
               and r["params"]["output_config"]["format"] == {"type": "json_schema", "schema": EJ.schema_of(r["custom_id"].split("-")[0])}
               for r in first)
    assert fc.options and all(o == {"max_retries": 0} for o in fc.options)
    # a rerun sends nothing and writes the same table
    assert judged.judge(fc) == 0 and len(fc.batches) == 1 and rj(judged.d / "judge" / "table.json") == t


def test_claude_judge_failures(judged, tmp_path):
    cfg, d = judged.cfg, judged.d
    sm = lambda cid, k: rj(d / "summaries" / f"{cid}.json")[k]["summary"]  # noqa: E731
    refuse, err, exp, garb, cut1, cut2 = (custom_id(cfg, tmp_path, "t2", sm(c, k)) for c, k in (
        ("V-01", "s0"), ("D-00", "s1"), ("V-00", "s2"), ("Pc-01", "s0"), ("N-00", "s1"), ("N-01", "s2")))

    def fate(c, n):
        return {refuse: "refusal", garb: "garbage", err: "errored" if n <= 2 else "succeeded",
                exp: "expired" if n <= 4 else "succeeded"}.get(c, "succeeded")
    fc = FakeClaude(fate, cut={cut1: 16000, cut2: 64000})
    assert judged.judge(fc) == 5  # the expired request spent its 1 + 3 attempts: left for a rerun
    t = rj(d / "judge" / "table.json")
    assert sorted(m["why"] for m in t["missing"]) == ["refusal", "unparseable", "unparseable (cut off at max_tokens 32000)"]
    assert t["failed"] == 1 and t["scores"]["V-00"]["s2"]["t2"] is None and t["scores"]["D-00"]["s1"]["t2"] is not None
    att = EL.read_records(d / "judge" / "claude_attempts.jsonl")
    assert {c: sum(a["custom_id"] == c for a in att) for c in (refuse, err, exp, garb, cut1, cut2)} == {
        refuse: 1, err: 3, exp: 4, garb: 1, cut1: 2, cut2: 3}
    costs = EL.read_records(d / "judge" / "costs.jsonl")
    assert len(costs) == sum(a["type"] == "succeeded" for a in att)
    n = len(fc.batches)
    assert judged.judge(fc) == 0 and len(fc.batches) == n + 1 and len(fc.batches[f"msgbatch_{n:04d}"]["requests"]) == 1
    assert rj(d / "judge" / "table.json")["scores"]["V-00"]["s2"]["t2"] is not None


def test_claude_judge_exits(judged_source, tmp_path):
    cfg, src = judged_source

    def fresh(name):
        runs, cp, d = world(tmp_path / name, cfg)
        shutil.copytree(src, d)
        return lambda client: ST.stage_judge(cp, runs, prog=QUIET, claude_client=client, claude_sleep=NO_SLEEP), d
    run, d = fresh("served")
    assert run(FakeClaude(served="claude-sonnet-5-0")) == 3
    assert not any(EL.read_records(p) for p in (d / "judge").glob("cache_claude_*.jsonl"))
    run, d = fresh("invalid")
    bad = custom_id(cfg, tmp_path, "t2", rj(d / "summaries" / "D-01.json")["s0"]["summary"])
    assert run(FakeClaude(lambda c, n: "invalid" if c == bad else "succeeded")) == 2
    fx = FakeClaude()
    fx.messages.batches.create = lambda requests: (_ for _ in ()).throw(type("AuthenticationError", (Exception,), {})("no credential"))
    run, d = fresh("auth")
    assert run(fx) == 2
    fy = FakeClaude()
    fy.messages.batches.create = lambda requests: (_ for _ in ()).throw(type("APIConnectionError", (Exception,), {})("network"))
    assert run(fy) == 5


def test_claude_judge_resumes_batches(judged_source, tmp_path, monkeypatch):
    """A stage stopped while its batch ran collects that batch next time; a create whose response was lost is matched to its batch
    (batches.list), never sent twice."""
    cfg, src = judged_source
    runs, cp, d = world(tmp_path / "pending", cfg)
    shutil.copytree(src, d)
    fc = FakeClaude()
    real_wait = EJ.ClaudeJudge._wait
    monkeypatch.setattr(EJ.ClaudeJudge, "_wait", lambda self, bid: (_ for _ in ()).throw(ConnectionError("stopped")))
    assert ST.stage_judge(cp, runs, prog=QUIET, claude_client=fc, claude_sleep=NO_SLEEP) == 5
    monkeypatch.setattr(EJ.ClaudeJudge, "_wait", real_wait)
    assert ST.stage_judge(cp, runs, prog=QUIET, claude_client=fc, claude_sleep=NO_SLEEP) == 0 and len(fc.batches) == 1

    runs, cp, d = world(tmp_path / "lost", cfg)
    shutil.copytree(src, d)
    fl = FakeClaude()
    real_create = EJ.ClaudeJudge._create

    def lost(self, reqs):
        real_create(self, reqs)
        raise TimeoutError("the response was lost")
    monkeypatch.setattr(EJ.ClaudeJudge, "_create", lost)
    assert ST.stage_judge(cp, runs, prog=QUIET, claude_client=fl, claude_sleep=NO_SLEEP) == 5
    monkeypatch.setattr(EJ.ClaudeJudge, "_create", real_create)
    assert ST.stage_judge(cp, runs, prog=QUIET, claude_client=fl, claude_sleep=NO_SLEEP) == 0 and len(fl.batches) == 1
    assert sum(bool(r.get("adopted")) for r in EL.read_records(d / "judge" / "claude_batches.jsonl")) == 1


def test_claude_parsing():
    """Braces inside JSON strings and enum capitalization do not change a parsed answer."""
    ans = json.dumps({"evidence": "it said {", "verdict": "Abandons"})
    assert EJ.brace_safe(ans) != ans and json.loads(EJ.brace_safe(ans)) == json.loads(ans)
    obj = EJ.canon(json.loads(ans), EJ.CHALLENGE_SCHEMA)
    assert obj["verdict"] == "abandons"
    assert EJ.parse_of("challenge")(EJ.brace_safe(json.dumps(obj)))["verdict"] == "abandons"
    assert {k: EJ.check_schema(EJ.schema_of(k)) for k in EJ.KINDS} == {"t2": [], "challenge": []}
    assert len(EJ.check_schema({"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}}})) >= 2


def test_judge_refuses_other_calls(judged):
    cfg = json.loads(json.dumps(judged.cfg))
    cfg["name"] = "endings_12b_rubric"
    cfg["judge"]["calls"]["rubric"] = ["s0"]
    runs, cp, _ = world(judged.d.parent.parent / "rubric", cfg)
    assert ST.stage_judge(cp, runs, prog=QUIET, claude_client=FakeClaude()) == 2


# ---------- the command lines ----------

def test_cli(pipeline, capsys):
    for main in (ST.main, ER.main):
        with pytest.raises(SystemExit) as e:
            main(["--help"])
        assert e.value.code == 0
    assert "generate,compact,perf,judge,status" in capsys.readouterr().out
    assert ST.main([str(pipeline.cp), "status", "--runs", str(pipeline.runs)]) == 0
    assert ST.main(["no-such-config.json", "status"]) == 2
    assert ER.main([str(pipeline.cp), "--runs", str(pipeline.runs), "--dry-run"]) == 0
