"""The Claude judge: Claude Sonnet 5.5 (claude-sonnet-5-5) through the Anthropic Message Batches API.

The calls. One request per document and call type, with no system prompt: the user message is the call type's prompt (T2,
endings.t2_prompt; the challenge verdict, endings.challenge_prompt); adaptive thinking with output_config.effort from the config ("medium");
structured output (output_config.format, type json_schema) with the call type's schema (T2: schemas/t2v2.json, byte for byte; the challenge
verdict: CHALLENGE_SCHEMA below); no sampling parameter and no fallback model. The schemas use only the documented structured-output
features (check_schema: anyOf, enum, const, required, additionalProperties false; no numeric, string-length or array constraints beyond
minItems 0 or 1; at most 24 optional and 16 union-typed parameters), so ranges, counts and uniqueness are checked by the parsers in code.
Enum values are compared case-insensitively (structured outputs do not guarantee an enum's capitalization) and mapped to the schema's
spelling before parsing, and every brace inside a JSON string is escaped before parsing (brace_safe).

The batch flow (ClaudeJudge.run). Every uncached (call type, document) pair becomes a request whose custom_id is the call type and a hash of
the model, effort, base max_tokens, schema and prompt. Requests go in batches of at most judge.batch_max. An intent record is written to
judge/claude_batches.jsonl before batches.create (sent with the SDK's own retries off) and a submit record (the batch id) after it, so an
interrupted stage collects its batch instead of submitting it again; an intent without a submit record is matched against the account's
recent batches (batches.list) before anything is sent again. Each batch is polled (batches.retrieve) until it has ended, then its results
(batches.results) are read by custom_id, never by position. Every result must come from the configured model, or nothing of the batch is
kept (exit 3). A reply cut off at max_tokens is sent again with the next cap (judge.max_tokens_retry: 16,000, then 32,000); cut off at the
last cap, it is stored as missing (unparseable), as is a refusal (stop_reason "refusal") or an output that does not parse. An
invalid_request_error stops the stage (exit 2). Any other error (errored, expired, canceled, or absent from the results) is sent again at
most judge.retries times in a run and then left unstored, so a rerun asks again (exit 5). Every attempt is logged
(judge/claude_attempts.jsonl), every request's parameters (judge/claude_requests.jsonl), and usage once per (custom_id, batch)
(judge/costs.jsonl). The requests hold no credential: the SDK client finds its own, and nothing here reads, prints or logs one.

The client is the zero-argument anthropic.Anthropic(); the tests pass a fake batches client and make no API call."""
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from . import endings as EL

POLL_LIMIT_S = 26 * 3600  # a batch ends within 24 hours (unfinished requests then expire); polling longer than this is an error

CHALLENGE_SCHEMA = {  # the challenge prompt's JSON form (endings.challenge_prompt)
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "verdict": {"type": "string", "enum": ["keeps", "abandons", "hedges"]}},
    "required": ["evidence", "verdict"],
    "additionalProperties": False,
}
KINDS = ("t2", "challenge")


def schema_of(kind):
    return {"t2": EL.t2_schema(), "challenge": CHALLENGE_SCHEMA}[kind]


def prompt_of(kind):
    return {"t2": EL.t2_prompt, "challenge": EL.challenge_prompt}[kind]


def parse_of(kind):
    return {"t2": EL.t2_parse, "challenge": EL.challenge_parse}[kind]


JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')  # a JSON string literal (json.dumps writes no quote outside one)


def brace_safe(text):
    """JSON text with every brace inside a string literal written as \\u007b or \\u007d: the same JSON value, with no brace but the
    structure's own. The parsers find objects by counting braces (judge._objects), and they count braces inside strings too, so an evidence
    quote with an unbalanced brace would otherwise lose the whole reply, and a "{}" inside one would be taken for the last object."""
    return JSON_STRING.sub(lambda m: m.group(0).replace("{", "\\u007b").replace("}", "\\u007d"), text)


UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "uniqueItems", "maxItems",
               "oneOf", "not", "if", "then", "else", "patternProperties", "dependentRequired")


def check_schema(schema):
    """The documented structured-output limits a schema must meet (problems, [] when none): every object additionalProperties false with every
    property required; no unsupported keyword; minItems only 0 or 1; at most 16 union-typed parameters (anyOf or a type list) and 24 optional
    ones; enums of strings, numbers, booleans or null only."""
    out, unions, optional = [], [0], [0]

    def walk(s, where):
        if not isinstance(s, dict):
            return
        for k in UNSUPPORTED:
            if k in s:
                out.append(f"{where}: {k} is not supported")
        if "minItems" in s and s["minItems"] not in (0, 1):
            out.append(f"{where}: minItems {s['minItems']}")
        if "enum" in s and not all(isinstance(v, (str, int, float, bool)) or v is None for v in s["enum"]):
            out.append(f"{where}: a complex enum")
        if "anyOf" in s or isinstance(s.get("type"), list):
            unions[0] += 1
        if s.get("type") == "object" or "properties" in s:
            props = s.get("properties", {})
            if s.get("additionalProperties") is not False:
                out.append(f"{where}: additionalProperties must be false")
            optional[0] += len(set(props) - set(s.get("required", [])))
            for k, v in props.items():
                walk(v, f"{where}.{k}")
        for k in ("items",):
            walk(s.get(k), f"{where}[]")
        for i, v in enumerate(s.get("anyOf", [])):
            walk(v, f"{where}|{i}")
    walk(schema, "$")
    if unions[0] > 16:
        out.append(f"{unions[0]} union-typed parameters (at most 16)")
    if optional[0] > 24:
        out.append(f"{optional[0]} optional parameters (at most 24)")
    return out


def canon(obj, schema):
    """obj with every string that matches one of its schema's enum values case-insensitively written in the schema's spelling (structured
    outputs do not guarantee an enum's capitalization); everything else unchanged."""
    if not isinstance(schema, dict):
        return obj
    if "anyOf" in schema:
        for s in schema["anyOf"]:
            if "enum" in s and isinstance(obj, str):
                hit = [v for v in s["enum"] if isinstance(v, str) and v.lower() == obj.lower()]
                if hit:
                    return hit[0]
        return obj
    if "enum" in schema and isinstance(obj, str):
        hit = [v for v in schema["enum"] if isinstance(v, str) and v.lower() == obj.lower()]
        return hit[0] if hit else obj
    if isinstance(obj, dict) and "properties" in schema:
        return {k: canon(v, schema["properties"].get(k)) for k, v in obj.items()}
    if isinstance(obj, list) and "items" in schema:
        return [canon(v, schema["items"]) for v in obj]
    return obj


def plain(x):
    """An SDK object (pydantic) or a fake as plain JSON-able data."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "model_dump"):
        return plain(x.model_dump())
    if hasattr(x, "__dict__"):
        return {k: plain(v) for k, v in vars(x).items() if not k.startswith("_")}
    return str(x)


def make_request(custom_id, params):
    """The batch request: the SDK's typed dicts when it is installed (Request, MessageCreateParamsNonStreaming; plain dicts at run time), else
    the same plain dicts (the tests' fake client)."""
    try:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
    except ImportError:
        return {"custom_id": custom_id, "params": dict(params)}
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))


class ClaudeJudge:
    """The Batches judge of one run's judge folder jd. settings: the config's judge block (endings.JUDGE_CLAUDE plus its calls); client: a
    batches client (None: anthropic.Anthropic(), made at the first submission); sleep: the wait between polls (the tests pass a no-op)."""

    def __init__(self, settings, jd, client=None, sleep=time.sleep, prog=print):
        self.s = dict(settings)
        assert self.s.get("backend") == "anthropic", self.s.get("backend")
        self.model, self.effort, self.max_tokens = self.s["model"], self.s["effort"], int(self.s["max_tokens"])
        self.caps = [self.max_tokens] + [int(x) for x in self.s.get("max_tokens_retry", [])]
        self.retries, self.poll_s, self.batch_max = int(self.s["retries"]), float(self.s["poll_s"]), int(self.s["batch_max"])
        self.jd = Path(jd)
        self.jd.mkdir(parents=True, exist_ok=True)
        self.client, self.sleep, self.prog = client, sleep, prog
        self.paths = {k: self.jd / f"cache_claude_{k}.jsonl" for k in KINDS}
        self.cache = {}
        for k, p in self.paths.items():
            EL.repair_cache(p)
            self.cache[k] = {r["key"]: r["result"] for r in EL.read_records(p)}
        self.attempts_path, self.batches_path = self.jd / "claude_attempts.jsonl", self.jd / "claude_batches.jsonl"
        self.requests_path, self.costs_path = self.jd / "claude_requests.jsonl", self.jd / "costs.jsonl"
        self.schema_md5 = {k: hashlib.md5(json.dumps(schema_of(k), sort_keys=True).encode()).hexdigest() for k in KINDS}
        self.provider, self.sdk = "anthropic", sdk_version()

    # ---------- keys, requests, the cache ----------

    def key(self, kind, text):
        prompt = prompt_of(kind)(text)
        return hashlib.sha256(json.dumps(["anthropic", self.model, self.effort, self.max_tokens, kind, self.schema_md5[kind], prompt]).encode()).hexdigest()

    @staticmethod
    def custom_id(kind, key):
        return f"{kind}-{key[:48]}"  # [a-zA-Z0-9_-], at most 64 characters

    def params(self, kind, text, cap=None):
        return {"model": self.model, "max_tokens": cap or self.max_tokens, "messages": [{"role": "user", "content": prompt_of(kind)(text)}],
                "thinking": {"type": "adaptive"}, "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema_of(kind)}}}

    def cached(self, kind, text):
        return self.cache[kind].get(self.key(kind, text))

    def _keep(self, kind, key, res):
        EL.append_record(self.paths[kind], {"key": key, "result": res})
        self.cache[kind][key] = res

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()  # the SDK finds the credentials itself; they are never read here
        return self.client

    # ---------- the batch flow ----------

    def _create(self, reqs):
        """batches.create with the SDK's own retries off (a create accepted server-side whose response was lost must not be sent twice)."""
        c = self._client()
        c = c.with_options(max_retries=0) if hasattr(c, "with_options") else c
        return c.messages.batches.create(requests=reqs)

    def _submit(self, jobs):
        """jobs: [(kind, key, text, cap)]; one batch. An intent record goes to judge/claude_batches.jsonl before batches.create and the submit
        record (the batch id) after it, both before the batch is polled, so an interrupted stage never submits (and pays for) a batch twice."""
        reqs, meta = [], {}
        for kind, key, text, cap in jobs:
            cid = self.custom_id(kind, key)
            reqs.append(make_request(cid, self.params(kind, text, cap)))
            meta[cid] = {"kind": kind, "key": key, "cap": cap}
        intent = uuid.uuid4().hex
        EL.append_record(self.batches_path, {"event": "intent", "intent": intent, "t": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             "jobs": meta, "n": len(reqs)})
        b = self._create(reqs)
        EL.append_record(self.batches_path, {"event": "submit", "intent": intent, "batch_id": b.id, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                             "jobs": meta, "n": len(reqs)})
        for r in reqs:
            EL.append_record(self.requests_path, {"batch_id": b.id, "custom_id": r["custom_id"], "params": plain(r["params"])})
        return b.id, meta

    def _wait(self, batch_id):
        t0 = time.time()
        while True:
            st = self._client().messages.batches.retrieve(batch_id).processing_status
            if st == "ended":
                return
            if time.time() - t0 > POLL_LIMIT_S:
                raise RuntimeError(f"batch {batch_id} has not ended after {POLL_LIMIT_S} s ({st})")
            self.sleep(self.poll_s)

    def _records(self):
        return EL.read_records(self.batches_path) if self.batches_path.exists() else []

    def _reconcile(self):
        """An intent with no submit record (a stage stopped between the two, or a create whose response was lost): the account's recent
        batches (batches.list) created within the hour after the intent, with as many requests and no record of their own, are looked
        for: exactly one is adopted (collected, never resubmitted); none: the intent is abandoned (its requests are sent again); more than
        one: AmbiguousBatch (exit 3, for a person to look)."""
        recs = self._records()
        closed = {r["intent"] for r in recs if r.get("event") in ("submit", "abandoned") and r.get("intent")}
        known = {r["batch_id"] for r in recs if r.get("event") == "submit"}
        dangling = [r for r in recs if r.get("event") == "intent" and r["intent"] not in closed]
        if not dangling:
            return
        listed = []
        for i, b in enumerate(self._client().messages.batches.list(limit=100)):
            if i >= 100:
                break
            listed.append(b)
        for r in dangling:
            cands = []
            for b in listed:
                ca = b.created_at.timestamp() if hasattr(b.created_at, "timestamp") else float(b.created_at)
                rc = b.request_counts
                total = sum(int(getattr(rc, k, 0) or 0) for k in ("processing", "succeeded", "errored", "canceled", "expired"))
                if b.id not in known and r["t"] - 120 <= ca <= r["t"] + 3600 and total == r["n"]:
                    cands.append(b.id)
            if len(cands) > 1:
                raise AmbiguousBatch(f"intent {r['intent']}: {len(cands)} unrecorded batches match ({cands}); look at the console before rerunning")
            if cands:
                EL.append_record(self.batches_path, {"event": "submit", "intent": r["intent"], "batch_id": cands[0], "adopted": True,
                                                     "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": r["jobs"], "n": r["n"]})
                known.add(cands[0])
                self.prog(f"claude: adopted batch {cands[0]} for an intent without its submit record")
            else:
                EL.append_record(self.batches_path, {"event": "abandoned", "intent": r["intent"], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    def _collect(self, batch_id, meta):
        """Every result of an ended batch, keyed by custom_id: the served model must be the configured one (else
        ServedModelMismatch, nothing of the batch kept); attempts and costs recorded once per (custom_id, batch); a reply cut off at
        max_tokens is resubmitted with the next cap (missing as unparseable at the last cap); a refusal or an unparseable output is cached as
        missing; an invalid_request_error stops the stage (InvalidRequest, exit 2) after the rest is kept; every other error (errored,
        expired, canceled, absent) is a technical failure, never stored. Returns (technical failures, cut off: custom_id -> next cap)."""
        results = list(self._client().messages.batches.results(batch_id))
        for r in results:
            msg = getattr(r.result, "message", None) if r.custom_id in meta and r.result.type == "succeeded" else None
            if msg is not None and msg.model != self.model:
                raise ServedModelMismatch(f"batch {batch_id}: served {msg.model!r}, configured {self.model!r}; nothing of it kept")
        seen, failed, cut, invalid = set(), [], {}, []
        logged = {(r["custom_id"], r["batch_id"]) for r in EL.read_records(self.attempts_path)}  # a re-collection never counts twice
        costed = {(r["custom_id"], r["batch_id"]) for r in EL.read_records(self.costs_path)}
        for r in results:
            cid = r.custom_id
            if cid not in meta:
                continue
            seen.add(cid)
            kind, key, cap = meta[cid]["kind"], meta[cid]["key"], meta[cid].get("cap", self.caps[0])
            res_ = r.result
            typ = res_.type
            msg = getattr(res_, "message", None) if typ == "succeeded" else None
            att = {"custom_id": cid, "kind": kind, "key": key, "batch_id": batch_id, "type": typ, "cap": cap, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            if msg is not None:
                att.update(model=msg.model, stop_reason=msg.stop_reason, usage=plain(msg.usage))
                if (cid, batch_id) not in costed:
                    EL.append_record(self.costs_path, {"custom_id": cid, "kind": kind, "batch_id": batch_id, "model": msg.model, "cap": cap,
                                                       "stop_reason": msg.stop_reason, "usage": plain(msg.usage)})
            else:
                att["error"] = plain(getattr(res_, "error", None))
                att["error_type"] = error_type(getattr(res_, "error", None))
            if (cid, batch_id) not in logged:
                EL.append_record(self.attempts_path, att)
            if typ != "succeeded":
                (invalid if att.get("error_type") == "invalid_request_error" else failed).append(cid)
                continue
            if key in self.cache[kind]:
                continue
            text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), None)
            base = {"raw": text, "model": self.model, "served": msg.model, "stop_reason": msg.stop_reason, "usage": plain(msg.usage), "cap": cap,
                    "client": f"anthropic {self.sdk}", "batch_id": batch_id, "custom_id": cid}
            if msg.stop_reason == "max_tokens" and cap != self.caps[-1]:
                cut[cid] = self.caps[self.caps.index(cap) + 1] if cap in self.caps else self.caps[-1]
                continue
            if msg.stop_reason == "refusal":
                res = {"rating": None, "evidence": None, "facts": None, "ok": False, "missing": "refusal", **base}
            elif msg.stop_reason == "max_tokens":
                res = {"rating": None, "evidence": None, "facts": None, "ok": False, "missing": f"unparseable (cut off at max_tokens {cap})", **base}
            else:
                try:
                    obj = canon(json.loads(text), schema_of(kind)) if text is not None else None
                except json.JSONDecodeError:
                    obj = None
                parsed = parse_of(kind)(brace_safe(json.dumps(obj))) if isinstance(obj, dict) else {"rating": None, "evidence": None, "facts": None, "ok": False}
                res = {**parsed, **base, **({} if parsed["ok"] else {"missing": "unparseable"})}
            self._keep(kind, key, res)
        for cid in meta:
            if cid not in seen:
                if (cid, batch_id) not in logged:
                    EL.append_record(self.attempts_path, {"custom_id": cid, "kind": meta[cid]["kind"], "key": meta[cid]["key"], "batch_id": batch_id,
                                                          "type": "absent", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                failed.append(cid)
        EL.append_record(self.batches_path, {"event": "collected", "batch_id": batch_id, "failed": len(failed), "cut_off": len(cut),
                                             "invalid": len(invalid)})
        if invalid:
            raise InvalidRequest(f"batch {batch_id}: {len(invalid)} requests rejected as invalid (e.g. {invalid[0]}); fix the request, then rerun")
        return failed, cut

    def _pending(self):
        """Submitted batches never collected (a stage that stopped while they ran)."""
        sub, done = {}, set()
        for r in self._records():
            if r["event"] == "submit":
                sub[r["batch_id"]] = r["jobs"]
            elif r["event"] == "collected":
                done.add(r["batch_id"])
        return {b: m for b, m in sub.items() if b not in done}

    def run(self, items, budget=None):
        """items: [(kind, document text)] in the order to send them; budget: at most this many calls (None: no limit). Returns (the number
        of pairs still unanswered after this run's resubmissions: technical failures, never stored, so a rerun asks again; the missing pairs
        [(kind, sha1 of the document, why)]: refusals and
        unparseable outputs only; the number deferred by the budget). A technical failure is resubmitted at most `retries` times in a run; a
        reply cut off at max_tokens is resubmitted with the next cap (max_tokens_retry)."""
        texts = {}
        for kind, t in items:
            assert kind in KINDS, kind
            texts[(kind, self.key(kind, t))] = t
        self._reconcile()
        cap_of, tries = {}, {}
        for b, meta in self._pending().items():  # collect what a stopped stage left running; never resubmit it
            self.prog(f"claude: collecting batch {b} submitted before ({len(meta)} requests)")
            self._wait(b)
            failed, cut = self._collect(b, meta)
            cap_of.update(cut)
        left = budget
        deferred = 0
        while True:
            todo = []
            for (kind, key), t in texts.items():
                cid = self.custom_id(kind, key)
                if key in self.cache[kind] or tries.get(cid, 0) > self.retries:
                    continue
                todo.append((kind, key, t, cap_of.get(cid, self.caps[0])))
            if left is not None:
                deferred = max(0, len(todo) - left)
                todo = todo[:left]
            if not todo:
                break
            for i in range(0, len(todo), self.batch_max):
                chunk = todo[i:i + self.batch_max]
                b, meta = self._submit(chunk)
                self.prog(f"claude: batch {b}: {len(chunk)} requests ({self.model}, effort {self.effort})")
                self._wait(b)
                failed, cut = self._collect(b, meta)
                for cid in failed:
                    tries[cid] = tries.get(cid, 0) + 1
                cap_of.update(cut)
            if left is not None:
                left -= len(todo)
                if left <= 0:
                    break
        missing = sorted((kind, hashlib.sha1(texts[(kind, key)].encode()).hexdigest(), self.cache[kind][key].get("missing"))
                         for (kind, key) in texts if key in self.cache[kind] and not self.cache[kind][key]["ok"])
        unanswered = sum(1 for (kind, key) in texts if key not in self.cache[kind]) - deferred
        return max(0, unanswered), missing, deferred


class InvalidRequest(Exception):
    """An invalid_request_error result (the stage stops with exit 2; nothing of it is stored)."""


class ServedModelMismatch(Exception):
    """A result served by another model than the configured one (exit 3; nothing of its batch is kept)."""


class AmbiguousBatch(Exception):
    """An intent whose create cannot be matched to one batch (exit 3)."""


def error_type(err):
    """An errored result's API error type: the SDK's ErrorResponse.error.type (or a flat .type, or the same keys in a dict)."""
    if err is None:
        return None
    if isinstance(err, dict):
        inner = err.get("error")
        return (inner or {}).get("type") if isinstance(inner, dict) else err.get("type")
    inner = getattr(err, "error", None)
    if inner is not None and not isinstance(inner, str) and getattr(inner, "type", None):
        return inner.type
    return getattr(err, "type", None)


def unavailable(exc):
    """A client error that means the judge cannot run here (exit 2: no credential set up, no access, a request the API refuses) rather than
    a transient one (exit 5: rerun later). Named by class, so the SDK need not be importable to decide."""
    return type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError", "UnprocessableEntityError")


def sdk_version():
    import importlib.metadata as md
    try:
        return md.version("anthropic")
    except md.PackageNotFoundError:
        return None


def claude_run(cj, items, budget, prog=print):
    """cj.run(items, budget) with its failures turned into exit codes: (result, None), or (None, exit code): 2 for a request the API rejects
    as invalid or a client error that means the judge cannot run here (no credential, no access); 3 for a reply served by another model
    than the configured one, or a submission that matches several batches; 5 for any other (transient) client error."""
    try:
        return cj.run(items, budget), None
    except InvalidRequest as e:
        prog(f"claude: {e}")
        return None, 2
    except (ServedModelMismatch, AmbiguousBatch) as e:
        prog(f"claude: {e}")
        return None, 3
    except Exception as e:  # noqa: BLE001 - the SDK's errors; the message carries no credential
        prog(f"claude: the Claude judge failed ({type(e).__name__}: {str(e)[:300]})")
        return None, 2 if unavailable(e) else 5
