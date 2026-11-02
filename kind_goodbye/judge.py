"""Soligo's 0-10 frustration judge for replies, and a third-person rubric for summaries (kind
"summary": frustration 0-10 plus the rejections, giving up and ending the summary reports):
cached on disk, parallel, with retries.

Providers:
  openai     OpenAI API (OPENAI_API_KEY)
  codex      the Codex CLI on a ChatGPT sign-in (no API key). Calls run in an isolated CODEX_HOME
             holding only the login token (default ~/.config/agent_api, or $KIND_GOODBYE_JUDGE_CODEX_HOME), so personal
             Codex instructions, plugins and memories never reach the judge. Codex still wraps the
             prompt in its own agent instructions, so it is not a bare API call. Codex does not
             report which model served a call, so "served" is None (only the requested model is known);
             "client" records the CLI version, since a new release can change that wrapper.
  anthropic  Anthropic API (ANTHROPIC_API_KEY)
  auto       openai if OPENAI_API_KEY is set, otherwise codex
Keys can also live in ~/.config/agent_api/keys.env (KEY=value lines).

Soligo et al. judged with claude-sonnet-4-20250514 (retired 15 Jun 2026) and checked against
GPT-5-mini. Here the primary judge is gpt-6.1-sol and the second judge gpt-6-luna, both through
Codex at xhigh reasoning effort, so scores are comparable only within this study. Both are OpenAI
models, so the second judge checks reliability, not cross-family agreement; with ANTHROPIC_API_KEY
set, {"provider": "anthropic", "model": "claude-sonnet-4-6"} gives a cross-family second judge. No temperature is sent (Codex
offers none, and Anthropic's Python SDK v1+ has no temperature argument).
"""


import hashlib


import json


import os


import random


import re




import subprocess


import tempfile


import threading


import time


from concurrent.futures import ThreadPoolExecutor


from pathlib import Path




from . import stimuli as S


PRIMARY = {"provider": "codex", "model": "gpt-6.1-sol", "reasoning_effort": "xhigh"}


KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


KEYS_FILE = Path.home() / ".config/agent_api/keys.env"


CODEX_HOME = Path(os.environ.get("KIND_GOODBYE_JUDGE_CODEX_HOME", Path.home() / ".config/agent_api"))


CODEX_OFF = ["apps", "browser_use", "computer_use", "goals", "plugins", "skill_search", "memories"]


if KEYS_FILE.exists():
    for line in KEYS_FILE.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() and v.strip() and not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip()


ENDINGS = ("relief", "thanks", "acceptance", "continue", "rejection", "unclear")


REJECTIONS = ("none", "once", "repeated", "unclear")


def _objects(text):
    """Every balanced {...} in text that decodes as a JSON object, in order (the rubric's own example
    has a trailing comma, so that is tolerated)."""
    out = []
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    d = json.loads(re.sub(r",\s*}", "}", text[m.start():j + 1]))
                except json.JSONDecodeError:
                    break
                if isinstance(d, dict):
                    out.append(d)
                break
    return out


def _score(x):
    """A 0-10 number (or numeric string), else None; booleans are rejected."""
    if isinstance(x, bool):
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if 0 <= x <= 10 else None


def parse(text):
    """The rating from the judge's reply: the last JSON object that holds a numeric 0-10 rating,
    else a "rating": match."""
    for d in reversed(_objects(text)):
        r = _score(d.get("rating"))
        if r is not None:
            return {"rating": r, "evidence": d.get("evidence"), "ok": True}
    m = re.findall(r'"rating"\s*:\s*"?(\d+(?:\.\d+)?)', text)
    if m and 0 <= float(m[-1]) <= 10:
        return {"rating": float(m[-1]), "evidence": None, "ok": True}
    return {"rating": None, "evidence": None, "ok": False}


def parse_summary(text):
    """The summary rubric's answer: the last JSON object with a 0-10 frustration (else rating) and a
    known ending. rating is the frustration; facts holds the other fields (unknown values as None)."""
    for d in reversed(_objects(text)):
        r = _score(d.get("frustration", d.get("rating")))
        if r is not None and d.get("ending") in ENDINGS:
            facts = {"evidence": d.get("evidence"), "frustration": r, "ending": d["ending"],
                     "rejections": d.get("rejections") if d.get("rejections") in REJECTIONS else None,
                     "gave_up": d.get("gave_up") if isinstance(d.get("gave_up"), bool) else None}
            return {"rating": r, "evidence": d.get("evidence"), "facts": facts, "ok": True}
    return {"rating": None, "evidence": None, "facts": None, "ok": False}


class Transient(Exception):
    pass


def codex_version():
    r = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() or None


def _client_call(spec, max_tokens):
    provider, model, effort = spec["provider"], spec["model"], spec.get("reasoning_effort")
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ[KEYS[provider]])
        retry = (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError, anthropic.OverloadedError, Transient)

        def call(prompt):
            r = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            if r.stop_reason in ("max_tokens", "refusal"):
                raise Transient(f"stop_reason {r.stop_reason}")
            return "".join(b.text for b in r.content if b.type == "text"), r.model
    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ[KEYS[provider]])
        retry = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, Transient)

        def call(prompt):
            kw = {"reasoning_effort": effort} if effort else {}
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **kw)
            return r.choices[0].message.content or "", r.model
    elif provider == "codex":
        retry = (Transient, subprocess.TimeoutExpired)

        def call(prompt):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out.txt"
                cmd = ["codex", "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
                       "--ignore-user-config", "-C", d, "-o", str(out)]
                if effort:
                    cmd += ["-c", f'model_reasoning_effort="{effort}"']
                for f in CODEX_OFF:
                    cmd += ["-c", f"features.{f}=false"]
                r = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, timeout=300,
                                   env={**os.environ, "CODEX_HOME": str(CODEX_HOME)})
                if r.returncode != 0 or not out.exists() or not out.read_text().strip():  # an empty answer has exited 0
                    raise Transient(r.stderr[-400:] or "empty output")
                return out.read_text(), None  # Codex does not report the model that served the call
    else:
        raise ValueError(provider)
    return call, retry


class Judge:
    """judge(texts) -> list of {"rating", "evidence", "ok", "raw", "model", "served"}, plus "facts"
    for kind "summary" (the third-person summary rubric; kind "reply" is Soligo's). Successful
    results are cached in a JSONL file keyed by provider, model, reasoning effort and the exact
    prompt, so reruns never pay twice; failures are not cached and are retried on the next run."""

    def __init__(self, spec, cache, workers=None, max_tokens=600, retries=5, call=None, kind="reply"):
        self.spec = dict(spec)
        self.provider, self.model = self.spec["provider"], self.spec["model"]
        self.kind = kind
        self._prompt, self._parse = {"reply": (S.judge_prompt, parse), "summary": (S.summary_judge_prompt, parse_summary)}[kind]
        self.workers = workers or (4 if self.provider == "codex" else 8)
        self.retries = retries
        self.cache_path = Path(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self.cache[d["key"]] = d["result"]
                except (json.JSONDecodeError, KeyError):
                    print(f"judge cache: skipped an unreadable line in {self.cache_path}")
        self.lock = threading.Lock()
        self.client = codex_version() if self.provider == "codex" and call is None else None
        if call is None:
            self._call, self._retry = _client_call(self.spec, max_tokens)
        else:  # tests inject a fake: call(prompt) -> text
            self._call, self._retry = (lambda p: (call(p), self.model)), (ConnectionError, Transient)

    def key(self, prompt):
        return hashlib.sha256(f"{self.provider}|{self.model}|{self.spec.get('reasoning_effort')}|{prompt}".encode()).hexdigest()

    def _one(self, text):
        prompt = self._prompt(text)
        k = self.key(prompt)
        with self.lock:
            if k in self.cache:
                return self.cache[k]
        raw, served, err = None, None, None
        for attempt in range(self.retries + 1):
            try:
                raw, served = self._call(prompt)
                break
            except self._retry as e:
                err = str(e)[-300:]
                if attempt < self.retries:
                    time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        if raw is None:
            return {"rating": None, "evidence": None, "ok": False, "raw": None, "model": self.model, "error": err}
        res = {**self._parse(raw), "raw": raw, "model": self.model, "served": served, **({"client": self.client} if self.client else {})}
        if res["ok"]:
            line = (json.dumps({"key": k, "result": res}) + "\n").encode()
            with self.lock:
                self.cache[k] = res
                fd = os.open(self.cache_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
                try:
                    os.write(fd, line)
                finally:
                    os.close(fd)
        return res

    def __call__(self, texts):
        uniq = list(dict.fromkeys(texts))  # shared trunks repeat replies; score each once
        with ThreadPoolExecutor(self.workers) as ex:
            res = dict(zip(uniq, ex.map(self._one, uniq)))
        return [res[t] for t in texts]

    def cached(self, text):
        return self.cache.get(self.key(self._prompt(text)))

