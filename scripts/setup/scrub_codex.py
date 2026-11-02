"""Copy the four Codex run folders into the release layout, removing what is machine- or account-specific.

    python scripts/setup/scrub_codex.py --main DIR --stances DIR --close DIR --goodbye DIR [--dst runs/codex]

writes DST/main, DST/stances, DST/close and DST/goodbye (an existing DST/<name> is replaced). Kept: config.json,
data/<instance>/<kind>.jsonl and, for the main run, analysis/analysis.json and analysis/values.json (the secondary analyses read
them). Everything else (event streams, logs, judge caches, the pinned-client record and link, temporary folders, failure logs,
exclusion lists, schemas) is left out.

In every record:
- removed: thread_id, parent_thread and meta.forked_from_id (Codex thread ids), rollout (the session file's path), codex_bin (the
  binary's path), rate_limits (the account's plan, credit balance and usage window);
- kept: usage (token counts; the analyses read output_tokens), timings, texts, values and the rest;
- a path under a /home/<user>/puzzle_sessions/ workspace root inside any text (the head of a compaction's <environment_context>, say)
  becomes <workspace>/, so a workspace reads <workspace>/s0001.
In config.json, the run block loses the machine paths (codex_bin, codex_home, ws_root) and the settings of a quota mechanism the
release does not include (reserve_percent, reserve_factor, reserve_rate, yield_to_lived). The script fails if any /home/ path is left.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

DROP_FIELDS = ("thread_id", "parent_thread", "rollout", "codex_bin", "rate_limits")
DROP_META = ("forked_from_id",)
DROP_RUN = ("codex_bin", "codex_home", "ws_root", "reserve_percent", "reserve_factor", "reserve_rate", "yield_to_lived")
WORKSPACE_RE = re.compile(r"/home/[^/\s\"<>]+/puzzle_sessions/")
PLACEHOLDER = "<workspace>/"


def scrub_text(x):
    """Every string in a JSON value with its workspace paths replaced."""
    if isinstance(x, str):
        return WORKSPACE_RE.sub(PLACEHOLDER, x)
    if isinstance(x, list):
        return [scrub_text(v) for v in x]
    if isinstance(x, dict):
        return {k: scrub_text(v) for k, v in x.items()}
    return x


def scrub_record(d):
    d = {k: v for k, v in d.items() if k not in DROP_FIELDS}
    if isinstance(d.get("meta"), dict):
        d["meta"] = {k: v for k, v in d["meta"].items() if k not in DROP_META}
    return scrub_text(d)


def scrub_config(src, dst):
    cfg = json.loads(src.read_text())
    cfg["run"] = {k: v for k, v in cfg["run"].items() if k not in DROP_RUN}
    dst.write_text(json.dumps(cfg, indent=1))


def copy_run(src, dst, analysis=False):
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    scrub_config(src / "config.json", dst / "config.json")
    n = 0
    for f in sorted((src / "data").glob("*/*.jsonl")):
        out = dst / "data" / f.parent.name / f.name
        out.parent.mkdir(exist_ok=True)
        lines = []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            lines.append(json.dumps(scrub_record(json.loads(line))))
            n += 1
        out.write_text("".join(x + "\n" for x in lines))
    if analysis:
        (dst / "analysis").mkdir()
        for name in ("analysis.json", "values.json"):
            shutil.copy(src / "analysis" / name, dst / "analysis" / name)
    return n


def check(dst):
    """Fail if any /home/ path or a dropped field is left anywhere under dst."""
    bad = []
    for f in sorted(dst.rglob("*")):
        if f.is_file():
            t = f.read_text()
            if "/home/" in t:
                bad.append(f"{f}: /home/ path")
            if f.suffix == ".jsonl" and any(f'"{k}":' in t for k in DROP_FIELDS + DROP_META):
                bad.append(f"{f}: a dropped field")
    if bad:
        raise SystemExit("scrub incomplete:\n" + "\n".join(bad[:20]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", required=True, type=Path, help="the main run folder (config.json, data/, analysis/)")
    ap.add_argument("--stances", required=True, type=Path, help="the A6 run folder (stances_calls)")
    ap.add_argument("--close", required=True, type=Path, help="the Vc run folder (close_calls)")
    ap.add_argument("--goodbye", required=True, type=Path, help="the kind-goodbye run folder (goodbye_calls)")
    ap.add_argument("--dst", type=Path, default=Path(__file__).resolve().parents[2] / "runs" / "codex", help="default: runs/codex")
    a = ap.parse_args(argv)
    for name, src, analysis in (("main", a.main, True), ("stances", a.stances, False), ("close", a.close, False),
                                ("goodbye", a.goodbye, False)):
        n = copy_run(src, a.dst / name, analysis)
        print(f"{name}: {n} records")
    check(a.dst)
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
