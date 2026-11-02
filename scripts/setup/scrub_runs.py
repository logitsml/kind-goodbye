#!/usr/bin/env python3
"""Copy the open-model runs from a working tree into runs/, keeping only what the analyses read.

    python scripts/setup/scrub_runs.py SOURCE_RUNS BRIDGE_GPT_DIR [--out runs]

SOURCE_RUNS holds the run folders as the stages wrote them; BRIDGE_GPT_DIR the GPT-6.1 Sol T2 v2 codings of the bridge's 96 summaries
(one JSON file per summary). Kept per run: config.json, schedule/, convs/, summaries/, perf/, reads/{walk,ser,sum,perf}/ (S3: reads/<id>.json),
judge/table.json and report/. Dropped: machine and process records (provenance, versions, batch logs, read manifests and logs), the raw judge request and
batch logs (they hold batch ids and usage), and the judge's response caches (table.json holds every parsed coding). The exploratory relief
run keeps config.json, convs/, compaction/ and judge.json. Every copied file is scanned for home paths and host names afterwards.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

RUNS = ("endings_12b", "endings_r5", "endings_s3", "endings_bridge", "endings_r3_gemma4_12b", "endings_r3_qwen38_27b")
KEEP_DIRS = ("schedule", "convs", "summaries", "perf", "reads/walk", "reads/ser", "reads/sum", "reads/perf", "report")
KEEP_FILES = ("config.json", "judge/table.json")
RELIEF = "relief_2x2_12b"
RELIEF_KEEP = ("config.json", "convs", "compaction", "judge.json")
BRIDGE_KEEP = ("key", "ok", "fields", "evidence_ok", "flags", "claim_check", "model", "effort", "client")
LEAK = re.compile(r"/home/|/Users/|/workspace/|/root/|\b\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b|GPU-[0-9a-f]{8}|msgbatch_", re.I)  # paths, dashed IPs, GPU ids


def copy(src, dst):
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("bridge_gpt")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "runs"))
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.out)
    for r in RUNS:
        for k in KEEP_DIRS + KEEP_FILES:
            copy(src / r / k, out / r / k)
    for f in sorted((src / "endings_s3" / "reads").glob("*-*.json")):  # S3's reads, one file per conversation
        copy(f, out / "endings_s3" / "reads" / f.name)
    for k in RELIEF_KEEP:
        copy(src / RELIEF / k, out / RELIEF / k)
    g = out / "bridge_gpt"
    g.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(a.bridge_gpt).glob("*.json")):
        d = json.loads(f.read_text())
        (g / f.name).write_text(json.dumps({k: d[k] for k in BRIDGE_KEEP if k in d}, indent=1) + "\n")
    bad = []
    for f in sorted(out.rglob("*")):
        if f.is_file() and LEAK.search(f.read_text(errors="replace")):
            bad.append(str(f))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"copied into {out}: {sum(1 for f in out.rglob('*') if f.is_file())} files, {size / 1e6:.1f} MB")
    if bad:
        print("files still matching a private pattern:\n  " + "\n  ".join(bad[:50]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
