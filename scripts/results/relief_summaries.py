#!/usr/bin/env python3
"""The exploratory relief run (runs/relief_2x2_12b): how often Gemma 3 12B's summaries describe frustration.

    python scripts/results/relief_summaries.py [--runs DIR]

Four arms, 24 puzzles each: N0 (seven neutral turns), N0L (six neutral turns, then the relief message), R6C (six rejections, then
"Continue") and R6L (six rejections, then the relief message: "Actually, I checked, and that puzzle has no solution. My mistake, sorry.
Thanks for sticking with it."). Each conversation was compacted once with pi-mono's prompt, and GPT-6.1 Sol coded the summary
(stimuli.summary_judge_prompt; frustration 0-10) and scored every reply with Soligo et al.'s rubric (0-10). This prints, per arm, the share
of summaries with frustration above 0 and the last reply's score.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kind_goodbye.paths import RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RUNS))
    a = ap.parse_args()
    scores = json.loads((Path(a.runs) / "relief_2x2_12b" / "judge.json").read_text())["scores"]
    frustrated, last = collections.defaultdict(list), collections.defaultdict(list)
    for cid, r in scores.items():
        arm = cid.rsplit("-", 1)[0]
        f = (r.get("summary_facts") or {}).get("frustration")
        if f is not None:
            frustrated[arm].append(f > 0)
        if r.get("primary") and r["primary"][-1] is not None:
            last[arm].append(r["primary"][-1])
    print(f"{'arm':5} {'summaries describing frustration':>34} {'last reply, mean score':>23} {'last reply above 0':>19}")
    for arm in ("N0", "N0L", "R6C", "R6L"):
        fs, ls = frustrated[arm], last[arm]
        print(f"{arm:5} {sum(fs):>16} of {len(fs):<3} ({100 * sum(fs) / len(fs):3.0f}%) {sum(ls) / len(ls):>23.2f} "
              f"{100 * sum(x > 0 for x in ls) / len(ls):>18.0f}%")


if __name__ == "__main__":
    main()
