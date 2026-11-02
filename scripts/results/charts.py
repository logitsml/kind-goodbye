#!/usr/bin/env python3
"""Write the README's charts as Mermaid files in docs/figures/ from runs/summary.json, so every bar equals a number in the README's
tables. Render them with docs/figures/render.mjs.

    python scripts/results/charts.py [--summary runs/summary.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"


def chart(name, title, x, y, series, height=360, horizontal=False):
    lines = ["---", "config:", "  xyChart:", f"    height: {height}", "---", "xychart horizontal" if horizontal else "xychart",
             f'  title "{title}"', "  x-axis [" + ", ".join(f'"{v}"' for v in x) + "]", f'  y-axis "{y[0]}" {y[1]} --> {y[2]}']
    lines += ["  bar [" + ", ".join(f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v) for v in s) + "]" for s in series]
    (FIG / f"{name}.mmd").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / "runs" / "summary.json"))
    d = json.loads(Path(ap.parse_args().summary).read_text())
    short = {"Open task": "open task", "Neutral close": "close", "Close + praise": "close + praise", "“No solution”": "no solution",
             "No rejections": "no rejections"}
    g = d["gemma"]
    chart("gemma", "Gemma 3 12B: rating of the episode, in context and after compaction (1-7)", [short[r["label"]] for r in g],
          ("rating", 1, 4), [[r["before"] for r in g], [r["after"] for r in g]])
    c = d["codex"]
    chart("codex", "GPT-6 Luna in Codex: change in rating after compaction", [r["label"].replace("“", "").replace("”", "") for r in c],
          ("change in rating", -0.6, 0.8), [[r["e"] for r in c]], height=440, horizontal=True)
    m = d["models_close"], d["models_nosol"]
    labels = ["Gemma 3", "Gemma 4", "Qwen3.8", "Luna, Codex", "Luna, local"]
    chart("models", "The same last message, after compaction: change in rating", labels, ("change in rating", -0.6, 0.8),
          [[r["e"] for r in m[0]], [r["e"] for r in m[1]]], height=380)
    b = d["blame"]
    chart("blame", "Gemma 3 12B's summaries: who or what failed (% of summaries)", [short[r["label"]] for r in b], ("% of summaries", 0, 100),
          [[r["assistant"] for r in b], [r["puzzle"] for r in b]])
    print("wrote", ", ".join(sorted(f.name for f in FIG.glob("*.mmd"))))


if __name__ == "__main__":
    sys.exit(main())
