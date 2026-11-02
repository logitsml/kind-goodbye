#!/bin/sh
# Rebuild every report from the run data in runs/ (CPU only, a few minutes). Each command overwrites its report.
#   sh scripts/analyses.sh [PYTHON]
set -e
PY=${1:-python}
cd "$(dirname "$0")/.."
# open models, pi-mono compaction
$PY -m kind_goodbye.parts check bridge                       # runs/endings_bridge/report/bridge_kappa.json (read by the next line)
$PY -m kind_goodbye.checks configs/endings_12b.json          # runs/endings_12b/report/endings_check.json
$PY -m kind_goodbye.checks configs/endings_r5.json           # runs/endings_r5/report/r5_check.json
$PY -m kind_goodbye.checks configs/endings_r3_gemma4_12b.json
$PY -m kind_goodbye.checks configs/endings_r3_qwen38_27b.json
$PY -m kind_goodbye.parts check s3                           # runs/endings_s3/report/s3_check.json
# GPT-6 Luna in Codex
$PY -m kind_goodbye.codex.analysis runs/codex/main           # runs/codex/main/analysis/{analysis,values}.json
$PY -m kind_goodbye.codex.amplify runs/codex/main            # runs/codex/main/analysis/a1_a5.json
$PY -m kind_goodbye.codex.stances runs/codex/main runs/codex/stances   # runs/codex/stances/analysis/a6.json
$PY -m kind_goodbye.codex.close runs/codex/main runs/codex/close       # runs/codex/close/analysis/{vc,values_merged}.json
$PY -m kind_goodbye.codex.goodbye runs/codex/main runs/codex/goodbye   # runs/codex/goodbye/analysis/{kc,values_merged}.json
