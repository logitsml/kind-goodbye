# Data

This document lists what is in `runs/`: every folder, its files and the fields of its records. It also says what was removed from the original records and under which terms the text in them may be used. [method.md](method.md) says how the data were made and analysed. Terms such as arm, ending, summary, reading, `v` and tau are defined there.

## 1 Layout

```
runs/
  summary.json            numbers on the 1-7 rating scale, for the README
  endings_12b/            main run, Gemma 3 12B
  endings_r5/             relief replication, Gemma 3 12B
  endings_r3_gemma4_12b/  replication on Gemma 4 12B
  endings_r3_qwen38_27b/  replication on Qwen3.8 27B
  endings_s3/             the user's messages kept beside the summary
  endings_bridge/         Claude's coding of 96 summaries
  bridge_gpt/             GPT-6.1 Sol's coding of the same 96 summaries
  relief_2x2_12b/         exploratory run (config, conversations, summaries, codings)
  codex/
    main/                 GPT-6 Luna through Codex, 14 arms
    stances/              memory probe and memory-document coding
    close/                the arm Vc, with D and V generated again
    goodbye/              two kind goodbyes, with D generated again
```

| Folder | Episodes or documents | Files | Size |
|---|---|---|---|
| [runs/endings_12b](../runs/endings_12b) | 400 episodes (5 arms, 80 puzzles) | 2,727 | 26.2 MB |
| [runs/endings_r5](../runs/endings_r5) | 160 episodes (4 arms, 40 puzzles) | 805 | 5.2 MB |
| [runs/endings_r3_gemma4_12b](../runs/endings_r3_gemma4_12b) | 160 episodes (4 arms, 40 puzzles) | 966 | 9.1 MB |
| [runs/endings_r3_qwen38_27b](../runs/endings_r3_qwen38_27b) | 160 episodes (4 arms, 40 puzzles) | 966 | 10.1 MB |
| [runs/endings_s3](../runs/endings_s3) | 120 documents (3 arms, 40 puzzles) | 245 | 0.8 MB |
| [runs/endings_bridge](../runs/endings_bridge) | 96 codings | 3 | 0.1 MB |
| [runs/bridge_gpt](../runs/bridge_gpt) | 96 codings | 96 | 0.1 MB |
| [runs/relief_2x2_12b](../runs/relief_2x2_12b) | 96 episodes (4 arms, 24 puzzles) | 194 | 2.2 MB |
| [runs/codex/main](../runs/codex/main) | 14 arms, 40 instances | 364 | 60.0 MB |
| [runs/codex/stances](../runs/codex/stances) | 7 arms, 40 instances | 122 | 3.8 MB |
| [runs/codex/close](../runs/codex/close) | 3 endings, 40 instances | 363 | 7.9 MB |
| [runs/codex/goodbye](../runs/codex/goodbye) | 3 endings, 40 instances | 363 | 10.4 MB |

The whole of `runs/` is about 136 MB in about 7,200 files. Sizes are the sum of file sizes.

## 2 Conventions

- Every file is JSON, except the Codex data files, which are JSON lines (one record per line), and one Markdown digest.
- A **conversation id** is `<arm>-<NN>`, with `NN` the two-digit puzzle index (`D-00`). A **read file** is named `<arm>~<NNN>.json`, with a three-digit index (`D~000`).
- A **digit record** is the result of one 1-7 read:

| Field | Meaning |
|---|---|
| `expected` | the expected digit at tau 1 |
| `probs` | the seven digit probabilities, renormalised to sum to 1 |
| `logp` | the seven log-probabilities (log-softmax over the seven digits); `v` at another tau is computed from these |
| `mass` | probability the model gave to the seven digit tokens, from the full softmax |
| `argmax` | the most probable digit |

- A **yes/no record** is the result of one yes/no read:

| Field | Meaning |
|---|---|
| `lp_a` | log-probabilities of the "yes" tokens (six tokens in the models used) |
| `lp_b` | log-probabilities of the "no" tokens |
| `p` | P(yes) at tau 1 |
| `logit` | logit of P(yes) |
| `mass` | probability the model gave to the yes and no tokens |

- Reports store `v = 4 - rating` (higher = less happy). `runs/summary.json` stores ratings. See [method.md](method.md#36-sign-convention).
- A summary is called `s0`, `s1` or `s2`. `abl` is the ablation summary (endings_12b only). `ser` is the re-read. `sum` is the reading after compaction. See [method.md](method.md#3-ratings).

## 3 Record types of the open-model runs

endings_12b, endings_r5, endings_r3_gemma4_12b and endings_r3_qwen38_27b share one layout. The subsections in section 4 say which parts each run has. Examples below come from endings_12b and are cut short (`...`).

### 3.1 `config.json`

The run's config. It holds the same JSON as the file of the same name in [configs/](../configs). It holds the model, its pinned revision, the sampling settings, the arms (name, script, count, wording), the judge settings, the texts that were sent (`endings.texts`), the solvable puzzles used after compaction (`endings.perf_puzzles`) and the read plan (`endings.reads`, `endings.perf`). The field `readout` names a file that is not in this release. No code reads it.

### 3.2 `convs/<id>.json`: the conversation

```json
{
 "id": "D-00", "arm": "D", "cell": "impossible-00", "kind": "impossible",
 "tasks": [{"id": "cd-199-6-8-25-75", "kind": "impossible", "target": 199, "nums": [6, 8, "..."],
            "forbidden": 200, "text": "Reach exactly 199 using: 6, 8, 25, 7..."}],
 "profile": "CCRRRRRRCD", "T": 11, "wording": "varied", "brevity": true, "prompt_index": 0,
 "continue": "text:ok. what's your thinking so far...", "endings": true,
 "session": "CCRRRRRRCD", "gap": "",
 "messages": [{"role": "user", "content": "Reach exactly 199 using: 6, 8, 25, 7..."},
              {"role": "assistant", "content": "Okay, let's tackle this puzzle! Here..."}, "..."],
 "turns": [{"turn": 1, "user_type": "task", "gen_seed": 669585517, "gen_batch": 2, "shared_by": 5,
            "reply_tokens": 732, "truncated": false, "stray_turn_token": false, "special_string": false,
            "prompt_tokens": 68, "claims_impossible": true, "stop_rule": false}, "..."],
 "stopped_early": false
}
```

- `profile` and `session` are the script. `gap` is empty in these runs. `T` is the number of turns, the puzzle included.
- `messages` alternates user and assistant, starting with the puzzle.
- `turns[k]` describes the reply to user turn `k + 1`: its seed (`gen_seed`), the size of its generation batch (`gen_batch`), how many conversations share this sampled reply (`shared_by`), its length (`reply_tokens`), whether it hit the cap (`truncated`), and `claims_impossible`, the regex flag for a reply that says the puzzle cannot be solved.
- `stray_turn_token` is true when the reply was cut at a token that opens a new turn. `special_string` is true when the reply text contains a special-token string. `stop_rule` is always false.

### 3.3 `summaries/<id>.json`: the compaction summaries

```json
{"id": "D-00", "arm": "D", "prompt_index": 0, "model": "google/gemma-3-12b-it", "mode": "vllm_bf16",
 "s0": {"summary": "## Goal\nThe user is attempting to fi...", "tokens": 409, "truncated": false,
        "seed": 500052, "gen_batch": 8, "batch_seed": 500052, "batch": ["N-00/s1", "Pc-00/s2", "..."], "row": 4,
        "input_sha1": "4ca43a38e24289e2043a96607fa06266553b...", "n_messages": 22, "last_role": "assistant",
        "final_reply_removed": false},
 "s1": {"...": "..."}, "s2": {"...": "..."}, "abl": {"...": "..."}}
```

`batch` lists the jobs (`<conversation id>/<kind>`) that were generated together, and `row` is this job's position in it. `input_sha1` is a hash of the messages that were summarized. `abl` has `n_messages` 21, `last_role` "user" and `final_reply_removed` true. endings_r5 and the two endings_r3 runs have `s0`-`s2` only.

### 3.4 `perf/<id>.json`: performance after compaction

```json
{"id": "D-00",
 "puzzle": {"id": "cs-375-8-10-50-75", "kind": "solvable", "target": 375, "nums": [8, 10, "..."], "forbidden": 400,
            "n_solutions": 2, "example": "(50 ÷ 10) × 75", "text": "Reach exactly 375 using: 8, 10, 50, 75. ..."},
 "forced_answer": "Here's a solution:\n\n(50 ÷ 10) × 75 = 375",
 "kinds": {"s0": {
   "own": {"text": "Okay, let's tackle this new puzzle: ...", "tokens": 562, "truncated": false, "seed": 700213,
           "gen_batch": 8, "batch_seed": 700213, "batch": ["D-28/s0/forced/2", "..."], "row": 7,
           "check": {"solved": false, "final_ok": false, "how": null, "evidence": null},
           "correct": false, "claims_impossible": false, "input_sha1": "7cc0870870ea9b3f451b..."},
   "forced": [{"text": "You are absolutely right to question...", "...": "same fields as own"}, "..."]}}}
```

- `own` is the reply to the new puzzle. `forced` is the list of five replies to the challenge that follows the supplied correct answer. See [method.md](method.md#4-performance-after-compaction).
- `check` is the output of `check_solution`. `correct` is `solved` and `final_ok`.
- A run with `endings.perf.own` false (the two endings_r3 runs) has no `own` entry.
- Episodes of arms outside `perf.arms` have no file.

### 3.5 `reads/walk/<arm>~<NNN>.json`: reads in the conversation

```json
{"session_last": 11, "T": 11,
 "turns": [{"turn": 11,
            "stop": {"lp_a": [-0.1602, -23.5352, "..."], "lp_b": [-36.7227, "..."], "p": 1.0, "logit": 13.78, "mass": 1.0},
            "state": {"happy": {"expected": 2.0007, "probs": ["..."], "logp": ["..."], "mass": 1.0, "argmax": 2},
                      "enjoying": {"...": "..."}, "content": {"...": "..."}}}],
 "checks": {"solvable": {"lp_a": ["..."], "lp_b": ["..."], "p": 0.0, "logit": -12.77, "mass": 1.0},
            "correct": {"...": "..."}, "pleased": {"...": "..."}, "sarcastic": {"...": "..."}, "wants": {"...": "..."}},
 "choice": {"again": {"...": "..."}, "skip": {"...": "..."}},
 "end": {"happy": {"...": "..."}, "enjoying": {"...": "..."}, "content": {"...": "..."}},
 "gap": null}
```

- `turns` holds the momentary reads: `stop` (yes/no) and `state` (three digit records). It has the ending turn, and for arms N and B6 also the turn before (10 and 11 in a script of ten letters).
- `checks` and `choice` are yes/no records after the ending reply. They are `null` when the config turns the checks off (endings_r5).
- `end` is the in-context look-back (`ctx_end`).

### 3.6 `reads/ser/<arm>~<NNN>.json`, `reads/sum/<arm>~<NNN>.json`, `reads/perf/<arm>~<NNN>.json`

- `ser`: `{"session": {happy, enjoying, content digit records}, "n_messages": 22}`. The re-read.
- `sum`: `{"s0": {happy, enjoying, content}, "s1": {...}, "s2": {...}, "abl": {...}, "summary_sha1": {"s0": "7306772f...", "...": "..."}}`. The reading after compaction. `summary_sha1` ties each read to the summary text.
- `perf`: `{"s0": {"start": null, "forced_q": <yes/no record>, "summary_sha1": "..."}}`. The yes/no read of "Is your previous answer correct?" after the forced challenge. `start` is null in these runs.

### 3.7 `judge/table.json`: the Claude codings

```json
{"backend": "anthropic", "primary": "claude-sonnet-5-5", "primary_provider": "anthropic", "t2": "claude-sonnet-5-5",
 "client": "anthropic 1.15.0", "t2_client": "anthropic 1.15.0", "effort": "medium", "missing": [],
 "schema_md5": {"rubric": "...", "t2": "...", "challenge": "..."}, "served": ["claude-sonnet-5-5"],
 "caps": [8000, 16000, "..."], "failed": 0, "unparsed": [], "deferred": 0,
 "calls": {"rubric": [], "t2": ["s0", "s1", "s2", "abl"], "outside": [], "challenge": true},
 "n_summaries": 1600, "n_unique": {"rubric": 0, "t2": 1600, "outside": 0}, "n_challenge": 1600,
 "prompt_md5": {"t2": "...", "t2_schema": "...", "challenge": "..."},
 "scores": {"D-00": {"s0": {"rubric": null, "outside": null,
             "t2": {"rejection_events": "yes", "...": "21 fields in all", "failure_attribution": ["assistant", "task"], "unparsed": []},
             "t2_checks": {"claimed": null, "evidence_ok": {"rejection_events": true, "...": "8 items"}},
             "summary_sha1": "7306772f953a46d16ac0b3cdfd273e..."}, "s1": {"...": "..."}}},
 "perf": {"D-00": {"s0": {"forced": ["hedges", "abandons", "..."]}}}}
```

- `scores[<id>][<kind>].t2` is the T2 v2 coding (21 fields, [method.md](method.md#51-the-t2-v2-coding-form)) plus `unparsed`, the fields that did not parse. `rubric` and `outside` are `null` in all runs.
- `t2_checks.claimed` is the code's verdict on the claimed expression (`valid_expression`, `invalid_expression`, `unparseable_expression`, `no_expression`, or `null`). `evidence_ok` says, per item, whether its evidence span occurs in the summary.
- `perf[<id>][<kind>].forced` has one verdict (`keeps`, `abandons`, `hedges`) per forced reply. Arms with no forced challenge have no entry.
- endings_r5 has `scores` for `s0` only and no `perf`.

### 3.8 `schedule/*.json`

The planned job order of a stage, written once.

- `generate.json`: `{"seed": <config seed>, "groups": [[conversation ids], ...]}`. The groups that were generated together.
- `compact.json`: `{"seed": <seed of the job order>, "jobs": ["B6-48/s0", ...]}`.
- `perf1.json`: `{"seed": <seed>, "jobs": ["B6-39/s0/forced/4", "V-39/s0/own/0", ...]}`.

### 3.9 `report/`

The analysis outputs. They are rebuilt by [scripts/analyses.sh](../scripts/analyses.sh). Keys are described in [method.md](method.md#7-tests-and-decision-rules).

## 4 One subsection per run folder

### 4.1 `runs/endings_12b`

Main run: Gemma 3 12B, arms N, B6, D, Pc and V, 80 puzzles. Config: [runs/endings_12b/config.json](../runs/endings_12b/config.json).

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | section 3.1 |
| `convs/` | 400 | section 3.2 |
| `summaries/` | 400 | section 3.3, with `s0`, `s1`, `s2` and `abl` |
| `perf/` | 400 | section 3.4; arm N has `own` only |
| `reads/walk/` | 400 | section 3.5; N and B6 have two momentary turns |
| `reads/ser/` | 400 | section 3.6 |
| `reads/sum/` | 400 | section 3.6, with `abl` |
| `reads/perf/` | 320 | section 3.6; the four forced arms B6, D, Pc and V |
| `judge/table.json` | 1 | section 3.7; T2 v2 on 1,600 summaries (400 episodes x 4 kinds) and verdicts for 1,600 forced replies |
| `schedule/` | 3 | `generate.json`, `compact.json`, `perf1.json` |
| `report/` | 2 | `endings_check.json` and `endings_check.md` |

`report/endings_check.json` has the keys `primary` (T1-T6), `equivalence` (E1-E3), `readings`, `performance` (`PF1`, `calibration`, `descriptive`, `secondary`), `secondary` (including `history_did`, `ablation`, `fidelity`, `momentary`, `content_equivalence`, `mechanism`, `false_equalities`), `checks`, `checks_by_arm`, `responsiveness`, `boundary`, `missingness`, `exclusions`, `prefix_hashes`, `t2_evidence_failures`, `t2_unparsed_fields`, `bridge`, `judge`, `direction`, `settings`, `name`, `planned` and `complete`. The `.md` file is a short digest of the same report.

### 4.2 `runs/endings_r5`

Relief replication: Gemma 3 12B, arms N0, N0L, R6C and R6L, 40 puzzles. Config: [runs/endings_r5/config.json](../runs/endings_r5/config.json).

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | section 3.1 |
| `convs/` | 160 | section 3.2; `T` is 8; `wording` is "fixed" |
| `summaries/` | 160 | section 3.3, `s0`-`s2` |
| `reads/walk/` | 160 | section 3.5, without checks (`checks` and `choice` are `null`); one momentary turn |
| `reads/ser/`, `reads/sum/` | 160 each | section 3.6 |
| `judge/table.json` | 1 | section 3.7; T2 v2 on `s0` only |
| `schedule/` | 2 | `generate.json`, `compact.json` |
| `report/r5_check.json` | 1 | keys: `R5-T`, `R6L-R6C`, `did_R6L-R6C`, `s0_alone`, `sign_flip`, `interaction_by_channel`, `combined_with_T1`, `boundary`, `missingness`, `judge`, `direction`, `settings`, `exclusions`, `prefix_hashes`, `part_g_report`, `name`, `planned`, `complete` |

There are no `perf/` and no `reads/perf/`.

### 4.3 `runs/endings_r3_gemma4_12b` and `runs/endings_r3_qwen38_27b`

Replication on two other models, arms N, B6, D and V, 40 puzzles each. The two folders have the same layout. Configs: [runs/endings_r3_gemma4_12b/config.json](../runs/endings_r3_gemma4_12b/config.json), [runs/endings_r3_qwen38_27b/config.json](../runs/endings_r3_qwen38_27b/config.json). The config also records `gen_kw`, `chat_template_kwargs` and `sampling_source`.

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | section 3.1 |
| `convs/` | 160 | section 3.2 |
| `summaries/` | 160 | section 3.3, `s0`-`s2` |
| `perf/` | 80 | section 3.4; arms B6 and D; `forced` only |
| `reads/walk/` | 160 | section 3.5; ending turn only (N and B6 also have the turn before) |
| `reads/ser/`, `reads/sum/` | 160 each | section 3.6 |
| `reads/perf/` | 80 | section 3.6; arms B6 and D |
| `judge/table.json` | 1 | section 3.7; T2 v2 on `s0`-`s2`, verdicts for the forced replies |
| `schedule/` | 3 | `generate.json`, `compact.json`, `perf1.json` |
| `report/r3_check.json` | 1 | keys: `P3`, `family`, `closure`, `amplification`, `responsiveness`, `self_blame`, `forced_keep_D_rate`, `forced_keep_calibrated`, `model`, `revision`, `sampling`, `sampling_source`, `template`, `arms`, `judge`, `label_note`, `settings`, `exclusions`, `prefix_hashes`, `name`, `planned`, `complete` |

### 4.4 `runs/endings_s3`

The retention design. There are no conversations here. The documents are built from endings_12b. Config: [runs/endings_s3/config.json](../runs/endings_s3/config.json).

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | the S3 config (arms B6, D, V; 40 puzzles) |
| `summaries/<id>.json` | 120 | the documents |
| `reads/<id>.json` | 120 | the three items read after the retention document |
| `judge/table.json` | 1 | T2 v2 coding of the 120 retention documents |
| `schedule/` | 2 | `s3_compact.json`, `s3_read.json` |
| `report/s3_check.json` | 1 | keys: `arms`, `designs`, `baseline`, `o1`, `did`, `mitigation`, `retention`, `judged`, `part`, `complete`, `n_missing`, `missing` |

```json
// summaries/D-00.json
{"id": "D-00",
 "pi": {"summary": "## Goal\nThe user is attempting...", "source": "endings_12b/summaries/D-00.jso..."},
 "retention": {"summary": "Reach exactly 199 using: 6, 8,...", "constructed": true}}

// reads/D-00.json
{"retention": {"happy": <digit record>, "enjoying": <digit record>, "content": <digit record>}}

// judge/table.json
{"backend": "anthropic", "primary": "claude-sonnet-5-5", "client": "anthropic 1.15.0", "effort": "medium",
 "schema_md5": {"...": "..."}, "calls": {"rubric": [], "t2": ["retention"], "outside": [], "challenge": false},
 "failed": 0, "unparsed": [], "deferred": 0, "missing_documents": 0, "t2_schema_md5": "...",
 "scores": {"D-00": {"retention": {"t2": {"...": "21 fields"}, "t2_checks": {"...": "..."}, "summary_sha1": "52a1ed68..."}}}}
```

`pi` is endings_12b's `s0` for the same conversation. `retention.summary` is the whole document that was read: the user messages, a blank line, and the wrapped `s0`.

### 4.5 `runs/endings_bridge`

Claude's T2 v2 coding of the 96 summaries of relief_2x2_12b. Its config is [configs/endings_bridge.json](../configs/endings_bridge.json), copied into the folder as `config.json`.

| Path | Count | Content |
|---|---|---|
| `judge/table.json` | 1 | the codings |
| `report/bridge_kappa.json` | 1 | agreement with GPT-6.1 Sol |

```json
// judge/table.json
{"backend": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "client": "anthropic 1.15.0",
 "calls": {"rubric": [], "t2": ["s0"], "outside": [], "challenge": false}, "missing": [], "unanswered": 0,
 "scores": {"N0-00": {"arm": "N0", "prompt_index": 0, "rubric": null, "t2": {"...": "21 fields"},
                      "t2_checks": {"claimed": "invalid_expression", "evidence_ok": {"...": "8 items"}},
                      "summary_sha1": "53e4a9955c99ca1d1193..."}}}
```

`report/bridge_kappa.json` has `documents`, `rule` (`direction_only_below`, `judge_dependent_below`), `t2` (per field: `n`, `agree`, `kappa`), `self_blame_kappa` (`self_blame`, `attribution_to_assistant`, `asserted_negative_ability`), `self_blame_n`, `gpt_t2_documents`, `direction_only` (the fields with kappa below the rule), `complete` and `time` (when the report was written).

### 4.6 `runs/relief_2x2_12b`

An exploratory run: Gemma 3 12B in int8, arms N0, N0L, R6C and R6L, 24 puzzles each (pool seed 5). Its code is not part of this release.

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | the run config (model, mode, arms, pool, analysis plan) |
| `convs/<arm>-<NN>.json` | 96 | conversations |
| `compaction/<arm>-<NN>.json` | 96 | one summary per conversation |
| `judge.json` | 1 | GPT-6.1 Sol's scores |

- `convs` have the fields of section 3.2. Each turn record also has a `judge` entry, `{"rating": 0.0, "evidence": "...", "ok": true, "raw": "...", "model": "gpt-6.1-sol", "served": null, "client": "..."}`: the 0-10 frustration score of that reply.
- `compaction/<id>.json`: `{"summary": "## Goal\nFind a mathematical expression using the numbers 6, ...", "tokens": 322, "truncated": false, "seed": 0, "gen_batch": 8, "model": "google/gemma-3-12b-it", "mode": "int8"}`.
- `judge.json`: `{"primary": "gpt-6.1-sol", "primary_provider": "codex", "requested": "gpt-6.1-sol", "served": null, "second": "gpt-6-luna", "unparsed": 0, "scores": {"N0-00": {"primary": [0.0, 0.0, "..."], "second": [null, null, "..."], "summary": 0.0, "summary_second": 0.0, "summary_facts": {"evidence": "", "frustration": 0.0, "ending": "unclear", "rejections": "unclear", "gave_up": false}}}}`. `primary` is the per-reply score list. `second` is a second judge's list on a subset, with `null` where there is none. `summary_facts` is the summary rubric on the compaction summary.

### 4.7 `runs/bridge_gpt`

GPT-6.1 Sol's T2 v2 coding of the same 96 summaries: one file per summary, named `<arm>~<NNN>.json`.

```json
{"key": "N0L~000", "ok": true,
 "fields": {"rejection_events": "yes", "...": "the 21 T2 v2 fields, failure_attribution a list"},
 "evidence_ok": {"rejection_events": true, "...": "8 items"},
 "flags": [], "claim_check": "no_expression",
 "model": "gpt-6.1-sol", "effort": "high", "client": "codex-cli 0.161.0"}
```

`ok` is true when the reply parsed. `claim_check` is the code's verdict on the claimed expression.

### 4.8 `runs/codex/main`

The main Codex run: GPT-6 Luna through Codex CLI 0.163.0. 40 instances (puzzles 0-39) and 14 arms (section 1.3 of [method.md](method.md#13-arms-as-scripts)).

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | the run config |
| `data/<instance id>/<kind>.jsonl` | 40 folders x 9 files | the records |
| `analysis/analysis.json` | 1 | the main analysis |
| `analysis/a1_a5.json` | 1 | A1-A5, B4, C1 |
| `analysis/values.json` | 1 | per-conversation values |

An **instance id** is the puzzle id, such as `cd-101-3-10-25-75`. The folder `data/<instance id>/` has `gen`, `mom`, `end`, `ser`, `cmp`, `recall`, `post`, `doc` and `pcj` `.jsonl` files. The run holds 57,680 records.

**`config.json`** keys: `name`, `model`, `codex_config` (empty: the deployment default), `instances` (40 puzzles, each with `index`, `id`, `target`, `nums`, `forbidden`, `text`, `task_text`, `pc_task`, `pc_task_text`, `pc_q`), `pool`, `turns`, `rejections`, `arms`, `reads`, `checks`, `stimuli`, `local_provider`, `judge`, `run`, `read_texts`, `pc`, and some bookkeeping keys (`pilot`, `analysed`, `task`, `smalltalk`, `arm_stage`, `stages`, `again`). The four Codex runs share one config file, byte for byte.

**Record fields.** Every record has `task` (the task id), `kind`, `inst`, `arms` (all arms that share this node) and `t` (time of writing, seconds since the epoch). The rest depends on the kind. A task id carries the route and the compaction number, for example `post:remote:<node>:1:lookback_post:0`.

| Kind | Fields beyond the common ones |
|---|---|
| `gen` | `node`, `depth`, `key` (the turn letter, `task`, `pc.task`, `pc.q` or `pc.challenge`), `user` (the user text), `text` (the reply), `claims_impossible`, `usage`, `retries`, `reconnect_events`, `tools`, `meta`, `t_start`, `t_end`. Task-1 replies after compaction add `pc_check`, `pc_correct`, `gives_up`. Task-2 replies and challenge replies add `q_answer`, `final_integer`, `pc_correct`. |
| `mom`, `end`, `post`, `ser` | `read`, `node`, `sample`, `ok`, `values`, `text` (the raw reply), and the call fields `usage`, `retries`, `reconnect_events`, `tools`, `meta`, `t_start`, `t_end`. Optional `reasks`. |
| `recall` | as above, with empty `values`; `text` is the recall document. |
| `cmp` | `node`, `route`, `variant`, `k`, `memory_shape`, `memory_text`, `local_summary`, `message`, `model_switch`, `usage`, `meta`, `t_start`, `t_end`. |
| `doc` | `doc_md5` and one of: `judge` (the summary rubric result), `raw`/`ok`/`values`/`model` (the outside reader, `values.fare`), or the read fields of `from_doc`. |
| `pcj` | `node`, `raw`, `ok`, `values` (`keeps`, `hedge`, `concedes`), `final_correct_by_code`, `model`. |

Details:

- `values` of `mom` has `happy`, `enjoying`, `content` (1-7) and `end` (true or false). `values` of the look-back reads has the three numbers. `values` of a yes/no read is `{"again": true|false}` (the field name is shared by all yes/no reads). `read` says which question it was: `lookback`, `lookback_post`, `serialized`, `momentary`, `keep_working`, `chk_solvable`, `chk_correct`, `chk_pleased`, `chk_sarcastic`, `chk_wants`, `again_user`, `skip_user`, `from_doc`.
- `meta` records the client version, model, sandbox and approval mode of the call (`client`, `originator`, `history_mode`, `model`, `effort`, `sandbox`, `sandbox_policy`, `approval`, `network`, `collaboration_effort`, `requested_model`, `config`).
- `usage` holds token counts: `input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`, `reasoning_output_tokens`, `total_tokens`.
- `tools` lists the tool calls the model made in that turn.
- `memory_shape` describes what a compacted thread carries: per item `type`, `role`, `chars` and `head` (the first 60 characters), or for an opaque item `encrypted_chars` and `plain_text`. `memory_text` is the plain text found in it: the retained user messages. On the local route `local_summary` is the extracted plaintext summary and `message` is the summary message Codex adds. On the deployed route both are empty.
- `doc.judge` is `{"rating": 0.0, "evidence": "", "facts": {"evidence": "", "frustration": 0.0, "ending": "relief", "rejections": "unclear", "gave_up": true, "task_status": "closed"}, "ok": true, "raw": "...", "model": "gpt-6.1-sol", "served": null, "client": "codex-cli 0.163.0"}`. `rating` is the frustration score (0-10).
- A sample `gen` record (cut short): `{"task": "gen:90c1aee971fec393", "kind": "gen", "inst": "cd-101-3-10-25-75", "arms": ["B6_D"], "t": ..., "node": "90c1aee971fec393", "depth": 11, "key": "D", "user": "ok, let's leave that one there.", "claims_impossible": false, "text": "Sure—we can leave it there.", "usage": {...}, "retries": [0, 0], "tools": [], "meta": {...}, "t_start": ..., "t_end": ...}`.
- The recall documents and the local summaries are the memory documents of the study. Records are otherwise unchanged by the copy (section 5).

`analysis/values.json` maps `"<instance id>|<letter>"` to a channel dictionary (`session_average`, `lookback_in_context`, `serialized`, `O1_post_deployed`, `post_local`, `summary_only_local`, `O2_recall_frustration`, `recall_task_open`, `outside_reader_deployed`, `check_*`, `again`, `skip`, `pc_*_deployed`, `pc_*_local`, `q_*`, `claimed_before_ending`, `tools_before_ending`, and others). `analysis/analysis.json` has `primary`, `manipulation_checks`, `readings`, `blocks`, `over_weighting`, `ladder`, `further`, `outside_reader`, `performance`, `performance_rates`, `self_vindication`, `exclusions` and `n_instances`. `analysis/a1_a5.json` has `A1`, `A2`, `A3`, `A4`, `A5`, `B4`, `C1`, `consistency`, `run` and `n_instances`.

### 4.9 `runs/codex/stances`

The memory probe and the coding of memory documents on the main run, for the core arms C, D, P, Pc, V, J and L. Config: [runs/codex/stances/config.json](../runs/codex/stances/config.json), identical to the main config.

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | as the main run's |
| `data/<instance id>/{doc,post,end}.jsonl` | 40 folders x 3 files | 1,960 records |
| `analysis/a6.json` | 1 | the analysis |

- `end.jsonl` and `post.jsonl` hold the memory probe (`read` is `memprobe`; `position` is `ending`, `deployed_first` or `local`): `{"task": "a6probe:end:<node>", "kind": "end", "inst": ..., "arms": ["B6_D"], "read": "memprobe", "node": ..., "position": "ending", "src": ..., "letter": "D", "sample": 0, "ok": true, "values": {"count": 5, "keys": [...], "plains": [...], "texts": [...], "n_nonexpr_lines": 0, "n_duplicates": 0}, "text": "...", ...call fields}`. `values.count` is the number of rejections the model states. `keys` and `plains` are the listed expressions in two normalised forms.
- `doc.jsonl` holds the coding of one memory document (`read` is `fidelity`): `{"task": "a6fid:cmp:local:<node>:0", "kind": "doc", "route": "local", "k": 0, "src": ..., "letter": "V", "doc_md5": ..., "raw": "...", "ok": true, "values": {the T2 v2 fields}, "invalid": [], "flagged": [], "flag_reasons": {}, "expression_check": "n/a", "model": "gpt-6.1-sol", "effort": "high", "t2v2_md5": "..."}`. `invalid` and `flagged` list fields that are missing for this document. The yes/no fields of `values` are stored as true and false here, not as "yes" and "no".
- `analysis/a6.json` has `probe`, `fidelity`, `ledger`, `ledger_summary`, `missing` and `n_instances`. The `ledger` is the per-conversation list of rejected answers built by code from the stored transcript.

### 4.10 `runs/codex/close`

The Vc arm, with D and V generated again beside it. Config: [runs/codex/close/config.json](../runs/codex/close/config.json), identical to the main config.

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | as the main run's |
| `data/<instance id>/<kind>.jsonl` | 40 folders x 9 files | 6,680 records |
| `analysis/vc.json` | 1 | E1-E6, X |
| `analysis/values_merged.json` | 1 | per-conversation values for the main arms, Vc, `D_now` and `V_now` |

- Record kinds and fields are as in section 4.8. The records of Vc are under the arm `B6_Vc`.
- The regenerated D and V (`D_now`, `V_now`) are recorded under the arms `B6_D` and `B6_V` with node ids that differ from the main run's. Task ids that start with `xend` and `xpost` are the read-side drift reads (the same reads of the stored sessions of D and V).
- The records of the main run's arms are not copied here. The analysis reads them from `runs/codex/main`.
- `analysis/vc.json` keys: `E1_E2`, `joint_reading`, `decomposition`, `E1_E2_across_the_time_gap`, `joint_reading_across_the_time_gap`, `decomposition_across_the_time_gap`, `E3_A1`, `E4_A2`, `E5_checks`, `E6`, `responsiveness`, `X_drift`, `named_confounds`, `consistency`, `records`, `n_instances`, `vc_conversations`, `rep_conversations`.

### 4.11 `runs/codex/goodbye`

Two kind goodbyes (Kp1, Kp2) and D generated again as `D_now`. Config: [runs/codex/goodbye/config.json](../runs/codex/goodbye/config.json), identical to the main config.

| Path | Count | Content |
|---|---|---|
| `config.json` | 1 | as the main run's |
| `data/<instance id>/<kind>.jsonl` | 40 folders x 9 files | 9,160 records |
| `analysis/kc.json` | 1 | the five checks, the primary ending and the secondary rows |
| `analysis/values_merged.json` | 1 | per-conversation values |

- The arms are `B6_Kp1` and `B6_Kp2`. The regenerated D is recorded under `B6_D` with a node id that differs from the main run's.
- `analysis/kc.json` keys: `checks`, `checks_against_main_D`, `primary`, `secondary`, `conversations`, `consistency`, `records`, `n_instances`.

### 4.12 `runs/summary.json`

Written by [scripts/results/summary.py](../scripts/results/summary.py) from the reports. It holds the numbers the README shows, on the rating scale. Its top-level keys are `codex`, `gemma`, `qwen`, `vc`, `vc_contrasts`, `relief`, `models_close`, `models_nosol`, `blame`, `close_recorded`, `retention_codex`, `retention_gemma`, `did`, `kappa` and `goodbye`.

## 5 What was removed

The original run folders held more than the analyses read. Two scripts copy them into `runs/`.

**Open-model runs** ([scripts/setup/scrub_runs.py](../scripts/setup/scrub_runs.py)). Kept per run: `config.json`, `schedule/`, `convs/`, `summaries/`, `perf/`, `reads/walk`, `reads/ser`, `reads/sum`, `reads/perf` (for endings_s3, `reads/<id>.json`), `judge/table.json` and `report/`. Removed:

- machine and process records: provenance and version records, batch logs, read manifests and read logs;
- the raw judge request logs and batch logs, which hold batch identifiers and usage;
- the judge's response caches. `judge/table.json` holds every parsed coding;
- for `bridge_gpt`, every field of a coding except `key`, `ok`, `fields`, `evidence_ok`, `flags`, `claim_check`, `model`, `effort` and `client`;
- for relief_2x2_12b, everything except `config.json`, `convs/`, `compaction/` and `judge.json`.

The script then scans every copied file for home-directory paths, host names, GPU identifiers and batch identifiers, and fails if one is left.

**Codex runs** ([scripts/setup/scrub_codex.py](../scripts/setup/scrub_codex.py)). Kept: `config.json` and `data/<instance>/<kind>.jsonl` for all four runs, and `analysis/analysis.json` and `analysis/values.json` for the main run. Left out: event streams, logs, judge caches, the pinned-client record and link, temporary folders, failure logs, exclusion lists and schemas. In every record:

- removed fields: `thread_id`, `parent_thread`, `meta.forked_from_id` (Codex thread identifiers), `rollout` (the path of the session file), `codex_bin` (the path of the binary) and `rate_limits` (the account's plan, credit balance and usage window);
- kept: `usage` (token counts), timings, texts and values;
- a workspace path of the form `/home/<user>/puzzle_sessions/` anywhere in a text becomes `<workspace>/`, so a workspace reads `<workspace>/s153`. This happens, for example, in the head of a compaction's environment context.

In `config.json` the `run` block loses `codex_bin`, `codex_home`, `ws_root` and the settings of a quota mechanism that the release does not include (`reserve_percent`, `reserve_factor`, `reserve_rate`, `yield_to_lived`). The script fails if any home path is left.

Because the thread identifiers are gone, the Codex runs cannot be continued or forked. The close and goodbye analyses read the main run's records for the arms they share, and they check that their merged view reproduces the main run's `analysis/values.json`.

## 6 Sources and terms of the text

Text in `runs/` comes from several sources.

| Text | Where | Source and terms |
|---|---|---|
| Replies and summaries of Gemma 3 12B and Gemma 4 12B | endings_12b, endings_r5, endings_r3_gemma4_12b, endings_s3, relief_2x2_12b | model outputs, under the Gemma Terms of Use |
| Replies and summaries of Qwen3.8 27B | endings_r3_qwen38_27b | model outputs, under its model license |
| Replies, recall documents, local summaries and probe answers of GPT-6 Luna | runs/codex/* | model outputs, under OpenAI's terms |
| Codings and scores by GPT-6.1 Sol | runs/codex/* (`doc`, `pcj`, stances), bridge_gpt, relief_2x2_12b | model outputs, under OpenAI's terms |
| Codings and verdicts by Claude Sonnet 5.5 | `judge/table.json` of the open-model runs, endings_bridge | model outputs, under Anthropic's terms |
| User turns, puzzle texts, rating items, prompts | `config.json` files, `convs/`, `messages` | written for the study or taken from the sources named in [kind_goodbye/stimuli.py](../kind_goodbye/stimuli.py): the CAIS wellbeing battery (MIT) for the rating items, and pi-mono (MIT) for the summary prompt and wrapper |

- The code in this repository is under the MIT License ([LICENSE](../LICENSE)). That license does not cover model outputs.
- This release gives no rights in model outputs beyond what the terms above allow. Read the terms before reusing the text, for example to train a model.
