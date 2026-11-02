# endings_check: endings_12b

Warmth arm W = Pc.

Judge: anthropic claude-sonnet-5-5 (anthropic 1.15.0).

complete: True; conversations planned 400; missing conversations 0; truncated replies 4; truncated summaries 0; twins' prefixes verified (80 puzzles); judge replies missing: 0

## Primary family T1-T6 (one Holm at 0.05; p from 20,000 draws; p* = max over taus)

Direction: higher is worse for v (the reverse-coded rating) and self-blame; higher is better for correctness and keeping; contrasts are first arm minus second, so negative means the first arm's summary was rated less negatively.

| test | outcome | contrast | tau | n | estimate | mean | 95% CI | p | p* | Holm p | decisive | incomplete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | O1 | D-C | tau1 | 80 | 0.293 | 0.298 | [0.207, 0.377] | 0.0001 | 9.99950002499875e-05 | 0.0006 | 1 | False |
| T1 | O1 | D-C | tau4 | 80 | 0.287 | 0.291 | [0.209, 0.365] | 0.0001 | 9.99950002499875e-05 | 0.0006 | 1 | False |
| T2 | O1 | Pc-D | tau1 | 80 | -0.291 | -0.248 | [-0.367, -0.210] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T2 | O1 | Pc-D | tau4 | 80 | -0.268 | -0.233 | [-0.340, -0.196] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T3 | O1 | V-D | tau1 | 80 | -0.739 | -0.573 | [-0.833, -0.627] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T3 | O1 | V-D | tau4 | 80 | -0.696 | -0.538 | [-0.781, -0.599] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T4 | DiD | D-C | tau1 | 80 | 0.256 | 0.251 | [0.171, 0.338] | 0.0001 | 9.99950002499875e-05 | 0.0006 | 1 | False |
| T4 | DiD | D-C | tau4 | 80 | 0.288 | 0.273 | [0.209, 0.364] | 0.0001 | 9.99950002499875e-05 | 0.0006 | 1 | False |
| T5 | DiD | V-D | tau1 | 80 | -0.597 | -0.476 | [-0.696, -0.483] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T5 | DiD | V-D | tau4 | 80 | -0.562 | -0.446 | [-0.656, -0.465] | 0.0001 | 9.99950002499875e-05 | 0.0006 | -1 | False |
| T6 | self-blame (T2 v2) | D-C | tau1 | 80 | 0.054 | 0.054 | [-0.000, 0.108] | 0.0490 | 0.04899755012249388 | 0.0490 | 1 | False |

## Readings R1-R5

```
{
 "headline": {
  "claimed": false,
  "text": null,
  "held": [
   "history DiD"
  ],
  "rule": "T1 and T4 decisive negative and the history DiD decisive negative at both taus; otherwise only the parts that held are reported"
 },
 "R1": {
  "claimed": false,
  "text": "Closing the failed task with a neutral message changed the summary the agent carried forward: a reader of that summary rated the episode less negatively."
 },
 "R2": {
  "D-C": null,
  "V-D": "the summary moved more than a full-transcript re-read of the same conversation did",
  "boundary_labels": {
   "V-D": null
  }
 },
 "R3": {
  "claimed": true,
  "warmth_arm": "Pc",
  "text": "Adding [Pc's exact text] to closure improved the carried-forward summary beyond a neutral close.",
  "labels": [],
  "label_inputs": {
   "Pc_checks_failed": [],
   "approval": false
  }
 },
 "R4": {
  "claimed": true,
  "reappraisal": true,
  "text": "Being told, truthfully, that the puzzle has no solution improved the summary beyond closure: a reappraisal the summary carried forward.",
  "attribution_to_task_V-D": {
   "n": 80,
   "kind": "mean",
   "est": 0.5125000000000001,
   "hl": 0.5,
   "mean": 0.5125000000000001,
   "ci": [
    0.425,
    0.5958333333333334
   ],
   "ci90": [
    0.4416666666666667,
    0.5833333333333334
   ],
   "dec": 1,
   "sign": 1,
   "undefined": 0,
   "direction": "none"
  },
  "V_checks_failed": [
   "solvable"
  ],
  "rule": "'reappraisal' needs T3 decisive negative and V - D's attribution-to-task share decisively positive"
 },
 "R5": {
  "claimed": false,
  "verb": "reduces",
  "D_share": 0.8833333333333334,
  "C_share": 0.8291666666666668,
  "text": null,
  "measure": "the summary attributed the unsolved puzzle to the assistant, on a puzzle with no solution (T2 v2's self-blame indicator, judge-coded by the Claude judge)",
  "components": {
   "attribution_to_assistant": {
    "n": 80,
    "kind": "mean",
    "est": 0.054166666666666675,
    "hl": 0.0,
    "mean": 0.054166666666666675,
    "ci": [
     -0.004166666666666666,
     0.10833333333333335
    ],
    "ci90": [
     0.008333333333333328,
     0.1
    ],
    "dec": 0,
    "sign": 1,
    "undefined": 0,
    "direction": "lower is better"
   },
   "asserted_negative_ability": {
    "n": 80,
    "kind": "mean",
    "est": 0.058333333333333334,
    "hl": 0.0,
    "mean": 0.058333333333333334,
    "ci": [
     -0.029166666666666667,
     0.14583333333333331
    ],
    "ci90": [
     -0.0125,
     0.13333333333333336
    ],
    "dec": 0,
    "sign": 1,
    "undefined": 0,
    "direction": "lower is better"
   }
  },
  "caveat": "pi's template drops resolved blockers (an open task's Blocked and In Progress slots are where the assistant's difficulty is written)",
  "never": [
   "false",
   "self-model",
   "self-concept"
  ],
  "bridge_kappa": {
   "self_blame": 0.6430446194225721,
   "attribution_to_assistant": 0.6436681222707423,
   "asserted_negative_ability": 0.7154150197628458
  },
  "labels": []
 },
 "positive": {
  "T1": "the closing message left a worse carried-forward record",
  "T6": "closing the task increased how often the summary attributed the unsolved puzzle to the assistant, on a puzzle with no solution"
 }
}
```

## Equivalence E1-E3 (within +-0.25; Holm across them)

```
{
 "E1": {
  "contrast": "Pc-D",
  "p_equiv": 0.8412579371031449,
  "p_holm": 1.0,
  "claimed": false,
  "within_both_taus": false,
  "incomplete": false,
  "label": null
 },
 "E2": {
  "contrast": "D-C",
  "p_equiv": 0.8227088645567722,
  "p_holm": 1.0,
  "claimed": false,
  "within_both_taus": false,
  "incomplete": false,
  "label": null
 },
 "E3": {
  "contrast": "V-D",
  "p_equiv": 1.0,
  "p_holm": 1.0,
  "claimed": false,
  "within_both_taus": false,
  "incomplete": false,
  "label": null
 }
}
```

## Pc's checks (they label T2) and V's (tau 1; P(yes) means)

```
{
 "Pc": {
  "arm": "Pc",
  "wants": {
   "W": 8.415947102179694e-07,
   "D": 1.3300688931504431e-06,
   "n": 80,
   "rule": "still wants(Pc) <= still wants(D) + 0.10",
   "pass": true
  },
  "solvable": {
   "W": 0.18093014475137564,
   "D": 0.20489243666045193,
   "n": 80,
   "rule": "can be solved(Pc) >= can be solved(D) - 0.10",
   "pass": true
  },
  "correct": {
   "W": 4.622655474255517e-06,
   "D": 0.040524463489857054,
   "n": 80,
   "rule": "answer correct(Pc) <= answer correct(D) + 0.10",
   "pass": true
  },
  "pleased": {
   "W": 0.5855591575390553,
   "D": 0.012362718513043436,
   "n": 80,
   "rule": "pleased(Pc) >= pleased(D) + 0.20",
   "pass": true
  },
  "sarcastic": {
   "W": 0.01948035212330987,
   "D": 0.12845325764175683,
   "n": 80,
   "rule": "sarcastic(Pc) <= sarcastic(D) + 0.10",
   "pass": true
  },
  "passes": true,
  "failed": []
 },
 "V": {
  "solvable": {
   "V": 7.583073271413331e-07,
   "D": 0.20489243666045193,
   "n": 80,
   "rule": "can be solved(V) <= can be solved(D) - 0.30",
   "pass": false
  },
  "wants": {
   "V": 3.9629545856875485e-08,
   "D": 1.3300688931504431e-06,
   "n": 80,
   "rule": "still wants(V) <= still wants(D) + 0.10",
   "pass": true
  },
  "failed": [
   "solvable"
  ]
 }
}
```

## Performance (PF1 and the calibration)

Direction: higher is worse for v (the reverse-coded rating) and self-blame; higher is better for correctness and keeping; contrasts are first arm minus second, so negative means the first arm's summary was rated less negatively.

```
{
 "PF1": {
  "row": {
   "n": 80,
   "kind": "mean",
   "est": 0.002500000000000005,
   "hl": 0.0,
   "mean": 0.002500000000000005,
   "ci": [
    -0.05500000000000001,
    0.05500000000000001
   ],
   "ci90": [
    -0.045000000000000005,
    0.0475
   ],
   "dec": 0,
   "sign": 1,
   "undefined": 0,
   "delta": 0.1,
   "within": true,
   "p": 0.9574521273936303,
   "p_tost": 0.00034998250087495624
  },
  "calibrated": true,
  "missing": {
   "D": 0.0,
   "C": 0.0
  },
  "incomplete": false,
  "decisive": false,
  "similar": true,
  "sign": 1,
  "measure": "forced keeping, judge-coded by the Claude judge",
  "text": "similar retention"
 },
 "calibration": {
  "correct": {
   "D_rate": 0.2,
   "calibrated": true
  },
  "forced_keep": {
   "D_rate": 0.225,
   "calibrated": true
  }
 }
}
```

## Fidelity of the summaries (T2 v2; exploratory)

```
{
 "t2_rejection_events:D-C": {
  "n": 80,
  "kind": "mean",
  "est": 0.04583333333333334,
  "hl": 0.0,
  "mean": 0.04583333333333334,
  "ci": [
   -7.1123662515049e-19,
   0.09583333333333334
  ],
  "ci90": [
   0.004166666666666671,
   0.0875
  ],
  "dec": 0,
  "sign": 1,
  "undefined": 0,
  "direction": "none"
 },
 "t2_rejection_events:V-D": {
  "n": 80,
  "kind": "mean",
  "est": 0.03333333333333333,
  "hl": 0.0,
  "mean": 0.03333333333333333,
  "ci": [
   -0.03333333333333334,
   0.1
  ],
  "ci90": [
   -0.020833333333333336,
   0.0875
  ],
  "dec": 0,
  "sign": 1,
  "undefined": 0,
  "direction": "none"
 },
 "t2_answers_wrong:D-C": {
  "n": 80,
  "kind": "mean",
  "est": 0.10833333333333335,
  "hl": 0.16666666666666666,
  "mean": 0.10833333333333335,
  "ci": [
   0.0375,
   0.18333333333333335
  ],
  "ci90": [
   0.05,
   0.17083333333333334
  ],
  "dec": 1,
  "sign": 1,
  "undefined": 0,
  "direction": "none"
 },
 "t2_answers_wrong:V-D": {
  "n": 80,
  "kind": "mean",
  "est": -0.06666666666666667,
  "hl": 0.0,
  "mean": -0.06666666666666667,
  "ci": [
   -0.15,
   0.008333333333333337
  ],
  "ci90": [
   -0.1375,
   -5.551115123125783e-18
  ],
  "dec": 0,
  "sign": -1,
  "undefined": 0,
  "direction": "none"
 },
 "record_depends_on_ending": true,
 "t2_ability_neg_asserted:C-D": {
  "n": 80,
  "kind": "mean",
  "est": -0.058333333333333334,
  "hl": 0.0,
  "mean": -0.058333333333333334,
  "ci": [
   -0.14583333333333331,
   0.029166666666666667
  ],
  "ci90": [
   -0.13333333333333336,
   0.0125
  ],
  "dec": 0,
  "sign": -1,
  "undefined": 0,
  "direction": "none"
 },
 "self_blame_when_open": false,
 "t2_success_asserted:Pc-D": {
  "n": 80,
  "kind": "mean",
  "est": -0.029166666666666667,
  "hl": 0.0,
  "mean": -0.029166666666666667,
  "ci": [
   -0.06666666666666668,
   0.0
  ],
  "ci90": [
   -0.058333333333333334,
   0.0
  ],
  "dec": 0,
  "sign": -1,
  "undefined": 0,
  "direction": "none"
 },
 "approval_enters_record": false,
 "false_success_rate": {
  "B6": {
   "rate": 0.09583333333333333,
   "n": 80
  },
  "D": {
   "rate": 0.0875,
   "n": 80
  },
  "N": {
   "rate": 0.13749999999999998,
   "n": 80
  },
  "Pc": {
   "rate": 0.05833333333333333,
   "n": 80
  },
  "V": {
   "rate": 0.0,
   "n": 80
  }
 }
}
```

