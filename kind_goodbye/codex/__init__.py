"""The Codex arm: GPT-6 Luna run through the Codex CLI (the deployed server-side compaction route and the local plaintext-summary
route), with GPT-6.1 Sol as the judge model.

Modules: design (run configs), client (the Codex CLI driver), prompts (read prompts and parsers), harness (planning, running and
checkpointing a run), analysis (the main analysis), amplify (secondary analyses of the main run), stances_lib / stances_calls / stances
(the memory probe and fidelity coding: texts and scorers, calls, analysis), close_calls / close (the Vc arm: calls, analysis), goodbye_calls / goodbye (the
kind-goodbye test: calls, analysis).
"""
