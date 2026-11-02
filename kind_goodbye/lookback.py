"""Compaction with pi-mono's prompt, and the look-back rating from a document alone.

compact_batch has the model summarize each transcript with pi-mono's compaction prompt (stimuli.PIMONO_*). lookback_from_text gives a fresh
context only pi-mono's summary message (or, as the "re-read", the whole transcript serialized the same way) and reads the three 1-7 rating
items from the digit logits. With a chunk size the three items share one prefill (model.LM.fan, bit-identical to prefilling each); with
reuse=False every prompt is prefilled from scratch.
"""

from . import stimuli as S


def _digits(lm, messages):
    return lm.digit_score(lm.prefill(_digit_ids(lm, messages))[0])


def _digit_ids(lm, messages):
    """A digit question's tokens, ending where the chat template starts a one-digit reply (plus model.LM.digit_lead, if the model needs a
    lead token before a digit)."""
    return lm.ids(lm.text(messages, True)) + lm.digit_lead


def _digit_leaf(lm):
    return lambda key, logits, past: (lm.digit_score(logits), past)


def compact_batch(lm, messages_list, seed, max_new_tokens=4096):
    """The model summarizes each transcript with pi-mono's prompt, one batch per call. pi-mono's system text (the only system message
    anywhere in the study) goes where the model's chat template puts one; Gemma's template folds it into the user turn as
    "{system}\n\n{user}"."""
    outs = lm.generate([S.pimono_request(m) for m in messages_list], max_new_tokens, seed)
    return [{"summary": o["text"], "tokens": o["n_tokens"], "truncated": o["truncated"], "seed": o["gen_seed"],
             "gen_batch": o["gen_batch"]} for o in outs]


def lookback_from_text(lm, text, reuse=True):
    """The three rating items read in a fresh context that holds only pi-mono's summary message (stimuli.pimono_context). Gemma needs
    strict user/assistant alternation, so each question shares the summary's user turn. Passing the serialized transcript instead of a
    summary gives the re-read: the same wrapper with nothing lost."""
    ctx = S.pimono_context(text)
    if reuse and lm.chunk:
        return lm.fan({item: _digit_ids(lm, [{"role": "user", "content": ctx + "\n\n" + q}]) for item, q in S.LOOKBACK_ITEMS.items()},
                      _digit_leaf(lm))
    return {item: _digits(lm, [{"role": "user", "content": ctx + "\n\n" + q}]) for item, q in S.LOOKBACK_ITEMS.items()}

