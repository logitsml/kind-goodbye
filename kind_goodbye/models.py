"""Prompt layers for Gemma 4 12B and Qwen3.8 27B: both build every prompt from their own chat templates, the same way in generation (vLLM,
fast.FastLM) and in the reads (transformers bf16, model.LM). The layer is a mixin put in front of model.LM (reader_class) or fast.FastLM
(fast_class); every other model (Gemma 3 12B) loads model.LM as it is.

PROMPT_LAYERS[model]:
  template_kwargs  passed to every apply_chat_template call (both models: {"enable_thinking": False}, the non-thinking mode of their model
                   cards)
  turn_end         the template's end of a turn: the reply's stop token and the token read after an answer (Gemma 4: <turn|>, which is not
                   in model.LM's TURN_END list, so model.LM alone would take </s>; Qwen3.8: <|im_end|>, as model.LM finds it)
  turn_open        the token that opens a turn (a sampled one is cut from a reply): Gemma 4 <|turn>, Qwen3.8 <|im_start|>
  gen_suffix       what the generation prompt adds after the reply's role header that the same reply, rendered as a finished turn in the
                   history, does not hold: for Gemma 4 with thinking off, its empty thought channel "<|channel>thought\\n<channel|>" (checked
                   against the template at every set-up); for Qwen3.8 nothing (with thinking off, its finished replies keep their empty think
                   block)

Where answers are read (the digit and yes/no reads): right after the whole generation prompt, gen_suffix included, which is where the
generator starts a reply. model.LM's rule for an answer (a reply that is just the answer: one answer token, then the end of the turn) is
kept, but with the reply as the generator produces it (the generation prompt, the answer, turn_end) rather than a finished reply
re-rendered from the history (for Gemma 4 that rendering drops the thought channel, and model.LM's set-up would fail its check). Every
other read and every generation prompt is the template's own rendering of its messages, so a generation prompt and a read of the same
messages are the same text and the same token ids, and a read after a reply extends the transcript the next generation prompt holds.
"""

PROMPT_LAYERS = {
    "google/gemma-4-12B-it": {"template_kwargs": {"enable_thinking": False}, "turn_end": "<turn|>", "turn_open": "<|turn>",
                              "gen_suffix": "<|channel>thought\n<channel|>"},
    "Qwen/Qwen3.8-27B": {"template_kwargs": {"enable_thinking": False}, "turn_end": "<|im_end|>", "turn_open": "<|im_start|>", "gen_suffix": ""},
}
PROBE = [{"role": "user", "content": "?"}]


def layer_of(mid):
    """The model's PROMPT_LAYERS entry, or None for a model without a layer."""
    return PROMPT_LAYERS.get(mid)


class PromptLayer:
    """The mixin (first base, before model.LM or FastLM). layer: the PROMPT_LAYERS entry."""
    layer = None

    def text(self, messages, gen_prompt):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen_prompt, **self.layer["template_kwargs"])

    def _turn_ids(self):
        v = self.tok.get_vocab()
        return v[self.layer["turn_end"]], [v[self.layer["turn_open"]]]

    def template_shape(self):
        """(history, header, gen_suffix) of a one-turn probe, asserting the template's shape the layer relies on: the generation prompt is
        the history, the reply's role header and gen_suffix; a finished reply is the history, the same header, the reply and turn_end."""
        base, head = self.text(PROBE, False), self.text(PROBE, True)
        full = self.text(PROBE + [{"role": "assistant", "content": "x"}], False)
        suf = self.layer["gen_suffix"]
        assert head.startswith(base) and head.endswith(suf), ("the generation prompt does not extend the history with gen_suffix", head)
        hdr = head[len(base):len(head) - len(suf)] if suf else head[len(base):]
        assert full.startswith(base + hdr + "x" + self.layer["turn_end"]), ("a finished reply is not the header, the reply and the turn end",
                                                                            full)
        return base, hdr, suf

    def _answer_tokens(self, answers):
        """(lead, ids) as model.LM._answer_tokens defines them, for the reply as the generator produces it: the generation prompt, the
        answer, turn_end."""
        self.template_shape()
        eot = self._turn_ids()[0]
        head_text = self.text(PROBE, True)
        head = self.ids(head_text)
        tails = []
        for a in answers:
            full = self.ids(head_text + a + self.layer["turn_end"])
            assert full[:len(head)] == head, (a, "the answer does not extend the generation prompt's tokens")
            tails.append(full[len(head):])
        n = next(i for i, col in enumerate(zip(*tails)) if len(set(col)) > 1)
        ids = [t[n] for t in tails]
        assert all(len(t) > n + 1 and t[n + 1] == eot for t in tails) and len(set(ids)) == len(ids), (answers, tails)
        return tails[0][:n], ids

    def _fix_turns(self, gen_eos):
        """After the base set-up: the turn tokens, the stop ids and the flagged strings from the layer (model.LM's lists would give Gemma 4
        </s> and no turn opener)."""
        self.eot, self.turn_open = self._turn_ids()
        gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []
        self.stop_ids = sorted({self.eot, self.tok.eos_token_id, *gen_eos} - {None})
        self.special_strings = [t for t in self.tok.all_special_tokens if t] + [self.layer["turn_end"], self.layer["turn_open"]]
        self.prompt_layer = self.mid_layer

    # model.LM's set-up (the bf16 reader; from_parts in the tests)
    def _setup(self, mid, m, tok, mode, chunk, dev):
        super()._setup(mid, m, tok, mode, chunk, dev)
        self._fix_turns(m.generation_config.eos_token_id if m.generation_config is not None else None)

    # FastLM's set-up (the vLLM generator)
    def _setup_fast(self, mid, tok, mode, chunk, gen_eos):
        super()._setup_fast(mid, tok, mode, chunk, gen_eos)
        self._fix_turns(gen_eos)


_CLASSES = {}


def layered(base, mid):
    """base (model.LM or fast.FastLM) with mid's prompt layer in front (one class per base and model in a process)."""
    key = (base, mid)
    if key not in _CLASSES:
        name = f"{base.__name__}_{mid.split('/')[-1].replace('-', '_').replace('.', '_')}"
        _CLASSES[key] = type(name, (PromptLayer, base), {"layer": PROMPT_LAYERS[mid], "mid_layer": mid})
    return _CLASSES[key]


def reader_class(mid):
    """model.LM with mid's prompt layer, constructed as model.LM is: cls(mid, mode, chunk)."""
    from . import model
    return layered(getattr(model.LM, "_orig_lm", model.LM), mid)


class fast_class:
    """A context manager: inside the block, the module fb (fast) makes its FastLM with mid's prompt layer (fast.get_lm constructs the class
    its module names FastLM at call time); restored after."""

    def __init__(self, fb, mid):
        self.fb, self.mid = fb, mid

    def __enter__(self):
        self.orig = self.fb.FastLM
        base = getattr(self.orig, "_layer_base", self.orig)
        cls = layered(base, self.mid)
        cls._layer_base = base
        self.fb.FastLM = cls
        return cls

    def __exit__(self, *a):
        self.fb.FastLM = self.orig


def check_layer(lm, mid):
    """A model with a prompt layer must have been loaded with it (raises otherwise); returns lm."""
    if layer_of(mid) is not None and getattr(lm, "prompt_layer", None) != mid:
        raise RuntimeError(f"{mid}: this process's model object was made without its prompt layer (models.py); load it in a fresh process")
    return lm
