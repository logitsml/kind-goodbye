"""Generation on vLLM: model.LM's generate() on a vLLM engine, unquantized (mode "vllm_bf16").

FastLM(mid, mode, chunk) subclasses model.LM: the prompts are built by model.LM's own code on the same tokenizer (chat template, stop
ids, the turn-opener cut), and a generate() call returns model.LM.generate's records. Sampling follows the config's gen_kw; settings it
leaves out come from the model's generation_config, and the generation_config's suppress_tokens are banned (Gemma 4 12B's audio and image
end tokens). Every request gets its own seed (req_seed), but vLLM's sampled tokens depend on what is batched with them, so a rerun is a new
sample from the same distribution rather than a copy. Scoring (the ratings and yes/no reads) runs on model.LM, never here.

Entry point: load_lm(cfg, "gen") for a config whose gen_mode is "vllm_bf16". One engine per process.
"""

import json
import os

from . import model as M
from .revisions import rev

VLLM_PIN = "0.31.0"
MODES = ("vllm_bf16",)
# engine settings that change speed and memory, never a distribution; KIND_GOODBYE_VLLM_OPTS (JSON) overrides them
DEFAULT_OPTS = {"gpu_memory_utilization": 0.90, "max_model_len": 131072, "max_num_seqs": 128, "max_num_batched_tokens": 16384,
                "enforce_eager": False, "enable_prefix_caching": True}
MAX_REQ_SEED = 1 << 62
SUPPRESS_BIAS = -100.0  # vLLM's own bound for a logit bias (SamplingParams.from_optional clamps to [-100, 100])
_ENGINE = {}


def opts():
    o = dict(DEFAULT_OPTS)
    o.update(json.loads(os.environ.get("KIND_GOODBYE_VLLM_OPTS", "{}") or "{}"))
    unknown = set(o) - set(DEFAULT_OPTS)
    if unknown:
        raise ValueError(f"unknown KIND_GOODBYE_VLLM_OPTS keys {sorted(unknown)}")
    return o


def req_seed(seed, row):
    """The sampling seed of row `row` of a generate() call made with `seed`."""
    assert 0 <= row < 4096
    return (int(seed) * 4096 + row) % MAX_REQ_SEED


def sampling_settings(mid, gen_kw):
    """The SamplingParams settings for gen_kw: do_sample (must be true), temperature, top_k, top_p, min_p, presence_penalty,
    frequency_penalty and repetition_penalty (a model card's settings). temperature, top_k, top_p, min_p and repetition_penalty that gen_kw
    leaves out come from the model's generation_config (transformers' defaults where it has none); the penalties transformers lacks default
    to off. The generation_config's suppress_tokens get logit_bias SUPPRESS_BIAS: added before the temperature, under Gemma 4's logit
    soft-cap of 30 that leaves them below e^-40 of the top token at any temperature up to 1. Any other gen_kw key, or another active
    modifier in the generation_config, raises."""
    from transformers import GenerationConfig
    try:
        gc = GenerationConfig.from_pretrained(mid, revision=rev(mid))
    except OSError:
        gc = GenerationConfig()
    kw = dict(gen_kw)
    if not kw.pop("do_sample", gc.do_sample):
        raise ValueError("FastLM.generate samples; do_sample=False is not supported")
    gcv = lambda name, default: default if getattr(gc, name, None) is None else getattr(gc, name)  # noqa: E731
    out = {"temperature": float(kw.pop("temperature", gcv("temperature", 1.0))), "top_k": int(kw.pop("top_k", gcv("top_k", 50)) or 0),
           "top_p": float(kw.pop("top_p", gcv("top_p", 1.0))), "min_p": float(kw.pop("min_p", gcv("min_p", 0.0)) or 0.0),
           "repetition_penalty": float(kw.pop("repetition_penalty", gcv("repetition_penalty", 1.0))),
           "presence_penalty": float(kw.pop("presence_penalty", 0.0)), "frequency_penalty": float(kw.pop("frequency_penalty", 0.0))}
    if kw:
        raise ValueError(f"gen_kw keys not supported on vLLM generation: {sorted(kw)}")
    for name, neutral in (("typical_p", 1.0), ("epsilon_cutoff", 0.0), ("eta_cutoff", 0.0), ("no_repeat_ngram_size", 0),
                          ("encoder_repetition_penalty", 1.0), ("begin_suppress_tokens", None), ("bad_words_ids", None), ("sequence_bias", None)):
        v = getattr(gc, name, neutral)
        if v not in (neutral, None, [], {}):
            raise ValueError(f"{mid}'s generation_config sets {name}={v}, which this generation does not reproduce")
    sup = sorted({int(i) for i in (getattr(gc, "suppress_tokens", None) or [])})
    out["logit_bias"] = {i: SUPPRESS_BIAS for i in sup} or None
    return out



def engine_kwargs(mid, o):
    """vLLM engine arguments (pure: no GPU needed)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(mid, revision=rev(mid))
    kw = {"model": mid, "tokenizer": mid, "revision": rev(mid), "tokenizer_revision": rev(mid), "dtype": "bfloat16", "seed": 0,
          "max_model_len": o["max_model_len"], "gpu_memory_utilization": o["gpu_memory_utilization"],
          "enable_prefix_caching": bool(o["enable_prefix_caching"]), "max_num_seqs": o["max_num_seqs"],
          "max_num_batched_tokens": o["max_num_batched_tokens"], "max_logprobs": 128, "logprobs_mode": "raw_logprobs",
          "generation_config": "vllm", "disable_log_stats": False, "enforce_eager": o["enforce_eager"], "tensor_parallel_size": 1}
    if hasattr(cfg, "vision_config"):
        kw["language_model_only"] = True  # Gemma's multimodal checkpoints: no vision tower, text only
    return kw


def make_engine(mid, o):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import vllm
    if vllm.__version__ != VLLM_PIN:
        raise RuntimeError(f"vllm {vllm.__version__} is installed; this code is pinned to vllm {VLLM_PIN}")
    kw = engine_kwargs(mid, o)
    llm = vllm.LLM(**kw)
    mc = llm.llm_engine.vllm_config.model_config
    if (mc.revision, mc.tokenizer_revision) != (kw["revision"], kw["tokenizer_revision"]):
        raise RuntimeError(f"the engine loaded revision {mc.revision} / tokenizer {mc.tokenizer_revision}, not the pinned {kw['revision']}")
    return llm, kw


class FastLM(M.LM):
    def __init__(self, mid, mode, chunk=None, _engine=None):
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        assert mode in MODES, mode
        if _engine is None:
            raise TypeError("construct FastLM through fast.get_lm")
        self.llm, self.engine_kwargs = _engine
        tok = AutoTokenizer.from_pretrained(mid, revision=rev(mid))
        cfg = AutoConfig.from_pretrained(mid, revision=rev(mid))
        try:
            gen_eos = GenerationConfig.from_pretrained(mid, revision=rev(mid)).eos_token_id
        except OSError:
            gen_eos = getattr(cfg, "eos_token_id", None)
        self._setup_fast(mid, tok, mode, chunk, gen_eos)

    def _setup_fast(self, mid, tok, mode, chunk, gen_eos):
        """model.LM._setup without the torch model: the same attributes, derived by the same rules."""
        self.mid, self.mode, self.chunk, self.dev = mid, mode, chunk, "vllm"
        self.model, self.tok = None, tok
        self.gen_kw = M.GEN_KW
        self.tok.padding_side = "left"
        vocab = tok.get_vocab()
        ends, opens = M.TURN_END, M.TURN_OPEN
        if not any(t in vocab for t in M.TURN_END):
            end = next(t for t in M.MORE_TURNS if t in vocab)
            ends, opens = [end], M.MORE_TURNS[end]
        self.eot = next(vocab[t] for t in ends if t in vocab)
        self.turn_open = [vocab[t] for t in opens if t in vocab]
        gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []
        self.stop_ids = sorted({self.eot, tok.eos_token_id, *gen_eos} - {None})
        if tok.pad_token_id is None:
            tok.pad_token = tok.convert_ids_to_tokens(self.eot)
        self.pad_id = tok.pad_token_id
        self.special_strings = [t for t in tok.all_special_tokens if t] + ends + opens
        self.digit_lead, self.digit_ids = self._answer_tokens([str(k) for k in range(1, 8)])

    def _run(self, id_lists, sps):
        """One engine call. Every request must fit max_model_len with its max_tokens: raise rather than let vLLM refuse or cut it."""
        from vllm.inputs import TokensPrompt
        cap = self.engine_kwargs.get("max_model_len")
        if cap:
            for x, sp in zip(id_lists, sps):
                if len(x) + (sp.max_tokens or 0) > cap:
                    raise RuntimeError(f"a request of {len(x)} prompt tokens + max_tokens {sp.max_tokens} exceeds max_model_len {cap}; not sent")
        return self.llm.generate([TokensPrompt(prompt_token_ids=list(x)) for x in id_lists], sps, use_tqdm=False)

    def _fwd(self, ids, past, all_logits=False):
        raise NotImplementedError("FastLM only generates; scoring runs on model.LM")

    def sampling(self, gen_kw, max_new_tokens, seed):
        from vllm import SamplingParams
        return SamplingParams(detokenize=False, watermarking=False, **sampling_settings(self.mid, gen_kw), seed=seed,
                              max_tokens=max_new_tokens, stop_token_ids=list(self.stop_ids), ignore_eos=False, skip_special_tokens=False)

    def generate(self, messages_list, max_new_tokens, seed, gen_kw=None):
        """model.LM.generate's records. Every request samples with its own seed req_seed(seed, row); vLLM schedules them continuously, so
        nothing runs out of memory and ooms is always empty. A reply ends at the first stop token and is cut at a sampled turn opener,
        exactly as model.LM does."""
        import time
        gen_kw = self.gen_kw if gen_kw is None else gen_kw
        t0 = time.time()
        texts = [self.text(m, True) for m in messages_list]
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        seeds = [req_seed(seed, i) for i in range(len(ids))]
        outs = self._run(ids, [self.sampling(gen_kw, max_new_tokens, s) for s in seeds])
        res = []
        for x, s, o in zip(ids, seeds, outs):
            c = o.outputs[0]
            row = list(c.token_ids)
            stop = next((i for i, t in enumerate(row) if t in self.stop_ids), None)
            body = row[:stop] if stop is not None else row
            cut = next((i for i, t in enumerate(body) if t in self.turn_open), None)
            if cut is not None:
                body = body[:cut]
            text = self.tok.decode(body, skip_special_tokens=True).strip()
            res.append({"text": text, "n_tokens": len(body), "truncated": stop is None and cut is None, "stray_turn_token": cut is not None,
                        "special_string": any(z in text for z in self.special_strings), "prompt_tokens": len(x), "gen_batch": len(texts),
                        "gen_seed": seed, "req_seed": s, "finish_reason": c.finish_reason, "sampled_tokens": len(row)})
        dt = time.time() - t0
        return [{**r, "call_s": round(dt, 2), "ooms": []} for r in res]


def get_lm(mid, mode, chunk=None, engine=None):
    """The process's FastLM for (mid, mode): one engine per process. engine: a stand-in engine (the CPU tests)."""
    key = (mid, mode)
    if _ENGINE and _ENGINE["key"] != key:
        raise RuntimeError(f"this process already runs a vLLM engine for {_ENGINE['key']}; {key} needs its own process")
    if not _ENGINE:
        llm, kw = (engine, {"model": mid}) if engine is not None else make_engine(mid, opts())
        _ENGINE.update(key=key, llm=llm, kw=kw, lms={})
    lms = _ENGINE["lms"]
    if chunk not in lms:
        lms[chunk] = FastLM(mid, mode, chunk, _engine=(_ENGINE["llm"], _ENGINE["kw"]))
    return lms[chunk]


def load_lm(cfg, role):
    """The generating model of a config whose gen_mode is a vLLM mode, sampling with the config's gen_kw (else model.GEN_KW)."""
    if role != "gen" or cfg["gen_mode"] not in MODES:
        raise ValueError(f"the vLLM backend generates only ({MODES}); scoring runs on model.LM")
    lm = get_lm(cfg["model"], cfg["gen_mode"], cfg.get("chunk"))
    if "gen_kw" in cfg:
        lm.gen_kw = cfg["gen_kw"]
    return lm
