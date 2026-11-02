"""Loading a model on the reference backend (transformers, model.LM), and running a step that may run out of GPU memory."""
import gc
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch loads: less fragmentation

import torch  # noqa: E402

from .model import LM  # noqa: E402


def load_lm(model, mode, chunk=None):
    return LM(model, mode, chunk)


def gen_lm(cfg):
    """The generating model (cfg's gen_mode), sampling with the config's "gen_kw" if it has one, else model.GEN_KW (Gemma 3's)."""
    lm = load_lm(cfg["model"], cfg["gen_mode"], cfg.get("chunk"))
    if "gen_kw" in cfg:
        lm.gen_kw = cfg["gen_kw"]
    return lm


def skip_oom(what, item, skipped):
    """After an out-of-memory error, outside the except block (its traceback holds the failed attempt's tensors until the block ends):
    free the cache and record the item."""
    print(f"{what} {item}: out of memory, skipped (nothing written)")
    skipped.append(item)
    gc.collect()
    torch.cuda.empty_cache()


def attempt(fn, *a, **k):
    """fn's result, or None if it ran out of memory (freed outside the except block, whose traceback holds the failed attempt's
    tensors)."""
    try:
        return fn(*a, **k)
    except torch.OutOfMemoryError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return None
