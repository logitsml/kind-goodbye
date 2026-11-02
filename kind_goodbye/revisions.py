"""The pinned model snapshots. Every load resolves to these commits; a load on the reference backend (model.LM, which passes no revision)
is allowed only while the Hugging Face cache resolves the model to its pinned commit."""
import os
from pathlib import Path

REVISIONS = {
    "google/gemma-3-12b-it": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def rev(mid):
    if mid not in REVISIONS:
        raise ValueError(f"{mid}: no pinned revision (add it to kind_goodbye/revisions.py)")
    return REVISIONS[mid]


def hub_cache():
    """The Hugging Face hub cache this process loads from (HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def check_cache_revision(mid):
    """Raises unless the cache's refs/main names the pinned commit and its snapshot exists (model.LM loads without a revision argument)."""
    want = rev(mid)
    base = hub_cache() / f"models--{mid.replace('/', '--')}"
    ref = base / "refs" / "main"
    got = ref.read_text().strip() if ref.exists() else None
    if got != want or not (base / "snapshots" / want).is_dir():
        raise RuntimeError(f"{mid}: the Hugging Face cache ({base}) resolves to {got}, not the pinned revision {want} with its snapshot; "
                           f"download it with `hf download {mid} --revision {want}`, then write {want} into {ref}")
    return want
