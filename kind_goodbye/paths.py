"""Where the repository's folders are. Runs are read from and written to runs/ unless KIND_GOODBYE_RUNS names another folder."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCHEMAS = ROOT / "kind_goodbye" / "schemas"
RUNS = Path(os.environ.get("KIND_GOODBYE_RUNS", ROOT / "runs"))
