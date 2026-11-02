"""Run one stage of a run: python scripts/pi/run.py CONFIG {generate,compact,perf,judge,status} [--runs DIR] [--limit CELLS].
The same as python -m kind_goodbye.stages; see its --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kind_goodbye.stages import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
