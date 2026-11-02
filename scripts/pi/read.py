"""Make a run's forward-pass reads: python scripts/pi/read.py CONFIG [--runs DIR] [--reference] [--dry-run].
The same as python -m kind_goodbye.reads; see its --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kind_goodbye.reads import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
