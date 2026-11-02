"""The same as python -m kind_goodbye.checks; see its --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kind_goodbye.checks import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
