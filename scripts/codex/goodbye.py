"""The analysis of the kind-goodbye test.

    python scripts/codex/goodbye.py ...   is   python -m kind_goodbye.codex.goodbye ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.goodbye import main

if __name__ == "__main__":
    sys.exit(main())
