"""The kind-goodbye test's calls.

    python scripts/codex/goodbye_calls.py ...   is   python -m kind_goodbye.codex.goodbye_calls ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.goodbye_calls import main

if __name__ == "__main__":
    sys.exit(main())
