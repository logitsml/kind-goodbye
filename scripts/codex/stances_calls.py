"""The A6 calls (memory probe, fidelity coding).

    python scripts/codex/stances_calls.py ...   is   python -m kind_goodbye.codex.stances_calls ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.stances_calls import main

if __name__ == "__main__":
    sys.exit(main())
