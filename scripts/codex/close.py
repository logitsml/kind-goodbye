"""The analysis of the Vc arm.

    python scripts/codex/close.py ...   is   python -m kind_goodbye.codex.close ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.close import main

if __name__ == "__main__":
    sys.exit(main())
