"""The main analysis of the Codex run.

    python scripts/codex/analyse.py ...   is   python -m kind_goodbye.codex.analysis ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.analysis import main

if __name__ == "__main__":
    sys.exit(main())
