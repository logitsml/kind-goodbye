"""Run, check or verify a Codex run (the harness).

    python scripts/codex/run.py ...   is   python -m kind_goodbye.codex.harness ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.harness import main

if __name__ == "__main__":
    sys.exit(main())
