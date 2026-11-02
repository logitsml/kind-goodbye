"""Secondary analyses of the main Codex run (A1-A5, B4, C1).

    python scripts/codex/amplify.py ...   is   python -m kind_goodbye.codex.amplify ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.amplify import main

if __name__ == "__main__":
    sys.exit(main())
