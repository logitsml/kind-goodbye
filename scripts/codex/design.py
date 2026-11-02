"""Write a Codex run's config.json.

    python scripts/codex/design.py ...   is   python -m kind_goodbye.codex.design ...

Run from the repository root with the package importable (pip install -e . or PYTHONPATH=.).
"""

import sys

from kind_goodbye.codex.design import main

if __name__ == "__main__":
    sys.exit(main())
