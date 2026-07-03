"""Shared pytest configuration.

Ensures the project's ``source/`` directory is importable for all test
modules from a single place, so individual tests do not need their own
``sys.path`` manipulation.
"""

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
