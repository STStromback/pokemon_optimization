"""Move-type classification sets loaded once from data/gen_all text files.

These sets replace inline hard-coded lists that were duplicated across
``generate_encounters`` and ``calculate_player_pokemon``.  Both modules now
import from here so there is a single source of truth.

All entries are normalized via :func:`~common.text_utils.normalize_text` and
stored as ``frozenset`` objects, which are hashable and safe to use as
``nonlocal`` references inside nested functions.
"""

from common import paths
from common.text_utils import normalize_text


def _load_normalized_set(filename: str) -> frozenset:
    """Read a one-entry-per-line text file and return a frozenset of normalized strings."""
    path = paths.GEN_ALL_DIR / filename
    with open(path, "r") as fh:
        return frozenset(normalize_text(line.strip()) for line in fh if line.strip())


PHYSICAL_MOVE_TYPES: frozenset = _load_normalized_set("physical_move_types.txt")
SPECIAL_MOVE_TYPES: frozenset = _load_normalized_set("special_move_types.txt")
HIGH_CRIT_MOVES: frozenset = _load_normalized_set("crit_moves.txt")
