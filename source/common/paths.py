"""Centralized filesystem path resolution.

Every module should import directory locations from here instead of recomputing
them with ``Path(__file__).parent...`` chains, which are fragile to file moves
and depend on the current working directory. All paths are anchored to the
project root, which is resolved relative to this file's location:

    <project_root>/source/common/paths.py  ->  parents[2] == <project_root>
"""

from pathlib import Path

# .../source/common/paths.py -> parents[0]=common, [1]=source, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "source"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATA_DIR = PROJECT_ROOT / "data"
GEN_ALL_DIR = DATA_DIR / "gen_all"
RESULTS_DIR = PROJECT_ROOT / "results"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_files"
REPORT_DIR = PROJECT_ROOT / "report"


def gen_data_dir(gen) -> Path:
    """Return the data directory for a given generation (e.g. ``data/gen_1``)."""
    return DATA_DIR / f"gen_{gen}"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    """Return the results directory, creating it if necessary."""
    return ensure_dir(RESULTS_DIR)


def intermediate_dir() -> Path:
    """Return the intermediate-files directory, creating it if necessary."""
    return ensure_dir(INTERMEDIATE_DIR)
