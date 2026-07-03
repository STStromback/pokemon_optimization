"""Centralized configuration loading and generation selection.

Replaces the several near-duplicate ``load_config`` helpers that previously
lived in individual modules with inconsistent defaults and path assumptions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from common import paths

VALID_GENERATIONS: List[int] = [1, 2, 3]

# Defaults applied when the config file is missing or a key is absent.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "initialize_repo": "n",
    "gen": list(VALID_GENERATIONS),
    "level_calc_method": "sequential_max",
    "trade_evolutions": "y",
    "legendaries": "y",
    "all_starters": "n",
    "restrictions": {},
    "exclusions": [],
    "easy_dog_catch": "n",
    "drop_first_rival_encounter": "y",
}

# Legacy/renamed keys mapped to their canonical names for backwards compatibility.
_LEGACY_KEYS = {
    "level_calc_method (sequential_max/independent)": "level_calc_method",
}


def load_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load configuration, applying defaults and migrating legacy keys.

    Args:
        config_path: Optional path to a config JSON file. Defaults to the
            project's ``config/config.json``.

    Returns:
        A configuration dict with defaults filled in.
    """
    path = Path(config_path) if config_path is not None else paths.CONFIG_FILE
    config: Dict[str, Any] = dict(_DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {path}. Using defaults.")
        return config
    except json.JSONDecodeError as e:
        print(f"Error parsing config file {path}: {e}. Using defaults.")
        return dict(_DEFAULT_CONFIG)

    # Migrate legacy keys before merging so explicit values override defaults.
    for old_key, new_key in _LEGACY_KEYS.items():
        if old_key in loaded:
            loaded.setdefault(new_key, loaded[old_key])
            del loaded[old_key]

    config.update(loaded)
    return config


def get_generations(config: Dict[str, Any] = None) -> List[int]:
    """Resolve the list of generations to process from a config dict."""
    if config is None:
        config = load_config()

    value = config.get("gen", VALID_GENERATIONS)

    if value == "all":
        return list(VALID_GENERATIONS)
    if isinstance(value, int):
        value = [value]
    if isinstance(value, list):
        valid = [g for g in value if g in VALID_GENERATIONS]
        if valid:
            return valid
        print(f"Warning: no valid generations in {value!r}; processing all.")
        return list(VALID_GENERATIONS)
    if str(value) in {"1", "2", "3"}:
        return [int(value)]

    print(f"Warning: invalid gen value {value!r}; processing all.")
    return list(VALID_GENERATIONS)


def get_level_calc_method(config: Dict[str, Any]) -> str:
    """Return the configured level calculation method (default ``sequential_max``)."""
    return config.get("level_calc_method", "sequential_max")
