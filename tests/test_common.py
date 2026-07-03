"""Tests for the shared ``common`` utilities (paths, config, text normalization)."""

import json

import numpy as np
import pandas as pd
import pytest

from common import paths
from common.config import (
    get_generations,
    get_level_calc_method,
    load_config,
)
from common.text_utils import normalize_text, normalize_text_columns


class TestNormalizeText:
    """Canonical normalization must satisfy both pipeline and availability rules."""

    def test_basic_lowercasing(self):
        assert normalize_text("Pikachu") == "pikachu"
        assert normalize_text("CHARIZARD") == "charizard"

    def test_spaces_to_underscore(self):
        assert normalize_text("Water Gun") == "water_gun"
        assert normalize_text("Pallet Town") == "pallet_town"
        assert normalize_text("Route 1") == "route_1"

    def test_hyphens_removed(self):
        assert normalize_text("Ho-Oh") == "hooh"

    def test_punctuation_removed(self):
        assert normalize_text("Mr. Mime") == "mr_mime"
        assert normalize_text("Mt. Moon") == "mt_moon"
        assert normalize_text("Diglett's Cave") == "digletts_cave"

    def test_gender_symbols(self):
        assert normalize_text("Nidoran\u2640") == "nidoran_f"
        assert normalize_text("Nidoran\u2642") == "nidoran_m"

    def test_accents_folded(self):
        assert normalize_text("Pok\u00e9mon") == "pokemon"
        assert normalize_text("Caf\u00e9") == "cafe"

    def test_none_and_nan(self):
        assert normalize_text(None) == ""
        assert normalize_text(np.nan) == ""
        assert normalize_text(pd.NA) == ""
        assert normalize_text("") == ""

    def test_collapses_and_strips_underscores(self):
        assert normalize_text("  fire   blast  ") == "fire_blast"


class TestNormalizeTextColumns:
    def test_preserves_nan_and_normalizes(self):
        df = pd.DataFrame({"name": ["Water Gun", None, "Ho-Oh"], "n": [1, 2, 3]})
        out = normalize_text_columns(df)
        assert out["name"].iloc[0] == "water_gun"
        assert pd.isna(out["name"].iloc[1])
        assert out["name"].iloc[2] == "hooh"
        assert list(out["n"]) == [1, 2, 3]


class TestPaths:
    def test_project_root_contains_known_dirs(self):
        assert (paths.PROJECT_ROOT / "source").is_dir()
        assert (paths.PROJECT_ROOT / "config").is_dir()

    def test_config_file_location(self):
        assert paths.CONFIG_FILE == paths.CONFIG_DIR / "config.json"

    def test_gen_data_dir(self):
        assert paths.gen_data_dir(1) == paths.DATA_DIR / "gen_1"


class TestConfig:
    def test_load_real_config(self):
        config = load_config()
        assert "gen" in config
        assert "level_calc_method" in config

    def test_defaults_when_missing(self, tmp_path):
        config = load_config(tmp_path / "does_not_exist.json")
        assert config["gen"] == [1, 2, 3]
        assert config["level_calc_method"] == "sequential_max"

    def test_legacy_key_migration(self, tmp_path):
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({
            "level_calc_method (sequential_max/independent)": "independent",
        }))
        config = load_config(legacy)
        assert config["level_calc_method"] == "independent"
        assert "level_calc_method (sequential_max/independent)" not in config

    def test_get_generations_variants(self):
        assert get_generations({"gen": "all"}) == [1, 2, 3]
        assert get_generations({"gen": 2}) == [2]
        assert get_generations({"gen": [3, 1]}) == [3, 1]
        assert get_generations({"gen": [9]}) == [1, 2, 3]

    def test_get_level_calc_method_default(self):
        assert get_level_calc_method({}) == "sequential_max"
        assert get_level_calc_method({"level_calc_method": "independent"}) == "independent"
