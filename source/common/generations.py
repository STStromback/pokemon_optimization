"""Single source of truth for all generation-specific behavior.

Historically, generation differences were encoded as ``if gen == 1 / 2 / 3``
branches scattered across the pipeline (stat formulas, critical-hit rates,
badge/item boosts, trade-evolution rules, legendary lists, national-dex ranges,
and trainer-data filenames). Adding a new generation therefore meant editing
many functions in several files and was error prone.

This module centralizes those differences into one :class:`GenerationConfig`
per generation, registered in :data:`GENERATIONS`. The pipeline reads parameters
from here instead of hard-coding literals, so the numeric behavior is unchanged.

Adding a new generation
-----------------------
1. Add the per-generation data files under ``data/gen_<n>/`` following the
   existing naming pattern.
2. Add one :class:`GenerationConfig` entry to :data:`GENERATIONS` below.
3. If the new generation introduces a novel trainer-data format, register a
   parser for it in ``pipeline.generate_encounters`` (keyed by generation).
4. Extend :data:`LEGENDARY_POKEMON` with any new legendaries.

This module contains only data and pure helpers; it must not import from the
``pipeline`` package so it can be imported anywhere without circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple

# Multipliers that are constant across all generations in this project.
STAB_MULTIPLIER: float = 1.5
ITEM_TYPE_BOOST_MULTIPLIER: float = 1.1
BADGE_TYPE_BOOST_MULTIPLIER: float = 1.125

# Stat-calculation DV/stat-exp constants (identical across generations here).
DV_NORMAL: int = 8
DV_ATTACK: int = 9
STAT_EXP: int = 0


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable description of a single Pokemon generation's mechanics."""

    number: int

    # --- Stat calculation -------------------------------------------------
    # "gen12" and "gen3" select the corresponding stat formula. Future
    # generations reuse an existing key unless their formula differs.
    stat_formula: str

    # --- Damage formula ---------------------------------------------------
    # Selects the base-damage expression (the position of the crit term differs
    # between generations). Future generations reuse an existing key unless
    # their formula differs.
    damage_formula: str  # "gen1" | "gen2" | "gen3"

    # --- Critical hits ----------------------------------------------------
    # Gen 1 derives its rate from base speed; later generations use fixed
    # normal/high rates. The crit contribution to damage is modeled as the
    # expected-value factor ``1 + crit_rate`` for every generation, so no
    # per-generation crit *multiplier* is stored (storing one would imply an
    # unused parameter and risk diverging from the actual damage model).
    speed_based_crit: bool
    crit_normal: float
    crit_high: float

    # --- Badge / item boosts ---------------------------------------------
    badge_boost_multiplier: float  # stat boost from badges (1.125 vs 1.1)
    has_item_type_boost: bool      # type-boosting held items exist
    has_badge_type_boost: bool     # type-boosting badges exist

    # --- Abilities --------------------------------------------------------
    has_abilities: bool

    # --- Trade evolutions -------------------------------------------------
    # Gen 1 has no items, so trade evolutions are detected by species name.
    # Later generations detect them via the evolution item ("trade" or a
    # specific held item).
    trade_evolution_mode: str  # "by_name" | "by_item"
    trade_evolution_names: FrozenSet[str] = field(default_factory=frozenset)
    trade_item_evolutions: FrozenSet[str] = field(default_factory=frozenset)

    # --- External data ----------------------------------------------------
    national_dex_max: int = 0  # highest national-dex id available in-game


_TRADE_ITEM_EVOLUTIONS: FrozenSet[str] = frozenset(
    {"metalcoat", "dragonscale", "kingsrock", "upgrade", "deepseatooth", "deepseascale"}
)


GENERATIONS: Dict[int, GenerationConfig] = {
    1: GenerationConfig(
        number=1,
        stat_formula="gen12",
        damage_formula="gen1",
        speed_based_crit=True,
        crit_normal=0.0,           # unused: gen 1 computes the rate from speed
        crit_high=0.0,
        badge_boost_multiplier=1.125,
        has_item_type_boost=False,
        has_badge_type_boost=False,
        has_abilities=False,
        trade_evolution_mode="by_name",
        trade_evolution_names=frozenset({"alakazam", "machamp", "golem", "gengar"}),
        trade_item_evolutions=frozenset(),
        national_dex_max=151,
    ),
    2: GenerationConfig(
        number=2,
        stat_formula="gen12",
        damage_formula="gen2",
        speed_based_crit=False,
        crit_normal=17 / 256,
        crit_high=1 / 4,
        badge_boost_multiplier=1.125,
        has_item_type_boost=True,
        has_badge_type_boost=True,
        has_abilities=False,
        trade_evolution_mode="by_item",
        trade_item_evolutions=_TRADE_ITEM_EVOLUTIONS,
        national_dex_max=251,
    ),
    3: GenerationConfig(
        number=3,
        stat_formula="gen3",
        damage_formula="gen3",
        speed_based_crit=False,
        crit_normal=1 / 16,
        crit_high=1 / 8,
        badge_boost_multiplier=1.1,
        has_item_type_boost=True,
        has_badge_type_boost=False,
        has_abilities=True,
        trade_evolution_mode="by_item",
        trade_item_evolutions=_TRADE_ITEM_EVOLUTIONS,
        national_dex_max=386,
    ),
}

# Tuple of supported generations, derived from the registry so there is a single
# place to update when adding a generation.
VALID_GENERATIONS: Tuple[int, ...] = tuple(sorted(GENERATIONS))


# Legendary Pokemon across all supported generations. The membership check uses
# the raw (un-normalized) species names exactly as they appear in the stats
# files, matching the previous inline list. Each generation's roster only
# contains species up to that generation, so a single combined set reproduces
# the original per-generation behavior.
LEGENDARY_POKEMON: FrozenSet[str] = frozenset({
    # Generation 1
    "Zapdos", "Moltres", "Articuno", "Mewtwo", "Mew",
    # Generation 2
    "Celebi", "Raikou", "Entei", "Suicune", "Lugia", "Ho_Oh",
    # Generation 3
    "Kyogre", "Groudon", "Rayquaza", "Jirachi", "Deoxys",
    "Regice", "Regirock", "Registeel", "Latias", "Latios",
})


def get_generation_config(gen: int) -> GenerationConfig:
    """Return the :class:`GenerationConfig` for ``gen``.

    Raises:
        ValueError: if ``gen`` is not a supported generation.
    """
    try:
        return GENERATIONS[int(gen)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"Unsupported generation: {gen!r}. Supported: {list(VALID_GENERATIONS)}"
        )


def is_legendary(pokemon_name: str) -> bool:
    """Return True if ``pokemon_name`` (raw stats-file spelling) is legendary."""
    return pokemon_name in LEGENDARY_POKEMON
