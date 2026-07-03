"""Ability-related constants for the Pokemon optimization pipeline.

These replace inline hard-coded collections that were scattered across
``calculate_player_pokemon`` and ``simulate_battles``.  All strings are
already normalized (lowercase, underscores for spaces) to match the join
keys used throughout the pipeline.

Note on ability activation
--------------------------
The ``ability`` / ``ability_enc`` DataFrame columns are currently never
populated (``data/gen_3/abilities_gen_3.csv`` is not read by the pipeline).
As a result only ``TRUANT_SPECIES`` — which is checked via a species-name
fallback — has any effect on outputs.  The remaining constants are correct
but dormant; wiring them to the abilities CSV is a separate, behavior-changing
task.
"""

TRUANT_SPECIES: frozenset = frozenset({"slakoth", "slaking"})

ATTACK_DOUBLING_ABILITIES: frozenset = frozenset({"pure_power", "huge_power"})

ELECTRIC_IMMUNE_ABILITIES: frozenset = frozenset({"volt_absorb", "lightning_rod"})

NO_CRIT_ABILITIES: frozenset = frozenset({"shell_armor", "battle_armor"})

THICK_FAT_WEAK_TYPES: frozenset = frozenset({"fire", "ice"})

WONDER_GUARD_SUPER_EFFECTIVE_TYPES: frozenset = frozenset(
    {"flying", "rock", "ghost", "fire", "dark"}
)
