import pandas as pd
import numpy as np
from pathlib import Path
import math
import json
from functools import lru_cache
from typing import Dict, Tuple, Set, Optional, Any, Callable, List, Iterable
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress function
    def tqdm(iterable, desc=None, total=None):
        return iterable

# Pre-compile character set for normalization
_KEEP_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_/")

@lru_cache(maxsize=50000)
def _norm(txt):
    """Cached text normalization for better performance."""
    if txt is None or pd.isna(txt):
        return ""
    txt = str(txt).lower().replace(" ", "_")
    # Handle special Nidoran characters
    txt = txt.replace("♀", "_f").replace("♂", "_m")
    return "".join(ch for ch in txt if ch in _KEEP_CHARS).replace("__", "_")

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized text normalization for DataFrame columns."""
    try:
        df = df.copy()
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        
        for c in text_cols:
            try:
                # Use vectorized operations where possible
                unique_vals = df[c].dropna().unique()
                norm_map = {val: _norm(val) for val in unique_vals}
                norm_map[np.nan] = np.nan  # Preserve NaN values
                df[c] = df[c].map(norm_map)
            except Exception as e:
                print(f"Warning: Failed to normalize column '{c}': {str(e)}")
                print(f"Column type: {df[c].dtype}, Column shape: {df[c].shape}")
                # Continue with other columns instead of failing completely
                continue
        
        return df
    except Exception as e:
        print(f"Critical error in normalize_text_columns: {str(e)}")
        print(f"DataFrame info - Shape: {df.shape if 'df' in locals() else 'Unknown'}")
        print(f"DataFrame columns: {list(df.columns) if 'df' in locals() else 'Unknown'}")
        raise

@lru_cache(maxsize=256)
def get_badge_boost_multiplier(gen, stage_enc, stat_name, stages_tuple):
    """
    Cached badge boost multiplier calculation.
    stages_tuple is a tuple of tuples for hashability.
    """
    for badge_stat, location_stage in stages_tuple:
        if badge_stat == stat_name and location_stage <= stage_enc:
            return 1.125 if gen <= 2 else 1.1
    return 1.0

def get_move_type_boost_multiplier(gen, stage_enc, move_type, stages_df):
    """
    Get the move type boost multiplier based on item_type_boost and badge_type_boost.
    """
    multiplier = 1.0
    
    # Ensure stages_df is a valid DataFrame
    if stages_df is None or not isinstance(stages_df, pd.DataFrame):
        return multiplier
    
    # Check item_type_boost (for gen 2 and 3)
    if gen >= 2 and 'item_type_boost' in stages_df.columns:
        try:
            applicable_stages = stages_df[
                (stages_df['location_stage'] <= stage_enc) &
                (stages_df['item_type_boost'].notna())
            ]
            
            # Use itertuples instead of iterrows for better performance and reliability
            for stage_row in applicable_stages.itertuples(index=False):
                item_types_value = getattr(stage_row, 'item_type_boost', '')
                item_types = str(item_types_value).split(',')
                item_types = [t.strip() for t in item_types if t.strip()]
                if move_type in item_types:
                    multiplier *= 1.1
                    break
        except Exception:
            # If there's any issue, just skip the boost
            pass
    
    # Check badge_type_boost (for gen 2 only)
    if gen == 2 and 'badge_type_boost' in stages_df.columns:
        try:
            applicable_stages = stages_df[
                (stages_df['location_stage'] <= stage_enc) &
                (stages_df['badge_type_boost'].notna())
            ]
            
            # Use itertuples instead of iterrows for better performance and reliability
            for stage_row in applicable_stages.itertuples(index=False):
                badge_types_value = getattr(stage_row, 'badge_type_boost', '')
                badge_types = str(badge_types_value).split(',')
                badge_types = [t.strip() for t in badge_types if t.strip()]
                if move_type in badge_types:
                    multiplier *= 1.125
                    break
        except Exception:
            # If there's any issue, just skip the boost
            pass
    
    return multiplier

@lru_cache(maxsize=512)
def calculate_critical_hit_rate(gen, move_name, base_speed, is_high_crit):
    """
    Calculate critical hit rate based on generation and move properties.
    
    Args:
        gen: Generation (1, 2, or 3)
        move_name: Name of the move
        base_speed: Base speed stat of the Pokemon
        is_high_crit: Boolean indicating if move has high critical hit ratio
        
    Returns:
        Critical hit rate as a decimal (e.g., 0.0625 for 1/16)
    """
    
    if gen == 1:
        # Gen 1: floor((base_speed / 2)/256) for normal, min(8 * floor(base_speed / 2)/256, 255/256) for high crit
        base_rate = math.floor(base_speed / 2) / 256
        if is_high_crit:
            crit_rate = min(8 * base_rate, 255/256)
        else:
            crit_rate = base_rate
    elif gen == 2:
        # Gen 2: 17/256 for normal, 1/4 for high crit
        if is_high_crit:
            crit_rate = 1/4
        else:
            crit_rate = 17/256
    elif gen == 3:
        # Gen 3: 1/16 for normal, 1/8 for high crit
        if is_high_crit:
            crit_rate = 1/8
        else:
            crit_rate = 1/16
    else:
        crit_rate = 0
    
    return crit_rate

def apply_ability_stat_modifiers(gen, pokemon_stats, ability, is_attacker=True):
    """
    Apply ability-based stat modifications for Gen 3.
    """
    if gen != 3 or pd.isna(ability):
        return pokemon_stats
    
    ability = str(ability).lower()
    modified_stats = pokemon_stats.copy()
    
    if ability in ['pure_power', 'huge_power']:
        # Double attack stat
        modified_stats['attack'] = int(modified_stats['attack'] * 2)
    elif ability == 'hustle' and is_attacker:
        # Increase attack by 50%
        modified_stats['attack'] = int(modified_stats['attack'] * 1.5)
    elif ability == 'intimidate' and not is_attacker:
        # Decrease opponent's attack (2/3 for Gen 3)
        modified_stats['attack'] = int(modified_stats['attack'] * 2/3)
    
    return modified_stats

def apply_ability_damage_modifiers(gen, damage, move_type, attacker_ability, defender_ability):
    """
    Apply ability-based damage modifications for Gen 3.
    """
    if gen != 3:
        return damage
    
    # Normalize abilities
    attacker_ability = str(attacker_ability).lower() if pd.notna(attacker_ability) else ""
    defender_ability = str(defender_ability).lower() if pd.notna(defender_ability) else ""
    
    # Defender abilities that provide immunity
    if defender_ability == 'wonder_guard' and move_type not in ['flying', 'rock', 'ghost', 'fire', 'dark']:
        return 0
    elif defender_ability == 'levitate' and move_type == 'ground':
        return 0
    elif defender_ability in ['volt_absorb', 'lightning_rod'] and move_type == 'electric':
        return 0
    elif defender_ability == 'water_absorb' and move_type == 'water':
        return 0
    elif defender_ability == 'flash_fire' and move_type == 'fire':
        return 0
    
    # Defender abilities that reduce damage
    if defender_ability == 'thick_fat' and move_type in ['fire', 'ice']:
        damage *= 0.5
    
    # Weather abilities (attacker)
    if attacker_ability == 'drizzle':
        if move_type == 'water':
            damage *= 1.5
        elif move_type == 'fire':
            damage *= 0.5
    elif attacker_ability == 'drought':
        if move_type == 'fire':
            damage *= 1.5
        elif move_type == 'water':
            damage *= 0.5
    
    return damage

def apply_ability_accuracy_modifiers(gen, accuracy, attacker_ability, move_name):
    """
    Apply ability-based accuracy modifications for Gen 3.
    """
    if gen != 3 or pd.isna(attacker_ability):
        return accuracy
    
    attacker_ability = str(attacker_ability).lower()
    
    if attacker_ability == 'hustle':
        # Reduce accuracy by 20%
        accuracy *= 0.8
    elif attacker_ability == 'compound_eyes':
        # Increase accuracy by 30% (1.3x), max 100%
        accuracy = min(accuracy * 1.3, 1.0)
    elif attacker_ability == 'drizzle' and _norm(move_name) == 'thunder':
        # Thunder has 100% accuracy in rain
        accuracy = 1.0
    
    return accuracy

def apply_ability_crit_modifiers(gen, crit_rate, defender_ability):
    """
    Apply ability-based critical hit modifications for Gen 3.
    """
    if gen != 3 or pd.isna(defender_ability):
        return crit_rate
    
    defender_ability = str(defender_ability).lower()
    
    if defender_ability in ['shell_armor', 'battle_armor']:
        # Reduce critical hit chance to 0%
        return 0
    
    return crit_rate

@lru_cache(maxsize=2048)
def calculate_pokemon_stats_cached(base_stats_tuple, level, generation, stage_enc=None, stages_tuple=None):
    """
    Cached version of calculate_pokemon_stats for better performance.
    base_stats_tuple and stages_tuple are tuples for hashability.
    """
    calculated_stats = {}
    
    # Constants
    dv_normal = 8
    dv_attack = 9
    stat_exp = 0
    
    # Convert tuple back to dict
    stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    base_stats = dict(zip(stat_names, base_stats_tuple))
    
    for stat_name in stat_names:
        base_stat = base_stats.get(stat_name, 0)
        
        # Determine DV value
        dv = dv_attack if stat_name == 'attack' else dv_normal
        
        if generation in [1, 2]:
            if stat_name == 'hp':
                # HP formula for Gen 1/2
                calculated_stat = math.floor(((base_stat + dv) * 2 + math.floor(math.sqrt(stat_exp) / 4)) * level / 100) + level + 10
            else:
                # Other stats formula for Gen 1/2
                calculated_stat = math.floor(((base_stat + dv) * 2 + math.floor(math.sqrt(stat_exp) / 4)) * level / 100) + 5
        
        elif generation == 3:
            if stat_name == 'hp':
                # HP formula for Gen 3
                calculated_stat = math.floor((2 * base_stat + dv + math.floor(stat_exp / 4)) * level / 100) + level + 10
            else:
                # Other stats formula for Gen 3
                calculated_stat = math.floor((math.floor(2 * base_stat + dv + math.floor(stat_exp / 4)) * level / 100) + 5)
        
        else:
            raise ValueError(f"Unsupported generation: {generation}")
        
        # Apply badge boost if applicable
        if stage_enc is not None and stages_tuple is not None:
            # Check for badge boost for this specific stat
            badge_multiplier = get_badge_boost_multiplier(generation, stage_enc, stat_name, stages_tuple)
            calculated_stat = int(calculated_stat * badge_multiplier)
            
            # Handle 'special' badge boost for sp_attack and sp_defense
            if stat_name in ['sp_attack', 'sp_defense']:
                special_multiplier = get_badge_boost_multiplier(generation, stage_enc, 'special', stages_tuple)
                calculated_stat = int(calculated_stat * special_multiplier)
            
        calculated_stats[stat_name] = calculated_stat
    
    return calculated_stats

def calculate_pokemon_stats(base_stats, level, generation, stage_enc=None, stages_df=None):
    """
    Wrapper for the cached version.
    """
    # Convert to tuple for caching
    stat_names = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    base_stats_tuple = tuple(base_stats.get(name, 0) for name in stat_names)
    
    # Convert stages_df to tuple if provided
    stages_tuple = None
    if stages_df is not None:
        stages_tuple = tuple(
            (row.get('badge_boost', ''), row.get('location_stage', 0))
            for _, row in stages_df.iterrows()
            if pd.notna(row.get('badge_boost'))
        )
    
    return calculate_pokemon_stats_cached(base_stats_tuple, level, generation, stage_enc, stages_tuple)


def apply_ability_move_power_modifiers(gen, power, move_name, attacker_ability):
    """Apply ability effects on move power for Gen 3"""
    if gen != 3 or pd.isna(attacker_ability):
        return power
    
    attacker_ability = str(attacker_ability).lower()
    move_name = str(move_name).lower().replace(' ', '_')
    
    if attacker_ability == 'drought' and move_name == 'solarbeam':
        return 120


# ---------------------------
# Fast Damage Computation Engine (Drop-in Replacement)
# ---------------------------

def prepare_enc_damage_context(
    *,
    gen: int,
    stats_lookup: Dict[str, Dict[str, Any]],
    _get_pokemon_stats_and_types: Callable[..., Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[set]]],
    _calculate_move_damage: Callable[..., float],
    apply_ability_stat_modifiers: Callable[..., Dict[str, Any]],
    min_stage_lookup: Dict[Any, float] = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build a fast calculator for encounter → player_pokemon damage.

    Preserves key differences vs player_pokemon→encounter:
      - Attacker is the encounter; defender is the player_pokemon.
      - Stats for attacker use `stage_enc` in the lookup; defender does not.
      - Moves are *only* the 1–4 displayed encounter moves.
      - `include_badge_boost=False`.
      - No extra 'intimidate' side-effect beyond baseline ability modifiers.
    """

    gl_gen = int(gen)
    gl_stats_lookup = stats_lookup
    gl_calc = _calculate_move_damage
    gl_get_stats = _get_pokemon_stats_and_types
    gl_apply_ability = apply_ability_stat_modifiers
    gl_min_stage_lookup = min_stage_lookup or {}

    # Cached stats/type lookups
    @lru_cache(maxsize=50_000)
    def _stats_attacker_enc(pokemon_enc: str, level_enc: int, stage_enc: int):
        return gl_get_stats(pokemon_enc, level_enc, gl_gen, stage_enc)

    @lru_cache(maxsize=50_000)
    def _stats_defender_player_pokemon(pokemon: str, level: int):
        return gl_get_stats(pokemon, level, gl_gen)

    # Ability-adjusted caches
    @lru_cache(maxsize=50_000)
    def _ability_stats_attacker_enc(pokemon_enc: str, level_enc: int, stage_enc: int, ability_enc: Optional[str]):
        stats, *_ = _stats_attacker_enc(pokemon_enc, level_enc, stage_enc)
        if stats is None:
            return None
        if gl_gen == 3:
            return gl_apply_ability(gl_gen, stats, ability_enc, is_attacker=True)
        return stats

    @lru_cache(maxsize=50_000)
    def _ability_stats_defender_player_pokemon(pokemon: str, level: int, ability: Optional[str]):
        stats, *_ = _stats_defender_player_pokemon(pokemon, level)
        if stats is None:
            return None
        if gl_gen == 3:
            return gl_apply_ability(gl_gen, stats, ability, is_attacker=False)
        return stats

    @lru_cache(maxsize=50_000)
    def _extract_enc_moves_cached(move1: str, move2: str, move3: str, move4: str) -> Tuple[str, ...]:
        """Cached version of move extraction for better performance."""
        names: List[str] = []
        for move in [move1, move2, move3, move4]:
            if move and pd.notna(move) and str(move).strip():
                name = str(move).strip()
                if name:
                    names.append(name)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for n in names:
            if n not in seen:
                uniq.append(n)
                seen.add(n)
        return tuple(uniq)
    
    def _extract_enc_moves(row: pd.Series) -> Tuple[str, ...]:
        """Extract moves from row using cached function."""
        move1 = row.get("move_name_1_enc", "")
        move2 = row.get("move_name_2_enc", "")
        move3 = row.get("move_name_3_enc", "")
        move4 = row.get("move_name_4_enc", "")
        return _extract_enc_moves_cached(move1, move2, move3, move4)

    def _row_calc_enc(
        pokemon_enc: str,
        level_enc: int,
        stage_enc: int,
        ability_enc: Optional[str],
        pokemon: str,
        level: int,
        ability: Optional[str],
        moves: Tuple[str, ...],
    ) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[int]]:
        if not pokemon_enc or not pokemon or not moves:
            return (None, None, None, None)
        try:
            lvl_e = int(level_enc); stg = int(stage_enc); lvl_v = int(level)
        except Exception:
            return (None, None, None, None)

        # Safety check: verify player_pokemon pokemon is available at this stage
        if gl_min_stage_lookup:
            base = gl_stats_lookup.get(pokemon)
            if base:
                evo_id = base.get("evo_id")
                if evo_id in gl_min_stage_lookup:
                    min_stage = gl_min_stage_lookup[evo_id]
                    if stg < min_stage:
                        return (None, None, None, None)

        atk_stats = _ability_stats_attacker_enc(pokemon_enc, lvl_e, stg, ability_enc)
        if atk_stats is None:
            return (None, None, None, None)
        def_stats = _ability_stats_defender_player_pokemon(pokemon, lvl_v, ability)
        if def_stats is None:
            return (None, None, None, None)

        # Bind locals for speed
        local_calc = gl_calc
        enc_types_set = _stats_attacker_enc(pokemon_enc, lvl_e, stg)[2]
        var_types = _stats_defender_player_pokemon(pokemon, lvl_v)[1]
        speed = int(gl_stats_lookup.get(pokemon_enc, {}).get("speed", atk_stats.get("speed", 0)))

        best_mv = None
        best_dmg = -1.0  # Initialize to -1 to ensure even 0-damage moves can be selected
        
        for m in moves:
            dmg = local_calc(
                m, atk_stats, def_stats, enc_types_set, var_types,
                lvl_e, speed, gl_gen, stg, ability_enc, ability,
                include_badge_boost=False, attacker_pokemon_name=pokemon_enc,
            )
            if dmg > best_dmg:
                best_dmg = dmg
                best_mv = m

        # If no move was selected, return None for all
        if best_mv is None:
            return (None, None, None, None)
        
        # Ensure consistency: if we have a move, we must have a damage value
        # If best damage is negative (shouldn't happen), treat as no valid move
        if best_dmg < 0:
            return (None, None, None, None)
        
        # Extract HP and speed from attacker stats (encounter Pokemon)
        hp_in = int(atk_stats.get('hp', 0))
        speed_in = int(atk_stats.get('speed', 0))
        
        # Return the move, damage, HP, and speed (including 0-damage cases for immunities)
        return (best_mv, float(best_dmg), hp_in, speed_in)

    def calculate_enc_best_moves(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({"move_in": pd.Series(dtype="string"), "damage_in": pd.Series(dtype="float"),
                                "hp_in": pd.Series(dtype="Int64"), "speed_in": pd.Series(dtype="Int64")}, 
                              index=df.index)

        cols = df.columns
        pokemon_enc = df["pokemon_enc"].astype("string") if "pokemon_enc" in cols else pd.Series([None]*len(df), index=df.index)
        level_enc = pd.to_numeric(df["level_enc"], errors="coerce").astype("Int64") if "level_enc" in cols else pd.Series([pd.NA]*len(df), dtype="Int64", index=df.index)
        stage_enc = pd.to_numeric(df["stage_enc"], errors="coerce").astype("Int64") if "stage_enc" in cols else pd.Series([pd.NA]*len(df), dtype="Int64", index=df.index)
        ability_enc = df.get("ability_enc")
        pokemon = df["pokemon"].astype("string") if "pokemon" in cols else pd.Series([None]*len(df), index=df.index)
        level = pd.to_numeric(df["level"], errors="coerce").astype("Int64") if "level" in cols else pd.Series([pd.NA]*len(df), dtype="Int64", index=df.index)
        ability = df.get("ability")

        # Precompute moves for all rows once to avoid double extraction
        all_moves = []
        has_move_mask = []
        
        # Vectorized move extraction - get all move columns at once
        move_cols = [f"move_name_{i}_enc" for i in range(1, 5)]
        available_cols = [col for col in move_cols if col in df.columns]
        
        if available_cols:
            # Extract all moves in one vectorized operation
            moves_data = df[available_cols].fillna("").astype(str)
            
            for i in range(len(df)):
                # Extract moves only from columns that actually exist in the DataFrame
                move_values = []
                for col in move_cols:
                    if col in available_cols:
                        move_values.append(moves_data.iloc[i][col])
                    else:
                        move_values.append("")
                row_moves = _extract_enc_moves_cached(*move_values)
                all_moves.append(row_moves)
                has_move_mask.append(len(row_moves) > 0)
        else:
            # Fallback if no move columns exist
            all_moves = [tuple() for _ in range(len(df))]
            has_move_mask = [False for _ in range(len(df))]
        
        has_move = np.array(has_move_mask, dtype=bool)
        
        valid = (
            pokemon_enc.notna() & level_enc.notna() & stage_enc.notna() &
            pokemon.notna() & level.notna() & has_move
        ).to_numpy()
        idx = np.where(valid)[0]

        n = len(df)
        move_out = np.array([None]*n, dtype=object)
        dmg_out = np.full(n, np.nan, dtype=float)
        hp_out = np.full(n, np.nan, dtype=float)  # Use float for np.full, convert to Int64 later
        speed_out = np.full(n, np.nan, dtype=float)  # Use float for np.full, convert to Int64 later

        # Process valid rows with tight Python loop (fast thanks to caches)
        # Use precomputed moves to avoid double extraction
        if TQDM_AVAILABLE and len(idx) > 1000:  # Only show progress bar for larger datasets
            for i in tqdm(idx, desc="Fast encounter damage calculation"):
                mv, dmg, hp, spd = _row_calc_enc(
                    pokemon_enc.iloc[i], int(level_enc.iloc[i]), int(stage_enc.iloc[i]),
                    ability_enc.iloc[i] if isinstance(ability_enc, pd.Series) else None,
                    pokemon.iloc[i], int(level.iloc[i]),
                    ability.iloc[i] if isinstance(ability, pd.Series) else None,
                    all_moves[i],  # Use precomputed moves
                )
                move_out[i] = mv
                dmg_out[i] = dmg if dmg is not None else np.nan
                hp_out[i] = hp if hp is not None else np.nan
                speed_out[i] = spd if spd is not None else np.nan
        else:
            for i in idx:
                mv, dmg, hp, spd = _row_calc_enc(
                    pokemon_enc.iloc[i], int(level_enc.iloc[i]), int(stage_enc.iloc[i]),
                    ability_enc.iloc[i] if isinstance(ability_enc, pd.Series) else None,
                    pokemon.iloc[i], int(level.iloc[i]),
                    ability.iloc[i] if isinstance(ability, pd.Series) else None,
                    all_moves[i],  # Use precomputed moves
                )
                move_out[i] = mv
                dmg_out[i] = dmg if dmg is not None else np.nan
                hp_out[i] = hp if hp is not None else np.nan
                speed_out[i] = spd if spd is not None else np.nan

        # CRITICAL: Preserve the original DataFrame's index
        # Fix: Create DataFrame with default index first, then reindex to match original
        result_df = pd.DataFrame({
            "move_in": pd.Series(move_out, dtype="string"),
            "damage_in": dmg_out,
            "hp_in": pd.Series(hp_out, dtype="Int64"),  # Convert to nullable Int64
            "speed_in": pd.Series(speed_out, dtype="Int64"),  # Convert to nullable Int64
        })
        
        # Reindex to match the original DataFrame's index
        result_df.index = df.index
        
        return result_df

    return calculate_enc_best_moves


def prepare_damage_context(  # <-- same name: drop-in replacement
    *,
    gen: int,
    moves_lookup: Dict[str, Dict[str, Any]],
    tm_stage_lookup: Dict[str, int],
    pokemon_moves_lookup: Dict[str, pd.DataFrame],
    stats_lookup: Dict[str, Dict[str, Any]],
    evo_chain_lookup: Dict[Any, List[Dict[str, Any]]],
    get_evo_stage_for_item: Callable[[Any, Any], Optional[int]],
    stages: Any,
    _get_pokemon_stats_and_types: Callable[..., Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[set]]],
    _calculate_move_damage: Callable[..., float],
    apply_ability_stat_modifiers: Callable[..., Dict[str, Any]],
    move_power_lookup: Dict[str, float],
    min_stage_lookup: Dict[Any, float] = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build an optimized damage calculator. Focus: minimize duplicate work across rows."""

    # ---------- immutable, small helpers (unchanged structure) ----------
    def _build_moves_meta(moves_lookup_in: Dict[str, Dict[str, Any]],
                          tm_stage_lookup_in: Dict[str, int],
                          move_power_lookup_in: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, m in moves_lookup_in.items():
            if not name:
                continue
            t = str(m.get("move_type", m.get("Type", ""))).lower()
            cat = str(m.get("Category", "")).lower()
            power = float(m.get("move_power", m.get("Power", 0)) or 0)
            key = name  # already normalized in caller
            # why: apply last-mile power overrides
            if key in move_power_lookup_in:
                power = float(move_power_lookup_in[key])
            # Normalize the key for TM stage lookup since tm_stage_lookup expects normalized names
            normalized_key = _norm(key)
            min_stage = int(tm_stage_lookup_in.get(normalized_key, 999) or 999)
            out[name] = {"power": power, "type": t, "category": cat, "min_tm_stage": min_stage}
        return out

    def _build_pokemon_moves_master(
        pokemon_moves_lookup_in: Dict[str, pd.DataFrame],
        moves_meta_in: Dict[str, Dict[str, Any]],
    ) -> Dict[str, pd.DataFrame]:
        master: Dict[str, pd.DataFrame] = {}
        for pkmn, df in pokemon_moves_lookup_in.items():
            if df is None or df.empty or "move_name" not in df.columns:
                master[pkmn] = pd.DataFrame(
                    {"move_name": pd.Series(dtype="string"),
                     "move_level_num": pd.Series(dtype="int64"),
                     "min_tm_stage": pd.Series(dtype="int64")}
                )
                continue
            local = df.copy()
            if "move_level_num" not in local.columns:
                local["move_level_num"] = pd.to_numeric(local.get("move_level"), errors="coerce").fillna(0).astype(int)
            local = local[local["move_name"].isin(moves_meta_in.keys())].copy()
            if local.empty:
                master[pkmn] = pd.DataFrame(
                    {"move_name": pd.Series(dtype="string"),
                     "move_level_num": pd.Series(dtype="int64"),
                     "min_tm_stage": pd.Series(dtype="int64")}
                )
                continue
            tm_names = local.get("tm")
            if tm_names is None:
                local["min_tm_stage"] = 0
            else:
                tm_str = tm_names.astype("string").str.strip().str.lower()
                is_tm = tm_str.notna() & (tm_str != "") & (tm_str != "nan")
                min_stage = np.zeros(len(local), dtype=int)
                if is_tm.any():
                    stage_vals = local.loc[is_tm, "move_name"].map(
                        lambda x: moves_meta_in.get(x, {}).get("min_tm_stage", 999)
                    ).astype(int)
                    min_stage[is_tm.to_numpy()] = stage_vals.to_numpy()
                local["min_tm_stage"] = min_stage
            master[pkmn] = local[["move_name", "move_level_num", "min_tm_stage"]].copy()
            master[pkmn]["move_name"] = master[pkmn]["move_name"].astype("string")
        return master

    moves_meta = _build_moves_meta(moves_lookup, tm_stage_lookup, move_power_lookup)
    pkmn_moves_master = _build_pokemon_moves_master(pokemon_moves_lookup, moves_meta)

    gl_gen = int(gen)
    gl_stats_lookup = stats_lookup
    gl_evo_chain_lookup = evo_chain_lookup
    gl_calc = _calculate_move_damage
    gl_get_stats = _get_pokemon_stats_and_types
    gl_apply_ability = apply_ability_stat_modifiers
    gl_get_item_stage = get_evo_stage_for_item
    gl_stages = stages
    gl_min_stage_lookup = min_stage_lookup or {}

    # ---------- caches ----------
    @lru_cache(maxsize=50_000)
    def _stats_key_attacker(pokemon: str, level: int, stage_enc: int):
        return gl_get_stats(pokemon, level, gl_gen, stage_enc)

    @lru_cache(maxsize=50_000)
    def _stats_key_defender(pokemon: str, level: int):
        return gl_get_stats(pokemon, level, gl_gen)

    @lru_cache(maxsize=50_000)
    def _valid_evo_chain(pokemon: str, level: int, stage_enc: int) -> Tuple[bool, Tuple[str, ...]]:
        base = gl_stats_lookup.get(pokemon)
        if not base:
            return (False, tuple())
        evo_id = base.get("evo_id")
        chain = gl_evo_chain_lookup.get(evo_id, []) or []
        valid: List[str] = []
        found_current = False
        for node in chain:
            name = node.get("pokemon")
            if not name:
                continue
            evo_lvl = int(node.get("evo_lvl", 0) or 0)
            if evo_lvl > level:
                continue
            evo_item = node.get("evo_item", "")
            has_item = bool(pd.notna(evo_item) and str(evo_item).strip())
            if has_item:
                stage_req = gl_get_item_stage(evo_item, gl_stages)
                if stage_req is not None and stage_req > stage_enc:
                    if name == pokemon:
                        valid.append(name); found_current = True
                    continue
            valid.append(name)
            if name == pokemon:
                found_current = True
        if not found_current and pokemon in gl_stats_lookup:
            valid.append(pokemon)
        return (len(valid) > 0, tuple(valid))

    @lru_cache(maxsize=50_000)
    def _ability_adjusted_stats_attacker(pokemon: str, level: int, stage_enc: int, ability: Optional[str]):
        stats, *_ = _stats_key_attacker(pokemon, level, stage_enc)
        if stats is None:
            return None
        if gl_gen == 3:
            return gl_apply_ability(gl_gen, stats, ability, is_attacker=True)
        return stats

    @lru_cache(maxsize=50_000)
    def _ability_adjusted_stats_defender(pokemon: str, level: int, ability: Optional[str]):
        stats, *_ = _stats_key_defender(pokemon, level)
        if stats is None:
            return None
        if gl_gen == 3:
            return gl_apply_ability(gl_gen, stats, ability, is_attacker=False)
        return stats

    @lru_cache(maxsize=200_000)
    def _available_moves_key(pokemon: str, level: int, stage_enc: int, player_types_key: Tuple[str, ...]) -> Tuple[str, ...]:
        """Cached available moves for (pkmn, level, stage, filtered_types)."""
        df = pkmn_moves_master.get(pokemon)
        if df is None or df.empty:
            return tuple()
        mask = (df["move_level_num"] <= level) & (df["min_tm_stage"] <= stage_enc)
        if not mask.any():
            return tuple()
        moves = df.loc[mask, "move_name"].unique()
        # Removed type filtering - player pokemon should be able to use all available moves
        # The filtering logic was incorrectly trying to restrict moves based on encounter types
        return tuple(sorted(moves))

    # ---------- hot path ----------
    def _row_calc(pokemon: str, level: int, stage_enc: int, pokemon_enc: str,
                  ability: Optional[str], ability_enc: Optional[str],
                  player_types_tuple: Optional[Tuple[str, ...]]) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[int]]:
        if not pokemon or not pokemon_enc:
            return (None, None, None, None)
        try:
            level_i = int(level); stage_i = int(stage_enc)
        except Exception:
            return (None, None, None, None)

        if gl_min_stage_lookup:
            base = gl_stats_lookup.get(pokemon)
            if base:
                evo_id = base.get("evo_id")
                if evo_id in gl_min_stage_lookup and stage_i < gl_min_stage_lookup[evo_id]:
                    return (None, None, None, None)

        evo_ok, _ = _valid_evo_chain(pokemon, level_i, stage_i)
        if not evo_ok:
            return (None, None, None, None)

        # No longer using type filtering for player pokemon moves
        moves = _available_moves_key(pokemon, level_i, stage_i, tuple())
        if not moves:
            return (None, None, None, None)

        atk_stats = _ability_adjusted_stats_attacker(pokemon, level_i, stage_i, ability)
        if atk_stats is None:
            return (None, None, None, None)
        def_stats = _ability_adjusted_stats_defender(pokemon_enc, level_i, ability_enc)
        if def_stats is None:
            return (None, None, None, None)

        if gl_gen == 3 and ability_enc and str(ability_enc).lower() == "intimidate":
            # why: preserve side-effect parity
            atk_stats = gl_apply_ability(gl_gen, atk_stats, "intimidate", is_attacker=False)

        local_calc = gl_calc
        v_types_set = _stats_key_attacker(pokemon, level_i, stage_i)[2]
        enc_types = _stats_key_defender(pokemon_enc, level_i)[1]
        speed = int(gl_stats_lookup.get(pokemon, {}).get("speed", atk_stats.get("speed", 0)))

        best_move = None
        best_dmg = -1.0
        for m in moves:
            dmg = local_calc(
                m, atk_stats, def_stats, v_types_set, enc_types,
                level_i, speed, gl_gen, stage_i, ability, ability_enc,
                include_badge_boost=True, attacker_pokemon_name=pokemon,
            )
            if dmg > best_dmg:
                best_dmg = dmg; best_move = m

        if best_move is None or best_dmg < 0:
            return (None, None, None, None)
        
        # Extract HP and speed from attacker stats
        hp_out = int(atk_stats.get('hp', 0))
        speed_out = int(atk_stats.get('speed', 0))
        
        return (best_move, float(best_dmg), hp_out, speed_out)

    # ---------- public API (optimized) ----------
    def calculate_best_moves(df: pd.DataFrame) -> pd.DataFrame:
        """Compute best move per row with heavy de-dup of repeated keys."""
        if df.empty:
            out = pd.DataFrame({"move_out": pd.Series(dtype="string"),
                                "damage_out": pd.Series(dtype="float"),
                                "hp_out": pd.Series(dtype="Int64"),
                                "speed_out": pd.Series(dtype="Int64")}, index=df.index)
            return out

        cols = df.columns
        # prebind series (avoid multiple .iloc on original frame)
        pokemon = df["pokemon"].astype("string") if "pokemon" in cols else pd.Series([None]*len(df), index=df.index, dtype="string")
        pokemon_enc = df["pokemon_enc"].astype("string") if "pokemon_enc" in cols else pd.Series([None]*len(df), index=df.index, dtype="string")
        level = pd.to_numeric(df["level"], errors="coerce").astype("float64") if "level" in cols else pd.Series(np.nan, index=df.index, dtype="float64")
        stage_enc = pd.to_numeric(df["stage_enc"], errors="coerce").astype("float64") if "stage_enc" in cols else pd.Series(np.nan, index=df.index, dtype="float64")
        ability = df.get("ability")
        ability_enc = df.get("ability_enc")

        # Remove problematic move type filtering - player pokemon should be able to use all available moves
        # The original code was trying to filter based on encounter move types, which is incorrect
        move_types_tuple = [None] * len(df)  # No type filtering

        valid_mask = pokemon.notna() & pokemon_enc.notna() & pd.notna(level) & pd.notna(stage_enc)
        valid_idx = np.flatnonzero(valid_mask.to_numpy())
        n = len(df)

        # dedup work: compute by unique composite keys once
        key_to_result: Dict[Tuple[Any, ...], Tuple[Optional[str], Optional[float], Optional[int], Optional[int]]] = {}
        uniq_keys: List[Tuple[Any, ...]] = []

        # prebind arrays for speed
        pokemon_arr = pokemon.to_numpy()
        pokemon_enc_arr = pokemon_enc.to_numpy()
        level_arr = level.to_numpy()
        stage_arr = stage_enc.to_numpy()
        ability_arr = ability.to_numpy() if isinstance(ability, pd.Series) else np.array([None]*n, dtype=object)
        ability_enc_arr = ability_enc.to_numpy() if isinstance(ability_enc, pd.Series) else np.array([None]*n, dtype=object)

        for i in valid_idx:
            key = (
                pokemon_arr[i],
                int(level_arr[i]),
                int(stage_arr[i]),
                pokemon_enc_arr[i],
                ability_arr[i],
                ability_enc_arr[i],
                move_types_tuple[i],
            )
            if key not in key_to_result:
                uniq_keys.append(key)

        # compute once per unique key
        if TQDM_AVAILABLE and len(uniq_keys) > 1000:
            for key in tqdm(uniq_keys, desc="Computing player pokemon moves"):
                mv, dmg, hp, spd = _row_calc(*key)
                key_to_result[key] = (mv, dmg, hp, spd)
        else:
            for key in uniq_keys:
                mv, dmg, hp, spd = _row_calc(*key)
                key_to_result[key] = (mv, dmg, hp, spd)
                

        # map back to rows
        move_out = np.empty(n, dtype=object)
        move_out[:] = None
        dmg_out = np.full(n, np.nan, dtype=float)
        hp_out = np.full(n, np.nan, dtype=float)  # Use float for np.full, convert to Int64 later
        speed_out = np.full(n, np.nan, dtype=float)  # Use float for np.full, convert to Int64 later

        # map back to rows with progress bar for large datasets
        if TQDM_AVAILABLE and len(valid_idx) > 10000:
            for i in tqdm(valid_idx, desc="Mapping results back to rows"):
                key = (
                    pokemon_arr[i],
                    int(level_arr[i]),
                    int(stage_arr[i]),
                    pokemon_enc_arr[i],
                    ability_arr[i],
                    ability_enc_arr[i],
                    move_types_tuple[i],
                )
                mv, dmg, hp, spd = key_to_result.get(key, (None, None, None, None))
                move_out[i] = mv
                dmg_out[i] = dmg if dmg is not None else np.nan
                hp_out[i] = hp if hp is not None else np.nan
                speed_out[i] = spd if spd is not None else np.nan
        else:
            for i in valid_idx:
                key = (
                    pokemon_arr[i],
                    int(level_arr[i]),
                    int(stage_arr[i]),
                    pokemon_enc_arr[i],
                    ability_arr[i],
                    ability_enc_arr[i],
                    move_types_tuple[i],
                )
                mv, dmg, hp, spd = key_to_result.get(key, (None, None, None, None))
                
                
                move_out[i] = mv
                dmg_out[i] = dmg if dmg is not None else np.nan
                hp_out[i] = hp if hp is not None else np.nan
                speed_out[i] = spd if spd is not None else np.nan

        out = pd.DataFrame(index=df.index)
        out["move_out"] = pd.Series(move_out, dtype="object", index=df.index)
        out["damage_out"] = dmg_out
        out["hp_out"] = pd.Series(hp_out, dtype="Int64", index=df.index)  # Convert to nullable Int64
        out["speed_out"] = pd.Series(speed_out, dtype="Int64", index=df.index)  # Convert to nullable Int64
        
        return out

    return calculate_best_moves


# ---------------------------
# Helpers (pure & cached)
# ---------------------------

def _build_moves_meta(moves_lookup: Dict[str, Dict[str, Any]], tm_stage_lookup: Dict[str, int], move_power_lookup: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Return a compact dict of per-move metadata with precomputed fields.

    Fields:
      - power (float), type (lower str), category ('physical'/'special'), min_tm_stage (int)
    """
    out: Dict[str, Dict[str, Any]] = {}
    get = dict.get
    for name, m in moves_lookup.items():
        if not name:
            continue
        # Handle both normalized and original column names
        t = str(get(m, "move_type", get(m, "Type", ""))).lower()
        cat = str(get(m, "Category", "")).lower()
        power = float(get(m, "move_power", get(m, "Power", 0)) or 0)
        
        # Apply move power adjustments (critical for moves like doubleedge)
        move_name_norm = _norm(name)
        if move_name_norm in move_power_lookup:
            power = float(move_power_lookup[move_name_norm])
        
        # Don't skip zero-power moves - they are valid status moves that can be selected
        # even if they do no damage (e.g., Sleep Powder, Stun Spore, Growl, etc.)
            
        # TMs not explicitly listed should have high stage requirements (default 999)
        # Only explicitly listed TMs in stages should have lower requirements
        # Use normalized name for tm_stage_lookup
        min_stage = int(tm_stage_lookup.get(move_name_norm, 999) or 999)
        out[name] = {
            "power": power,
            "type": t,
            "category": cat,
            "min_tm_stage": min_stage,
        }
    return out


def _build_pokemon_moves_master(
    pokemon_moves_lookup: Dict[str, pd.DataFrame],
    moves_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """For each pokemon, return a small DataFrame of its learnset with numeric level and min TM stage.

    Keeps only necessary columns: ['move_name','move_level_num','min_tm_stage'].
    """
    master: Dict[str, pd.DataFrame] = {}
    for pkmn, df in pokemon_moves_lookup.items():
        if df is None or df.empty:
            master[pkmn] = pd.DataFrame(columns=["move_name", "move_level_num", "min_tm_stage"]).astype(
                {"move_name": "string", "move_level_num": "int64", "min_tm_stage": "int64"}
            )
            continue
        local = df.copy()
        if "move_name" not in local.columns:
            # Defensive: skip malformed
            master[pkmn] = pd.DataFrame(columns=["move_name", "move_level_num", "min_tm_stage"]).astype(
                {"move_name": "string", "move_level_num": "int64", "min_tm_stage": "int64"}
            )
            continue
        if "move_level_num" not in local.columns:
            local["move_level_num"] = pd.to_numeric(local.get("move_level"), errors="coerce").fillna(0).astype(int)
        # Filter out moves not in moves_meta (only moves with valid metadata)
        valid_moves_mask = local["move_name"].isin(moves_meta.keys())
        local = local[valid_moves_mask].copy()
        
        if local.empty:
            master[pkmn] = pd.DataFrame(columns=["move_name", "move_level_num", "min_tm_stage"]).astype(
                {"move_name": "string", "move_level_num": "int64", "min_tm_stage": "int64"}
            )
            continue
        
        # Precompute per-row TM minimum stage; non-TM moves get 0.
        tm_names = local.get("tm")
        if tm_names is None:
            local["min_tm_stage"] = 0
        else:
            tm_str = tm_names.astype("string").str.strip().str.lower()
            is_tm = tm_str.notna() & (tm_str != "") & (tm_str != "nan")
            min_stage = np.zeros(len(local), dtype=int)
            if is_tm.any():
                # Vectorized map using moves_meta
                # Why: avoid repeating dict lookups per row later
                # For TM moves, use min_tm_stage from moves_meta; default to 999 for unlisted TMs
                stage_vals = local.loc[is_tm, "move_name"].map(lambda x: moves_meta.get(x, {}).get("min_tm_stage", 999)).astype(int)
                min_stage[is_tm.to_numpy()] = stage_vals.to_numpy()
            local["min_tm_stage"] = min_stage

        master[pkmn] = local[["move_name", "move_level_num", "min_tm_stage"]].copy()
        master[pkmn]["move_name"] = master[pkmn]["move_name"].astype("string")
    return master


def calculate_player_pokemons(gen: int, player_pokemon_encounter_pairs_df: pd.DataFrame, config: dict = None, verbose: bool = False) -> pd.DataFrame:
    """
    Calculate player_pokemon levels from player_pokemon-encounter pairs for a specific generation.
{{ ... }}
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        player_pokemon_encounter_pairs_df (pd.DataFrame): Combined player_pokemons and encounters dataframe
        
    Returns:
        pd.DataFrame: player_pokemon-encounter pairs with calculated levels
    """
    if TQDM_AVAILABLE:
        print(f"\n=== Calculating player_pokemons for Generation {gen} ===")
        print(f"Processing {len(player_pokemon_encounter_pairs_df)} player_pokemon-encounter pairs...")
        print("[Loading Data] Loading and normalizing data files...")
        print("*** USING UPDATED CODE WITH ZERO-POWER MOVE FIX ***")
    
    
    # Load and normalize all required data files once
    base_path = Path(__file__).parent.parent
    try:
        combo = normalize_text_columns(player_pokemon_encounter_pairs_df.copy())
    except Exception as e:
        print(f"Error normalizing input data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise
    
    # Convert frequently used string columns to categorical for memory efficiency
    categorical_cols = ['pokemon_enc', 'location_enc']
    for col in categorical_cols:
        if col in combo.columns:
            combo[col] = combo[col].astype('category')
    
    # Load generation-specific data - Load raw first for parsing comma-separated fields
    stats_raw = pd.read_csv(base_path / f"data/gen_{gen}/stats_gen_{gen}.csv")
    stages_raw = pd.read_csv(base_path / f"data/gen_{gen}/stages_gen_{gen}.csv")
    moves = pd.read_csv(base_path / f"data/gen_{gen}/unique_moves_alt_gen_{gen}.csv")
    learnsets = pd.read_csv(base_path / f"data/gen_{gen}/learnsets_gen_{gen}.csv")
    
    # Add Struggle as a universal fallback move (use correct column names matching CSV format)
    struggle_move = pd.DataFrame({
        'Move': ['Struggle'],
        'Type': ['Normal'],
        'Category': ['Physical'], 
        'Power': [1],
        'Accuracy': [100],
        'PP': [1]
    })
    moves = pd.concat([moves, struggle_move], ignore_index=True)
    moves = moves.drop_duplicates(subset=['Move'], keep='first')  # Remove duplicates if Struggle already exists
    
    # Parse comma-separated fields from raw stages BEFORE normalization
    # Evolution item to stage lookup
    evo_item_stage_lookup = {}
    for _, stage_row in stages_raw.iterrows():
        evo_items = stage_row.get("evo_items", "")
        if pd.notna(evo_items) and str(evo_items).strip():
            # Split on comma before any normalization happens
            items = [item.strip() for item in str(evo_items).split(",")]
            for item in items:
                if item:
                    # Normalize the item name before storing in lookup
                    evo_item_stage_lookup[_norm(item)] = stage_row["location_stage"]
    
    # TM move to stage lookup
    tm_stage_lookup = {}
    for _, stage_row in stages_raw.iterrows():
        tms_str = stage_row.get("tms", "")
        if pd.notna(tms_str) and str(tms_str).strip():
            # Split on comma before any normalization happens
            tm_list = [tm.strip() for tm in str(tms_str).split(",")]
            for tm_move_name in tm_list:
                if tm_move_name:
                    normalized_move_name = _norm(tm_move_name)
                    tm_stage_lookup[normalized_move_name] = stage_row["location_stage"]
    
    # Load generation-agnostic data
    etable = pd.read_csv(base_path / "data/gen_all/exp_table.csv")
    exp_types = pd.read_csv(base_path / "data/gen_all/exp_types.csv")
    pokemon_availability = pd.read_csv(base_path / "data/gen_all/pokemon_availability.csv")
    typechart_raw = pd.read_csv(base_path / f"data/gen_{gen}/typechart_gen_{gen}.csv")
    
    # Filter availability data for current generation and normalize pokemon names
    availability_gen = pokemon_availability[pokemon_availability['gen'] == gen].copy()
    availability_gen['pokemon'] = availability_gen['pokemon'].apply(_norm)
    
    # Load stats_gen_3 for evo_id mapping (consistent across generations)
    if gen != 3:
        stats_gen_3 = pd.read_csv(base_path / "data/gen_3/stats_gen_3.csv")
        # Normalize the pokemon column for consistent lookups
        stats_gen_3['pokemon'] = stats_gen_3['pokemon'].apply(_norm)
        evo_id_lookup = stats_gen_3[['pokemon', 'evo_id']].drop_duplicates()
    else:
        # Use current stats for gen 3 - normalize pokemon column
        stats_temp = stats_raw.copy()
        stats_temp['pokemon'] = stats_temp['pokemon'].apply(_norm)
        evo_id_lookup = stats_temp[['pokemon', 'evo_id']].drop_duplicates()
    
    # Join availability data with evo_id - this will duplicate rows for Pokemon with multiple evo_ids (like Eevee)
    availability_with_evo = availability_gen.merge(evo_id_lookup, on='pokemon', how='left')
    
    # For each evo_id within this gen, find the minimum stage_available, treating NaN as 999
    availability_with_evo['stage_available_filled'] = availability_with_evo['stage_available'].fillna(999)
    min_stage_by_evo = availability_with_evo.groupby('evo_id', observed=True)['stage_available_filled'].min().reset_index()
    min_stage_by_evo.columns = ['evo_id', 'min_stage_available']
    
    # Create lookup dictionary for quick access
    min_stage_lookup = dict(zip(min_stage_by_evo['evo_id'], min_stage_by_evo['min_stage_available']))
    
    # Create individual Pokemon availability lookup for _identify_pokemon function
    # This prevents evolving to Pokemon that aren't available yet (e.g., Raichu before stage 38)
    pokemon_availability_lookup = {}
    for _, row in availability_gen.iterrows():
        pokemon_name = row['pokemon']
        stage_available = row['stage_available']
        if pd.notna(stage_available):
            # For each Pokemon, track the minimum stage where it's available
            if pokemon_name not in pokemon_availability_lookup or stage_available < pokemon_availability_lookup[pokemon_name]:
                pokemon_availability_lookup[pokemon_name] = stage_available
    
    # Now normalize all dataframes after extracting comma-separated values
    try:
        stats = normalize_text_columns(stats_raw)
    except Exception as e:
        print(f"Error normalizing stats data: {str(e)}")
        raise
    
    try:
        stages = normalize_text_columns(stages_raw)
    except Exception as e:
        print(f"Error normalizing stages data: {str(e)}")
        raise
    
    try:
        moves = normalize_text_columns(moves)
    except Exception as e:
        print(f"Error normalizing moves data: {str(e)}")
        raise
    
    try:
        learnsets = normalize_text_columns(learnsets)
    except Exception as e:
        print(f"Error normalizing learnsets data: {str(e)}")
        raise
    
    try:
        exp_types = normalize_text_columns(exp_types)
    except Exception as e:
        print(f"Error normalizing exp_types data: {str(e)}")
        raise
    
    # Convert string columns to categorical where appropriate for memory efficiency
    stats['pokemon'] = stats['pokemon'].astype('category')
    learnsets['pokemon'] = learnsets['pokemon'].astype('category')
    learnsets['move_name'] = learnsets['move_name'].astype('category')
    
    # Rename columns for consistency
    if "move" in moves.columns:
        moves = moves.rename(columns={"move": "move_name"})
    if "type" in moves.columns:
        moves = moves.rename(columns={"type": "move_type"})
    if "power" in moves.columns:
        moves = moves.rename(columns={"power": "move_power"})
    if "accuracy" in moves.columns:
        moves = moves.rename(columns={"accuracy": "move_accuracy"})
    
    # build lookup: normalized exp_type to (exp_array, level_array)
    exp_dict = {}
    level_arr = etable["Level"].to_numpy(dtype="int64", copy=False)
    for col in etable.columns:
        if col == "Level":
            continue
        exp_dict[_norm(col)] = (etable[col].to_numpy(dtype="int64", copy=False), level_arr)

    # Pre-compute lookup dictionaries for performance
    if TQDM_AVAILABLE:
        print("[Pre-processing] Building lookup dictionaries...")
    
    # Create pokemon availability lookup within the function scope for direct access
    # Normalize pokemon names to lowercase for consistent lookups
    local_pokemon_availability_lookup = {}
    for _, row in availability_gen.iterrows():
        pokemon_name = row['pokemon'].lower()  # Convert to lowercase for consistent lookups
        stage_available = row['stage_available']
        if pd.notna(stage_available):
            if pokemon_name not in local_pokemon_availability_lookup or stage_available < local_pokemon_availability_lookup[pokemon_name]:
                local_pokemon_availability_lookup[pokemon_name] = stage_available
    
    def _get_exp_type_for_evo_id(evo_id):
        """Get exp_type for an evo_id by finding the first pokemon in that evo_id"""
        if pd.isna(evo_id):
            return 'medium_slow'  # Default fallback
        
        # Find first pokemon in this evo_id from stats
        evo_stats = stats[stats["evo_id"] == evo_id]
        if evo_stats.empty:
            return 'medium_slow'
        
        first_pokemon = evo_stats.iloc[0]["pokemon"]
        
        # Use lookup dictionary for O(1) access
        return exp_type_lookup.get(first_pokemon, 'medium_slow')
    
    def _calculate_level_from_exp(exp_enc, exp_type):
        """Calculate level from exp_enc using exp_table"""
        if pd.isna(exp_enc):
            return pd.NA
        
        exp_type_norm = _norm(exp_type)
        if exp_type_norm not in exp_dict:
            exp_type_norm = 'medium_slow'
        if exp_type_norm not in exp_dict:
            return pd.NA
        
        # Filter exp_table for rows where exp_enc >= exp_type column value
        exp_arr, level_arr = exp_dict[exp_type_norm]
        
        # Find the highest level where exp requirement <= exp_enc
        valid_indices = exp_arr <= exp_enc
        if not valid_indices.any():
            return pd.NA
        
        # Get the max level (last valid index)
        max_valid_idx = np.where(valid_indices)[0][-1]
        level = int(level_arr[max_valid_idx])
        
        # Ensure level is never less than 5
        return max(level, 5)
    
    def _get_required_stage_for_item(evo_item):
        """Get required stage for evolution item from stages data"""
        if pd.isna(evo_item) or not evo_item.strip():
            return 0
        
        # Use pre-computed lookup dictionary
        evo_item_norm = _norm(evo_item)
        return evo_item_stage_lookup.get(evo_item_norm, 0)
    
    # Define trade evolution lists based on generation
    trade_evolutions_gen1 = ['alakazam', 'machamp', 'golem', 'gengar']
    trade_item_evolutions = ['metalcoat', 'dragonscale', 'kingsrock', 'upgrade', 'deepseatooth', 'deepseascale']
    
    # Check if trade evolutions are allowed from config
    allow_trade_evolutions = True
    if config and config.get('trade_evolutions', 'y').lower() == 'n':
        allow_trade_evolutions = False
        if verbose:
            print(f"Trade evolutions DISABLED - will block evolution into trade Pokemon")
    
    
    def _identify_pokemon(row):
        evo_id = row['evo_id_pp']
        exp_enc = row['exp_enc']
        stage_enc = row['stage_enc']
        
        if pd.isna(evo_id) or pd.isna(exp_enc) or pd.isna(stage_enc):
            return pd.NA
            
        # Step 1: Get exp_type for this evo_id
        exp_type = _get_exp_type_for_evo_id(evo_id)
        
        # Step 2: Calculate level from exp_enc and exp_type
        level = _calculate_level_from_exp(exp_enc, exp_type)
        if pd.isna(level):
            return pd.NA
        
        # Ensure stage_enc is numeric
        try:
            stage_enc = int(stage_enc)
        except (ValueError, TypeError):
            return pd.NA
        
        # Step 4: Filter stats for evo_id, level >= evo_lvl, stage_enc >= required_stage
        evo_stats = stats[stats["evo_id"] == evo_id]
        if evo_stats.empty:
            return pd.NA
        
        # Step 2: Get all pokemon in this evo_id from stats  
        valid_pokemon = []
        for _, pokemon_row in evo_stats.iterrows():
            evo_lvl = pokemon_row.get("evo_lvl", 0)
            evo_item = pokemon_row.get("evo_item", "")
            pokemon_name = pokemon_row["pokemon"]
            
            # Check level requirement
            if pd.notna(evo_lvl) and level < evo_lvl:
                continue
            
            # Step 3: Check stage requirement for evolution items
            required_stage = _get_required_stage_for_item(evo_item)
            
            # Step 4: Check evolution requirements
            # Check if this is a trade evolution
            is_trade_evolution = False
            
            if gen == 1:
                # Gen 1: Check by Pokemon name
                is_trade_evolution = _norm(pokemon_name) in trade_evolutions_gen1
            else:
                # Gen 2-3: Check by evo_item
                if pd.notna(evo_item):
                    evo_item_norm = _norm(evo_item)
                    is_trade_evolution = (evo_item_norm == 'trade' or 
                                         evo_item_norm in trade_item_evolutions)
            
            # Block trade evolutions if config says so
            if is_trade_evolution and not allow_trade_evolutions:
                continue
            
            # For item-based evolutions, only check if the item is available
            # Individual Pokemon availability is irrelevant for evolutions
            has_evo_item = pd.notna(evo_item) and str(evo_item).strip()
            
            # Only evolve if:
            # 1. No evolution item required (level-based evolution), OR
            # 2. Evolution item is required AND stage meets requirement
            item_available = not has_evo_item or stage_enc >= required_stage
            
            if item_available:
                valid_pokemon.append({
                    "pokemon": pokemon_name,
                    "evo_lvl": evo_lvl if pd.notna(evo_lvl) else 0,
                    "required_stage": required_stage,
                    "evo_item": evo_item
                })
        
        if not valid_pokemon:
            return pd.NA
        
        # Step 5: Take the highest evolution that meets stage requirements
        # For item-based evolutions, only evolve if the stage requirement is met
        valid_pokemon.sort(key=lambda x: (x["evo_lvl"], x["required_stage"]))
        
        # Find the highest evolution where stage requirements are actually met
        best_pokemon = None
        for pokemon in valid_pokemon:
            if pokemon["required_stage"] == 0 or stage_enc >= pokemon["required_stage"]:
                best_pokemon = pokemon
        
        return best_pokemon["pokemon"] if best_pokemon else valid_pokemon[0]["pokemon"]
    
    # Build pre-computed lookups for performance
    # Evo_id to exp_type lookup
    exp_type_lookup = dict(zip(exp_types["pokemon"], exp_types["exp_type"]))
    
    # Pokemon to evo_id lookup  
    pokemon_evo_lookup = dict(zip(stats["pokemon"], stats["evo_id"]))
    
    # Note: evo_item_stage_lookup and tm_stage_lookup were already built above from raw data
    
    # Pre-compute evo_id to exp_type mapping for all evo_ids in stats
    evo_id_to_exp_type = {}
    for evo_id in stats['evo_id'].dropna().unique():
        evo_stats = stats[stats['evo_id'] == evo_id]
        if not evo_stats.empty:
            first_pokemon = evo_stats.iloc[0]['pokemon']
            evo_id_to_exp_type[evo_id] = exp_type_lookup.get(first_pokemon, 'medium_slow')
    # Vectorized pokemon identification and level calculation
    if TQDM_AVAILABLE:
        print("\n[Step 1/5] Identifying Pokemon and calculating levels (vectorized)...")
    
    # Create masks for valid data
    if TQDM_AVAILABLE:
        print("  - Creating data validity masks...")
    valid_mask = combo['evo_id_pp'].notna() & combo['exp_enc'].notna() & combo['stage_enc'].notna()
    
    # Initialize pokemon and level columns
    combo['pokemon'] = pd.NA
    combo['level'] = pd.NA
    
    if valid_mask.sum() == 0:
        if TQDM_AVAILABLE:
            print("  - No valid data found, skipping pokemon identification")
    else:
        if TQDM_AVAILABLE:
            print(f"  - Processing {valid_mask.sum()} valid player_pokemon-encounter pairs...")
        
        # Use views where possible to avoid copying
        valid_combo = combo.loc[valid_mask].copy()  # Copy needed here for modifications
        
        # Calculate levels for all valid rows at once
        if TQDM_AVAILABLE:
            print("  - Mapping evo_id to exp_type...")
        valid_combo['exp_type'] = valid_combo['evo_id_pp'].map(evo_id_to_exp_type).fillna('medium_slow')
        
        # Fully optimized level calculation using vectorized operations
        if TQDM_AVAILABLE:
            print("  - Calculating Pokemon levels (fully vectorized)...")
        
        # Build level lookup for all unique (exp_enc, exp_type) combinations
        unique_combinations = valid_combo[['exp_enc', 'exp_type']].drop_duplicates()
        level_lookup = {}
        
        if TQDM_AVAILABLE:
            tqdm.pandas(desc="Building level lookup")
            for _, row in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc="Calculating levels"):
                exp_enc = row['exp_enc']
                exp_type = row['exp_type']
                level = _calculate_level_from_exp(exp_enc, exp_type)
                level_lookup[(exp_enc, exp_type)] = level
        else:
            for _, row in unique_combinations.iterrows():
                exp_enc = row['exp_enc']
                exp_type = row['exp_type']
                level = _calculate_level_from_exp(exp_enc, exp_type)
                level_lookup[(exp_enc, exp_type)] = level
        
        # Vectorized level assignment using map
        valid_combo['lookup_key'] = list(zip(valid_combo['exp_enc'], valid_combo['exp_type']))
        valid_combo['level'] = valid_combo['lookup_key'].map(level_lookup)
        valid_combo = valid_combo.drop('lookup_key', axis=1)
        
        # Fully optimized pokemon identification using pre-computed lookups and vectorization
        if TQDM_AVAILABLE:
            print("  - Identifying Pokemon species (fully optimized)...")
        
        # Build comprehensive pokemon identification lookup table
        pokemon_id_cache = {}
        
        # Pre-build evolution chain lookup if not exists
        local_evo_chain_lookup = {}
        for evo_id in stats['evo_id'].dropna().unique():
            chain_pokemon = stats[stats['evo_id'] == evo_id]
            chain_data = []
            for _, row in chain_pokemon.iterrows():
                pokemon_name = row['pokemon']
                evo_lvl = row.get('evo_lvl', 0)
                try:
                    evo_lvl = int(evo_lvl) if pd.notna(evo_lvl) else 0
                except (ValueError, TypeError):
                    evo_lvl = 0
                    
                evo_item = row.get('evo_item', '')
                required_stage = evo_item_stage_lookup.get(_norm(evo_item), 0) if pd.notna(evo_item) and str(evo_item).strip() else 0
                
                # Check if this is a trade evolution and should be blocked
                is_trade_evolution = False
                if gen == 1:
                    # Gen 1: Check by Pokemon name
                    is_trade_evolution = _norm(pokemon_name) in trade_evolutions_gen1
                else:
                    # Gen 2-3: Check by evo_item
                    if pd.notna(evo_item):
                        evo_item_norm = _norm(evo_item)
                        is_trade_evolution = (evo_item_norm == 'trade' or 
                                             evo_item_norm in trade_item_evolutions)
                
                # Skip trade evolutions if config says so
                if is_trade_evolution and not allow_trade_evolutions:
                    continue
                
                chain_data.append({
                    'pokemon': pokemon_name,
                    'evo_lvl': evo_lvl,
                    'required_stage': required_stage
                })
            
            # Sort by evolution level for each chain
            chain_data.sort(key=lambda x: x['evo_lvl'])
            local_evo_chain_lookup[evo_id] = chain_data
            
        
        # Pre-compute pokemon identification for all unique (evo_id, level, stage_enc) combinations
        unique_id_combinations = valid_combo[['evo_id_pp', 'level', 'stage_enc']].dropna().drop_duplicates()
        
        if TQDM_AVAILABLE:
            for _, row in tqdm(unique_id_combinations.iterrows(), total=len(unique_id_combinations), desc="Identifying Pokemon"):
                evo_id = row['evo_id_pp']
                level = int(row['level'])
                stage_enc = int(row['stage_enc'])
                
                chain_data = local_evo_chain_lookup.get(evo_id, [])
                if not chain_data:
                    pokemon_id_cache[(evo_id, level, stage_enc)] = pd.NA
                    continue
                
                # Find the highest valid evolution
                best_pokemon = None
                best_evo_lvl = -1
                
                for poke_data in chain_data:
                    # Check level requirement
                    if poke_data['evo_lvl'] > level:
                        continue
                    
                    # Check evolution item stage requirement
                    if poke_data['required_stage'] > stage_enc:
                        continue
                    
                    # This Pokemon is valid, update if it's better than current best
                    if poke_data['evo_lvl'] >= best_evo_lvl:
                        best_pokemon = poke_data['pokemon']
                        best_evo_lvl = poke_data['evo_lvl']
                
                final_pokemon = best_pokemon if best_pokemon else pd.NA
                pokemon_id_cache[(evo_id, level, stage_enc)] = final_pokemon
                
        else:
            for _, row in unique_id_combinations.iterrows():
                evo_id = row['evo_id_pp']
                level = int(row['level'])
                stage_enc = int(row['stage_enc'])
                
                chain_data = local_evo_chain_lookup.get(evo_id, [])
                if not chain_data:
                    pokemon_id_cache[(evo_id, level, stage_enc)] = pd.NA
                    continue
                
                # Find the highest valid evolution
                best_pokemon = None
                best_evo_lvl = -1
                
                for poke_data in chain_data:
                    # Check level requirement
                    if poke_data['evo_lvl'] > level:
                        continue
                    
                    # Check evolution item stage requirement
                    if poke_data['required_stage'] > stage_enc:
                        continue
                    
                    # This Pokemon is valid, update if it's better than current best
                    if poke_data['evo_lvl'] >= best_evo_lvl:
                        best_pokemon = poke_data['pokemon']
                        best_evo_lvl = poke_data['evo_lvl']
                
                final_pokemon = best_pokemon if best_pokemon else pd.NA
                pokemon_id_cache[(evo_id, level, stage_enc)] = final_pokemon
                
        
        # Vectorized pokemon identification using cached results
        def get_pokemon_from_cache(row):
            if pd.isna(row['level']) or pd.isna(row['evo_id_pp']):
                return pd.NA
            try:
                level = int(row['level'])
                stage_enc = int(row['stage_enc'])
                result = pokemon_id_cache.get((row['evo_id_pp'], level, stage_enc), pd.NA)
                return result
            except (ValueError, TypeError):
                return pd.NA
        
        valid_combo['pokemon'] = valid_combo.apply(get_pokemon_from_cache, axis=1)
        
        # Update original dataframe
        combo.loc[valid_mask, 'pokemon'] = valid_combo['pokemon']
        combo.loc[valid_mask, 'level'] = valid_combo['level']
    
    # Apply normalization to pokemon column
    combo["pokemon"] = combo["pokemon"].map(lambda x: _norm(x) if pd.notna(x) else x)

    # Filter duplicate evo_id_pp entries based on evolution requirements
    if TQDM_AVAILABLE:
        print("\n[Step 2/5] Filtering duplicate evo_id_pp entries by evolution requirements...")
    
    initial_count = len(combo)
    
    # Step 1: Filter by evolution requirements
    # Create mask for rows that meet evolution requirements
    evolution_mask = pd.Series([True] * len(combo), index=combo.index)
    
    # Filter by level >= evo_lvl_pp (treating NaN evo_lvl_pp as 0)
    if 'evo_lvl_pp' in combo.columns and 'level' in combo.columns:
        evo_lvl_filled = combo['evo_lvl_pp'].fillna(0)
        level_mask = combo['level'].notna() & (combo['level'] >= evo_lvl_filled)
        evolution_mask = evolution_mask & level_mask
    
    # Filter by stage_enc >= evo_item_stage_pp (treating NaN evo_item_stage_pp as 0)
    if 'evo_item_stage_pp' in combo.columns and 'stage_enc' in combo.columns:
        evo_item_stage_filled = combo['evo_item_stage_pp'].fillna(0)
        item_stage_mask = combo['stage_enc'].notna() & (combo['stage_enc'] >= evo_item_stage_filled)
        evolution_mask = evolution_mask & item_stage_mask
    
    # Filter by stage_enc >= stage_available_pp (treating NaN stage_available_pp as 0)
    if 'stage_available_pp' in combo.columns and 'stage_enc' in combo.columns:
        stage_available_filled = combo['stage_available_pp'].fillna(0)
        availability_mask = combo['stage_enc'].notna() & (combo['stage_enc'] >= stage_available_filled)
        evolution_mask = evolution_mask & availability_mask
    
    # Apply evolution requirements filter
    combo = combo.loc[evolution_mask].copy()
    
    # Step 2: For duplicate evo_id_pp + enc_id combinations, keep only the one with highest evo_index_pp
    if 'evo_id_pp' in combo.columns and 'enc_id' in combo.columns and 'evo_index_pp' in combo.columns:
        # Group by evo_id_pp and enc_id, then keep the row with maximum evo_index_pp
        combo = combo.loc[combo.groupby(['evo_id_pp', 'enc_id'])['evo_index_pp'].idxmax()]
    
    filtered_count = len(combo)
    if TQDM_AVAILABLE:
        removed_count = initial_count - filtered_count
        print(f"  - Removed {removed_count} rows that didn't meet evolution requirements")
        print(f"  - Removed duplicate evo_id_pp entries, keeping highest evo_index_pp")
        print(f"  - Filtered dataset: {filtered_count} rows remaining")

    # Note: evo_id column should already exist in the input dataframe
    # If it doesn't exist, add it by mapping from pokemon column
    # Ensure pokemon column exists and is properly normalized
    if 'pokemon' not in combo.columns:
        if TQDM_AVAILABLE:
            print("  - Warning: Pokemon column missing, reinitializing...")
        combo['pokemon'] = pd.NA
    
    # Ensure pokemon column is normalized
    combo["pokemon"] = combo["pokemon"].map(lambda x: _norm(x) if pd.notna(x) else x)
    
    # Create move lookup dictionary for O(1) access
    # Use the normalized move name as the key for consistent lookups
    moves_lookup = {}
    for _, row in moves.iterrows():
        move_name = row.get('move_name', row.get('Move', ''))
        if move_name:
            move_name_norm = _norm(move_name)
            moves_lookup[move_name_norm] = row.to_dict()
    
    # Create type effectiveness lookup for O(1) access
    type_eff_lookup = {}
    
    # The typechart is in matrix format with types as row/column headers
    # First column contains attacker types, first row contains defender types
    attacker_types = typechart_raw.iloc[:, 0].str.lower().tolist()
    defender_types = typechart_raw.columns[1:].str.lower().tolist()
    
    for i, attacker in enumerate(attacker_types):
        for j, defender in enumerate(defender_types):
            effectiveness = typechart_raw.iloc[i, j + 1]
            try:
                effectiveness = float(effectiveness)
            except (ValueError, TypeError):
                effectiveness = 1.0
            type_eff_lookup[(attacker, defender)] = effectiveness
    
    def get_type_effectiveness(move_type, defender_types):
        """Calculate type effectiveness for a move against defender types."""
        effectiveness = 1.0
        for def_type in defender_types:
            eff = type_eff_lookup.get((move_type, def_type), 1.0)
            effectiveness *= eff
        return effectiveness
    
    # Load move type classifications
    with open(base_path / "data/gen_all/physical_move_types.txt", 'r') as f:
        physical_moves = set(line.strip().lower().replace(" ", "_") for line in f if line.strip())
    
    with open(base_path / "data/gen_all/special_move_types.txt", 'r') as f:
        special_moves = set(line.strip().lower().replace(" ", "_") for line in f if line.strip())
    
    # Load critical hit moves as frozenset for hashability
    with open(base_path / "data/gen_all/crit_moves.txt", 'r') as f:
        crit_moves = frozenset(_norm(line.strip()) for line in f if line.strip())
    
    # Load move power adjustments and filter out zero-power moves
    move_power_lookup = {}
    try:
        power_adj_path = base_path / "data/gen_all/move_power_adjustments.csv"
        if power_adj_path.exists():
            power_adj_df = pd.read_csv(power_adj_path)
            if "Name" in power_adj_df.columns and "new_power" in power_adj_df.columns:
                # Apply normalization to move names for consistent matching
                power_adj_df["normalized_name"] = power_adj_df["Name"].apply(_norm)
                move_power_lookup = dict(zip(power_adj_df["normalized_name"], power_adj_df["new_power"]))
                if verbose:
                    print(f"[Optimization] Loaded {len(move_power_lookup)} move power adjustments")
                    zero_power_moves = power_adj_df[power_adj_df["new_power"] == 0]["Name"].tolist()
                    if zero_power_moves:
                        print(f"[Optimization] Zero-power moves to filter: {zero_power_moves}")
            elif "move_name" in power_adj_df.columns and "move_power" in power_adj_df.columns:
                power_adj_df["normalized_name"] = power_adj_df["move_name"].apply(_norm)
                move_power_lookup = dict(zip(power_adj_df["normalized_name"], power_adj_df["move_power"]))
    except FileNotFoundError:
        pass
    
    # Apply power adjustments to moves and filter out zero-power moves
    if 'move' in moves.columns:  
        initial_move_count = len(moves)
        # Apply power adjustments first - moves are already normalized by normalize_text_columns
        for idx, row in moves.iterrows():
            move_name_norm = row.get('move', '')  
            if move_name_norm in move_power_lookup:
                moves.at[idx, 'power'] = move_power_lookup[move_name_norm]
        
        # Filter out zero-power moves - use normalized column names
        moves = moves[(pd.notna(moves['power'])) & (moves['power'] != 0)]
        filtered_count = len(moves)
        if verbose:
            print(f"[Optimization] Filtered out {initial_move_count - filtered_count} zero-power move entries from moves")
    
    if verbose:
        print(f"[Optimization] Data loaded successfully. Moves: {len(moves)}, Pokemon: {len(stats)}")
        print(f"[Optimization] Setting up optimized lookups for damage calculation...")

    if 'move_name' in learnsets.columns and 'move' in moves.columns:
        initial_learnset_count = len(learnsets)
        # Create set of valid move names (non-zero power moves) - use normalized column names
        valid_moves = set(moves['move'])
        learnsets = learnsets[learnsets['move_name'].isin(valid_moves)]
        filtered_learnset_count = initial_learnset_count - len(learnsets)
        if verbose and filtered_learnset_count > 0:
            print(f"[Optimization] Removed {filtered_learnset_count} zero-power move entries from learnsets")
    
    # Helper function to get evolution stage for a pokemon with an evolution item
    def get_evo_stage_for_item(evo_item, stages_df):
        """Get the stage when an evolution item becomes available."""
        if pd.isna(evo_item) or not str(evo_item).strip():
            return None
            
        evo_item_norm = _norm(evo_item)
        
        # Use itertuples for better performance and reliability
        try:
            for stage_row in stages_df.itertuples(index=False):
                evo_items_str = getattr(stage_row, 'evo_items', '')
                if pd.notna(evo_items_str) and str(evo_items_str).strip():
                    # Handle comma-separated items
                    evo_items_list = [_norm(item.strip()) for item in str(evo_items_str).split(",")]
                    if evo_item_norm in evo_items_list:
                        return getattr(stage_row, 'location_stage', None)
        except Exception:
            # If there's any issue with iteration, return None
            pass
        return None
    
    # Pre-compute expensive lookups for better performance
    if TQDM_AVAILABLE:
        print("[Pre-processing] Building optimized lookups for damage calculation...")
    
    # Create fast lookup dictionaries
    stats_lookup = {}
    evo_chain_lookup = {}
    pokemon_moves_lookup = {}
    
    # Build pokemon stats lookup (O(1) access)
    for _, row in stats.iterrows():
        pokemon_name = row['pokemon']
        stats_lookup[pokemon_name] = {
            'evo_id': row['evo_id'],
            'hp': row.get('hp', 0),
            'attack': row.get('attack', 0),
            'defense': row.get('defense', 0),
            'sp_attack': row.get('sp_attack', 0),
            'sp_defense': row.get('sp_defense', 0),
            'speed': row.get('speed', 0),
            'types': row.get('types', ''),
            'evo_lvl': row.get('evo_lvl', 0),
            'evo_item': row.get('evo_item', '')
        }
    
    # Build evolution chain lookup (filtering out trade evolutions if disabled)
    for evo_id in stats['evo_id'].dropna().unique():
        chain_pokemon = stats[stats['evo_id'] == evo_id]
        evo_chain_lookup[evo_id] = []
        for _, row in chain_pokemon.iterrows():
            pokemon_name = row['pokemon']
            evo_item = row.get('evo_item', '')
            
            # Check if this is a trade evolution and should be blocked
            is_trade_evolution = False
            if gen == 1:
                # Gen 1: Check by Pokemon name
                is_trade_evolution = _norm(pokemon_name) in trade_evolutions_gen1
            else:
                # Gen 2-3: Check by evo_item
                if pd.notna(evo_item):
                    evo_item_norm = _norm(evo_item)
                    is_trade_evolution = (evo_item_norm == 'trade' or 
                                         evo_item_norm in trade_item_evolutions)
            
            # Skip trade evolutions if config says so
            if is_trade_evolution and not allow_trade_evolutions:
                continue
            
            evo_chain_lookup[evo_id].append({
                'pokemon': pokemon_name,
                'evo_lvl': row.get('evo_lvl', 0),
                'evo_item': evo_item
            })
    
    # Build pokemon moves lookup (pre-process complex move filtering)
    for pokemon_name in stats['pokemon'].unique():
        if pd.isna(pokemon_name):
            continue
            
        # Get evo_id for this pokemon
        pokemon_evo_id = stats_lookup.get(pokemon_name, {}).get('evo_id')
        if pd.isna(pokemon_evo_id):
            continue
            
        # Get moves from this pokemon AND all previous evolutions in the same evo_id
        # This allows evolved Pokemon like Kakuna to use moves from Weedle
        evo_chain = evo_chain_lookup.get(pokemon_evo_id, [])
        current_pokemon_evo_lvl = int(stats_lookup.get(pokemon_name, {}).get('evo_lvl', 0) or 0)
        current_pokemon_evo_item = str(stats_lookup.get(pokemon_name, {}).get('evo_item', '') or '').strip().lower()
        
        # Sort the evolution chain by evo_lvl to determine evolution order
        # This is critical for correctly identifying which forms come before the current one
        sorted_evo_chain = sorted(evo_chain, key=lambda x: (int(x.get('evo_lvl', 0) or 0), x.get('pokemon', '')))
        
        # Find the position of the current Pokemon in the sorted chain
        current_position = -1
        for idx, evo_entry in enumerate(sorted_evo_chain):
            if evo_entry.get('pokemon') == pokemon_name:
                current_position = idx
                break
        
        # Include the current Pokemon and all forms that come BEFORE it in the evolution chain
        accessible_pokemon = [pokemon_name]  # Always include current Pokemon
        
        if current_position > 0:
            # Include all Pokemon that appear before the current one in the sorted chain
            # This correctly handles all cases:
            # - Level evolutions: earlier forms have lower evo_lvl
            # - Stone evolutions: Pikachu (lvl=0) comes before Raichu (lvl=0, item=thunderstone)
            # - Trade evolutions: already filtered out from evo_chain if trade_evolutions="n"
            for idx in range(current_position):
                evo_pokemon = sorted_evo_chain[idx].get('pokemon')
                if evo_pokemon:
                    accessible_pokemon.append(evo_pokemon)
        
        # Get moves from all accessible Pokemon in the evolution line
        pokemon_moves_raw = learnsets[learnsets["pokemon"].isin(accessible_pokemon)].copy()
        
        # Always ensure Struggle is available as a fallback move for every Pokemon
        struggle_move = pd.DataFrame({
            'pokemon': [pokemon_name],
            'move_name': ['Struggle'],
            'move_level': [1],
            'tm': [pd.NA]
        })
        
        if not pokemon_moves_raw.empty:
            # Separate TM/HM moves from level-up moves
            tm_moves = pokemon_moves_raw[pokemon_moves_raw['tm'].notna() & (pokemon_moves_raw['tm'].str.strip() != '')].copy()
            level_moves = pokemon_moves_raw[pokemon_moves_raw['tm'].isna() | (pokemon_moves_raw['tm'].str.strip() == '')].copy()
            
            # For level-up moves, keep only the highest level for each move (across all accessible Pokemon)
            if not level_moves.empty:
                level_moves['move_level_numeric'] = pd.to_numeric(level_moves['move_level'], errors='coerce').fillna(0)
                level_moves = level_moves.loc[level_moves.groupby('move_name', observed=True)['move_level_numeric'].idxmax()]
                level_moves = level_moves.drop('move_level_numeric', axis=1)
            
            # For TM moves, drop duplicates
            if not tm_moves.empty:
                tm_moves = tm_moves.drop_duplicates(subset=['move_name'])
            
            # Combine moves from evolution line with Struggle fallback
            pokemon_moves = pd.concat([level_moves, tm_moves, struggle_move], ignore_index=True)
            # Remove duplicates in case Struggle was already in the learnset
            pokemon_moves = pokemon_moves.drop_duplicates(subset=['move_name'], keep='first')
            pokemon_moves_lookup[pokemon_name] = pokemon_moves
        else:
            # If no moves found, at least provide Struggle
            pokemon_moves_lookup[pokemon_name] = struggle_move
    
    # Update base stats columns to match the evolved Pokemon form
    if TQDM_AVAILABLE:
        print("  - Updating base stats columns to match evolved Pokemon...")
    
    # Apply base stats update to all rows with valid pokemon
    valid_pokemon_mask = combo['pokemon'].notna()
    if valid_pokemon_mask.any():
        # Process each unique pokemon to update stats
        unique_pokemon = combo.loc[valid_pokemon_mask, 'pokemon'].unique()
        
        for pokemon_name in unique_pokemon:
            if pd.notna(pokemon_name):
                # Look up the evolved Pokemon's base stats
                pokemon_stats = stats_lookup.get(pokemon_name)
                if pokemon_stats:
                    # Get mask for all rows with this pokemon
                    pokemon_mask = combo['pokemon'] == pokemon_name
                    
                    # Update all the base stat columns to match the evolved form
                    combo.loc[pokemon_mask, 'hp_pp_base'] = pokemon_stats.get('hp', 0)
                    combo.loc[pokemon_mask, 'attack_pp_base'] = pokemon_stats.get('attack', 0)
                    combo.loc[pokemon_mask, 'defense_pp_base'] = pokemon_stats.get('defense', 0)
                    combo.loc[pokemon_mask, 'sp_attack_pp_base'] = pokemon_stats.get('sp_attack', 0)
                    combo.loc[pokemon_mask, 'sp_defense_pp_base'] = pokemon_stats.get('sp_defense', 0)
                    combo.loc[pokemon_mask, 'speed_pp_base'] = pokemon_stats.get('speed', 0)
                    
                    # Also update type information to match evolved form
                    types_str = pokemon_stats.get('types', '')
                    if types_str:
                        types_list = [t.strip() for t in str(types_str).split('/') if t.strip()]
                        combo.loc[pokemon_mask, 'type_1_pp'] = types_list[0] if len(types_list) > 0 else ''
                        combo.loc[pokemon_mask, 'type_2_pp'] = types_list[1] if len(types_list) > 1 else ''
    
    # Shared helper functions for damage calculation
    def _get_pokemon_stats_and_types(pokemon_name, level, gen, stage_enc=None):
        """Get pokemon stats and types using pre-computed lookups."""
        pokemon_base_stats = stats_lookup.get(pokemon_name)
        if not pokemon_base_stats:
            return None, None, None
        
        base_stat_dict = {
            'hp': pokemon_base_stats['hp'],
            'attack': pokemon_base_stats['attack'],
            'defense': pokemon_base_stats['defense'],
            'sp_attack': pokemon_base_stats['sp_attack'],
            'sp_defense': pokemon_base_stats['sp_defense'],
            'speed': pokemon_base_stats['speed']
        }
        
        if stage_enc is not None:
            pokemon_stats = calculate_pokemon_stats(base_stat_dict, level, gen, stage_enc, stages)
        else:
            pokemon_stats = calculate_pokemon_stats(base_stat_dict, level, gen)
        
        # Special case: Shedinja always has 1 HP regardless of calculated value
        if _norm(pokemon_name) == 'shedinja':
            pokemon_stats['hp'] = 1
        
        # Get types for STAB/effectiveness calculation
        types_str = pokemon_base_stats.get("types", "")
        types = [t.strip().lower() for t in types_str.split("/")] if types_str else []
        types_set = set(types)
        
        return pokemon_stats, types, types_set

    def _calculate_move_damage(move_name, attacker_stats, defender_stats, attacker_types_set, defender_types, 
                              attacker_level, attacker_base_speed, gen, stage_enc=None, ability_attacker=None, 
                              ability_defender=None, include_badge_boost=False, attacker_pokemon_name=None):
        """Calculate damage for a single move with all modifiers."""
        nonlocal move_power_lookup, crit_moves, physical_moves, special_moves, get_type_effectiveness
        
        # Get move details from lookup
        if move_name not in moves_lookup:
            return 0
        move_info = moves_lookup[move_name]
        
        move_type = move_info.get("Type", "").lower()
        power = move_info.get("Power", 0)
        accuracy = move_info.get("Acc", 1.0)
        
        try:
            power = float(power)
            accuracy = float(accuracy) if pd.notna(accuracy) else 1.0
        except (ValueError, TypeError):
            return 0
        
        # Apply move power adjustments BEFORE checking if power is zero
        # This is critical for moves like Return that have 0 base power but are adjusted
        move_name_norm = _norm(move_name)
        if move_name_norm in move_power_lookup:
            power = move_power_lookup[move_name_norm]
        
        # NOW check if the move has zero power after adjustments
        if move_type == "none" or pd.isna(power) or power == 0:
            return 0
        
        # Special case: Truant Pokemon using Hyper Beam gets doubled power (75 -> 150)
        # This compensates for the 0.5x damage penalty applied in battle simulation
        has_truant = False
        if gen == 3:
            # Check if ability is explicitly set to truant
            if ability_attacker and str(ability_attacker).lower() == 'truant':
                has_truant = True
            # If ability is not available or empty, determine from pokemon name (Slakoth and Slaking have Truant)
            elif (not ability_attacker or pd.isna(ability_attacker) or str(ability_attacker) == '') and attacker_pokemon_name:
                pokemon_name_norm = _norm(str(attacker_pokemon_name))
                has_truant = pokemon_name_norm in ['slakoth', 'slaking']
        
        if has_truant and move_name_norm == 'hyper_beam':
            power = 150
        
        # Determine physical or special attack/defense
        is_physical = move_type in physical_moves
        is_special = move_type in special_moves
        
        if is_physical:
            attack = attacker_stats.get('attack', 0)
            defense = defender_stats.get('defense', 0)
        elif is_special:
            if gen <= 2:
                attack = attacker_stats.get('sp_attack', attacker_stats.get('special', 0))
                defense = defender_stats.get('sp_defense', defender_stats.get('special', 0))
            else:
                attack = attacker_stats.get('sp_attack', 0)
                defense = defender_stats.get('sp_defense', 0)
        else:
            return 0
        
        try:
            attack = float(attack)
            defense = float(defense) if defense > 0 else 1
        except (ValueError, TypeError):
            return 0
        
        ad_ratio = attack / defense
        
        # Calculate STAB
        stab = 1.5 if move_type in attacker_types_set else 1.0
        
        # Calculate type effectiveness
        type_effectiveness = get_type_effectiveness(move_type, defender_types)
        se1 = type_effectiveness
        se2 = 1.0
        
        # Calculate critical hit rate and multiplier
        is_high_crit = move_name_norm in crit_moves
        crit_rate = calculate_critical_hit_rate(gen, move_name, attacker_base_speed, is_high_crit)
        
        # Apply ability critical hit modifiers for Gen 3
        if gen == 3 and ability_attacker:
            crit_rate = apply_ability_crit_modifiers(gen, crit_rate, ability_attacker)
        
        crit = 1 + crit_rate
        
        # Calculate base damage by generation
        if gen == 1:
            damage = ((2 * attacker_level * crit / 5) + 2) * power * ad_ratio / 50 + 2
            damage = damage * stab * se1 * se2
        elif gen == 2:
            damage = (((2 * attacker_level / 5) + 2) * power * ad_ratio / 50) * crit + 2
            damage = damage * stab * se1 * se2
        elif gen == 3:
            damage = (((2 * attacker_level / 5) + 2) * power * ad_ratio / 50) + 2
            damage = damage * crit * stab * se1 * se2
        else:
            damage = 0
        
        # Apply ability damage modifiers for Gen 3
        if gen == 3:
            damage = apply_ability_damage_modifiers(gen, damage, move_type, ability_attacker, ability_defender)
        
        # Apply badge type boost (only for player_pokemon attacks)
        if include_badge_boost and stage_enc is not None:
            badge_type_boost = get_move_type_boost_multiplier(gen, stage_enc, move_type, stages)
            damage = damage * badge_type_boost
        
        # Apply ability accuracy modifiers for Gen 3
        if gen == 3 and ability_attacker:
            accuracy = apply_ability_accuracy_modifiers(gen, accuracy, ability_attacker, move_name)
        
        # Apply accuracy multiplier for effective damage
        damage = float(int(damage * accuracy))
        
        return damage

    # Legacy _calculate_highest_damage and _calculate_enc_damage functions have been replaced 
    # by the fast damage engine above. All damage calculations now use the optimized context system
    
    # Initialize the fast damage computation engines
    if TQDM_AVAILABLE:
        print("[Optimization] Initializing fast damage computation engines...")
    
    # Create the fast damage context with all pre-computed lookups (for player_pokemon damage)
    fast_damage_calculator = prepare_damage_context(
        gen=gen,
        moves_lookup=moves_lookup,
        tm_stage_lookup=tm_stage_lookup,
        pokemon_moves_lookup=pokemon_moves_lookup,
        stats_lookup=stats_lookup,
        evo_chain_lookup=evo_chain_lookup,
        get_evo_stage_for_item=get_evo_stage_for_item,
        stages=stages,
        _get_pokemon_stats_and_types=_get_pokemon_stats_and_types,
        _calculate_move_damage=_calculate_move_damage,
        apply_ability_stat_modifiers=apply_ability_stat_modifiers,
        move_power_lookup=move_power_lookup,
        min_stage_lookup=min_stage_lookup,
    )
    
    # Create the fast encounter damage context (for encounter damage)
    fast_enc_damage_calculator = prepare_enc_damage_context(
        gen=gen,
        stats_lookup=stats_lookup,
        _get_pokemon_stats_and_types=_get_pokemon_stats_and_types,
        _calculate_move_damage=_calculate_move_damage,
        apply_ability_stat_modifiers=apply_ability_stat_modifiers,
        min_stage_lookup=min_stage_lookup,
    )
    
    # Apply player_pokemon damage calculation (move_out, damage_out) with pre-filtering
    # Pre-filter to separate available vs unavailable player_pokemons
    def is_player_pokemon_available(row):
        evo_id = row['evo_id_pp']
        stage_enc = row['stage_enc']
        pokemon = row.get('pokemon', '')
        
        # Exclude specific Pokemon that should never have damage calculations
        excluded_pokemon = ['mew']
        if not pd.isna(pokemon) and pokemon and str(pokemon).lower() in excluded_pokemon:
            return False
            
        # player_pokemon is available only if all conditions are met:
        # 1. evo_id is not missing
        # 2. evo_id is tracked in min_stage_lookup
        # 3. stage_enc is not missing  
        # 4. current stage >= minimum required stage for evo_id
        # Note: We do NOT check individual Pokemon availability here
        # because evolved forms can be obtained through evolution
        evo_id_available = (not pd.isna(evo_id) and 
                           evo_id in min_stage_lookup and 
                           not pd.isna(stage_enc) and
                           stage_enc >= min_stage_lookup[evo_id])
        
        return evo_id_available
    
    if TQDM_AVAILABLE:
        print("\n[Step 3/6] Pre-filtering player_pokemons by availability...")
        tqdm.pandas(desc="Filtering player_pokemons")
        available_mask = combo.progress_apply(is_player_pokemon_available, axis=1)
    else:
        available_mask = combo.apply(is_player_pokemon_available, axis=1)
    
    # Use views to avoid unnecessary copying
    available_df = combo.loc[available_mask]
    unavailable_df = combo.loc[~available_mask]
    
    if TQDM_AVAILABLE:
        print(f"  - Found {len(available_df)} available and {len(unavailable_df)} unavailable player_pokemons")
    
    # Apply function only to available player_pokemons using the fast damage engine
    if not available_df.empty:
        if TQDM_AVAILABLE:
            print(f"\n[Step 4/6] Calculating player_pokemon damage for {len(available_df)} available player_pokemons (FAST ENGINE)...")
        
        # Use the fast damage computation engine instead of the old row-by-row approach
        available_results = fast_damage_calculator(available_df)
    else:
        available_results = pd.DataFrame(columns=['move_out', 'damage_out', 'hp_out', 'speed_out'])
    
    # Create NA results for unavailable player_pokemons
    if not unavailable_df.empty:
        unavailable_results = pd.DataFrame({
            'move_out': [pd.NA] * len(unavailable_df),
            'damage_out': [pd.NA] * len(unavailable_df),
            'hp_out': [pd.NA] * len(unavailable_df),
            'speed_out': [pd.NA] * len(unavailable_df)
        }, index=unavailable_df.index)
    else:
        unavailable_results = pd.DataFrame(columns=['move_out', 'damage_out', 'hp_out', 'speed_out'])
    
    # Combine results and sort by original index
    player_pokemon_damage_results = pd.concat([available_results, unavailable_results]).sort_index()
    
    
    # Ensure proper alignment and avoid dtype conflicts during concatenation
    # Use direct column assignment instead of pd.concat to avoid index/dtype issues
    
    combo = combo.copy()  # Avoid modifying the original
    combo['move_out'] = player_pokemon_damage_results['move_out']
    combo['damage_out'] = player_pokemon_damage_results['damage_out']
    combo['hp_out'] = player_pokemon_damage_results['hp_out']
    combo['speed_out'] = player_pokemon_damage_results['speed_out']
    
    # Apply encounter damage calculation (move_in, damage_in) to ALL rows
    # Don't split by availability - let the fast calculator handle it internally
    if TQDM_AVAILABLE:
        print(f"\n[Step 5/6] Calculating encounter damage for {len(combo)} player_pokemons (FAST ENGINE)...")
    
    
    # Use the fast encounter damage computation engine on the full DataFrame
    enc_damage_results = fast_enc_damage_calculator(combo)
    
    
    # Instead of concatenating (which can cause conflicts), add columns directly
    combo['move_in'] = enc_damage_results['move_in']
    combo['damage_in'] = enc_damage_results['damage_in']
    combo['hp_in'] = enc_damage_results['hp_in']
    combo['speed_in'] = enc_damage_results['speed_in']
    
    
    # Remove pokemon_pp column to avoid confusion - pokemon column is the authoritative one
    # that contains the correct evolution form based on stage and level
    if 'pokemon_pp' in combo.columns:
        combo = combo.drop(columns=['pokemon_pp'])
        if TQDM_AVAILABLE:
            print("  - Removed pokemon_pp column to avoid confusion (pokemon column is authoritative)")
    
    if TQDM_AVAILABLE:
        print(f"\n[SUCCESS] Generation {gen} processing complete!")
        print(f"  - Final dataset: {len(combo)} rows with damage calculations")
        print(f"  - Performance: Vectorized operations + pre-computed lookups + categorical optimization")
    
    return combo

# Optional: Test the function
if __name__ == "__main__":
    # Load configuration from config.json file (same as main pipeline)
    def load_config():
        """Load configuration from config.json file."""
        config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using defaults.")
            return {"gen": [1, 2, 3], "trade_evolutions": "y"}
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}. Using defaults.")
            return {"gen": [1, 2, 3], "trade_evolutions": "y"}
    
    # Load config once at the beginning
    config = load_config()
    print(f"Loaded config: {config}")
    
    # Determine which generations to process based on config
    generations_to_process = config.get("gen", [1, 2, 3])
    if not isinstance(generations_to_process, list):
        generations_to_process = [generations_to_process]
    
    print(f"Processing generations from config: {generations_to_process}")
    
    for g in generations_to_process:
        # Load from CSV file for testing - look in intermediate_files directory
        intermediate_files_dir = Path(__file__).parent.parent / "intermediate_files"
        input_path = intermediate_files_dir / f"player_pokemon_encounter_pairs_gen{g}.csv"
        
        try:
            # Read and normalize CSV data with error handling
            try:
                raw_df = pd.read_csv(input_path)
                combo_df = normalize_text_columns(raw_df)
                print(f"Successfully loaded and normalized data for generation {g}")
            except Exception as e:
                print(f"Error in normalize_text_columns for generation {g}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                continue
            
            # Calculate player pokemons with error handling, passing config parameter
            try:
                df = calculate_player_pokemons(g, combo_df, config=config, verbose=True)
                print(f"Successfully calculated player pokemons for generation {g}")
                print(f"Config settings applied: trade_evolutions={config.get('trade_evolutions', 'y')}")
            except Exception as e:
                print(f"Error in calculate_player_pokemons for generation {g}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                continue
            
            # Save to intermediate_files directory when run manually for debugging
            output_path = intermediate_files_dir / f"player_pokemon_encounters_with_level_gen{g}.csv"
            df.to_csv(output_path, index=False)
            print(f"\n*** DEBUG MODE: Gen {g} output saved to {output_path} ***")
            
        except FileNotFoundError:
            print(f"Error: Missing input file for gen {g}")
            print(f"Expected file: {input_path}")
            print("Please run generate_player_pokemons_x_encounters.py first to create the required input file.")
            break
        except Exception as e:
            print(f"Unexpected error processing generation {g}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
