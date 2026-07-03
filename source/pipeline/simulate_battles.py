import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import os
import math
import zlib

from common import paths
from common.config import load_config
from common.logging_setup import get_logger
from common.abilities import TRUANT_SPECIES

logger = get_logger(__name__)

# Scoring constants
M = 1000
TIE_SCORE = 1000.0
UNAVAILABLE_SCORE = 1000000000.0

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, total=None):
        return iterable

def filter_unwinnable_columns_pivot(pivot_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove columns (encounters) that have no winning combination (a value in [0, 1)) for any row.

    This is the pivot-table equivalent of ``optimize.milp.filter_unwinnable_columns``, adapted to
    operate directly on a DataFrame indexed by ``evo_id_pp`` with ``enc_id`` columns, so it can run
    in ``simulate_battles`` *before* dominance filtering (see module docstring / plan notes on why
    column pruning must happen before dominance filtering rather than after).
    """
    if pivot_df.empty or len(pivot_df.columns) == 0:
        return pivot_df

    cols_to_drop = []
    for col in pivot_df.columns:
        col_values = pivot_df[col].values
        has_winning_combination = np.any((col_values >= 0) & (col_values < 1))
        if not has_winning_combination:
            cols_to_drop.append(col)

    if cols_to_drop:
        if verbose:
            logger.debug(f"Removed {len(cols_to_drop)} unwinnable columns (no values between 0 and 1)")
        return pivot_df.drop(columns=cols_to_drop)

    if verbose:
        logger.debug("No unwinnable columns found (all columns have at least one value between 0 and 1)")
    return pivot_df


def apply_drop_first_rival_encounter(pivot_df: pd.DataFrame, config: dict = None, verbose: bool = True) -> pd.DataFrame:
    """
    Drop the first 3 (sorted by enc_id) columns if ``drop_first_rival_encounter`` is enabled in config.

    Pivot-table equivalent of the same-named logic in ``optimize.milp.calculate_best_party_milp``,
    moved here so it runs before dominance filtering.
    """
    config = config or {}
    if str(config.get('drop_first_rival_encounter', 'n')).lower() != 'y':
        return pivot_df

    if len(pivot_df.columns) < 3:
        return pivot_df

    try:
        sorted_cols = sorted(pivot_df.columns, key=lambda x: int(x))
    except (ValueError, TypeError):
        sorted_cols = sorted(pivot_df.columns, key=lambda x: str(x))

    cols_to_drop = sorted_cols[:3]
    if verbose:
        logger.info(f"Dropped first 3 rival encounter columns: {cols_to_drop}")
    return pivot_df.drop(columns=cols_to_drop)


def simulate_battles(
    gen: int,
    player_pokemon_encounters_df: pd.DataFrame,
    config: dict = None,
    party_size: int = None,
) -> pd.DataFrame:
    """
    Simulate Pokemon battles between variants and encounters to calculate performance scores.
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        player_pokemon_encounters_df (pd.DataFrame): DataFrame with variant-encounter pairs, damage calculations,
                                                    and pre-calculated HP/speed stats (hp_out, speed_out, hp_in, speed_in)
        config (dict, optional): Loaded config dict (used for ``drop_first_rival_encounter``). Loaded
            from disk if not provided.
        party_size (int, optional): Party size this batch of rows belongs to. Only used to namespace
            debug/summary artifact filenames so multiple party-size runs for the same generation don't
            overwrite each other.
        
    Returns:
        pd.DataFrame: Pivot table with evo_id_pp as rows, enc_id as columns, and scores as values
    """
    if config is None:
        config = load_config()
    if TQDM_AVAILABLE:
        logger.info(f"Simulating Battles for Generation {gen}")
        logger.info(f"Processing {len(player_pokemon_encounters_df)} variant-encounter pairs...")
    
    # Create a copy to avoid modifying the original dataframe
    battles_df = player_pokemon_encounters_df.copy()

    # Initialize the score column
    battles_df['score'] = pd.NA
    
    if TQDM_AVAILABLE:
        logger.info("Simulating individual battles...")
    
    def simulate_single_battle(row):
        """Mathematically calculate battle outcome using pre-calculated HP/speed stats and full ability support."""
        
        # Check if variant is available (damage_out is not null)
        if pd.isna(row['damage_out']):
            # Default score for unavailable variants
            return UNAVAILABLE_SCORE
        
        # Extract battle parameters
        damage_out = row['damage_out']  # Variant's damage to encounter
        damage_in = row['damage_in']    # Encounter's damage to variant
        
        # Handle missing damage values
        if pd.isna(damage_in):
            damage_in = 0
        if pd.isna(damage_out):
            damage_out = 0
            
        # Get Pokemon stats for HP and speed calculation
        pokemon = row.get('pokemon', '')
        pokemon_enc = row.get('pokemon_enc', '')
        level = row.get('level', 50)
        level_enc = row.get('level_enc', 50)
        
        # Check for truant ability in Gen 3
        has_truant = False
        enc_has_truant = False
        if gen == 3:
            # Check player Pokemon ability
            ability = row.get('ability', '')
            pokemon_species = row.get('pokemon_pp', row.get('pokemon', ''))
            
            # If ability is not in data, determine from species (Slakoth and Slaking have Truant)
            if pd.isna(ability) or str(ability) == '':
                has_truant = str(pokemon_species).lower() in TRUANT_SPECIES
            else:
                has_truant = str(ability).lower() == 'truant'
            
            # Check encounter Pokemon ability
            ability_enc = row.get('ability_enc', '')
            enc_has_truant = str(ability_enc).lower() == 'truant' if pd.notna(ability_enc) else False
        
        # Get HP values from the pre-calculated columns (hp_out, hp_in)
        # These are calculated during the damage calculation phase for accurate stats
        player_pokemonhp = row.get('hp_out', max(1, int(level * 2 + 50)))  # Use pre-calculated HP or fallback
        encounter_hp = row.get('hp_in', max(1, int(level_enc * 2 + 50)))  # Use pre-calculated HP or fallback
        
        # Store starting HP for score calculation
        player_pokemonhp_start = player_pokemonhp
        encounter_hp_start = encounter_hp
        
        # Get speed values from the pre-calculated columns (speed_out, speed_in)
        # These are calculated during the damage calculation phase for accurate stats
        # Use deterministic fallback based on CRC32 to avoid per-run hash randomization
        player_pokemonspeed = row.get('speed_out', level + (zlib.crc32(str(pokemon).encode('utf-8')) % 20))
        encounter_speed = row.get('speed_in', level_enc + (zlib.crc32(str(pokemon_enc).encode('utf-8')) % 20))
        
        # Calculate effective damage per turn accounting for abilities
        player_pokemon_effective_damage = damage_out
        encounter_effective_damage = damage_in
        
        # Adjust for Truant ability (halves effective damage since Pokemon attacks every other turn)
        if has_truant:
            player_pokemon_effective_damage *= 0.5
        if enc_has_truant:
            encounter_effective_damage *= 0.5
        
        # Handle speed tie case
        if player_pokemonspeed == encounter_speed:
            # Both attack simultaneously with half damage
            player_pokemon_effective_damage *= 0.5
            encounter_effective_damage *= 0.5
            
            # Calculate turns to KO
            if player_pokemon_effective_damage <= 0 and encounter_effective_damage <= 0:
                # Neither can damage the other (tie scenario)
                return TIE_SCORE
            elif player_pokemon_effective_damage <= 0:
                # Only encounter can deal damage (loss scenario: add M*M)
                return 0 + M*M  # Variant loses
            elif encounter_effective_damage <= 0:
                # Only variant can deal damage
                return 0  # Variant wins with full HP
            else:
                # Both can deal damage
                player_pokemonturns_to_ko = math.ceil(encounter_hp / player_pokemon_effective_damage)
                encounter_turns_to_ko = math.ceil(player_pokemonhp / encounter_effective_damage)
                
                if player_pokemonturns_to_ko < encounter_turns_to_ko:
                    # Variant wins
                    remaining_hp = player_pokemonhp - (encounter_effective_damage * player_pokemonturns_to_ko)
                    remaining_hp = max(0, remaining_hp)
                    return 1 - (remaining_hp / player_pokemonhp_start)
                elif encounter_turns_to_ko < player_pokemonturns_to_ko:
                    # Variant loses (loss scenario: add M*M)
                    remaining_enc_hp = encounter_hp - (player_pokemon_effective_damage * encounter_turns_to_ko)
                    remaining_enc_hp = max(0, remaining_enc_hp)
                    return (remaining_enc_hp / encounter_hp_start) + M*M
                else:
                    # Tie (both KO at same time) (tie scenario)
                    return TIE_SCORE
        
        # Handle normal turn order
        player_pokemongoes_first = player_pokemonspeed > encounter_speed
        
        # Calculate battle outcome mathematically
        if player_pokemon_effective_damage <= 0 and encounter_effective_damage <= 0:
            # Neither can damage the other (tie scenario)
            return TIE_SCORE
        elif player_pokemon_effective_damage <= 0:
            # Only encounter can deal damage (loss scenario: add M*M)
            return 0 + M*M  # Variant loses
        elif encounter_effective_damage <= 0:
            # Only variant can deal damage  
            return 0  # Variant wins with full HP
        
        # Both can deal damage - calculate turns needed
        player_pokemonturns_to_ko = math.ceil(encounter_hp / player_pokemon_effective_damage)
        encounter_turns_to_ko = math.ceil(player_pokemonhp / encounter_effective_damage)
        
        if player_pokemongoes_first:
            # Variant attacks first each round
            if player_pokemonturns_to_ko <= encounter_turns_to_ko:
                # Variant wins (KOs encounter before or at same time as being KO'd)
                # Calculate remaining HP after encounter's attacks
                encounter_attacks = min(player_pokemonturns_to_ko - 1, encounter_turns_to_ko - 1)
                if player_pokemonturns_to_ko == encounter_turns_to_ko:
                    encounter_attacks = encounter_turns_to_ko - 1  # Encounter gets one less attack
                
                remaining_hp = player_pokemonhp - (encounter_effective_damage * encounter_attacks)
                remaining_hp = max(0, remaining_hp)
                return 1 - (remaining_hp / player_pokemonhp_start)
            else:
                # Variant loses
                # Calculate remaining encounter HP after variant's attacks (loss scenario: add M*M)
                player_pokemonattacks = min(encounter_turns_to_ko, player_pokemonturns_to_ko - 1)
                remaining_enc_hp = encounter_hp - (player_pokemon_effective_damage * player_pokemonattacks)
                remaining_enc_hp = max(0, remaining_enc_hp)
                return (remaining_enc_hp / encounter_hp_start) + M*M
        else:
            # Encounter attacks first each round
            if encounter_turns_to_ko <= player_pokemonturns_to_ko:
                # Variant loses
                # Calculate remaining encounter HP after variant's attacks (loss scenario: add M*M)
                player_pokemonattacks = min(encounter_turns_to_ko - 1, player_pokemonturns_to_ko - 1)
                if encounter_turns_to_ko == player_pokemonturns_to_ko:
                    player_pokemonattacks = player_pokemonturns_to_ko - 1  # Variant gets one less attack
                
                remaining_enc_hp = encounter_hp - (player_pokemon_effective_damage * player_pokemonattacks)
                remaining_enc_hp = max(0, remaining_enc_hp)
                return (remaining_enc_hp / encounter_hp_start) + M*M
            else:
                # Variant wins
                # Calculate remaining HP after encounter's attacks
                encounter_attacks = min(player_pokemonturns_to_ko, encounter_turns_to_ko - 1)
                remaining_hp = player_pokemonhp - (encounter_effective_damage * encounter_attacks)
                remaining_hp = max(0, remaining_hp)
                return 1 - (remaining_hp / player_pokemonhp_start)
    
    # Apply battle simulation
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="Battle simulation")
        battles_df['score'] = battles_df.progress_apply(simulate_single_battle, axis=1)
    else:
        battles_df['score'] = battles_df.apply(simulate_single_battle, axis=1)
    
    if TQDM_AVAILABLE:
        logger.debug("Filtering to required columns...")
    
    # Filter down to required columns
    required_columns = ['evo_id_pp', 'enc_id', 'score']
    available_columns = [col for col in required_columns if col in battles_df.columns]
    
    if len(available_columns) != len(required_columns):
        missing_cols = set(required_columns) - set(available_columns)
        if TQDM_AVAILABLE:
            logger.warning(f"Missing columns {missing_cols}. Available columns: {list(battles_df.columns)}")
        
        # Create missing columns if needed
        if 'evo_id_pp' not in battles_df.columns:
            battles_df['evo_id_pp'] = battles_df.index
        if 'enc_id' not in battles_df.columns:
            battles_df['enc_id'] = battles_df.get('pokemon_enc', battles_df.index)
    
    # Select final columns
    final_df = battles_df[['evo_id_pp', 'enc_id', 'score']].copy()
    
    if TQDM_AVAILABLE:
        logger.debug("Creating pivot table...")
    
    # Create pivot table with evo_id_pp as rows, enc_id as columns, score as values
    try:
        pivot_df = final_df.pivot(index='evo_id_pp', columns='enc_id', values='score')
        
        # Deterministic ordering of rows and columns
        pivot_df = pivot_df.sort_index()
        try:
            sorted_cols = sorted(pivot_df.columns, key=lambda x: int(x))
        except Exception:
            sorted_cols = sorted(pivot_df.columns, key=lambda x: str(x))
        pivot_df = pivot_df.reindex(columns=sorted_cols)

        if TQDM_AVAILABLE:
            logger.debug("Pruning unwinnable columns and rival-encounter columns before dominance filtering...")

        # Column pruning MUST happen before dominance/equivalence filtering: dominance depends on
        # the full set of columns being compared, and pruning columns first can reveal additional
        # dominated/equivalent rows that would otherwise be missed (or incorrectly kept) if dominance
        # were computed on the larger, unpruned column set.
        pivot_df = filter_unwinnable_columns_pivot(pivot_df, verbose=TQDM_AVAILABLE)
        pivot_df = apply_drop_first_rival_encounter(pivot_df, config=config, verbose=TQDM_AVAILABLE)

        if TQDM_AVAILABLE:
            logger.debug("Applying dominance filtering (removing strictly worse rows)...")
        
        # Build evo_id_pp -> pokemon name mapping for summaries
        def _build_evo_id_pp_to_name_map(df: pd.DataFrame) -> dict:
            name_col = None
            if 'pokemon_pp' in df.columns and df['pokemon_pp'].notna().any():
                name_col = 'pokemon_pp'
            elif 'pokemon' in df.columns and df['pokemon'].notna().any():
                name_col = 'pokemon'
            mapping = {}
            try:
                if name_col is not None and 'evo_id_pp' in df.columns:
                    tmp = df[['evo_id_pp', name_col]].dropna().drop_duplicates('evo_id_pp')
                    mapping = {row['evo_id_pp']: str(row[name_col]) for _, row in tmp.iterrows()}
            except Exception:
                mapping = {}
            return mapping

        evo_id_pp_to_name = _build_evo_id_pp_to_name_map(battles_df)

        # Prepare a shared summary file in intermediate_files for this generation (and party size,
        # when running the multi-party-size pipeline, to avoid different party sizes overwriting
        # each other's summaries for the same generation).
        ps_suffix = f"_ps{party_size}" if party_size is not None else ""
        summary_file = paths.intermediate_dir() / f"dominated_and_equivalent_gen{gen}{ps_suffix}.txt"

        # Apply dominance filtering to remove strictly worse rows and write dominance summary
        pivot_df = filter_dominated_rows(pivot_df, gen=gen, evo_id_pp_to_name=evo_id_pp_to_name, summary_file=summary_file)
        
        if TQDM_AVAILABLE:
            logger.debug("Identifying equivalent rows...")
        
        # Identify equivalent rows and append to the same summary file
        pivot_df = identify_and_remove_equivalent_rows(pivot_df, gen, evo_id_pp_to_name=evo_id_pp_to_name, summary_file=summary_file, party_size=party_size)
        
        if TQDM_AVAILABLE:
            logger.debug("Filling null values with additive unavailable score...")
        
        # Fill any null values in the transverse table with unavailable score
        pivot_df = pivot_df.fillna(UNAVAILABLE_SCORE)
        
        if TQDM_AVAILABLE:
            logger.info("Battle simulation complete!")
            logger.debug(f"Final pivot table shape: {pivot_df.shape}")
            logger.debug(f"Variants: {len(pivot_df.index)}")
            logger.debug(f"Encounters: {len(pivot_df.columns)}")
            logger.debug(f"Null values filled: All null values replaced with {UNAVAILABLE_SCORE}")
        
        return pivot_df
        
    except ValueError as e:
        if TQDM_AVAILABLE:
            logger.error(f"Error creating pivot table: {e}")
            logger.error("Returning dataframe without pivot...")
        return final_df


def filter_dominated_rows(pivot_df: pd.DataFrame, gen: int = None, evo_id_pp_to_name: dict = None, summary_file: Path = None) -> pd.DataFrame:
    """
    Remove rows that are strictly worse than other rows (dominance filtering).
    A row is considered strictly worse if all its values are >= another row's values,
    with at least one value being strictly greater.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table with evo_id_pp as index and encounter scores
        gen (int, optional): Pokemon generation number for summary output
        evo_id_pp_to_name (dict, optional): Mapping from evo_id_pp to pokemon display name
        summary_file (Path, optional): Path to write dominance/equivalence summary. If provided, this
            function writes a Dominance section (overwrites/creates file).
        
    Returns:
        pd.DataFrame: Filtered pivot table without dominated rows
    """
    initial_rows = len(pivot_df)
    
    # Convert to numpy for faster computation
    values = pivot_df.values
    n_rows = len(pivot_df)
    index_arr = pivot_df.index.to_numpy()
    indices_to_keep = []
    # Track dominance relationships: dominator -> set(dominated)
    dominates_map = {}
    
    # Vectorized dominance check: for each row i, compare it against *all* other rows
    # j in a single NumPy broadcast instead of an inner Python for-loop. This preserves
    # the exact same O(n^2) comparison semantics (and NaN behavior) as the original
    # nested-loop implementation, but replaces n^2 individual Python-level np.all/np.any
    # calls with n vectorized O(n * m) array operations evaluated in C, which is a large
    # constant-factor speedup without changing what gets filtered.
    #
    # Dominance rule (unchanged): row_j dominates row_i if every value in row_j is <=
    # the corresponding value in row_i, and at least one value is strictly less.
    # When multiple rows dominate row_i, the lowest-index dominator is recorded, matching
    # the original loop's "break on first match" (j iterating 0..n-1, skipping i).
    row_range = tqdm(range(n_rows), desc="Checking dominance") if TQDM_AVAILABLE else range(n_rows)
    
    for i in row_range:
        row_i = values[i]
        
        all_le = np.all(values <= row_i, axis=1)
        any_lt = np.any(values < row_i, axis=1)
        dominators = all_le & any_lt
        dominators[i] = False
        
        if dominators.any():
            j = int(np.argmax(dominators))  # first (lowest-index) dominator, matches original break order
            dom_idx = index_arr[j]
            sub_idx = index_arr[i]
            dominates_map.setdefault(dom_idx, set()).add(sub_idx)
        else:
            indices_to_keep.append(i)
    
    filtered_df = pivot_df.iloc[indices_to_keep].copy()
    
    if TQDM_AVAILABLE:
        removed_count = initial_rows - len(filtered_df)
        if initial_rows > 0:
            logger.debug(f"Removed {removed_count} dominated rows ({removed_count/initial_rows*100:.1f}%)")
        else:
            logger.debug(f"Removed {removed_count} dominated rows (no initial rows to process)")
        logger.debug(f"Remaining rows: {len(filtered_df)}")
    
    # Write dominance summary if a summary_file is provided
    try:
        if summary_file is not None:
            # Start fresh: write dominance section header
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Dominance Analysis for Generation {gen if gen is not None else ''}\n")
                f.write("=" * 60 + "\n\n")
                if dominates_map:
                    # Deterministic ordering of dominators using current index order
                    index_order = {idx: pos for pos, idx in enumerate(pivot_df.index)}
                    for dom in sorted(dominates_map.keys(), key=lambda v: index_order.get(v, float('inf'))):
                        dominated_list = sorted(list(dominates_map[dom]), key=lambda v: index_order.get(v, float('inf')))
                        dom_name = (evo_id_pp_to_name or {}).get(dom, str(dom))
                        f.write(f"Dominator: {dom_name} [{dom}]\n")
                        for sub in dominated_list:
                            sub_name = (evo_id_pp_to_name or {}).get(sub, str(sub))
                            f.write(f"  - Dominated: {sub_name} [{sub}]\n")
                        f.write("-" * 40 + "\n")
                else:
                    f.write("No dominated rows found.\n")
                f.write("\n")
    except Exception:
        # Do not fail pipeline due to logging issues
        pass
    
    return filtered_df


def identify_and_remove_equivalent_rows(pivot_df: pd.DataFrame, gen: int, evo_id_pp_to_name: dict = None, summary_file: Path = None, party_size: int = None) -> pd.DataFrame:
    """
    Identify rows with identical values across all columns, save equivalent groups to file,
    and remove duplicates keeping only one representative from each group.
    
    Args:
        pivot_df (pd.DataFrame): Pivot table with evo_id_pp as index
        gen (int): Pokemon generation number
        evo_id_pp_to_name (dict, optional): Mapping from evo_id_pp to pokemon display name
        summary_file (Path, optional): Path to append equivalent groups summary
        party_size (int, optional): Party size, used only to namespace the equivalent-rows filename
        
    Returns:
        pd.DataFrame: Pivot table with duplicate equivalent rows removed
    """
    initial_rows = len(pivot_df)
    
    # Find equivalent rows by grouping on all column values
    # Convert to string representation to handle potential floating point precision issues
    pivot_df_rounded = pivot_df.round(6)  # Round to 6 decimal places for comparison
    
    # Create a string representation of each row for grouping
    row_signatures = pivot_df_rounded.apply(lambda row: tuple(row.values), axis=1)
    
    # Preserve a stable index order mapping for deterministic representative selection
    index_order = {idx: pos for pos, idx in enumerate(pivot_df.index)}
    
    # Group evo_id_pps by their row signatures
    equivalence_groups = {}
    # Use tqdm for progress bar if available
    signatures_items = tqdm(row_signatures.items(), desc="Grouping equivalent rows") if TQDM_AVAILABLE else row_signatures.items()
    
    for evo_id_pp, signature in signatures_items:
        if signature not in equivalence_groups:
            equivalence_groups[signature] = []
        equivalence_groups[signature].append(evo_id_pp)
    
    # Find groups with more than one evo_id_pp (equivalent rows)
    equivalent_groups = {sig: variants for sig, variants in equivalence_groups.items() if len(variants) > 1}
    
    if equivalent_groups:
        # Save equivalent groups to an intermediate artifact (not into data/).
        ps_suffix = f"_ps{party_size}" if party_size is not None else ""
        equivalent_file = paths.intermediate_dir() / f"equivalent_rows_gen{gen}{ps_suffix}.txt"
        
        with open(equivalent_file, 'w') as f:
            f.write(f"Equivalent Variant Groups for Generation {gen}\n")
            f.write("=" * 50 + "\n\n")
            
            # Use tqdm for progress bar if available when writing groups
            groups_items = tqdm(equivalent_groups.items(), desc="Writing equivalent groups") if TQDM_AVAILABLE else equivalent_groups.items()
            
            for i, (signature, variants) in enumerate(groups_items, 1):
                f.write(f"Group {i}: {len(variants)} equivalent variants\n")
                f.write(f"Variant IDs: {', '.join(map(str, variants))}\n")
                f.write(f"Score values: {signature}\n")
                f.write("-" * 30 + "\n")
        
        if TQDM_AVAILABLE:
            logger.debug(f"Found {len(equivalent_groups)} groups of equivalent rows")
            total_equivalent = sum(len(variants) for variants in equivalent_groups.values())
            logger.debug(f"Total equivalent variants: {total_equivalent}")
            logger.debug(f"Equivalent groups saved to: {equivalent_file}")
        
        # Also append a human-readable equivalent rows summary with names to the shared intermediate summary file
        try:
            if summary_file is not None:
                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write(f"Equivalent Rows Analysis for Generation {gen}\n")
                    f.write("=" * 60 + "\n\n")
                    # Deterministic ordering by original index order
                    index_order = {idx: pos for pos, idx in enumerate(pivot_df.index)}
                    group_items = sorted(equivalent_groups.items(), key=lambda item: min(index_order.get(v, float('inf')) for v in item[1]))
                    for gi, (signature, variants) in enumerate(group_items, 1):
                        representative = min(variants, key=lambda v: index_order.get(v, float('inf')))
                        kept_name = (evo_id_pp_to_name or {}).get(representative, str(representative))
                        f.write(f"Group {gi}: {len(variants)} equivalent variants\n")
                        f.write(f"  Kept: {kept_name} [{representative}]\n")
                        removed = [v for v in variants if v != representative]
                        for r in sorted(removed, key=lambda v: index_order.get(v, float('inf'))):
                            r_name = (evo_id_pp_to_name or {}).get(r, str(r))
                            f.write(f"  Removed: {r_name} [{r}]\n")
                        f.write("-" * 40 + "\n")
                    f.write("\n")
        except Exception:
            # Do not fail pipeline due to logging issues
            pass
    else:
        # Ensure the shared summary file records there were no equivalent rows
        try:
            if summary_file is not None:
                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write(f"Equivalent Rows Analysis for Generation {gen}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("No equivalent rows found.\n\n")
        except Exception:
            pass
    
    # Keep only one representative from each equivalence group
    variants_to_keep = []
    
    # Process all groups (both equivalent and unique)
    for signature, variants in equivalence_groups.items():
        # Choose representative deterministically by original index order
        representative = min(variants, key=lambda v: index_order.get(v, float('inf')))
        variants_to_keep.append(representative)
    
    # Add all unique (non-equivalent) variants that don't appear in equivalent_groups
    # These are variants that appear only once in the original equivalence_groups dict
    all_equivalent_variants = set()
    for variants in equivalence_groups.values():
        all_equivalent_variants.update(variants)
    
    # Add unique variants that weren't part of any equivalent group
    all_variants = set(pivot_df.index)
    unique_variants = all_variants - all_equivalent_variants
    # Add unique variants with deterministic ordering based on original index order
    unique_variants_sorted = sorted(unique_variants, key=lambda v: index_order.get(v, float('inf')))
    variants_to_keep.extend(unique_variants_sorted)
    
    # Filter the pivot table to keep only selected variants in deterministic order
    filtered_df = pivot_df.loc[variants_to_keep].copy()
    
    if TQDM_AVAILABLE:
        removed_count = initial_rows - len(filtered_df)
        logger.debug(f"Removed {removed_count} duplicate equivalent rows")
        logger.debug(f"Remaining unique rows: {len(filtered_df)}")
    
    return filtered_df


def load_and_simulate_battles(gen: int, input_file: str = None) -> pd.DataFrame:
    """
    Load variant-encounter data and simulate battles.
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        input_file (str): Path to input CSV file. If None, uses default naming pattern.
        
    Returns:
        pd.DataFrame: Battle simulation results
    """
    
    if input_file is None:
        # When run manually, look in intermediate_files directory
        input_file = paths.INTERMEDIATE_DIR / f"player_pokemon_encounters_with_level_gen{gen}.csv"
    else:
        input_file = Path(input_file)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if TQDM_AVAILABLE:
        logger.debug(f"Loading data from: {input_file}")
    
    # Load the data
    player_pokemon_encounters_df = pd.read_csv(input_file)
    
    # Run battle simulation
    results_df = simulate_battles(gen, player_pokemon_encounters_df)
    
    return results_df


# Optional: Test the function
if __name__ == "__main__":
    import sys
    
    # Load config to get generation
    config = load_config()
    config_gen = config.get("gen", 1)
    
    # Handle command line argument or use config
    if len(sys.argv) > 1:
        generation_arg = sys.argv[1]
    else:
        generation_arg = str(config_gen)
    
    try:
        # Determine which generations to process
        generation_str = str(generation_arg).strip()
        
        if generation_str.lower() == "all":
            generations = [1, 2, 3]
            logger.debug("Processing all generations (1, 2, 3) as specified")
        elif generation_str.startswith('[') and generation_str.endswith(']'):
            # Parse bracket notation like [1,2,3] or [1,2]
            bracket_content = generation_str[1:-1]  # Remove brackets
            try:
                generations = [int(x.strip()) for x in bracket_content.split(',')]
                # Validate generations are in valid range
                for gen in generations:
                    if gen not in [1, 2, 3]:
                        raise ValueError(f"Invalid generation: {gen}. Must be 1, 2, or 3.")
                logger.debug(f"Processing generations {generations} as specified")
            except ValueError as e:
                logger.error(f"Error parsing generation list '{generation_str}': {e}")
                logger.error("Valid formats: [1,2,3], [1,2], [2,3], etc.")
                sys.exit(1)
        else:
            # Single generation number
            generations = [int(generation_arg)]
            logger.debug(f"Processing generation {generation_arg}")
        
        # Process each generation
        for generation in generations:
            logger.info(f"Processing Generation {generation}")
            
            # Load and simulate battles
            results = load_and_simulate_battles(generation)
            
            # Save results to intermediate_files directory when run manually for debugging
            output_file = paths.intermediate_dir() / f"battle_results_gen{generation}.csv"
            results.to_csv(output_file)
            
            logger.debug(f"Gen {generation} battle simulation results saved to: {output_file}")
        
        logger.info("All requested generations processed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
