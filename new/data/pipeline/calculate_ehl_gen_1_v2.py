import numpy as np
import logging
import warnings
from multiprocessing import cpu_count
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = pd.read_csv('../config/config.csv')
gen = int(config[config.rule == 'gen'].value.values[0])

def read_csv_files(gen):
    logging.info("Reading CSV files...")
    variants_df = pd.read_csv(f'../data_curated/data_curated_gen_{gen}/variants_gen_{gen}.csv')
    stats_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv')
    exp_table_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/exp_table.csv')
    encounters_df = pd.read_csv(f'../data_curated/data_curated_gen_{gen}/encounters_gen_{gen}.csv')
    moves_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/moves_gen_{gen}.csv')
    stages_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/stages_gen_{gen}.csv')
    logging.info("CSV files read successfully.")

    moves_df = pd.merge(moves_df, stats_df[['pokemon', 'evo_id']], on='pokemon', how='left')

    # DEBUG
    # variants_df = variants_df[variants_df.variant_id == 8297]

    stats_df = stats_df.rename(columns={
        'hp': 'base_hp',
        'attack': 'base_attack',
        'defense': 'base_defense',
        'sp_attack': 'base_sp_attack',
        'sp_defense': 'base_sp_defense',
        'speed': 'base_speed'
    })

    stats_df['evo_index'] = stats_df.groupby('evo_id').cumcount() + 1

    expected_stat_columns = ['evo_id', 'base_hp', 'base_attack', 'base_defense', 'base_sp_attack', 'base_sp_defense',
                             'base_speed', 'exp_type', 'evo_index']

    if not set(expected_stat_columns).issubset(stats_df.columns):
        logging.error(f"Missing expected columns in stats_gen_{gen}.csv. Columns found: {stats_df.columns}")
        raise KeyError("Missing expected columns in stats DataFrame.")

    return variants_df, stats_df, exp_table_df, encounters_df, moves_df, stages_df


def create_cross_join(variants_df, encounters_df):
    logging.info("Creating cross join of variants and encounters...")
    variants_df['key'] = 1
    encounters_df['key'] = 1
    cross_join_df = pd.merge(variants_df, encounters_df, on='key').drop('key', axis=1)
    logging.info("Cross join created successfully.")
    return cross_join_df


def prepare_exp_table(exp_table_df):
    logging.info("Preparing experience table...")
    exp_table_melted = pd.melt(exp_table_df, id_vars=['Level'], var_name='exp_type', value_name='cumulative_exp')
    logging.info("Experience table prepared successfully.")
    return exp_table_melted


def merge_and_calculate_levels(cross_join_df, stats_df, exp_table_melted, gen):
    logging.info("Merging and calculating player levels...")

    merged_df = pd.merge(cross_join_df, stats_df[[
        'evo_id', 'exp_type', 'base_hp', 'base_attack', 'base_defense',
        'base_sp_attack', 'base_sp_defense', 'base_speed', 'evo_index'
    ]], on='evo_id')

    exp_levels = {}
    for exp_type in exp_table_melted['exp_type'].unique():
        exp_type_data = exp_table_melted[exp_table_melted['exp_type'] == exp_type]
        exp_levels[exp_type] = exp_type_data.set_index('cumulative_exp')['Level'].to_dict()

    def get_level(exp_enc, exp_type):
        exp_type_levels = exp_levels[exp_type]
        cumulative_exps = np.array(list(exp_type_levels.keys()))
        levels = np.array(list(exp_type_levels.values()))
        idx = np.searchsorted(cumulative_exps, exp_enc, side='right') - 1
        idx = np.clip(idx, 0, len(levels) - 1)
        return levels[idx]

    merged_df['player_level'] = merged_df.apply(lambda row: get_level(row['exp_enc'], row['exp_type']), axis=1)

    def calc_level(df, gen):
        wild_locations = pd.read_csv(f'../data_curated/data_curated_gen_{gen}/wild_locations_gen_{gen}.csv')

        if 'evo_id' not in wild_locations.columns:
            temp_stats_df = pd.read_csv('../data_raw/data_raw_gen_2/stats_gen_2.csv')
            wild_locations = pd.merge(wild_locations, temp_stats_df[['pokemon','evo_id']], on='pokemon', how='left')
        df = pd.merge(df, wild_locations, how='left', on='evo_id')
        # Filter merged dataframe for only entries with
        df = df[df.stage_enc >= df.wild_location_stage]
        # Group by variant_id and enc_id, and find the min_level info from merged wild columns.
        df = df.loc[df.groupby(['variant_id', 'enc_id'])['wild_level'].idxmax()]
        df['prior_max_level'] = df.groupby('variant_id')['player_level'].expanding().max().shift().reset_index(level=0, drop=True)
        df['player_level'] = df[['player_level', 'prior_max_level', 'wild_level']].max(axis=1)

        return df

    merged_df = calc_level(merged_df, gen)

    logging.info("Player levels calculated successfully.")
    return merged_df


def bulk_calculate_stats(df, gen, trainer_dv_file=None):
    """
    Vectorized version of calculate_stats for Gen != 2.
    For Gen 2, dynamically reads DV values based on trainer_name_enc from a CSV file.
    """
    # ------------------------------------------
    # 0. If Gen != 2, keep the original logic
    # ------------------------------------------
    if gen != 2:
        iv = 8
        iv_attack = 9

        # Gen 1 or Gen >= 3, same formula as before
        if gen <= 2:
            hp = np.floor((df['base_hp'] + iv) * 2 * df['player_level'] / 100) + df['player_level'] + 10
            attack = np.floor((df['base_attack'] + iv_attack) * 2 * df['player_level'] / 100) + 5
            defense = np.floor((df['base_defense'] + iv) * 2 * df['player_level'] / 100) + 5
            sp_attack = np.floor((df['base_sp_attack'] + iv) * 2 * df['player_level'] / 100) + 5
            sp_defense = np.floor((df['base_sp_defense'] + iv) * 2 * df['player_level'] / 100) + 5
            speed = np.floor((df['base_speed'] + iv) * 2 * df['player_level'] / 100) + 5
        else:
            hp = np.floor((2 * df['base_hp'] + iv) * df['player_level'] / 100) + df['player_level'] + 10
            attack = np.floor((2 * df['base_attack'] + iv_attack) * df['player_level'] / 100) + 5
            defense = np.floor((2 * df['base_defense'] + iv) * df['player_level'] / 100) + 5
            sp_attack = np.floor((2 * df['base_sp_attack'] + iv) * df['player_level'] / 100) + 5
            sp_defense = np.floor((2 * df['base_sp_defense'] + iv) * df['player_level'] / 100) + 5
            speed = np.floor((2 * df['base_speed'] + iv) * df['player_level'] / 100) + 5

        result = pd.DataFrame({
            'hp': hp.astype(int),
            'attack': attack.astype(int),
            'defense': defense.astype(int),
            'sp_attack': sp_attack.astype(int),
            'sp_defense': sp_defense.astype(int),
            'speed': speed.astype(int)
        }, index=df.index)

        return result

    # ------------------------------------------
    # 1. For Gen 2, load trainer DV data from the CSV file
    # ------------------------------------------
    # trainer_dv_file = '../data_raw/data_raw_gen_2/trainer_dvs.csv'

    if trainer_dv_file is None:
        raise ValueError("For Gen 2, a valid trainer_dv_file must be provided.")

    # Load trainer DV data
    trainer_dv_df = pd.read_csv(trainer_dv_file)
    trainer_dv_df.columns = trainer_dv_df.columns.str.strip()  # Strip column names of whitespace
    trainer_dv_df['Trainer'] = trainer_dv_df['Trainer'].str.lower().str.strip()  # Normalize for matching

    # We'll define a function that, for a single trainer_name_enc string,
    # returns (atk_dv, def_dv, spd_dv, spc_dv) or a default if not found.
    def match_trainer_dv(trainer_name):
        """
        Finds a match if any trainer substring in the DV table is contained
        in trainer_name_enc (case-insensitive).
        If multiple match, the first one in the trainer_dv_df is used.
        If none match, return default (8,8,8,8).
        """
        if pd.isna(trainer_name):
            return (9, 8, 8, 8)  # Default DV values if trainer is NaN

        name_lower = trainer_name.lower()
        for _, row in trainer_dv_df.iterrows():
            if row['Trainer'] in name_lower:
                return (row['Attack DV'], row['Defense DV'], row['Speed DV'], row['Special DV'])
        # If no match found, return default
        return (9, 8, 8, 8)

    # Apply the DV lookup function row-wise
    dv_vals = df['trainer_name_enc'].apply(match_trainer_dv)

    # Split the DV tuples into individual columns
    df['atk_dv'] = dv_vals.apply(lambda x: x[0])
    df['def_dv'] = dv_vals.apply(lambda x: x[1])
    df['spd_dv'] = dv_vals.apply(lambda x: x[2])
    df['spc_dv'] = dv_vals.apply(lambda x: x[3])

    # We still assume HP's DV = 8 for Gen 2 if not specified
    df['hp_dv'] = 8

    # ------------------------------------------
    # 2. Vectorized formula for Gen 2 using dynamic DVs
    # ------------------------------------------
    hp = np.floor((df['base_hp'] + df['hp_dv']) * 2 * df['player_level'] / 100) + df['player_level'] + 10
    attack = np.floor((df['base_attack'] + df['atk_dv']) * 2 * df['player_level'] / 100) + 5
    defense = np.floor((df['base_defense'] + df['def_dv']) * 2 * df['player_level'] / 100) + 5
    speed = np.floor((df['base_speed'] + df['spd_dv']) * 2 * df['player_level'] / 100) + 5
    sp_attack = np.floor((df['base_sp_attack'] + df['spc_dv']) * 2 * df['player_level'] / 100) + 5
    sp_defense = np.floor((df['base_sp_defense'] + df['spc_dv']) * 2 * df['player_level'] / 100) + 5

    # Build the result DataFrame
    result = pd.DataFrame({
        'hp': hp.astype(int),
        'attack': attack.astype(int),
        'defense': defense.astype(int),
        'sp_attack': sp_attack.astype(int),
        'sp_defense': sp_defense.astype(int),
        'speed': speed.astype(int)
    }, index=df.index)

    # Clean up helper columns if you no longer need them
    df.drop(['atk_dv', 'def_dv', 'spd_dv', 'spc_dv', 'hp_dv'], axis=1, inplace=True)

    return result

def determine_pokemon_by_level(stats_df, merged_df, gen):
    """
    Optimized version of determine_pokemon_by_level:
      1. Merges and filters once.
      2. Avoids row-by-row loops (vectorized stat calculations).
      3. Minimizes usage of DataFrame.apply(..., axis=1).
      4. Uses efficient lookups for min_level, max_level, and trade evolutions.
      5. Only takes the needed rows (final evolutions).
      6. Combines everything into as few passes as possible.
    """
    logging.info("Determining Pokémon by level...")

    # Read config once (instead of multiple times)
    config = pd.read_csv('../config/config.csv')
    trade_evos = (config[config.rule == 'trade_evos'].value.values[0].lower() == 'y')

    # Ensure evo_stage is not null
    stats_df['evo_stage'] = stats_df['evo_stage'].fillna(0)

    # Sort stats by evo_id and evo_lvl (used for min/max level logic)
    stats_df_sorted = stats_df.sort_values(['evo_id', 'evo_lvl'])

    # Define min_level and the "naive" max_level from the next evo_lvl
    stats_df_sorted['min_level'] = stats_df_sorted['evo_lvl'].fillna(0)
    stats_df_sorted['max_level'] = stats_df_sorted.groupby('evo_id')['evo_lvl'].shift(-1)

    # Use a groupby "shift" to look ahead for the next evo_stage
    stats_df_sorted['next_evo_stage'] = stats_df_sorted.groupby('evo_id')['evo_stage'].shift(-1)
    # If next_evo_stage > 0, this indicates a staged evolution => set max_level to infinity
    mask = stats_df_sorted['next_evo_stage'] > 0
    stats_df_sorted.loc[mask, 'max_level'] = np.inf
    # Fill any remaining NaNs with infinity
    stats_df_sorted['max_level'] = stats_df_sorted['max_level'].fillna(np.inf)

    # Drop the helper column
    stats_df_sorted.drop(columns=['next_evo_stage'], inplace=True)

    # If trade evolutions are disallowed, artificially raise their min_level
    if not trade_evos:
        trade_mons = ['Alakazam', 'Gengar', 'Golem', 'Machamp']
        stats_df_sorted.loc[stats_df_sorted['pokemon'].isin(trade_mons), 'min_level'] = 999

    # Merge once
    merged = pd.merge(
        merged_df,
        stats_df_sorted,
        on='evo_id',
        how='left',
        suffixes=('_merged', '')
    )

    # Filter rows in one pass
    # Only keep rows where:
    #   player_level >= min_level
    #   player_level < max_level
    #   stage_enc >= evo_stage
    cond = (
        (merged['player_level'] >= merged['min_level']) &
        (merged['player_level'] < merged['max_level']) &
        (merged['stage_enc'] >= merged['evo_stage'])
    )
    merged = merged[cond]

    # Pick only the final evolution row for each (variant_id, enc_id)
    merged = (
        merged
        .sort_values(['enc_id', 'evo_index'], ascending=[True, True])
        .groupby(['variant_id', 'enc_id'], as_index=False, sort=False)
        .tail(1)
    )

    # Calculate final stats for these Pokémon in a single, vectorized pass
    final_stats = bulk_calculate_stats(merged, gen, '../data_raw/data_raw_gen_2/trainer_dvs.csv')
    merged[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']] = final_stats

    # At this point, we already cast to int in bulk_calculate_stats,
    # but if you want to ensure consistency:
    # int_columns = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    # merged[int_columns] = merged[int_columns].astype(int)

    logging.info("Pokémon determined by level and stats calculated successfully.")
    return merged


def prepare_moves_df(moves_df, stages_df):
    logging.info("Preparing moves DataFrame...")

    # Merge moves_df with stages_df to update move_stage
    if gen == 1:
        moves_df = pd.merge(moves_df, stages_df[['location', 'location_stage']], left_on='move_stage', right_on='location', how='left')
        moves_df['move_stage'] = moves_df['location_stage'].fillna(0).astype(int)
        moves_df = moves_df.drop(columns=['location', 'location_stage'])
    else:
        moves_df['move_stage'] = moves_df['move_stage'].fillna(0).astype(int)
    moves_df.loc[moves_df['move_stage'].notna() & moves_df['move_level'].isna(), 'move_level'] = 0

    # Remove moves for evolutions that have move_level = 1
    filtered_moves = []
    grouped = moves_df.groupby('evo_id', group_keys=False)
    for evo_id, group in grouped:
        # Identify non-base evolutions by excluding the first occurrence per evo_id
        is_base_evolution = group.index == group.index.min()
        group['is_base_evolution'] = is_base_evolution
        # Keep move_level != 1 for non-base evolutions
        group = group[~((~group['is_base_evolution']) & (group['move_level'] == 1))]
        filtered_moves.append(group)

    # Concatenate all filtered groups back together
    moves_df = pd.concat(filtered_moves, ignore_index=True)
    moves_df = moves_df.drop(columns=['is_base_evolution'], errors='ignore')  # Clean up intermediate column

    logging.info("Moves DataFrame prepared successfully.")
    return moves_df



def calculate_expected_damage(move, base_speed, gen):
    crit_binary = 1 if move.crit else 0

    if gen == 1:
        expected_damage = move.power * move.accuracy * ((1 + (7 * crit_binary)) * (base_speed / 2))
    elif gen == 2:
        expected_damage = move.power * move.accuracy * (1 + (crit_binary * 0.1721680))
    elif gen == 3:
        expected_damage = move.power * move.accuracy * (1 + (crit_binary * 0.0549512))
    else:
        expected_damage = move.power * move.accuracy * ((1 + (7 * crit_binary)) * (base_speed / 2))

    return expected_damage


def assign_best_moves(row, moves_df, gen):
    base_speed = row['base_speed']
    variant_types = [
        row[f'move_type_{i}']
        for i in range(1, 5)
        if pd.notna(row[f'move_type_{i}']) and row[f'move_type_{i}'] != 'none'
    ]
    best_moves = {}

    # Get all moves for the current Pokémon and its prior evolutions
    valid_moves = moves_df[
        (moves_df['evo_id'] == row['evo_id']) &
        ((moves_df['move_level'].isna()) | (moves_df['move_level'] <= row['player_level'])) &
        (moves_df['move_stage'] <= row['stage_enc'])
        ]

    for move_type in variant_types:
        type_moves = valid_moves[valid_moves['move_type'] == move_type]
        if not type_moves.empty:
            type_moves = type_moves.copy()
            type_moves['expected_damage'] = type_moves.apply(lambda m: calculate_expected_damage(m, base_speed, gen), axis=1)
            best_move = type_moves.loc[type_moves['expected_damage'].idxmax()]
            best_moves[move_type] = best_move

    # Initialize all move columns to None or appropriate default values
    for i in range(1, 5):
        row[f'move{i}_name'] = None
        row[f'move{i}_type'] = None
        row[f'move{i}_power'] = np.nan
        row[f'move{i}_accuracy'] = np.nan
        row[f'move{i}_crit'] = False

    for i, (move_type, best_move) in enumerate(best_moves.items(), start=1):
        row[f'move{i}_name'] = best_move['move_name']
        row[f'move{i}_type'] = best_move['move_type']
        row[f'move{i}_power'] = best_move['power']
        row[f'move{i}_accuracy'] = best_move['accuracy']
        row[f'move{i}_crit'] = best_move['crit']

    return row


new_columns = {
    'move1_name': 'object',
    'move1_type': 'object',
    'move1_power': 'float64',
    'move1_accuracy': 'float64',
    'move1_crit': 'bool',
    'move2_name': 'object',
    'move2_type': 'object',
    'move2_power': 'float64',
    'move2_accuracy': 'float64',
    'move2_crit': 'bool',
    'move3_name': 'object',
    'move3_type': 'object',
    'move3_power': 'float64',
    'move3_accuracy': 'float64',
    'move3_crit': 'bool',
    'move4_name': 'object',
    'move4_type': 'object',
    'move4_power': 'float64',
    'move4_accuracy': 'float64',
    'move4_crit': 'bool',
}


def initialize_new_columns(df, new_columns):
    """
    Initialize all of the 'move*_...' columns once, outside the Dask partitions.
    This avoids repeating the column creation in every partition.
    """
    for col, dtype in new_columns.items():
        if dtype == 'object':
            df[col] = None
        elif dtype == 'bool':
            df[col] = False
        else:
            df[col] = np.nan
    return df



def assign_moves_partition(df, moves_df, gen):
    """
    Optimized partition function for assigning moves.
    Uses vectorization and groupby operations to minimize row-wise computations.
    """
    # Precompute all valid moves for the Pokémon in this partition
    relevant_evo_ids = df['evo_id'].unique()
    valid_moves = moves_df[
        moves_df['evo_id'].isin(relevant_evo_ids) &  # Only relevant evolutions
        ((moves_df['move_level'].isna()) | (moves_df['move_level'] <= df['player_level'].max())) &  # Player-level check
        (moves_df['move_stage'] <= df['stage_enc'].max())  # Encounter stage check
    ]

    # Calculate expected damage for all valid moves in this partition
    valid_moves['expected_damage'] = (
        valid_moves['power'] *
        valid_moves['accuracy'] *
        (1 + (7 * valid_moves['crit'].astype(int))) *  # Crit multiplier for Gen 1
        (df['base_speed'].max() / 2 if gen == 1 else 1)  # Speed factor for Gen 1 only
    )

    # Group moves by evo_id and move_type, then pick the best move (highest expected_damage)
    best_moves = (
        valid_moves
        .sort_values('expected_damage', ascending=False)  # Sort moves by expected damage
        .drop_duplicates(subset=['evo_id', 'move_type'])  # Keep only the best move for each type
    )

    # Prepare a mapping of (evo_id, move_type) -> best move details
    best_moves_mapping = best_moves.set_index(['evo_id', 'move_type']).to_dict('index')

    # Assign the best moves to each Pokémon in the partition
    for i in range(1, 5):  # Move slots 1 to 4
        move_type_col = f'move_type_{i}'
        mask = df[move_type_col].notna()

        # Extract move information from the mapping for the corresponding move type
        move_details = df.loc[mask, ['evo_id', move_type_col]].apply(
            lambda row: best_moves_mapping.get((row['evo_id'], row[move_type_col]), {}),
            axis=1
        )

        # Assign each column based on the best moves found
        for attr in ['name', 'power', 'accuracy', 'crit']:
            df.loc[mask, f'move{i}_{attr}'] = move_details.map(lambda x: x.get(attr, None))

    return df


def process_variants(merged_df, moves_df, gen):
    """
    Improved version of the process_variants function:
      1) Initializes new columns only once.
      2) Uses a partition function with vectorized/groupby logic.
      3) Avoids redundant row-wise computations where possible.
    """
    total_variants = merged_df['variant_id'].nunique()
    logging.info(f"Total variants to process: {total_variants}")

    # 1. Initialize new columns just once
    merged_df = initialize_new_columns(merged_df, new_columns)

    # 2. Convert to a Dask DataFrame
    merged_ddf = dd.from_pandas(merged_df, npartitions=cpu_count())

    # 3. Create an empty DataFrame (with the same columns) to use as meta
    output_meta = merged_df.iloc[:0].copy()

    # Explicitly set column data types in metadata to match expected schema
    for col, dtype in new_columns.items():
        output_meta[col] = pd.Series(dtype=dtype)

    # 4. Apply the partition function
    with ProgressBar():
        results = (
            merged_ddf
            .map_partitions(
                assign_moves_partition,
                moves_df=moves_df,
                gen=gen,
                meta=output_meta
            )
            .compute(scheduler='threads')
        )

    logging.info("Best moves assigned successfully.")
    return results


def calc_se(df, gen):
    logging.info("Calculating se columns")

    # Read the type chart once
    type_chart = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/typechart_gen_{gen}.csv', index_col=0)

    # Initialize columns to 1.0
    for i in range(1, 5):
        df[f'move{i}_se'] = 1.0
        df[f'move{i}_se_enc'] = 1.0

    # Pre-define badge_boost_dict and held_item_dict for Gen 2
    badge_boost_dict = {
        'Flying': 5, 'Bug': 9, 'Normal': 12, 'Ghost': 17,
        'Fighting': 23, 'Steel': 24, 'Ice': 31, 'Dragon': 38,
        'Electric': 50, 'Psychic': 52, 'Water': 62, 'Grass': 65,
        'Poison': 69, 'Rock': 77, 'Fire': 87, 'Ground': 89
    }
    held_item_dict = {
        'Fighting': 29, 'Dark': 17, 'Fire': 9, 'Dragon': 39,
        'Rock': 15, 'Electric': 16, 'Steel': 18, 'Grass': 6,
        'Water': 17, 'Ice': 36, 'Normal': 5, 'Poison': 6,
        'Flying': 21, 'Bug': 10, 'Ground': 11, 'Ghost': 37
    }

    # Single function to compute se and apply Gen 2 boosts for all four moves in one pass
    def calc_se_for_row(row):
        """
        Calculates move1_se, move1_se_enc, move2_se, etc. in a single pass,
        plus Gen 2 boosts if applicable.
        """
        # Helper to get base effectiveness
        def get_effectiveness(move_type, target_types):
            if pd.isna(move_type) or pd.isna(target_types) or move_type == '-' or target_types == '-':
                return 1.0
            # Split target types e.g. "Grass/Poison"
            tlist = target_types.split('/')
            eff = 1.0
            for t in tlist:
                eff *= type_chart.loc[move_type, t]
            return eff

        # Helper to get Gen 2 boosts
        def get_boosts(original, move_type, enc_stage, pokemon):
            if enc_stage > badge_boost_dict.get(move_type, 0):
                original *= 1.125
            if enc_stage > held_item_dict.get(move_type, 0):
                original *= 1.1
            if pokemon == 'Farfetchd':
                original *= 1.172168
            if pokemon in ['Cubone', 'Marowak'] and move_type in [
                'Normal','Fighting','Flying','Poison','Ground','Rock','Bug','Ghost','Steel'
            ]:
                original *= 2
            return original

        # For each move slot, compute se and se_enc
        for i in range(1, 5):
            # Base effectiveness
            se_col = f'move{i}_se'
            se_enc_col = f'move{i}_se_enc'

            move_type = row.get(f'move{i}_type', None)
            target_types = row.get('types', None)
            row[se_col] = get_effectiveness(move_type, target_types)

            move_type_enc = row.get(f'move{i}_type_enc', None)
            target_types_enc = row.get('types_enc', None)
            row[se_enc_col] = get_effectiveness(move_type_enc, target_types_enc)

            # Apply Gen 2 boosts if needed
            if gen == 2:
                row[se_col] = get_boosts(
                    row[se_col],
                    move_type,
                    row.get('stage_enc', 0),
                    row.get('pokemon', None)
                )

        return row

    # Perform a single pass, calculating all SE values (and Gen 2 boosts if applicable)
    df = df.apply(calc_se_for_row, axis=1)

    return df


def calc_ehl(row):
    """
    Calculates an 'ehl' value (some form of metric) for the given row.
    Avoids errors when move types or power/crit columns are pd.NA.
    """

    phys_types = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel']
    enc_stage = row['stage_enc']

    level = row['player_level']
    level_enc = row['level_enc']

    # Split the 'types' and 'types_enc' columns safely
    types = row['types'].split('/') if pd.notna(row['types']) else []
    types_enc = row['types_enc'].split('/') if pd.notna(row['types_enc']) else []
    pokemon_type1 = types[0] if len(types) > 0 else None
    pokemon_type2 = types[1] if len(types) > 1 else None
    pokemon_type1_enc = types_enc[0] if len(types_enc) > 0 else None
    pokemon_type2_enc = types_enc[1] if len(types_enc) > 1 else None

    # Stats
    hp = row['hp']
    hp_enc = row['hp_enc']
    speed = row['speed']
    speed_enc = row['speed_enc']

    hp_calc = hp
    hp_calc_enc = hp_enc

    # Possible speed buff
    speed_calc = speed
    if enc_stage > 27:
        speed_calc *= 1.125
    speed_calc_enc = speed_enc

    # Helper function: safely check if a move_type is "physical"
    def is_physical_type(move_type):
        """
        Returns True if move_type is one of the physical types,
        handling the case where move_type might be pd.NA or None.
        """
        if pd.isna(move_type):
            return False
        return move_type in phys_types

    def calc_stats(move_type, atk, defense, crit, power, acc, stab, se, level, enc_stage, phys_spec, speed, gen):
        """
        Safely compute damage for a single move, avoiding errors if power or move_type is pd.NA.
        """
        # If power is missing or '-', treat as 0
        if pd.isna(power) or power == '-':
            return 0

        # Badge stat bonuses or other multipliers
        atk_calc = atk
        def_calc = defense

        # Example badge logic
        if gen == 1:
            # Physical
            if phys_spec == 'physical' and enc_stage > 5:
                atk_calc *= 1.125
            # Special
            if phys_spec == 'special' and enc_stage > 33:
                atk_calc *= 1.125
            # Physical
            if phys_spec == 'physical' and enc_stage > 14:
                def_calc *= 1.125
            # Special
            if phys_spec == 'special' and enc_stage > 33:
                def_calc *= 1.125
        elif gen == 2:
            # Physical
            if phys_spec == 'physical' and enc_stage > 5:
                atk_calc *= 1.125
            # Special
            if phys_spec == 'special' and enc_stage > 31:
                atk_calc *= 1.125
            # Physical
            if phys_spec == 'physical' and enc_stage > 24:
                def_calc *= 1.125
            # Special
            if phys_spec == 'special' and enc_stage > 31:
                def_calc *= 1.125
        # (Add more generations if needed)

        # Different crit formulas and multipliers
        if gen == 1:
            if crit:
                # Crit formula (Gen 1 style)
                crit = ((min(8 * speed / 2, 255) / 256) + 1)
            else:
                # Non-crit
                crit = (((speed / 2) / 256) + 1)
            calc_dam = (
                (
                    ((2 * float(level) * float(crit) / 5) + 2)
                    * float(power)
                    * float(atk_calc)
                    / float(def_calc)
                ) / 50
                + 2
            ) * float(stab) * float(se) * float(acc)
        elif gen == 2:
            if crit:
                crit = 1.25
            else:
                crit = 1.0664
            calc_dam = (
                (
                    ((2 * float(level) / 5) + 2)
                    * float(power)
                    * float(atk_calc)
                    / float(def_calc)
                )
                / 50
            ) * crit + 2
            calc_dam *= float(stab) * float(se) * float(acc)
        else:
            # Gen >= 3 (example)
            if crit:
                crit = 1.125
            else:
                crit = 1.0664
            # Add your formula or just set calc_dam = 0 if not implemented
            calc_dam = 0
            # ...
            # Example from above (uncomment if you have an actual formula):
            # calc_dam = ...
            # calc_dam *= float(crit) * float(stab) * float(se) * float(acc)

        return np.nan_to_num(calc_dam, nan=0)

    def get_stab(move_type, ptype1, ptype2):
        """
        Returns 1.5 if move_type matches either of the Pokémon's types.
        """
        if pd.isna(move_type):
            return 1.0
        if move_type == ptype1 or move_type == ptype2:
            return 1.5
        return 1.0

    # Rebuild the (moveX_power, moveX_accuracy, etc.) tuples
    moves = []
    for i in range(1, 5):
        power = row.get(f'move{i}_power', np.nan)
        acc = row.get(f'move{i}_accuracy', np.nan)
        crit = row.get(f'move{i}_crit', False)
        mtype = row.get(f'move{i}_type', None)

        power_enc = row.get(f'move{i}_power_enc', np.nan)
        acc_enc = row.get(f'move{i}_accuracy_enc', np.nan)
        crit_enc = row.get(f'move{i}_crit_enc', False)
        mtype_enc = row.get(f'move{i}_type_enc', None)

        se = row.get(f'move{i}_se', 1.0)
        se_enc = row.get(f'move{i}_se_enc', 1.0)

        moves.append((power, acc, crit, mtype, power_enc, acc_enc, crit_enc, mtype_enc, se, se_enc))

    max_dam, max_dam_enc = 0, 0

    for (power, acc, crit, move_type,
         power_enc, acc_enc, crit_enc, move_type_enc,
         se, se_enc) in moves:

        # Determine physical/special for player's move
        phys_spec = 'physical' if is_physical_type(move_type) else 'special'
        stab = get_stab(move_type, pokemon_type1, pokemon_type2)
        atk = row['attack'] if phys_spec == 'physical' else row['sp_attack']

        # Determine physical/special for enemy's move
        phys_spec_enc = 'physical' if is_physical_type(move_type_enc) else 'special'
        stab_enc = get_stab(move_type_enc, pokemon_type1_enc, pokemon_type2_enc)
        atk_enc = row['attack_enc'] if phys_spec_enc == 'physical' else row['sp_attack_enc']

        # Defenses are swapped:
        #   the player's move hits the enemy's defense,
        #   the enemy's move hits the player's defense
        defense_enc = row['defense_enc'] if phys_spec == 'physical' else row['sp_defense_enc']
        defense = row['defense'] if phys_spec_enc == 'physical' else row['sp_defense']

        calc_dam = calc_stats(
            move_type, atk, defense_enc, crit, power, acc, stab, se,
            level, enc_stage, phys_spec, speed, gen
        )
        calc_dam_enc = calc_stats(
            move_type_enc, atk_enc, defense, crit_enc, power_enc, acc_enc, stab_enc, se_enc,
            level_enc, enc_stage, phys_spec_enc, speed_enc, gen
        )

        max_dam = max(max_dam, calc_dam)
        max_dam_enc = max(max_dam_enc, calc_dam_enc)

    max_dam = max(1, math.floor(max_dam))
    max_dam_enc = max(1, math.floor(max_dam_enc))

    hp_calc = math.floor(hp_calc)
    current_hp = hp_calc
    current_hp_enc = math.floor(hp_calc_enc)
    speed_calc = math.floor(speed_calc)
    rounds = 0

    # If the stage_enc is lower than some required threshold, return large EHL
    if enc_stage < row['wild_location_stage']:
        return 1_000_000_000  # or some large sentinel value

    # Who attacks first?
    if speed_calc > speed_calc_enc:
        turn = 1
    elif speed_calc < speed_calc_enc:
        turn = 0
    else:
        turn = -1  # speed tie => simultaneous?

    while current_hp > 0 and current_hp_enc > 0:
        rounds += 1
        if rounds > 20:
            return 1_000_000
        if turn == 1:
            current_hp_enc -= max_dam
        elif turn == 0:
            current_hp -= max_dam_enc
        else:
            current_hp -= max_dam_enc / 2
            current_hp_enc -= max_dam / 2
        turn = 1 - turn

    # If the player's HP is 0 or less => enemy is still alive => ehl = ...
    if current_hp <= 0:
        current_hp_enc = max(0, current_hp_enc)
        ehl = (current_hp_enc / hp_calc_enc) + 1000
    else:
        # else the enemy's HP is 0 => ehl = ...
        current_hp = max(0, current_hp)
        ehl = 1 - (current_hp / hp_calc)

    return ehl


def create_ehl_pivot(df):
    logging.info("Generating ehl pivot dataframe...")

    # Determine the full ranges for variant_id and enc_id
    max_variant_id = df['variant_id'].max()
    max_enc_id = df['enc_id'].max()

    # 1) Create pivot table from existing data
    pivot_df = df.pivot_table(
        index='variant_id',
        columns='enc_id',
        values='ehl',
        aggfunc='first'
    )

    # 2) Reindex rows and columns to include the full range
    variant_ids = np.arange(1, max_variant_id + 1)
    enc_ids = np.arange(1, max_enc_id + 1)

    pivot_df = pivot_df.reindex(index=variant_ids, columns=enc_ids, fill_value=1e9)

    # 3) Convert back to a “normal” DataFrame
    pivot_df.index.name = 'variant_id'
    pivot_df.reset_index(inplace=True)

    # 4) Optionally rename columns (e.g., 'enc_id_1', 'enc_id_2', etc.)
    pivot_df.columns.name = None  # Remove the pivot index name
    pivot_df = pivot_df.rename(columns=lambda x: f'enc_id_{x}' if isinstance(x, int) else x)

    return pivot_df


def filter_strictly_worse_rows(df):
    """
    Returns a subset of rows that are not dominated by any other row,
    using a 'sort & sweep' (Pareto front) approach.

    Excludes the first column from comparisons (assumed to be 'variant_id').
    Sorts by the second column ascending, then incrementally builds the frontier.
    """
    logging.info("Filtering strictly worse rows via Pareto approach...")

    # Convert to NumPy for speed
    arr = df.to_numpy()
    n_rows, n_cols = arr.shape

    # We'll treat arr[:, 0] as 'variant_id' => not compared
    # The columns to compare are arr[:, 1:]
    data = arr[:, 1:]

    # Sort by the first comparison column (i.e., data[:, 0]) ascending
    # We keep track of original indices so we can re-map
    sorted_idx = np.argsort(data[:, 0])
    data_sorted = data[sorted_idx]

    # We'll maintain a list of indices (in the sorted array) that are "winners"
    winners = []

    def dominates(row_a, row_b):
        """
        Check if row_a dominates row_b:
          row_a <= row_b in all columns,
          row_a <  row_b in at least one column.
        """
        # Vectorized check
        diff = row_a - row_b
        if np.any(diff > 0):
            # row_a has some dimension > row_b => cannot dominate
            return False
        # row_a <= row_b in all dims, now check for strict < in at least one
        return np.any(diff < 0)

    # Incremental sweep
    for i in range(n_rows):
        row_i = data_sorted[i]
        # Check if row_i is dominated by any existing winner
        dominated = False
        to_remove = []
        for w in winners:
            row_w = data_sorted[w]
            # if w dominates i => skip i
            if dominates(row_w, row_i):
                dominated = True
                break
            # if i dominates w => remove w from winners
            if dominates(row_i, row_w):
                to_remove.append(w)
        if not dominated:
            # Remove the now-dominated winners
            winners = [w for w in winners if w not in to_remove]
            # Add i as a new winner
            winners.append(i)

    # 'winners' now contains the indices in data_sorted that survived
    # Map back to original row indices (sorted_idx)
    final_indices = sorted_idx[winners]
    # Sort them so the output is consistent with the original order
    final_indices.sort()

    return df.iloc[final_indices].reset_index(drop=True)


def main():
    logging.info("Starting the program...")
    config = pd.read_csv('../config/config.csv')
    gen = int(config[config.rule == 'gen'].value.values[0])
    variants_df, stats_df, exp_table_df, encounters_df, moves_df, stages_df = read_csv_files(gen)
    cross_join_df = create_cross_join(variants_df, encounters_df)
    exp_table_melted = prepare_exp_table(exp_table_df)
    cross_join_df['level_enc'] = cross_join_df['level_enc']
    merged_df = merge_and_calculate_levels(cross_join_df, stats_df, exp_table_melted, gen)
    merged_df = determine_pokemon_by_level(stats_df, merged_df, gen)
    moves_df = prepare_moves_df(moves_df, stages_df)
    merged_df['base_speed'] = merged_df['speed']
    merged_df = process_variants(merged_df, moves_df, gen)
    merged_df = calc_se(merged_df, gen)
    logging.info(f"Calculating EHL values...")
    merged_df['ehl'] = merged_df.apply(calc_ehl, axis=1)
    ehl_pivot = create_ehl_pivot(merged_df)
    ehl_pivot = filter_strictly_worse_rows(ehl_pivot)
    logging.info(f"Saving ehl pivot dataframe...")
    ehl_pivot.to_csv(f'../data_curated/data_curated_gen_{gen}/ehl_pivot_gen_{gen}.csv', index=False)
    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
