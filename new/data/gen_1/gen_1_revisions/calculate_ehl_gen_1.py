import pandas as pd
import numpy as np
import logging
from multiprocessing import cpu_count
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_csv_files():
    logging.info("Reading CSV files...")
    variants_df = pd.read_csv('../gen_1_data_curated/variants_gen_1.csv')
    stats_df = pd.read_csv('../gen_1_data_raw/stats_gen_1.csv')
    exp_table_df = pd.read_csv('../gen_1_data_raw/exp_table.csv')
    encounters_df = pd.read_csv('../gen_1_data_curated/encounters_gen_1.csv')
    moves_df = pd.read_csv('../gen_1_data_raw/moves_gen_1.csv')
    stages_df = pd.read_csv('../gen_1_data_raw/stages_gen_1.csv')
    logging.info("CSV files read successfully.")

    moves_df = pd.merge(moves_df, stats_df[['pokemon', 'evo_id']], on='pokemon', how='left')

    # DEBUG
    # variants_df = variants_df[variants_df.variant_id == 300]

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
        logging.error(f"Missing expected columns in stats_gen_1.csv. Columns found: {stats_df.columns}")
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


def merge_and_calculate_levels(cross_join_df, stats_df, exp_table_melted):
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

    merged_df['player_level'] = merged_df.apply(
        lambda row: get_level(row['exp_enc'], row['exp_type']), axis=1
    )

    def calc_level(df):
        wild_locations = pd.read_csv('../gen_1_data_curated/wild_locations_gen_1.csv')
        df = pd.merge(df, wild_locations, how='left', on='evo_id')
        # Filter merged dataframe for only entries with
        df = df[df.stage_enc >= df.wild_location_stage]
        # Group by variant_id and enc_id, and find the min_level info from merged wild columns.
        df = df.loc[df.groupby(['variant_id', 'enc_id'])['wild_level'].idxmax()]
        df['prior_max_level'] = df.groupby('variant_id')['player_level'].expanding().max().shift().reset_index(level=0, drop=True)
        df['player_level'] = df[['player_level', 'prior_max_level', 'wild_level']].max(axis=1)

        return df

    merged_df = calc_level(merged_df)

    logging.info("Player levels calculated successfully.")
    return merged_df


def calculate_stats(base_stats, level):
    iv = 8
    iv_attack = 9
    calculated_stats = {}
    calculated_stats['hp'] = int(np.floor((base_stats['base_hp'] + iv) * 2 * level / 100) + level + 10)
    calculated_stats['attack'] = int(np.floor((base_stats['base_attack'] + iv_attack) * 2 * level / 100) + 5)
    calculated_stats['defense'] = int(np.floor((base_stats['base_defense'] + iv) * 2 * level / 100) + 5)
    calculated_stats['sp_attack'] = int(np.floor((base_stats['base_sp_attack'] + iv) * 2 * level / 100) + 5)
    calculated_stats['sp_defense'] = int(np.floor((base_stats['base_sp_defense'] + iv) * 2 * level / 100) + 5)
    calculated_stats['speed'] = int(np.floor((base_stats['base_speed'] + iv) * 2 * level / 100) + 5)
    return calculated_stats


def determine_pokemon_by_level(stats_df, merged_df):
    logging.info("Determining Pokémon by level...")

    stats_df['evo_stage'] = stats_df['evo_stage'].fillna(0)

    stats_df_sorted = stats_df.sort_values(['evo_id', 'evo_lvl'])
    stats_df_sorted['min_level'] = stats_df_sorted['evo_lvl'].fillna(0)
    stats_df_sorted['max_level'] = stats_df_sorted.groupby('evo_id')['evo_lvl'].shift(-1)

    # Adjust max_level based on evo_stage condition
    def adjust_max_level(row, df):
        next_rows = df[(df['evo_id'] == row['evo_id']) & ((df['evo_lvl'] > row['evo_lvl']) | (df['evo_stage'] > row['evo_stage']))]
        if not next_rows.empty and next_rows.iloc[0]['evo_stage'] > 0:
            return np.inf
        return row['max_level']

    stats_df_sorted['max_level'] = stats_df_sorted.apply(lambda row: adjust_max_level(row, stats_df_sorted), axis=1)
    stats_df_sorted['max_level'] = stats_df_sorted['max_level'].fillna(np.inf)

    # merged_df = pd.merge(merged_df, stats_df_sorted, on=['evo_id', 'evo_index'], how='left', suffixes=('_merged', ''))
    merged_df = pd.merge(merged_df, stats_df_sorted, on='evo_id', how='left', suffixes=('_merged', ''))

    # Filter out entries that are not possible
    condition = (
            (merged_df['player_level'] >= merged_df['min_level']) &
            (merged_df['player_level'] < merged_df['max_level']) &
            (merged_df['stage_enc'] >= merged_df['evo_stage'])
    )

    merged_df = merged_df[condition]
    merged_df = merged_df.sort_values(['enc_id', 'evo_index'], ascending=[True, True]).groupby(
        ['variant_id', 'enc_id']).tail(1)

    for index, row in merged_df.iterrows():
        base_stats = {
            'base_hp': row['base_hp'],
            'base_attack': row['base_attack'],
            'base_defense': row['base_defense'],
            'base_sp_attack': row['base_sp_attack'],
            'base_sp_defense': row['base_sp_defense'],
            'base_speed': row['base_speed']
        }
        calculated_stats = calculate_stats(base_stats, row['player_level'])
        merged_df.at[index, 'hp'] = calculated_stats['hp']
        merged_df.at[index, 'attack'] = calculated_stats['attack']
        merged_df.at[index, 'defense'] = calculated_stats['defense']
        merged_df.at[index, 'sp_attack'] = calculated_stats['sp_attack']
        merged_df.at[index, 'sp_defense'] = calculated_stats['sp_defense']
        merged_df.at[index, 'speed'] = calculated_stats['speed']

    # Ensure the columns are of int type
    int_columns = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    merged_df[int_columns] = merged_df[int_columns].astype(int)

    logging.info("Pokémon determined by level and stats calculated successfully.")
    return merged_df


def prepare_moves_df(moves_df, stages_df):
    logging.info("Preparing moves DataFrame...")
    moves_df = pd.merge(moves_df, stages_df[['location', 'location_stage']], left_on='move_stage', right_on='location',
                        how='left')
    moves_df['move_stage'] = moves_df['location_stage'].fillna(0).astype(int)
    moves_df = moves_df.drop(columns=['location', 'location_stage'])
    moves_df.loc[moves_df['move_stage'].notna() & moves_df['move_level'].isna(), 'move_level'] = 0
    logging.info("Moves DataFrame prepared successfully.")
    return moves_df


def calculate_expected_damage(move, base_speed):
    crit_binary = 1 if move.crit else 0
    expected_damage = move.power * move.accuracy * ((1 + 7 * crit_binary) * (base_speed / 2))
    return expected_damage


def assign_best_moves(row, moves_df):
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
            type_moves['expected_damage'] = type_moves.apply(
                lambda m: calculate_expected_damage(m, base_speed), axis=1
            )
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


def process_variants(merged_df, moves_df):
    total_variants = merged_df['variant_id'].nunique()
    logging.info(f"Total variants to process: {total_variants}")

    merged_ddf = dd.from_pandas(merged_df, npartitions=cpu_count())

    # Keep moves_df as a Pandas DataFrame

    def initialize_new_columns(df):
        for col, dtype in new_columns.items():
            df[col] = pd.Series(dtype=dtype)
        return df

    # Initialize new columns in merged_df
    merged_df = initialize_new_columns(merged_df)

    # Now create output_meta from merged_df
    output_meta = merged_df.head(0).copy()

    with ProgressBar():
        results = merged_ddf.map_partitions(
            lambda df: df.apply(assign_best_moves, moves_df=moves_df, axis=1),
            meta=output_meta
        ).compute(scheduler='processes')

    logging.info("Best moves assigned successfully.")
    return results


def calc_se(df):
    # Read the type chart
    type_chart = pd.read_csv('../gen_1_data_raw/typechart_gen_1.csv', index_col=0)

    # Initialize new columns
    df['se1'] = 1.0
    df['se1_enc'] = 1.0
    df['se2'] = 1.0
    df['se2_enc'] = 1.0
    df['se3'] = 1.0
    df['se3_enc'] = 1.0
    df['se4'] = 1.0
    df['se4_enc'] = 1.0

    # Function to get effectiveness
    def get_effectiveness(move_type, target_types):
        if pd.isna(move_type) or pd.isna(target_types) or move_type == '-' or target_types == '-':
            return 1.0
        target_types = target_types.split('/')
        effectiveness = 1.0
        for target_type in target_types:
            effectiveness *= type_chart.loc[move_type, target_type]
        return effectiveness

    # Calculate effectiveness for each move
    for i in range(1, 5):
        df[f'move{i}_se'] = df.apply(lambda row: get_effectiveness(row[f'move{i}_type'], row['types']), axis=1)
        df[f'move{i}_se_enc'] = df.apply(lambda row: get_effectiveness(row[f'move{i}_type_enc'], row['types_enc']), axis=1)

    return df

def calc_ehl(row):

    phys_types = ['Normal', 'Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel']
    enc_stage = row['stage_enc']
    level, level_enc = row['player_level'], row['level_enc']
    types = row['types'].split('/')
    types_enc = row['types_enc'].split('/')
    pokemon_type1, pokemon_type2 = types[0], types[1] if len(types) > 1 else None
    pokemon_type1_enc, pokemon_type2_enc = types_enc[0], types_enc[1] if len(types_enc) > 1 else None
    hp, hp_enc = row['hp'], row['hp_enc']
    speed, speed_enc = row['speed'], row['speed_enc']

    hp_calc = (((hp + 8) * 2 * level) / 100) + level + 10
    hp_calc_enc = hp_enc
    speed_calc = (((speed + 8) * 2 * level) / 100) + 5
    if enc_stage > 26:
        speed_calc *= 1.125
    speed_calc_enc = speed_enc

    def calc_stats(move_type, atk, defense, crit, power, acc, stab, se, level, enc_stage, phys_spec, speed):
        if power != '-':
            dv_atk = 9 if phys_spec == 'physical' else 8
            atk_calc = ((atk + dv_atk) * 2 * level / 100) + 5
            if phys_spec == 'physical' and enc_stage > 5:
                atk_calc *= 1.125
            if phys_spec == 'special' and enc_stage > 32:
                atk_calc *= 1.125
            def_calc = ((defense + 8) * 2 * level / 100) + 5
            if phys_spec == 'physical' and enc_stage > 14:
                def_calc *= 1.125
            if phys_spec == 'special' and enc_stage > 32:
                def_calc *= 1.125
            if crit:
                crit = ((min(8 * speed / 2, 255) / 256) + 1)
            else:
                crit = (((speed / 2) / 256) + 1)
            calc_dam = (((((2 * float(level) * (float(crit) + 1) / 5) + 2) * float(power) * float(atk_calc) / float(def_calc)) / 50) + 2) * float(stab) * float(se) * float(acc)
            return np.nan_to_num(calc_dam, nan=0)
        else:
            return 0

    def get_stab(move_type, pokemon_type1, pokemon_type2):
        return 1.5 if move_type == pokemon_type1 or move_type == pokemon_type2 else 1


    moves = [(row[f'move{i}_power'], row[f'move{i}_accuracy'], row[f'move{i}_crit'], row[f'move{i}_type'], row[f'move{i}_power_enc'], row[f'move{i}_accuracy_enc'], row[f'move{i}_crit_enc'], row[f'move{i}_type_enc'], row[f'move{i}_se'], row[f'move{i}_se_enc']) for i in range(1, 5)]
    max_dam, max_dam_enc = 0, 0

    for power, acc, crit, move_type, power_enc, acc_enc, crit_enc, move_type_enc, se, se_enc in moves:
        phys_spec = 'physical' if move_type in phys_types else 'special'
        phys_spec_enc = 'physical' if move_type_enc in phys_types else 'special'
        stab = get_stab(move_type, pokemon_type1, pokemon_type2)
        stab_enc = get_stab(move_type_enc, pokemon_type1_enc, pokemon_type2_enc)
        atk = row['attack'] if phys_spec == 'physical' else row['sp_attack']
        defense_enc = row['defense_enc'] if phys_spec == 'physical' else row['sp_defense_enc']
        atk_enc = row['attack_enc'] if phys_spec_enc == 'physical' else row['sp_attack_enc']
        defense = row['defense'] if phys_spec_enc == 'physical' else row['sp_defense']

        calc_dam = calc_stats(move_type, atk, defense_enc, crit, power, acc, stab, se, level, enc_stage, phys_spec, speed)
        calc_dam_enc = calc_stats(move_type_enc, atk_enc, defense, crit_enc, power_enc, acc_enc, stab_enc, se_enc, level_enc, enc_stage, phys_spec_enc, speed_enc)

        max_dam = max(max_dam, calc_dam)
        max_dam_enc = max(max_dam_enc, calc_dam_enc)

    max_dam = max(1, math.floor(max_dam))
    max_dam_enc = max(1, math.floor(max_dam_enc))

    hp_calc = math.floor(hp_calc)
    current_hp = hp_calc
    current_hp_enc = hp_calc_enc
    speed_calc = math.floor(speed_calc)
    rounds = 0

    if enc_stage < row['wild_location_stage']: # or enc_stage < row['method_stage']:
        return 1000000000

    turn = 1 if speed_calc > speed_calc_enc else 0 if speed_calc < speed_calc_enc else -1

    while current_hp > 0 and current_hp_enc > 0:
        rounds += 1
        if rounds > 20:
            return 1000000
        if turn == 1:
            current_hp_enc -= max_dam
        elif turn == 0:
            current_hp -= max_dam_enc
        else:
            current_hp -= max_dam_enc / 2
            current_hp_enc -= max_dam / 2
        turn = 1 - turn

    if current_hp <= 0:
        current_hp_enc = max(0, current_hp_enc)
        ehl = (current_hp_enc / hp_calc_enc) + 1000
    else:
        current_hp = max(0, current_hp)
        ehl = 1 - (current_hp / hp_calc)

    return ehl


def create_ehl_pivot(df):
    logging.info(f"Generating ehl pivot dataframe...")

    # Step 1: Find the maximum enc_id and variant_id
    max_variant_id = df['variant_id'].max()
    max_enc_id = df['enc_id'].max()

    # Step 2: Create a full set of combinations of variant_id and enc_id
    variant_ids = np.arange(1, max_variant_id + 1)
    enc_ids = np.arange(1, max_enc_id + 1)
    full_index = pd.MultiIndex.from_product([variant_ids, enc_ids], names=['variant_id', 'enc_id'])

    # Existing combinations
    existing_index = pd.MultiIndex.from_arrays([df['variant_id'], df['enc_id']])

    # Identify missing combinations
    missing_index = full_index.difference(existing_index)

    # Step 3: For each missing combination, add a row with proper variant_id and enc_id, EHL=1e9
    missing_df = pd.DataFrame(index=missing_index).reset_index()
    missing_df['ehl'] = 1e9  # Set EHL value
    # Set other columns to None
    for col in df.columns:
        if col not in ['variant_id', 'enc_id', 'ehl']:
            missing_df[col] = None

    # Ensure all columns are in the same order
    missing_df = missing_df[df.columns]

    # Append missing_df to df
    df = pd.concat([df, missing_df], ignore_index=True)

    # Step 4: Create a new dataframe with variant_id as rows, enc_id as columns, ehl as values
    pivot_df = df.pivot_table(index='variant_id', columns='enc_id', values='ehl', aggfunc='first')

    # Reset index to have variant_id as a column
    pivot_df.reset_index(inplace=True)

    # Optional: Rename columns to make enc_id explicit
    pivot_df.columns.name = None  # Remove the name from columns
    pivot_df = pivot_df.rename(columns=lambda x: f'enc_id_{x}' if isinstance(x, int) else x)

    return pivot_df


def save_final_output(merged_df):
    logging.info("Saving final output to CSV...")
    final_columns = [
        'variant_id', 'enc_id', 'evo_id', 'pokemon', 'level_enc', 'player_level', 'stage_enc',
        'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'types',
        'move1_name', 'move1_type', 'move1_power', 'move1_accuracy', 'move1_crit',
        'move2_name', 'move2_type', 'move2_power', 'move2_accuracy', 'move2_crit',
        'move3_name', 'move3_type', 'move3_power', 'move3_accuracy', 'move3_crit',
        'move4_name', 'move4_type', 'move4_power', 'move4_accuracy', 'move4_crit',
        'hp_enc', 'attack_enc', 'defense_enc', 'sp_attack_enc', 'sp_defense_enc', 'speed_enc', 'types_enc',
        'exp_type_enc', 'exp_enc', 'move1_name_enc', 'move1_type_enc', 'move1_power_enc', 'move1_accuracy_enc',
        'move1_crit_enc', 'move2_name_enc', 'move2_type_enc', 'move2_power_enc', 'move2_accuracy_enc', 'move2_crit_enc',
        'move3_name_enc', 'move3_type_enc', 'move3_power_enc', 'move3_accuracy_enc', 'move3_crit_enc', 'move4_name_enc',
        'move4_type_enc', 'move4_power_enc', 'move4_accuracy_enc', 'move4_crit_enc', 'ehl'
    ]
    final_df = merged_df[final_columns]
    final_df.columns = final_df.columns.str.lower()
    final_df.sort_values(['variant_id', 'enc_id'], ascending=[True, True], inplace=True)
    final_df.to_csv('variants_x_enc_gen_1.csv', index=False)
    logging.info("Final output saved successfully.")


# def main():
#     try:
#         logging.info("Starting the program...")
#         variants_df, stats_df, exp_table_df, encounters_df, moves_df, stages_df = read_csv_files()
#         cross_join_df = create_cross_join(variants_df, encounters_df)
#         exp_table_melted = prepare_exp_table(exp_table_df)
#         cross_join_df['level_enc'] = cross_join_df['level_enc']
#         merged_df = merge_and_calculate_levels(cross_join_df, stats_df, exp_table_melted)
#         merged_df = determine_pokemon_by_level(stats_df, merged_df)
#         moves_df = prepare_moves_df(moves_df, stages_df)
#         merged_df['base_speed'] = merged_df['speed']
#         merged_df = process_variants(merged_df, moves_df)
#         save_final_output(merged_df)
#         logging.info("Processing completed successfully.")
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")


def main():
    logging.info("Starting the program...")
    variants_df, stats_df, exp_table_df, encounters_df, moves_df, stages_df = read_csv_files()
    cross_join_df = create_cross_join(variants_df, encounters_df)
    exp_table_melted = prepare_exp_table(exp_table_df)
    cross_join_df['level_enc'] = cross_join_df['level_enc']
    merged_df = merge_and_calculate_levels(cross_join_df, stats_df, exp_table_melted)
    merged_df = determine_pokemon_by_level(stats_df, merged_df)
    moves_df = prepare_moves_df(moves_df, stages_df)
    merged_df['base_speed'] = merged_df['speed']
    merged_df = process_variants(merged_df, moves_df)
    merged_df = calc_se(merged_df)
    logging.info(f"Calculating EHL values...")
    merged_df['ehl'] = merged_df.apply(calc_ehl, axis=1)
    # save_final_output(merged_df)
    ehl_pivot = create_ehl_pivot(merged_df)
    logging.info(f"Saving ehl pivot dataframe...")
    ehl_pivot.to_csv('ehl_pivot.csv', index=False)
    logging.info("Processing completed successfully.")


if __name__ == "__main__":
    main()
