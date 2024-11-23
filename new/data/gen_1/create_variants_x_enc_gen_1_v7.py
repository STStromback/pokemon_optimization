import pandas as pd
import numpy as np
import logging
from multiprocessing import cpu_count
from dask import dataframe as dd
from dask.diagnostics import ProgressBar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv_files():
    logging.info("Reading CSV files...")
    variants_df = pd.read_csv('variants_gen_1.csv')
    stats_df = pd.read_csv('stats_gen_1.csv')
    exp_table_df = pd.read_csv('exp_table.csv')
    encounters_df = pd.read_csv('encounters_gen_1.csv')
    moves_df = pd.read_csv('moves_gen_1.csv')
    stages_df = pd.read_csv('stages_gen_1.csv')
    logging.info("CSV files read successfully.")

    moves_df = pd.merge(moves_df, stats_df[['pokemon', 'evo_id']], on='pokemon', how='left')

    # DEBUG
    # variants_df = variants_df[variants_df.variant_id == 4]

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
    stats_df_sorted['max_level'] = stats_df_sorted['max_level'].fillna(np.inf)

    merged_df = pd.merge(merged_df, stats_df_sorted, on=['evo_id', 'evo_index'], suffixes=('_merged', ''))

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
    variant_types = [row[f'move_type_{i}'] for i in range(1, 5) if row[f'move_type_{i}'] != 'none']
    best_moves = {}

    # Get all moves for the current Pokémon and its prior evolutions
    valid_moves = moves_df[(moves_df['evo_id'] == row['evo_id']) &
                           ((moves_df['move_level'].isna()) | (moves_df['move_level'] <= row['player_level'])) &
                           (moves_df['move_stage'] <= row['stage_enc'])]

    for move_type in variant_types:
        type_moves = valid_moves[valid_moves['move_type'] == move_type]
        if not type_moves.empty:
            type_moves = type_moves.copy()
            type_moves.loc[:, 'expected_damage'] = type_moves.apply(
                lambda m: calculate_expected_damage(m, base_speed), axis=1)
            best_move = type_moves.loc[type_moves['expected_damage'].idxmax()]
            best_moves[move_type] = best_move

    for i, (move_type, best_move) in enumerate(best_moves.items(), start=1):
        row[f'move{i}_name'] = best_move['move_name']
        row[f'move{i}_type'] = best_move['move_type']
        row[f'move{i}_power'] = best_move['power']
        row[f'move{i}_accuracy'] = best_move['accuracy']
        row[f'move{i}_crit'] = best_move['crit']

    for i in range(len(best_moves) + 1, 5):
        row[f'move{i}_name'] = None
        row[f'move{i}_type'] = None
        row[f'move{i}_power'] = None
        row[f'move{i}_accuracy'] = None
        row[f'move{i}_crit'] = None

    return row

def process_variants(merged_df, moves_df):
    total_variants = merged_df['variant_id'].nunique()
    logging.info(f"Total variants to process: {total_variants}")

    merged_ddf = dd.from_pandas(merged_df, npartitions=cpu_count())
    moves_ddf = dd.from_pandas(moves_df, npartitions=cpu_count())

    with ProgressBar():
        results = merged_ddf.map_partitions(
            lambda df: df.apply(assign_best_moves, moves_df=moves_ddf.compute(), axis=1)).compute(scheduler='processes')

    logging.info("Best moves assigned successfully.")
    return results

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
        'move4_type_enc', 'move4_power_enc', 'move4_accuracy_enc', 'move4_crit_enc'
    ]
    final_df = merged_df[final_columns]
    final_df.columns = final_df.columns.str.lower()
    final_df.sort_values(['variant_id', 'enc_id'], ascending=[True, True], inplace=True)
    final_df.to_csv('final_output.csv', index=False)
    logging.info("Final output saved successfully.")

def main():
    try:
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
        save_final_output(merged_df)
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()