import pandas as pd
import numpy as np

# Step 1: Read the CSV files into pandas DataFrames
<<<<<<< HEAD
variants_df = pd.read_csv('../variants_gen_1.csv')
stats_df = pd.read_csv('../stats_gen_1.csv')
exp_table_df = pd.read_csv('../exp_table.csv')
encounters_df = pd.read_csv('../encounters_gen_1.csv')
moves_df = pd.read_csv('../moves_gen_1.csv')
stages_df = pd.read_csv('../stages_gen_1.csv')
=======
variants_df = pd.read_csv('../gen_1_data_curated/variants_gen_1.csv')
stats_df = pd.read_csv('../gen_1_data_raw/stats_gen_1.csv')
exp_table_df = pd.read_csv('../gen_1_data_raw/exp_table.csv')
encounters_df = pd.read_csv('../gen_1_data_curated/encounters_gen_1.csv')
moves_df = pd.read_csv('../gen_1_data_raw/moves_gen_1.csv')
stages_df = pd.read_csv('../gen_1_data_raw/stages_gen_1.csv')
>>>>>>> 36bb595 (many changes)

# DEBUG
# variants_df = variants_df[variants_df.evo_id == 3]

# Step 2: Create all possible combinations of variant_id and enc_id (cross join)
variants_df['key'] = 1
encounters_df['key'] = 1
cross_join_df = pd.merge(variants_df, encounters_df, on='key').drop('key', axis=1)

# Step 3: Prepare the experience table
exp_table_melted = pd.melt(exp_table_df, id_vars=['Level'], var_name='exp_type', value_name='cumulative_exp')

# Step 4: Set `level_enc` directly from the encounters_df to represent the level of the encountered Pokémon
cross_join_df['level_enc'] = cross_join_df['level_enc']  # This comes from the encounter data


# Step 5: Calculate the player's Pokémon level based on its own experience type
def calculate_player_level(exp_enc, exp_type, exp_table_melted):
    exp_type_data = exp_table_melted[exp_table_melted['exp_type'] == exp_type]
    level = exp_type_data[exp_type_data['cumulative_exp'] <= exp_enc]['Level'].max()
    return level if pd.notnull(level) else 1


# Merge the player's Pokémon experience type into the cross_join_df
merged_df = pd.merge(cross_join_df, stats_df[['evo_id', 'exp_type']], on='evo_id')

# Calculate `player_level` based on the player's Pokémon experience type and exp_enc
merged_df['player_level'] = merged_df.apply(
    lambda row: calculate_player_level(row['exp_enc'], row['exp_type'], exp_table_melted), axis=1)

# Step 6: Determine which Pokémon is present within each evo_id based on the player's Level
stats_df_sorted = stats_df.sort_values(['evo_id', 'evo_stage'])
stats_df_sorted['min_level'] = stats_df_sorted.groupby('evo_id')['evo_lvl'].shift(1).fillna(0)
stats_df_sorted['max_level'] = stats_df_sorted.groupby('evo_id')['evo_lvl'].shift(-1)
stats_df_sorted['max_level'] = stats_df_sorted['max_level'].fillna(np.inf)

# Step 7: Merge stats with merged_df and filter based on the player's Level
merged_df = pd.merge(merged_df, stats_df_sorted, on='evo_id')
condition = (merged_df['player_level'] >= merged_df['min_level']) & (merged_df['player_level'] < merged_df['max_level'])
merged_df = merged_df[condition].drop_duplicates(subset=['variant_id', 'enc_id', 'evo_id'])

# Step 8: Convert `move_stage` from text to integer using `location` and `location_stage` from the stages file
moves_df = pd.merge(moves_df, stages_df[['location', 'location_stage']], left_on='move_stage', right_on='location',
                    how='left')
moves_df['move_stage'] = moves_df['location_stage'].fillna(0).astype(
    int)  # Fill missing stages as 0 (for moves with no location) and convert to integer
moves_df = moves_df.drop(columns=['location', 'location_stage'])  # Clean up extra columns

# Set move_level to zero where move_stage is not NaN but move_level is NaN
moves_df.loc[moves_df['move_stage'].notna() & moves_df['move_level'].isna(), 'move_level'] = 0

# Step 9: Save base speed temporarily for expected damage calculation
merged_df['base_speed'] = merged_df['speed']


# Step 10: Calculate the expected damage for each move
def calculate_expected_damage(move, base_speed):
    crit_binary = 1 if move.crit else 0
    expected_damage = move.power * move.accuracy * ((1 + 7 * crit_binary) * (base_speed / 2))
    return expected_damage


# Apply the function to assign moves to each Pokémon in merged_df
total_variants = merged_df['variant_id'].nunique()  # Get the total number of unique variant_ids
variant_counter = 0


# Step 11: Assign the best moves for each variant based on the expected damage calculation
def assign_best_moves_for_variant_with_progress(row, moves_df):
    global variant_counter
    base_speed = row['base_speed']
    variant_types = [row[f'move_type_{i}'] for i in range(1, 5) if row[f'move_type_{i}'] != 'none']

    # Check if the variant_id has changed
    if variant_counter < row['variant_id']:
        variant_counter = row['variant_id']
        print(f"Processing variant {variant_counter}/{total_variants}")

    best_moves = {}

    for move_type in variant_types:
        # Filter moves based on Pokémon, type, level, and stage_enc
        valid_moves = moves_df[(moves_df['pokemon'] == row['pokemon']) &
                               (moves_df['move_type'] == move_type) &
                               ((moves_df['move_level'].isna()) | (moves_df['move_level'] <= row['player_level'])) &
                               (moves_df['move_stage'] <= row['stage_enc'])]

        if not valid_moves.empty:
            # Calculate expected damage for each move
            valid_moves['expected_damage'] = valid_moves.apply(lambda m: calculate_expected_damage(m, base_speed),
                                                               axis=1)

            # Get the move with the highest expected damage
            best_move = valid_moves.loc[valid_moves['expected_damage'].idxmax()]
            best_moves[move_type] = best_move

    # Assign the best moves to the row, handling cases where there are fewer than 4 moves
    for i, (move_type, best_move) in enumerate(best_moves.items(), start=1):
        row[f'move{i}_name'] = best_move['move_name']
        row[f'move{i}_type'] = best_move['move_type']
        row[f'move{i}_power'] = best_move['power']
        row[f'move{i}_accuracy'] = best_move['accuracy']
        row[f'move{i}_crit'] = best_move['crit']

    # Fill empty move slots if there are fewer than 4 moves
    for i in range(len(best_moves) + 1, 5):
        row[f'move{i}_name'] = None
        row[f'move{i}_type'] = None
        row[f'move{i}_power'] = None
        row[f'move{i}_accuracy'] = None
        row[f'move{i}_crit'] = None

    return row


# Apply the function to assign moves to each Pokémon in merged_df with progress tracking
merged_df = merged_df.apply(assign_best_moves_for_variant_with_progress, axis=1, moves_df=moves_df)

# Step 12: Convert column names to lowercase and save the final DataFrame to a CSV file
final_columns = [
    'variant_id', 'enc_id', 'evo_id', 'pokemon', 'level_enc', 'player_level', 'stage_enc',
    'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'types',
    'move1_name', 'move1_type', 'move1_power', 'move1_accuracy', 'move1_crit',
    'move2_name', 'move2_type', 'move2_power', 'move2_accuracy', 'move2_crit',
    'move3_name', 'move3_type', 'move3_power', 'move3_accuracy', 'move3_crit',
    'move4_name', 'move4_type', 'move4_power', 'move4_accuracy', 'move4_crit'
]
final_df = merged_df[final_columns]

# Convert column names to lowercase
final_df.columns = final_df.columns.str.lower()

# Save to CSV
final_df.to_csv('final_output.csv', index=False)
