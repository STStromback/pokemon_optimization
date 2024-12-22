import csv
import string

# PART 1

def clean_pokemon_name(name):
    # Remove punctuation
    name = name.translate(str.maketrans('', '', string.punctuation))
    # Replace gender symbols more robustly
    name = name.replace('â™€', '_f').replace('â™‚', '_m')  # Using Unicode code points for female and male symbols
    # Replace spaces with underscores and make lowercase
    return name.replace(' ', '_').lower()


def clean_location_name(location_line):
    """
    Extracts the location and trainer name from a line like "Sprout Tower - 2F - Sage Nico".
    """
    # Split the line by '-' and trim whitespace
    parts = [part.strip() for part in location_line.split('-')]

    if len(parts) > 2:
        # Last part is the trainer name; the rest form the location
        trainer_name = parts[-1]
        location = ' - '.join(parts[:-1])
    elif len(parts) == 2:
        # Two parts: assume the second is the trainer name
        location, trainer_name = parts
    else:
        # Single part: location only, unknown trainer
        location = parts[0]
        trainer_name = 'Unknown'

    # Normalize names
    location = location.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_').lower()
    trainer_name = trainer_name.translate(str.maketrans('', '', string.punctuation))

    return location, trainer_name


def parse_pokemon_data(lines, location, trainer_name):
    pokemon_data = []
    pokemon_names = []
    levels = []
    moves = {1: [], 2: [], 3: [], 4: []}
    stats = {'Max HP': [], 'Attack': [], 'Defense': [], 'Sp. Atk.': [], 'Sp. Def.': [], 'Speed': []}

    for i, line in enumerate(lines):
        parts = line.split('\t')
        if "Number of Pokemon:" in line:
            if i + 1 < len(lines):
                pokemon_names = [clean_pokemon_name(part.strip()) for part in lines[i + 1].split('\t') if part.strip()]
        elif "Level" in parts[0]:
            levels = [part.strip().split(' ')[1] if 'Level' in part and len(part.strip().split(' ')) > 1 else '-' for
                      part in parts]
        elif any(mv in parts[0] for mv in ["Move 1", "Move 2", "Move 3", "Move 4"]):
            move_number = int(parts[0].split(' ')[1])
            moves[move_number] = [part.strip() if part.strip() != '-' else '-' for part in parts[1:]]
        elif any(stat in parts[0] for stat in stats.keys()):
            stat_name = parts[0].strip()
            stats[stat_name] = [part.strip() if part.strip() != '-' else '-' for part in parts[1:]]
        elif "EXP. Yield" in line:
            # Compile data
            for i, name in enumerate(pokemon_names):
                pokemon_info = {
                    'pokemon': name,
                    'level': levels[i + 1] if i < len(levels) else '-',
                    'move1': moves[1][i] if i < len(moves[1]) else '-',
                    'move2': moves[2][i] if i < len(moves[2]) else '-',
                    'move3': moves[3][i] if i < len(moves[3]) else '-',
                    'move4': moves[4][i] if i < len(moves[4]) else '-',
                    'hp': stats['Max HP'][i] if i < len(stats['Max HP']) else '-',
                    'attack': stats['Attack'][i] if i < len(stats['Attack']) else '-',
                    'defense': stats['Defense'][i] if i < len(stats['Defense']) else '-',
                    'sp_attack': stats['Sp. Atk.'][i] if i < len(stats['Sp. Atk.']) else '-',
                    'sp_defense': stats['Sp. Def.'][i] if i < len(stats['Sp. Def.']) else '-',
                    'speed': stats['Speed'][i] if i < len(stats['Speed']) else '-',
                    'location_enc': location,
                    'trainer_name_enc': trainer_name
                }
                pokemon_data.append(pokemon_info)
            break

    return pokemon_data


def process_file(filename, output_csv):
    with open(filename, 'r') as file:
        lines = file.readlines()

    pokemon_entries = []
    current_section = []
    current_location = None
    current_trainer_name = None

    for i, line in enumerate(lines):
        if 'Battle Rewards:' in line:
            if current_section:  # Process the previous section if exists
                pokemon_entries.extend(parse_pokemon_data(current_section, current_location, current_trainer_name))
            current_section = []  # Start a new section
            current_location, current_trainer_name = clean_location_name(lines[i - 1].strip())
        current_section.append(line.strip())

    if current_section:  # Catch any remaining data
        pokemon_entries.extend(parse_pokemon_data(current_section, current_location, current_trainer_name))

    # Write to CSV if entries exist
    if pokemon_entries:
        fieldnames = ['pokemon', 'level', 'move1', 'move2', 'move3', 'move4', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'location_enc', 'trainer_name_enc']
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in pokemon_entries:
                writer.writerow(entry)
    else:
        print("No Pokemon data was parsed. Please check the input file format.")

process_file('../data_raw/data_raw_gen_2/TrainerDataGen2Raw.txt', '../data_curated/data_curated_gen_2/trainer_data_gen2.csv')

# PART 2

import pandas as pd

# Load data
trainer_data = pd.read_csv('../data_curated/data_curated_gen_2/trainer_data_gen2.csv')
unique_moves = pd.read_csv('../data_raw/data_raw_gen_2/unique_moves_gen_2.csv')
exp_table = pd.read_csv('../data_raw/data_raw_gen_2/exp_table.csv')
exp_types = pd.read_csv('../data_raw/data_raw_gen_2/exp_types.csv')
stages = pd.read_csv('../data_raw/data_raw_gen_2/stages_gen_2.csv')
stats = pd.read_csv('../data_raw/data_raw_gen_2/stats_gen_2.csv')

exp_types['pokemon'] = exp_types['pokemon'].str.lower()

trainer_data.columns = ['pokemon_enc', 'level_enc', 'move1_enc', 'move2_enc', 'move3_enc', 'move4_enc', 'hp_enc', 'atk_enc',
       'def_enc', 'spatk_enc', 'spdef_enc', 'spd_enc', 'location_enc', 'trainer_name_enc']

# Step 1: Add the location_stage column from gen2_stages to the trainer_data_gen2
trainer_data['location_enc'] = trainer_data['location_enc'].replace({'__': '_'}, regex=True)
trainer_data['location_enc'] = trainer_data['location_enc'].replace({
    '_1f':'',
    '_2f':'',
    '_3f':'',
    '_4f':'',
    '_5f':'',
    '_b1f':'',
    '_b2f':'',
    '_b3f':''
}, regex=True)
trainer_data = trainer_data.merge(stages[['location', 'location_stage']], left_on='location_enc', right_on='location', how='left')

# Step 2: Add the exp_type_enc column
trainer_data = trainer_data.merge(exp_types[['pokemon', 'exp_type']], left_on='pokemon_enc', right_on='pokemon', how='left')
trainer_data.rename(columns={'exp_type': 'exp_type_enc'}, inplace=True)

# Step 3: Generate the exp_enc column
def get_exp_enc(row):
    exp_type = row['exp_type_enc']
    level = row['level_enc']
    return exp_table.loc[exp_table['Level'] == level, exp_type].values[0]

trainer_data['exp_enc'] = trainer_data.apply(get_exp_enc, axis=1)
trainer_data.sort_values(by=['location_stage', 'exp_enc'], ascending=[True, True], inplace=True)
trainer_data['exp_enc_max'] = trainer_data['exp_enc'].expanding().max().astype(int)

# Step 4: Add move details columns
def get_move_details(move):
    if move == '-':
        return pd.Series([None, None, None, None])
    move_details = unique_moves.loc[unique_moves['move'] == move]
    if not move_details.empty:
        return pd.Series([move_details.iloc[0]['type'], move_details.iloc[0]['power'], move_details.iloc[0]['accuracy'], move_details.iloc[0]['crit']])
    else:
        return pd.Series([None, None, None, None])

for i in range(1, 5):
    trainer_data[[f'move_type_{i}', f'power{i}_enc', f'acc{i}_enc', f'crit{i}_enc']] = trainer_data[f'move{i}_enc'].apply(get_move_details)

# Reorder and rename columns to match the desired format
trainer_data = trainer_data[['pokemon_enc', 'trainer_name_enc', 'level_enc', 'exp_type_enc', 'location_enc', 'location_stage', 'exp_enc', 'exp_enc_max', 'hp_enc', 'atk_enc', 'def_enc', 'spatk_enc', 'spdef_enc', 'spd_enc',
                             'acc1_enc', 'acc2_enc', 'acc3_enc', 'acc4_enc', 'crit1_enc', 'crit2_enc', 'crit3_enc', 'crit4_enc',
                             'move1_enc', 'move2_enc', 'move3_enc', 'move4_enc', 'move_type_1', 'move_type_2', 'move_type_3', 'move_type_4',
                             'power1_enc', 'power2_enc', 'power3_enc', 'power4_enc']]

trainer_data.rename(columns={'move1_enc': 'move_name_1_enc', 'move2_enc': 'move_name_2_enc', 'move3_enc': 'move_name_3_enc', 'move4_enc': 'move_name_4_enc'}, inplace=True)

# trainer_data.sort_values(by=['location_stage', 'level_enc'], ascending=[True, True], inplace=True)
trainer_data['enc_id'] = range(1, len(trainer_data)+1)

types = stats[['pokemon','types']]
types.rename(columns={'types': 'types_enc', 'pokemon': 'pokemon_enc'}, inplace=True)
types['pokemon_enc'] = types['pokemon_enc'].str.lower()

trainer_data = trainer_data.merge(types, on='pokemon_enc', how='left')

trainer_data.drop_duplicates(inplace=True)


# Convert to standard df format for pipeline
def transform_dataframe(input_df):
    # Rename columns to standardize and prepare for transformation
    input_df.rename(columns={
        'atk_enc': 'attack_enc',
        'def_enc': 'defense_enc',
        'spatk_enc': 'sp_attack_enc',
        'spdef_enc': 'sp_defense_enc',
        'spd_enc': 'speed_enc'
    }, inplace=True)

    # Fill missing values with '-'
    input_df.fillna('-', inplace=True)

    # Create the output dataframe structure
    output_columns = [
        'enc_id', 'trainer_name_enc', 'location_enc', 'pokemon_enc', 'level_enc', 'stage_enc',
        'hp_enc', 'attack_enc', 'defense_enc', 'sp_attack_enc', 'sp_defense_enc', 'speed_enc',
        'types_enc', 'exp_type_enc', 'exp_enc',
        'move1_name_enc', 'move1_type_enc', 'move1_power_enc', 'move1_accuracy_enc', 'move1_crit_enc',
        'move2_name_enc', 'move2_type_enc', 'move2_power_enc', 'move2_accuracy_enc', 'move2_crit_enc',
        'move3_name_enc', 'move3_type_enc', 'move3_power_enc', 'move3_accuracy_enc', 'move3_crit_enc',
        'move4_name_enc', 'move4_type_enc', 'move4_power_enc', 'move4_accuracy_enc', 'move4_crit_enc'
    ]

    # Map input data to output format
    output_df = pd.DataFrame(columns=output_columns)

    for _, row in input_df.iterrows():
        # Extract move information
        moves = []
        for i in range(1, 5):
            power = row.get(f'power{i}_enc', '-')
            name = '-' if power in ['-', None] else row.get(f'move_name_{i}_enc', '-') or '-'
            crit = row.get(f'crit{i}_enc', '-')
            crit = False if power not in ['-', None] and crit in ['-', None] else crit
            move = {
                'name': name,
                'type': row.get(f'move_type_{i}', '-') or '-',
                'power': power,
                'accuracy': row.get(f'acc{i}_enc', '-') or '-',
                'crit': crit
            }
            moves.append(move)

        # Append the transformed row to the output dataframe
        output_df = pd.concat([
            output_df,
            pd.DataFrame([{
                'enc_id': row['enc_id'],
                'trainer_name_enc': row['trainer_name_enc'],
                'location_enc': row['location_enc'],
                'pokemon_enc': row['pokemon_enc'],
                'level_enc': row['level_enc'],
                'stage_enc': row['location_stage'],
                'hp_enc': row['hp_enc'],
                'attack_enc': row['attack_enc'],
                'defense_enc': row['defense_enc'],
                'sp_attack_enc': row['sp_attack_enc'],
                'sp_defense_enc': row['sp_defense_enc'],
                'speed_enc': row['speed_enc'],
                'types_enc': row['types_enc'],
                'exp_type_enc': row['exp_type_enc'],
                'exp_enc': row['exp_enc'],
                **{f'move{i+1}_name_enc': moves[i]['name'] for i in range(4)},
                **{f'move{i+1}_type_enc': moves[i]['type'] for i in range(4)},
                **{f'move{i+1}_power_enc': moves[i]['power'] for i in range(4)},
                **{f'move{i+1}_accuracy_enc': moves[i]['accuracy'] for i in range(4)},
                **{f'move{i+1}_crit_enc': moves[i]['crit'] for i in range(4)}
            }])
        ], ignore_index=True)

    return output_df


input_df = trainer_data
output_df = transform_dataframe(input_df)

# Save to a new CSV file
output_df.to_csv('../data_curated/data_curated_gen_2/encounters_gen_2.csv', index=False)
