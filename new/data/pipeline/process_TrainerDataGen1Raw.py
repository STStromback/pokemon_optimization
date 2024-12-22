import re
import csv
import numpy as np
import pandas as pd

config = pd.read_csv('../config/config.csv')
gen = int(config[config.rule == 'gen'].value.values[0])

location_mapping = {}
if gen == 1:
    # Location mapping dictionary
    location_mapping = {
        'Route 3': 'route_3',
        'Route 4': 'route_4',
        'Mt. Moon 1F': 'mt_moon',
        'Mt. Moon B2F': 'mt_moon',
        'Route 24': 'route_24',
        'Route 25': 'route_25',
        'SS Anne 1F Rooms': 'ss_anne',
        'SS Anne B1F Rooms': 'ss_anne',
        'SS Anne 2F Rooms': 'ss_anne',
        'SS Anne 2F': 'ss_anne',
        'SS Anne Stern': 'ss_anne',
        'Vermilion Gym': 'vermilion_gym',
        'Pewter Gym': 'pewter_gym',
        'Route 24/Route 25': 'route_24',
        'Route 11': 'route_11',
        'Viridian Forest': 'viridian_forest',
        'Route 6': 'route_6',
        'Route 9': 'route_9',
        'Cerulean City': 'cerulean_city',
        'Cerulean Gym': 'cerulean_gym',
        'Celadon Gym': 'celadon_gym',
        'Route 8': 'route_8',
        'Rock Tunnel B1F': 'rock_tunnel',
        'Rock Tunnel 1F': 'rock_tunnel',
        'Route 9/Rock Tunnel B1F': 'rock_tunnel',
        'Route 4 Surf': 'route_4_surf',
        'Route 10': 'route_10',
        'Route 13': 'route_13',
        'Route 14': 'route_14',
        'Route 15': 'route_15',
        'Route 12': 'route_12',
        'Route 16': 'route_16',
        'Route 17': 'route_17',
        'Route 18': 'route_18',
        'Route 19': 'route_19',
        'Route 20': 'route_20',
        'Route 21': 'route_21',
        'Route 22': 'route_22',
        'Route 23': 'route_23',
        'Cinnabar Gym': 'cinnabar_gym',
        'Victory Road 1F': 'victory_road',
        'Victory Road 2F': 'victory_road',
        'Victory Road 3F': 'victory_road',
        'Saffron Gym': 'saffron_gym',
        'Fighting Dojo': 'dojo',
        'Silph Co.': 'silph_co',
        'Silph Co. 2F': 'silph_co',
        'Silph Co. 3F':'silph_co',
        'Silph Co. 4F': 'silph_co',
        'Silph Co. 5F':'silph_co',
        'Silph Co. 6F': 'silph_co',
        'Silph Co. 7F': 'silph_co',
        'Silph Co. 8F': 'silph_co',
        'Silph Co. 9F': 'silph_co',
        'Silph Co. 10F': 'silph_co',
        'Silph Co. 11F': 'silph_co',
        'Mansion 1F': 'pokemon_mansion',
        'Mansion 2F': 'pokemon_mansion',
        'Mansion 3F': 'pokemon_mansion',
        'Mansion B1F': 'pokemon_mansion',
        'Game Corner': 'rocket_hideout',
        'Rocket Hideout B1F': 'rocket_hideout',
        'Rocket Hideout B2F': 'rocket_hideout',
        'Rocket Hideout B3F': 'rocket_hideout',
        'Rocket Hideout B4F': 'rocket_hideout',
        'Fuchsia Gym': 'fuchsia_gym',
        'Viridian Gym': 'viridian_gym',
        'PokÃ©mon Tower 1F': 'pokemon_tower',
        'PokÃ©mon Tower 2F': 'pokemon_tower',
        'PokÃ©mon Tower 3F': 'pokemon_tower',
        'PokÃ©mon Tower 4F': 'pokemon_tower',
        'PokÃ©mon Tower 5F': 'pokemon_tower',
        'PokÃ©mon Tower 6F': 'pokemon_tower',
        'PokÃ©mon Tower 7F': 'pokemon_tower',
        'Power Plant': 'power_plant'
    }
if gen == 2:
    location_mapping = {}

# Function to load the stage mapping from a CSV file
def load_stage_mapping(csv_filename):
    stage_mapping = {}
    # Open the CSV file
    with open(csv_filename, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)  # Read the CSV into a dictionary

        for row in reader:
            # Extract 'location' and 'location_stage'
            if gen == 2:
                location = row['ï»¿location'].strip().lower().replace(' ', '_')
            else:
                location = row['location'].strip().lower().replace(' ', '_')
            location_stage = int(row['location_stage'].strip())
            # Add to the stage mapping dictionary
            stage_mapping[location] = location_stage
    return stage_mapping

csv_file = f'../data_raw/data_raw_gen_{gen}/stages_gen_{gen}.csv'  # Path to your CSV file
stage_mapping = load_stage_mapping(csv_file)

null_location_mapping = {}
if gen == 1:
    # Null location mapping dictionary
    null_location_mapping = {
        "Green1": "pallet_town",
        "Brock": "pewter_gym",
        "Misty": "cerulean_gym",
        "LtSurge": "vermilion_gym",
        "Erika": "celadon_gym",
        "Sabrina": "saffron_gym",
        "Koga": "fuchsia_gym",
        "Blaine": "cinnabar_gym",
        "Lorelei": "indigo_plateau",
        "Bruno": "indigo_plateau",
        "Agatha": "indigo_plateau",
        "Lance": "indigo_plateau",
        "Green3": "indigo_plateau"
    }


# Special case handler for Green2 on route_22
def special_case_handling(processed_data):
    for entry in processed_data:
        if entry['trainer_name_enc'] == "Green2" and entry['location_enc'] == "route_22":
            entry['location_enc'] = "route_23"
            entry['stage_enc'] = stage_mapping['route_23']


# Load the stats from stats_gen_1.csv
def load_stats_mapping(stats_file):
    stats_mapping = {}
    with open(stats_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon = row['pokemon'].strip().lower()
            stats_mapping[pokemon] = {
                'base_hp': int(row['hp']),
                'base_attack': int(row['attack']),
                'base_defense': int(row['defense']),
                'base_sp_attack': int(row['sp_attack']),
                'base_sp_defense': int(row['sp_defense']),
                'base_speed': int(row['speed']),
                'types': row['types'],
                'exp_type': row['exp_type']
            }
    return stats_mapping


# Load the experience table from exp_table.csv
def load_exp_table(exp_table_file):
    exp_table = {}
    with open(exp_table_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            level = int(row['Level'])  # Use 'Level' as the key
            exp_table[level] = {
                'Erratic': int(row['Erratic']),
                'Fast': int(row['Fast']),
                'Medium Fast': int(row['Medium Fast']),
                'Medium Slow': int(row['Medium Slow']),
                'Slow': int(row['Slow']),
                'Fluctuating': int(row['Fluctuating'])
            }
    return exp_table


# Load moves from moves_gen_1.csv and return a mapping of pokemon -> list of move data
def load_moves_mapping(moves_file):
    moves_mapping = {}
    with open(moves_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon = row['pokemon'].strip().lower()
            if row['move_level']:  # Only consider moves that have a move_level value
                move = {
                    'move_level': float(row['move_level']),
                    'move_name': row['move_name'].strip(),
                    'move_type': row['move_type'].strip(),
                    'power': row['power'].strip(),
                    'accuracy': row['accuracy'].strip(),
                    'crit': row['crit'].strip()
                }
                if pokemon not in moves_mapping:
                    moves_mapping[pokemon] = []
                moves_mapping[pokemon].append(move)

    # Sort each Pokémon's move list by move_level
    for pokemon in moves_mapping:
        moves_mapping[pokemon].sort(key=lambda x: x['move_level'], reverse=True)

    return moves_mapping


# Filter moves based on Pokémon level and return up to the top 4 moves
def get_top_moves_for_pokemon(pokemon, level, moves_mapping):
    if pokemon not in moves_mapping:
        return []
    all_moves = moves_mapping[pokemon]
    # Filter moves that are valid for the Pokémon's level
    valid_moves = [move for move in all_moves if move['move_level'] <= level]
    # Return only the top 4 moves
    return valid_moves[:4]


# Function to calculate actual stats based on formulas
def calculate_stats(base_stats, level, gen):
    iv = 8
    iv_attack = 9
    calculated_stats = {}

    if gen <= 2:
        calculated_stats['hp_enc'] = int(np.floor((base_stats['base_hp'] + iv) * 2 * level / 100) + level + 10)
        calculated_stats['attack_enc'] = int(np.floor((base_stats['base_attack'] + iv_attack) * 2 * level / 100) + 5)
        calculated_stats['defense_enc'] = int(np.floor((base_stats['base_defense'] + iv) * 2 * level / 100) + 5)
        calculated_stats['sp_attack_enc'] = int(np.floor((base_stats['base_sp_attack'] + iv) * 2 * level / 100) + 5)
        calculated_stats['sp_defense_enc'] = int(np.floor((base_stats['base_sp_defense'] + iv) * 2 * level / 100) + 5)
        calculated_stats['speed_enc'] = int(np.floor((base_stats['base_speed'] + iv) * 2 * level / 100) + 5)
        return calculated_stats
    else:
        calculated_stats['hp_enc'] = int(np.floor((2 * base_stats['base_hp'] + iv) * level / 100) + level + 10)
        calculated_stats['attack_enc'] = int(np.floor((2 * base_stats['base_attack'] + iv_attack) * level / 100) + 5)
        calculated_stats['defense_enc'] = int(np.floor((2 * base_stats['base_defense'] + iv) * level / 100) + 5)
        calculated_stats['sp_attack_enc'] = int(np.floor((2 * base_stats['base_sp_attack'] + iv) * level / 100) + 5)
        calculated_stats['sp_defense_enc'] = int(np.floor((2 * base_stats['base_sp_defense'] + iv) * level / 100) + 5)
        calculated_stats['speed_enc'] = int(np.floor((2 * base_stats['base_speed'] + iv) * level / 100) + 5)
        return calculated_stats

    return calculated_stats


# Special case handler for Green2 on route_22
def special_case_handling(processed_data):
    for entry in processed_data:
        if entry['trainer_name_enc'] == "Green2" and entry['location_enc'] == "route_22":
            entry['location_enc'] = "route_23"
            entry['stage_enc'] = stage_mapping['route_23']


# Add moves, stats, and experience values to the result
def add_stats_and_exp_and_moves_to_pokemon_data(result, stats_mapping, exp_table, moves_mapping):
    for entry in result:
        pokemon = entry['pokemon_enc']

        # Check if the level is already an integer or convert from string if necessary
        if isinstance(entry['level_enc'], int):
            level = entry['level_enc']
        elif isinstance(entry['level_enc'], str) and entry['level_enc'].isdigit():
            level = int(entry['level_enc'])
        else:
            level = None

        if pokemon in stats_mapping and level is not None:
            stats = stats_mapping[pokemon]

            # Calculate actual stats using the provided formulas
            calculated_stats = calculate_stats(stats, level, gen)
            entry.update(calculated_stats)

            entry['types_enc'] = stats['types']
            entry['exp_type_enc'] = stats['exp_type']

            # Find experience value based on level and experience type
            if level in exp_table and stats['exp_type'] in exp_table[level]:
                entry['exp_enc'] = exp_table[level][stats['exp_type']]
            else:
                entry['exp_enc'] = '-'

            # Get top 4 moves
            top_moves = get_top_moves_for_pokemon(pokemon, level, moves_mapping)
            for idx, move in enumerate(top_moves):
                entry[f'move{idx + 1}_name_enc'] = move['move_name']
                entry[f'move{idx + 1}_type_enc'] = move['move_type']
                entry[f'move{idx + 1}_power_enc'] = move['power']
                entry[f'move{idx + 1}_accuracy_enc'] = move['accuracy']
                entry[f'move{idx + 1}_crit_enc'] = move['crit']

            # Fill empty move columns for Pokémon with fewer than 4 moves
            for idx in range(len(top_moves), 4):
                entry[f'move{idx + 1}_name_enc'] = '-'
                entry[f'move{idx + 1}_type_enc'] = '-'
                entry[f'move{idx + 1}_power_enc'] = '-'
                entry[f'move{idx + 1}_accuracy_enc'] = '-'
                entry[f'move{idx + 1}_crit_enc'] = '-'
        else:
            entry['hp_enc'] = entry['attack_enc'] = entry['defense_enc'] = '-'
            entry['sp_attack_enc'] = entry['sp_defense_enc'] = entry['speed_enc'] = '-'
            entry['types_enc'] = entry['exp_type_enc'] = entry['exp_enc'] = '-'

    return result


# Process the file and load the stats, exp_table, and moves data
def process_file_with_stats_exp_and_moves(file_path, stats_file, exp_table_file, moves_file):
    # Load the stats mapping, exp table, and moves
    stats_mapping = load_stats_mapping(stats_file)
    exp_table = load_exp_table(exp_table_file)
    moves_mapping = load_moves_mapping(moves_file)

    pokemon_entries = []  # This will store each pokemon appearance as a dictionary
    pokemon_line_pattern = re.compile(r'\s*db (\$?[0-9A-Fa-f]+),(.+?)(?:,\s*0)?$')
    location_pattern = re.compile(r';\s*(.+)')  # Pattern to capture location comments
    current_location = "Unknown"  # Default location if none specified

    result = []  # Final result to store formatted data
    trainer_name = ''

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for trainer block
            if line.endswith('Data:'):
                trainer_name = line.replace('Data:', '').strip()
                current_location = "Unknown"  # Reset location when a new trainer starts
                # print(f"New trainer found: {trainer_name}")

            # Capture location changes based on the location pattern
            location_match = location_pattern.search(line)
            if location_match:
                current_location = location_match.group(1)  # Update current location
                # print(f"Location updated: {current_location}")

            # Match Pokémon data lines using the regular expression
            match = pokemon_line_pattern.search(line)
            if match:
                level_str = match.group(1)  # Capture the level
                entries = match.group(2).split(',')  # Capture Pokémon names

                if level_str.strip().lower() == '$ff':  # Unique trainer data format
                    last_level = None
                    for entry in entries:
                        entry = entry.strip()
                        if entry.isdigit():  # New level found before a Pokémon name
                            last_level = int(entry)
                        elif entry.isalpha() or '_' in entry:  # Pokémon name
                            pokemon = entry.lower()
                            if pokemon and last_level is not None:
                                locations = current_location.split('/')
                                for loc in locations:
                                    loc = loc.strip()
                                    result.append({
                                        'trainer_name_enc': trainer_name,
                                        'location_enc': location_mapping.get(loc, loc),
                                        'pokemon_enc': pokemon,
                                        'level_enc': last_level,
                                        'stage_enc': stage_mapping.get(location_mapping.get(loc, loc), -1)
                                    })
                                    # print(f"Appended: {trainer_name}, {loc}, {pokemon}, {last_level}")
                else:  # General data format
                    level = int(level_str, 16) if level_str.startswith('$') else int(level_str)
                    for entry in entries:
                        entry = entry.strip()
                        if entry.isalpha() or '_' in entry:  # Pokémon name
                            pokemon = entry.lower()
                            if pokemon:
                                locations = current_location.split('/')
                                for loc in locations:
                                    loc = loc.strip()
                                    result.append({
                                        'trainer_name_enc': trainer_name,
                                        'location_enc': location_mapping.get(loc, loc),
                                        'pokemon_enc': pokemon,
                                        'level_enc': level,
                                        'stage_enc': stage_mapping.get(location_mapping.get(loc, loc), -1)
                                    })
                                    # print(f"Appended: {trainer_name}, {loc}, {pokemon}, {level}")

    # Add the stats, exp, and move data to each Pokémon entry
    result_with_stats_and_exp_and_moves = add_stats_and_exp_and_moves_to_pokemon_data(result, stats_mapping, exp_table,
                                                                                      moves_mapping)

    # Handle null locations using null_location_mapping
    for entry in result_with_stats_and_exp_and_moves:
        if entry['location_enc'] == 'Unknown':
            mapped_location = null_location_mapping.get(entry['trainer_name_enc'])
            if mapped_location:
                entry['location_enc'] = mapped_location
                entry['stage_enc'] = stage_mapping[mapped_location]

    # Apply special case handling for Green2 on route_22
    special_case_handling(result_with_stats_and_exp_and_moves)

    # Sort by stage_enc and exp_enc
    result_with_stats_and_exp_and_moves = sorted(result_with_stats_and_exp_and_moves,
                                                 key=lambda x: (x['stage_enc'], x['exp_enc']))

    # Add unique enc_id
    for idx, entry in enumerate(result_with_stats_and_exp_and_moves, start=1):
        entry['enc_id'] = idx

    return result_with_stats_and_exp_and_moves


# Save to CSV including the new columns for moves, calculated stats, and exp
def save_to_csv_with_moves(output_path, processed_data):
    with open(output_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        # Include additional calculated stat, exp, and move columns in the CSV header
        writer.writerow(
            ["enc_id", "trainer_name_enc", "location_enc", "pokemon_enc", "level_enc", "stage_enc", "hp_enc",
             "attack_enc", "defense_enc", "sp_attack_enc", "sp_defense_enc", "speed_enc", "types_enc", "exp_type_enc",
             "exp_enc",
             "move1_name_enc", "move1_type_enc", "move1_power_enc", "move1_accuracy_enc", "move1_crit_enc",
             "move2_name_enc", "move2_type_enc", "move2_power_enc", "move2_accuracy_enc", "move2_crit_enc",
             "move3_name_enc", "move3_type_enc", "move3_power_enc", "move3_accuracy_enc", "move3_crit_enc",
             "move4_name_enc", "move4_type_enc", "move4_power_enc", "move4_accuracy_enc", "move4_crit_enc"])
        for entry in processed_data:
            writer.writerow([
                entry['enc_id'], entry['trainer_name_enc'], entry['location_enc'], entry['pokemon_enc'],
                entry['level_enc'], entry['stage_enc'],
                entry['hp_enc'], entry['attack_enc'], entry['defense_enc'], entry['sp_attack_enc'],
                entry['sp_defense_enc'], entry['speed_enc'], entry['types_enc'], entry['exp_type_enc'],
                entry['exp_enc'],
                entry['move1_name_enc'], entry['move1_type_enc'], entry['move1_power_enc'], entry['move1_accuracy_enc'],
                entry['move1_crit_enc'],
                entry['move2_name_enc'], entry['move2_type_enc'], entry['move2_power_enc'], entry['move2_accuracy_enc'],
                entry['move2_crit_enc'],
                entry['move3_name_enc'], entry['move3_type_enc'], entry['move3_power_enc'], entry['move3_accuracy_enc'],
                entry['move3_crit_enc'],
                entry['move4_name_enc'], entry['move4_type_enc'], entry['move4_power_enc'], entry['move4_accuracy_enc'],
                entry['move4_crit_enc']
            ])


# Specify the input and output file paths
input_file = f'../data_raw/data_raw_gen_{gen}/TrainerDataGen{gen}Raw.txt'
output_file = f'../data_curated/data_curated_gen_{gen}/encounters_gen_{gen}.csv'
stats_file = f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv'
exp_table_file = f'../data_raw/data_raw_gen_{gen}/exp_table.csv'
moves_file = f'../data_raw/data_raw_gen_{gen}/moves_gen_{gen}.csv'

# Process the file with stats, exp, and moves and save the result
processed_data = process_file_with_stats_exp_and_moves(input_file, stats_file, exp_table_file, moves_file)
save_to_csv_with_moves(output_file, processed_data)

print(f"Processed data with calculated stats, moves, and experience saved to {output_file}")
