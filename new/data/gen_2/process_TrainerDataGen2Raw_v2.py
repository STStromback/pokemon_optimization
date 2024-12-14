import csv
import string
import unicodedata


# Clean location name: take only the part before the first dash, remove punctuation, format the location
def clean_location_name(location):
    location_cleaned = location.split('-')[0].strip()  # Take only the part before the first dash
    return location_cleaned.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_').lower()


# Extract the trainer name: take the part after the last dash and clean it
def extract_trainer_name(location_line):
    if '-' in location_line:
        trainer_name = location_line.split('-')[-1].strip()
        trainer_name = clean_trainer_name(trainer_name)
        return trainer_name
    return None


# Load stage mapping from CSV file
def load_stage_mapping(csv_filename):
    stage_mapping = {}
    with open(csv_filename, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            location = row['location'].strip().lower().replace(' ', '_')
            location_stage = int(row['location_stage'].strip())
            stage_mapping[location] = location_stage
    return stage_mapping


# Load the experience table from exp_table.csv
def load_exp_table(exp_table_file):
    exp_table = {}
    with open(exp_table_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            level = int(row['Level'])
            exp_table[level] = {
                'Erratic': int(row['Erratic']),
                'Fast': int(row['Fast']),
                'Medium Fast': int(row['Medium Fast']),
                'Medium Slow': int(row['Medium Slow']),
                'Slow': int(row['Slow']),
                'Fluctuating': int(row['Fluctuating'])
            }
    return exp_table


# Load the stats from stats_gen_2.csv and map pokemon names to their exp_type
def load_stats_mapping(stats_file):
    stats_mapping = {}
    with open(stats_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon = clean_pokemon_name(row['pokemon'])
            stats_mapping[pokemon] = {
                'exp_type': row['exp_type'],
                'types': row['types']
            }
    return stats_mapping


# Load the moves from moves_gen_2.csv and map them to the Pokémon
def load_moves_mapping(moves_file):
    moves_mapping = {}
    with open(moves_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon = clean_pokemon_name(row['pokemon'])
            if pokemon not in moves_mapping:
                moves_mapping[pokemon] = []
            if row['move_level']:
                moves_mapping[pokemon].append({
                    'move_level': float(row['move_level']),
                    'move_name': row['move_name'],
                    'move_type': row['move_type'],
                    'power': row['power'],
                    'accuracy': row['accuracy'],
                    'crit': row['crit']
                })
    return moves_mapping


# Get the top 4 moves for the Pokémon based on their level
def get_top_4_moves(pokemon_name, pokemon_level, moves_mapping):
    if pokemon_name not in moves_mapping:
        return ['-', '-', '-', '-'], ['-', '-', '-', '-'], ['-', '-', '-', '-'], ['-', '-', '-', '-'], ['-', '-', '-', '-']  # Ensure lists are initialized

    all_moves = moves_mapping[pokemon_name]
    valid_moves = [move for move in all_moves if move['move_level'] <= pokemon_level]
    valid_moves = sorted(valid_moves, key=lambda x: x['move_level'], reverse=True)[:4]

    move_names = [move['move_name'] for move in valid_moves] + ['-'] * (4 - len(valid_moves))
    move_types = [move['move_type'] for move in valid_moves] + ['-'] * (4 - len(valid_moves))
    move_powers = [move['power'] for move in valid_moves] + ['-'] * (4 - len(valid_moves))
    move_accuracies = [move['accuracy'] for move in valid_moves] + ['-'] * (4 - len(valid_moves))
    move_crits = [move['crit'] for move in valid_moves] + ['-'] * (4 - len(valid_moves))

    return move_names, move_types, move_powers, move_accuracies, move_crits




def parse_pokemon_line(pokemon_line):
    parts = pokemon_line.split(' Lv.')
    name_part = parts[0].strip()
    level_part = parts[1].strip()
    level = int(level_part)

    # Remove gender symbols from the name, since they shouldn't be part of the base name for matching
    name_part = name_part.replace('♂', '').replace('♀', '')

    # Clean the name for consistent matching
    name = clean_pokemon_name(name_part)
    return name, level


# Load unique moves from unique_moves_gen_2.csv
def load_unique_moves(unique_moves_file):
    unique_moves = {}
    with open(unique_moves_file, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            move_name = row['move'].strip()
            unique_moves[move_name] = {
                'type': row['type'],
                'power': row['power'],
                'accuracy': row['accuracy'],
                'crit': row['crit']
            }
    return unique_moves


# Clean Pokemon name: remove punctuation, handle gender symbols, format the name
def clean_pokemon_name(name):
    # Normalize the string to handle different unicode representations
    name = unicodedata.normalize('NFKD', name)
    # Replace male and female symbols with '_m' and '_f'
    name = name.replace('♀', '_f').replace('♂', '_m')
    # Remove any punctuation
    name = name.translate(str.maketrans('', '', string.punctuation))
    # Replace spaces with underscores
    name = name.replace(' ', '_').lower()
    name = name.replace('nidoranf', 'nidoran_f')
    name = name.replace('nidoranm', 'nidoran_m')
    return name


# Clean trainer name
def clean_trainer_name(name):
    # Normalize and lower case the name
    name = unicodedata.normalize('NFKD', name)
    name = name.strip().lower().replace(' ', '_')
    return name


# Clean location name: take only the part before the first dash, remove punctuation, format the location
def clean_location_name(location):
    location_cleaned = location.split('-')[0].strip()  # Take only the part before the first dash
    return location_cleaned.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_').lower()


# Extract the trainer name: take the part after the last dash and clean it
def extract_trainer_name(location_line):
    if '-' in location_line:
        trainer_name = location_line.split('-')[-1].strip()
        trainer_name = clean_trainer_name(trainer_name)
        return trainer_name
    return None


def parse_alternate_moves_file(filename):
    trainer_moves = {}
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    current_trainer = None
    current_pokemon = None
    i = 0
    pokemon_counter = {}  # To track the count of duplicate Pokémon species

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('$'):  # Detect the start of a new trainer section
            raw_trainer_name = line[1:].strip()
            current_trainer = clean_trainer_name(raw_trainer_name)
            trainer_moves[current_trainer] = {}
            print(f"Debug: Parsing alternate moves for trainer: {current_trainer}")  # Debugging
            i += 1
            continue

        elif ' Lv.' in line:  # Detect the start of a new Pokémon section
            pokemon_line = line
            current_pokemon, level = parse_pokemon_line(pokemon_line)

            # Count and handle multiple Pokémon of the same species
            if current_pokemon not in pokemon_counter:
                pokemon_counter[current_pokemon] = 1
            else:
                pokemon_counter[current_pokemon] += 1
            unique_pokemon = f"{current_pokemon}_{pokemon_counter[current_pokemon]}"

            moves = []
            i += 1

            # Read moves until we hit another Pokémon or trainer
            while i < len(lines) and not lines[i].strip().startswith(('Spr', '$', '...')):
                move_name = lines[i].strip()
                if move_name == "":
                    i += 1
                    continue

                move_type = lines[i + 1].strip()
                i += 2

                # Stop parsing moves if another trainer or invalid data is detected
                if move_name.startswith('$') or 'Spr' in move_name:
                    break

                if move_name != '--':
                    moves.append({'move_name': move_name, 'move_type': move_type})

            trainer_moves[current_trainer][unique_pokemon] = {'level': level, 'moves': moves}
            print(f"Debug: {unique_pokemon} (Level {level}) moves: {moves}")  # Debugging
        else:
            i += 1

    return trainer_moves


# Load unique moves from unique_moves_gen_2.csv
def load_unique_moves(unique_moves_file):
    unique_moves = {}
    with open(unique_moves_file, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            move_name = row['move'].strip()
            unique_moves[move_name] = {
                'type': row['type'],
                'power': row['power'],
                'accuracy': row['accuracy'],
                'crit': row['crit']
            }
            print(f"Debug: Loaded move {move_name} from unique_moves_gen_2.csv")  # Debugging
    return unique_moves


def filter_moves_with_zero_power(move_names, move_types, move_powers, move_accuracies, move_crits):
    # Ensure all inputs are valid lists; if None, replace with a list of placeholders
    move_names = move_names if move_names is not None else ['-'] * 4
    move_types = move_types if move_types is not None else ['-'] * 4
    move_powers = move_powers if move_powers is not None else ['-'] * 4
    move_accuracies = move_accuracies if move_accuracies is not None else ['-'] * 4
    move_crits = move_crits if move_crits is not None else ['-'] * 4

    # Filter out moves where power is '0', None, or '-'
    valid_moves = [
        (name, m_type, power, accuracy, crit)
        for name, m_type, power, accuracy, crit in zip(move_names, move_types, move_powers, move_accuracies, move_crits)
        if power not in ('0', None, '-')
    ]

    # Unzip the filtered moves back into separate lists, or use placeholders if no valid moves are left
    if valid_moves:
        move_names, move_types, move_powers, move_accuracies, move_crits = zip(*valid_moves)
        move_names, move_types, move_powers, move_accuracies, move_crits = (
            list(move_names), list(move_types), list(move_powers), list(move_accuracies), list(move_crits)
        )
    else:
        move_names, move_types, move_powers, move_accuracies, move_crits = ['-'] * 4, ['-'] * 4, ['-'] * 4, ['-'] * 4, [
            '-'] * 4

    # Ensure exactly 4 moves
    move_names = (move_names + ['-'] * 4)[:4]
    move_types = (move_types + ['-'] * 4)[:4]
    move_powers = (move_powers + ['-'] * 4)[:4]
    move_accuracies = (move_accuracies + ['-'] * 4)[:4]
    move_crits = (move_crits + ['-'] * 4)[:4]

    return move_names, move_types, move_powers, move_accuracies, move_crits


def parse_pokemon_data(lines, location, trainer_name, stage_mapping, exp_table, stats_mapping, moves_mapping,
                       trainer_moves, unique_moves):
    pokemon_data = []
    pokemon_names = []
    levels = []
    stats = {'Max HP': [], 'Attack': [], 'Defense': [], 'Sp. Atk.': [], 'Sp. Def.': [], 'Speed': []}

    # Counter for each species to distinguish between multiple Pokémon with the same name
    species_counter = {}

    # Lookup stage_enc from the stage_mapping
    stage_enc = stage_mapping.get(location, "-")

    for i, line in enumerate(lines):
        parts = line.split('\t')
        if "Number of Pokemon:" in line:
            if i + 1 < len(lines):
                pokemon_names = [clean_pokemon_name(part.strip()) for part in lines[i + 1].split('\t') if part.strip()]
        elif "Level" in parts[0]:
            levels = [part.strip().split(' ')[1] if 'Level' in part and len(part.strip().split(' ')) > 1 else '-' for
                      part in parts]
        elif any(stat in parts[0] for stat in stats.keys()):
            stat_name = parts[0].strip()
            stats[stat_name] = [part.strip() if part.strip() != '-' else '-' for part in parts[1:]]
        elif "EXP. Yield" in line:
            for idx, name in enumerate(pokemon_names):
                pokemon = name.lower()
                exp_type = stats_mapping.get(pokemon, {}).get('exp_type', '-')
                types = stats_mapping.get(pokemon, {}).get('types', '-')
                level = int(levels[idx + 1]) if idx + 1 < len(levels) and levels[idx + 1].isdigit() else None

                exp_enc = exp_table[level][exp_type] if level and exp_type in exp_table[level] else '-'

                # Default values in case no moves are found
                move_names = ['-', '-', '-', '-']
                move_types = ['-', '-', '-', '-']
                move_powers = ['-', '-', '-', '-']
                move_accuracies = ['-', '-', '-', '-']
                move_crits = ['-', '-', '-', '-']

                # Track species count and create a unique identifier for duplicates
                if pokemon not in species_counter:
                    species_counter[pokemon] = 1
                else:
                    species_counter[pokemon] += 1
                unique_pokemon = f"{pokemon}_{species_counter[pokemon]}"

                # Debugging: Print trainer and Pokémon names before matching
                print(f"Debug: Checking moves for Trainer: '{trainer_name}', Pokémon: '{unique_pokemon}'")
                print(
                    f"Debug: Available moves in trainer_moves for '{trainer_name}': {trainer_moves.get(trainer_name, {})}")

                # Check if the trainer has specific moves for this Pokémon in the alternate moves file
                if trainer_name in trainer_moves and unique_pokemon in trainer_moves[trainer_name]:
                    print(f"Debug: Found moves to replace for {unique_pokemon} under trainer {trainer_name}")
                    moves_data = trainer_moves[trainer_name][unique_pokemon]['moves']
                    move_names = []
                    move_types = []
                    move_powers = []
                    move_accuracies = []
                    move_crits = []
                    for move in moves_data:
                        move_name = move['move_name']
                        move_type = move['move_type']
                        move_info = unique_moves.get(move_name, {})
                        move_power = move_info.get('power', '-') if move_info else '-'
                        move_accuracy = move_info.get('accuracy', '-') if move_info else '-'
                        move_crit = move_info.get('crit', '-') if move_info else '-'
                        move_names.append(move_name)
                        move_types.append(move_type)
                        move_powers.append(move_power)
                        move_accuracies.append(move_accuracy)
                        move_crits.append(move_crit)

                    print(f"Debug: Replacing moves for {unique_pokemon} (Trainer: {trainer_name}): {move_names}")

                    # Ensure exactly 4 moves (filling with '-' if fewer than 4)
                    while len(move_names) < 4:
                        move_names.append('-')
                        move_types.append('-')
                        move_powers.append('-')
                        move_accuracies.append('-')
                        move_crits.append('-')
                else:
                    print(f"Debug: No moves found to replace for {unique_pokemon} under trainer {trainer_name}")

                    # Get the top 4 default moves based on level if no alternate moves are found
                    move_names, move_types, move_powers, move_accuracies, move_crits = get_top_4_moves(pokemon, level,
                                                                                                       moves_mapping)

                # Filter out moves with zero power after all moves are collected
                move_names, move_types, move_powers, move_accuracies, move_crits = filter_moves_with_zero_power(
                    move_names, move_types, move_powers, move_accuracies, move_crits
                )

                # Assemble Pokémon data
                pokemon_info = {
                    'trainer_name_enc': trainer_name,
                    'location_enc': location,
                    'pokemon_enc': name,
                    'level_enc': level if level else '-',
                    'stage_enc': stage_enc,
                    'hp_enc': stats.get('Max HP', [None])[idx] if idx < len(stats.get('Max HP', [])) else '-',
                    'attack_enc': stats.get('Attack', [None])[idx] if idx < len(stats.get('Attack', [])) else '-',
                    'defense_enc': stats.get('Defense', [None])[idx] if idx < len(stats.get('Defense', [])) else '-',
                    'sp_attack_enc': stats.get('Sp. Atk.', [None])[idx] if idx < len(
                        stats.get('Sp. Atk.', [])) else '-',
                    'sp_defense_enc': stats.get('Sp. Def.', [None])[idx] if idx < len(
                        stats.get('Sp. Def.', [])) else '-',
                    'speed_enc': stats.get('Speed', [None])[idx] if idx < len(stats.get('Speed', [])) else '-',
                    'types_enc': types,
                    'exp_type_enc': exp_type,
                    'exp_enc': exp_enc,
                    'move1_name_enc': move_names[0], 'move2_name_enc': move_names[1], 'move3_name_enc': move_names[2],
                    'move4_name_enc': move_names[3],
                    'move1_type_enc': move_types[0], 'move2_type_enc': move_types[1], 'move3_type_enc': move_types[2],
                    'move4_type_enc': move_types[3],
                    'move1_power_enc': move_powers[0], 'move2_power_enc': move_powers[1],
                    'move3_power_enc': move_powers[2], 'move4_power_enc': move_powers[3],
                    'move1_accuracy_enc': move_accuracies[0], 'move2_accuracy_enc': move_accuracies[1],
                    'move3_accuracy_enc': move_accuracies[2], 'move4_accuracy_enc': move_accuracies[3],
                    'move1_crit_enc': move_crits[0], 'move2_crit_enc': move_crits[1], 'move3_crit_enc': move_crits[2],
                    'move4_crit_enc': move_crits[3]
                }
                pokemon_data.append(pokemon_info)
            break

    return pokemon_data


# Process the file and save data to CSV
def process_file(filename, output_csv, stage_mapping_csv, exp_table_file, stats_file, moves_file, alternate_moves_file, unique_moves_file):
    # Load data
    stage_mapping = load_stage_mapping(stage_mapping_csv)
    exp_table = load_exp_table(exp_table_file)
    stats_mapping = load_stats_mapping(stats_file)
    moves_mapping = load_moves_mapping(moves_file)
    trainer_moves = parse_alternate_moves_file(alternate_moves_file)
    unique_moves = load_unique_moves(unique_moves_file)

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    pokemon_entries = []
    current_section = []
    current_location = None
    current_trainer = None

    for i, line in enumerate(lines):
        if 'Battle Rewards:' in line:
            if current_section:
                pokemon_entries.extend(parse_pokemon_data(current_section, current_location, current_trainer, stage_mapping, exp_table, stats_mapping, moves_mapping, trainer_moves, unique_moves))
            current_section = []
            location_line = lines[i - 1].strip()
            current_location = clean_location_name(location_line)
            current_trainer = extract_trainer_name(location_line)
        current_section.append(line.strip())

    if current_section:
        pokemon_entries.extend(parse_pokemon_data(current_section, current_location, current_trainer, stage_mapping, exp_table, stats_mapping, moves_mapping, trainer_moves, unique_moves))

    # Sort and write to CSV
    def convert_to_int(val, default=-1):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    pokemon_entries.sort(key=lambda x: (convert_to_int(x['stage_enc']), convert_to_int(x['exp_enc'])))

    for idx, entry in enumerate(pokemon_entries):
        entry['enc_id'] = idx + 1

    fieldnames = [
        'enc_id', 'trainer_name_enc', 'location_enc', 'pokemon_enc', 'level_enc', 'stage_enc',
        'hp_enc', 'attack_enc', 'defense_enc', 'sp_attack_enc', 'sp_defense_enc', 'speed_enc',
        'types_enc', 'exp_type_enc', 'exp_enc',
        'move1_name_enc', 'move1_type_enc', 'move1_power_enc', 'move1_accuracy_enc', 'move1_crit_enc',
        'move2_name_enc', 'move2_type_enc', 'move2_power_enc', 'move2_accuracy_enc', 'move2_crit_enc',
        'move3_name_enc', 'move3_type_enc', 'move3_power_enc', 'move3_accuracy_enc', 'move3_crit_enc',
        'move4_name_enc', 'move4_type_enc', 'move4_power_enc', 'move4_accuracy_enc', 'move4_crit_enc'
    ]

    if pokemon_entries:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in pokemon_entries:
                for field in fieldnames:
                    if field not in entry or entry[field] is None:
                        entry[field] = '-'
                writer.writerow(entry)
    else:
        print("No Pokémon data was parsed. Please check the input file format.")


# Call the process_file function with the input and output file names and mapping files
process_file(
    'TrainerDataGen2Raw.txt',
    'encounters_gen_2.csv',
    'stages_gen_2.csv',
    'exp_table.csv',
    'stats_gen_2.csv',
    'moves_gen_2.csv',
    'alternate_moves_gen_2.txt',
    'unique_moves_gen_2.csv'
)
