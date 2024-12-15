import pandas as pd
import re

config = pd.read_csv('../config/config.csv')
all_starters = True if config[config.rule == 'all_starters'].value.values[0].lower() == 'y' else False
legendaries = True if config[config.rule == 'legendaries'].value.values[0].lower() == 'y' else False
gen = int(config[config.rule == 'gen'].value.values[0])

if all_starters:
    stats = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv')
    stats = stats.groupby('evo_id').first().reset_index()
    df = pd.DataFrame()
    df['pokemon'] = stats.pokemon
    df['wild_level'] = 5
    df['wild_location'] = None
    df['wild_item'] = None
    df['wild_location'] = None
    df['wild_version'] = None
    df['wild_method'] = None
    df['wild_location'] = None
    df['wild_location_stage'] = 0
    df['evo_id'] = stats.evo_id

    if not legendaries:
        df = df[~df.pokemon.isin(['Mewtwo','Mew','Zapdos','Moltres','Articuno'])]

    df = df.sort_values(by=['wild_location_stage', 'wild_level', 'pokemon'])

else:

    # Initialize a list to store the data
    data = []

    # Open and read the text file
    with open(f'../data_raw/data_raw_gen_{gen}/wild_locations_gen_{gen}_raw.txt', 'r') as file:
        lines = file.readlines()

    # Initialize variables to keep track of the current location, rod type, catch method, and versions
    current_location = None
    catch_method = None
    item = None
    version_columns = []

    for line in lines:
        line = line.strip()

        # Detect and set the current location for each encounter section
        location_match = re.match(r"^(.*) encounters$", line)
        if location_match:
            current_location = location_match.group(1)
            catch_method = None  # Reset catch method for new location
            continue

        # Detect fishing method (Rod types) or general catch method
        if line in ["Old Rod", "Good Rod", "Super RodRB", "Super RodY", "Tall grass", "Cave", "Surfing"]:
            catch_method = "Fishing" if "Rod" in line else line  # Set "Fishing" for rods, direct for others
            item = line if "Rod" in line else "None"
            version_columns = []  # Reset version columns for each method section
            continue

        # Identify versions by column presence (Red, Blue, Yellow)
        if re.search(r"\bRed\b.*\bBlue\b.*\bYellow\b", line):
            version_columns = ['r', 'b', 'y']
            continue
        elif re.search(r"\bRed\b.*\bBlue\b", line):
            version_columns = ['r', 'b']
            continue
        elif re.search(r"\bYellow\b", line):
            version_columns = ['y']
            continue

        # Parse Pokémon data lines with format "Pokemon Lv.X" across different columns for each version
        encounter_matches = re.findall(r"([\w'♀♂]+) Lv\.(\d+)", line)
        if encounter_matches:
            # Ensure the number of matches aligns with version columns for this section
            for i, (pokemon, level) in enumerate(encounter_matches):
                if i < len(version_columns):
                    version = version_columns[i]
                    data.append([pokemon, level, current_location, item, version, catch_method])

    # Second File: wild_locations_unique_gen_1_raw.txt
    with open(f'../data_raw/data_raw_gen_{gen}/wild_locations_unique_gen_{gen}_raw.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Variables specific to the second file
    for i in range(len(lines)):
        line = lines[i].strip()

        # Check for lines starting with "Poké"
        if line.startswith("Poké"):
            # Extract the Pokémon name and location from this line
            pokemon_match = re.match(r"Poké.* ([A-Z'♀♂]+)\s+Dex No\.\s+\d+\s+(.+)", line)
            if pokemon_match:
                pokemon = pokemon_match.group(1).capitalize()
                location = pokemon_match.group(2).strip()  # Keep the full location part

                # Check the next line for the level
                if i + 1 < len(lines):
                    level_line = lines[i + 1].strip()
                    level_match = re.search(r"Lv\.(\d+)", level_line)
                    if level_match:
                        level = level_match.group(1)

                        # Handle multiple routes (e.g., "Route 12/Route 16")
                        if "Route" in location:
                            routes = re.findall(r"Route (\d+)", location)
                            if routes:
                                # Create an entry for each route individually
                                for route in routes:
                                    formatted_location = f"Route {route}"
                                    data.append([pokemon, level, formatted_location, "None", "r", "event"])
                            else:
                                # Use the full location if no specific route numbers are found
                                data.append([pokemon, level, location, "None", "r", "event"])
                        else:
                            # Split the location by tab and take the first item
                            formatted_location = location.split('\t')[0]
                            data.append([pokemon, level, formatted_location, "None", "r", "event"])

    # Function to clean location names
    def clean_location(location):
        # Remove parenthesis and everything between them, including the space before
        location = re.sub(r'\s*\(.*?\)', '', location)
        # Replace specific location names
        location = location.replace('Safari Zone', 'safari_zone')
        location = location.replace('Seafoam Islands', 'seafoam_islands')
        location = location.replace('PokÃ©mon Tower', 'pokemon_tower')
        location = location.replace('PokÃ©mon Mansion', 'pokemon_mansion')
        location = location.replace('Indigo Plateau', 'indigo_plateau')
        location = location.replace('Vermilion Harbor', 'vermilion_city')
        location = location.replace('Sea Route', 'route')
        location = location.replace("Diglett's Cave", 'diglet_cave')
        # Convert any location containing "Safari Zone" to "safari_zone"
        if 'safari_zone' in location:
            location = 'safari_zone'
        # Convert to lower case, replace spaces with underscores, and remove punctuation
        location = re.sub(r'[^\w\s]', '', location).lower().replace(' ', '_')
        return location

    # Function to update wild_location_stage based on item
    def update_wild_location_stage(row):
        if row['item'] == 'Old Rod':
            return max(row['wild_location_stage'], 12)
        elif row['item'] in ['Good Rod', 'Super Rod']:
            return max(row['wild_location_stage'], 25)
        return row['wild_location_stage']

    # Convert to DataFrame, filter for only red version, and save to CSV
    df = pd.DataFrame(data, columns=['pokemon', 'level', 'wild_location', 'item', 'version', 'method'])
    df = df[df['version'] == 'r']  # Filter for red version only
    df['wild_location'] = df['wild_location'].apply(clean_location)  # Clean location names
    df['item'] = df['item'].replace('Super RodRB', 'Super Rod')

    # Load stages_gen_1.csv and rename columns
    stages_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/stages_gen_{gen}.csv', usecols=['location', 'location_stage'])
    stages_df.rename(columns={'location': 'wild_location', 'location_stage': 'wild_location_stage'}, inplace=True)

    # Merge the dataframes on wild_location
    df = df.merge(stages_df, on='wild_location', how='left')

    # Update wild_location_stage based on item
    df['wild_location_stage'] = df.apply(update_wild_location_stage, axis=1)

    df.drop_duplicates(inplace=True)
    df['wild_location_stage'] = df['wild_location_stage'].astype(int)
    df['level'] = df['level'].astype(int)  # Ensure level is an integer

    # Keep only the highest level for each pokemon per wild_location_stage
    df = df.loc[df.groupby(['pokemon', 'wild_location_stage'])['level'].idxmax()]

    # Sort the DataFrame by pokemon, wild_location_stage, and level
    df = df.sort_values(by=['pokemon', 'wild_location_stage', 'level'])

    # Process the trades data
    # Step 1: Read the trades data
    trades_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/wild_locations_trades_gen_{gen}_raw.csv')

    # Clean the location names in trades_df to match stages_df
    trades_df['wild_location'] = trades_df['location'].apply(clean_location)

    # Step 1: Determine the stage of availability for the pokemon_out by joining stages on location
    trades_df = trades_df.merge(stages_df[['wild_location', 'wild_location_stage']], on='wild_location', how='left')
    trades_df.rename(columns={'wild_location_stage': 'trade_location_stage'}, inplace=True)

    # Get the earliest wild_location_stage for pokemon_out from df
    wild_pokemon_stages = df.groupby('pokemon')['wild_location_stage'].min().reset_index()
    wild_pokemon_stages.rename(columns={'wild_location_stage': 'pokemon_stage'}, inplace=True)

    trades_df = trades_df.merge(wild_pokemon_stages, left_on='pokemon_out', right_on='pokemon', how='left')
    trades_df.rename(columns={'pokemon_stage': 'pokemon_out_stage'}, inplace=True)
    trades_df.drop(columns=['pokemon'], inplace=True)

    # Step 2: Determine the stage of availability for the pokemon_in using the pokemon_in name and wild locations
    # Since some pokemon_in may not be in df, we need to handle them carefully
    # For pokemon_in not in df, we will assume they are only available via trade, so their initial availability stage is infinite
    # We'll set their initial availability stage to a large number (e.g., 99)
    trades_df = trades_df.merge(wild_pokemon_stages, left_on='pokemon_in', right_on='pokemon', how='left')
    trades_df.rename(columns={'pokemon_stage': 'pokemon_in_stage'}, inplace=True)
    trades_df.drop(columns=['pokemon'], inplace=True)

    # Set pokemon_in_stage to 99 if NaN (i.e., not available in the wild)
    trades_df['pokemon_in_stage'].fillna(99, inplace=True)

    # Step 3: Take the max of the availability stages between pokemon_out and the trade location
    trades_df['trade_availability_stage'] = trades_df[['trade_location_stage', 'pokemon_out_stage']].max(axis=1)

    # Step 4: Adjust the availability stage for pokemon_in
    # The adjusted availability stage for pokemon_in is the minimum between their current stage and the trade_availability_stage
    trades_df['adjusted_availability_stage'] = trades_df[['pokemon_in_stage', 'trade_availability_stage']].min(axis=1)

    # Now, for each pokemon_in, we need to update df or add new entries if they don't exist
    # Create a DataFrame of adjusted pokemon_in stages
    adjusted_pokemon_in = trades_df[['pokemon_in', 'adjusted_availability_stage']].drop_duplicates()
    adjusted_pokemon_in.rename(columns={'pokemon_in': 'pokemon', 'adjusted_availability_stage': 'wild_location_stage'}, inplace=True)

    # For pokemon_in not in df, create new entries
    pokemon_in_not_in_df = adjusted_pokemon_in[~adjusted_pokemon_in['pokemon'].isin(df['pokemon'])]

    # For these pokemon, create new entries in df
    if not pokemon_in_not_in_df.empty:
        # We'll create entries with default values
        new_entries = pd.DataFrame({
            'pokemon': pokemon_in_not_in_df['pokemon'],
            'level': 5,  # Level is unknown from trade data; set to 0 or any default
            'wild_location': 'trade',
            'item': 'None',
            'version': 'r',
            'method': 'trade',
            'wild_location_stage': pokemon_in_not_in_df['wild_location_stage']
        })
        df = pd.concat([df, new_entries], ignore_index=True)

    # For pokemon_in already in df, update their wild_location_stage if the adjusted stage is lower
    df = df.merge(adjusted_pokemon_in, on='pokemon', how='left', suffixes=('', '_adjusted'))

    # Update wild_location_stage where adjusted stage is lower
    # df['wild_location_stage'] = df.apply(
    #     lambda row: min(row['wild_location_stage'], row['wild_location_stage_adjusted']) if pd.notnull(row['wild_location_stage_adjusted']) else row['wild_location_stage'],
    #     axis=1
    # )

    # Drop the temporary columns
    df.drop(columns=['wild_location_stage_adjusted'], inplace=True)

    # Remove duplicates and keep highest level per pokemon and wild_location_stage
    # df.drop_duplicates(subset=['pokemon', 'wild_location_stage'], keep='last', inplace=True)

    # Ensure data types are correct
    df['wild_location_stage'] = df['wild_location_stage'].astype(int)
    df['level'] = df['level'].astype(int)

    # Load stats_gen_1.csv for evo_id information
    stats_df = pd.read_csv(f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv')

    # Merge evo_id into the main dataframe
    df = df.merge(stats_df[['evo_id', 'pokemon']], on='pokemon', how='left')

    # Sort the DataFrame
    df['wild_location_stage'] = df['wild_location_stage'].astype(int)
    # Sort by evo_id, wild_location_stage, and level for proper processing
    df = df.sort_values(by=['evo_id', 'wild_location_stage', 'level'])

    # Initialize a list to store the filtered rows
    filtered_rows = []

    # Process each evo_id group
    # for evo_id, group in df.groupby('evo_id'):
    #     previous_stage = None
    #     previous_level = None
    #
    #     for wild_location_stage, stage_group in group.groupby('wild_location_stage'):
    #         # Retain only the Pokémon with the highest level for the same wild_location_stage
    #         highest_level_row = stage_group.loc[stage_group['level'].idxmax()]
    #         if previous_stage is None or wild_location_stage > previous_stage:
    #             filtered_rows.append(highest_level_row)
    #             previous_stage = wild_location_stage
    #             previous_level = highest_level_row['level']
    #         elif wild_location_stage == previous_stage and highest_level_row['level'] > previous_level:
    #             # Replace the previous row if a higher-level evolution is found at the same stage
    #             filtered_rows[-1] = highest_level_row
    #             previous_level = highest_level_row['level']
    #
    # # Convert the filtered rows back into a DataFrame
    # df = pd.DataFrame(filtered_rows)

    # Sort the final DataFrame for clean output
    df = df.sort_values(by=['wild_location_stage', 'level', 'pokemon'])

df.rename({'level':'wild_level', 'item':'wild_item', 'version':'wild_version', 'method':'wild_method'}, axis=1, inplace=True)

df.to_csv(f'../data_curated/data_curated_gen_{gen}/wild_locations_gen_{gen}.csv', index=False)
