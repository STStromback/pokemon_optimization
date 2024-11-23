import pandas as pd
import re

# Initialize a list to store the data
data = []

# Open and read the text file
with open('wild_locations_gen_1_raw.txt', 'r') as file:
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

# Second File: /mnt/data/wild_locations_unique_gen_1_raw.txt
with open('wild_locations_unique_gen_1_raw.txt', 'r', encoding='utf-8') as file:
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
stages_df = pd.read_csv('stages_gen_1.csv', usecols=['location', 'location_stage'])
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

# Filter for relevant rows
relevant_rows = []
for pokemon, group in df.groupby('pokemon'):
    previous_stage = None
    previous_level = None
    for _, row in group.iterrows():
        if previous_stage is None or (row['wild_location_stage'] > previous_stage and row['level'] > previous_level):
            relevant_rows.append(row)
            previous_stage = row['wild_location_stage']
            previous_level = row['level']

df = pd.DataFrame(relevant_rows)

df.drop_duplicates(inplace=True)
df = df.sort_values(by=['wild_location_stage', 'level', 'pokemon'])
df.to_csv('wild_locations_gen_1.csv', index=False)