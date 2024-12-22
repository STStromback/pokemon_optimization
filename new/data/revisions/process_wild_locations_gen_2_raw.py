import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

SITEMAP_URL = "https://pokemondb.net/static/sitemaps/pokemondb.xml"
BASE_URL = "https://pokemondb.net"
CRAWL_DELAY = 2  # Respecting the crawl-delay from robots.txt

# Step 1: Parse the sitemap and get all /location/johto URLs
response = requests.get(SITEMAP_URL)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'lxml')
urls = [loc.get_text() for loc in soup.find_all("loc")]

johto_urls = [url for url in urls if "/location/johto" in url]
url_count = len(johto_urls)

all_tables = []
n = 0

for loc_url in johto_urls:
    n = n + 1
    print(f'searching url {n}/{url_count}: {loc_url}')
    time.sleep(CRAWL_DELAY)
    resp = requests.get(loc_url)
    resp.raise_for_status()

    page_soup = BeautifulSoup(resp.text, 'lxml')

    # Find the H2 that contains "Generation 2"
    h2_tags = page_soup.find_all('h2')
    gen2_h2 = None
    for h2 in h2_tags:
        if "Generation 2" in h2.get_text():
            gen2_h2 = h2
            break

    if gen2_h2 is None:
        # If no Gen 2 data, skip this page
        continue

    # Traverse siblings after gen2_h2 to find h3 (methods) and tables
    next_sibling = gen2_h2.find_next_sibling()
    method = None

    while next_sibling:
        # Stop if we hit another h2 (end of Gen 2 section)
        if next_sibling.name == 'h2':
            break

        if next_sibling.name == 'h3':
            # This defines a new method
            method = next_sibling.get_text(strip=True)

        # If we find a div with class resp-scroll, extract tables inside it
        if next_sibling.name == 'div' and 'resp-scroll' in (next_sibling.get('class') or []):
            tables = next_sibling.find_all('table', class_='data-table')
            for tbl in tables:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    # Your code that triggers the warning
                    df = pd.read_html(str(tbl))[0]
                # Add identifying columns
                df['Method'] = method
                df['Location'] = loc_url
                all_tables.append(df)

        next_sibling = next_sibling.find_next_sibling()

# Combine all DataFrames
if not all_tables:
    print("No tables found.")
    exit()

final_df = pd.concat(all_tables, ignore_index=True)

# ---------------------------
# Post-processing steps
# ---------------------------

# DEBUG
# final_df = pd.read_csv('../data_curated/data_curated_gen_2/wild_locations_gen_2.csv')

# 1. Remove rows with null or 'Swarm' in the Pokemon column
#    (Adjust the column name if the table has a different name. Here we assume 'Pokemon'.)
if 'Pokemon' in final_df.columns:
    final_df = final_df.dropna(subset=['Pokemon'])
    final_df = final_df[final_df['Pokemon'] != 'Swarm']

# 2. Replace é with e, ♀ with _f, and ♂ with _m in the Pokemon column
#    If special characters appear in other columns as well, apply these replacements more broadly.
# Rename the column 'Pokémon' to 'Pokemon' if it exists
if 'Pokémon' in final_df.columns:
    final_df.rename(columns={'Pokémon': 'Pokemon'}, inplace=True)

# Replace é with e, ♀ with _f, and ♂ with _m in the Pokemon column
if 'Pokemon' in final_df.columns:
    final_df['Pokemon'] = (
        final_df['Pokemon']
        .str.encode('utf-8')  # Handle potential encoding issues
        .str.decode('utf-8')
        .str.replace('é', 'e', regex=False)
        .str.replace('♀', '_f', regex=False)
        .str.replace('♂', '_m', regex=False)
    )

# 3. For values in the 'Level' column that contain a '-', isolate the number after the dash.
#    Assuming the column is named 'Level'. If it's named differently (e.g., 'Levels'), change accordingly.
# Ensure column name is correct and handle levels with a dash
if 'Levels' in final_df.columns:
    def fix_level(val):
        # Convert to string to handle mixed types
        val = str(val).strip()

        # Check if a dash is present and split by the dash
        if '-' in val:
            return val.split('-')[-1].strip()  # Take the part after the dash
        elif '–' in val:  # Handle en dash
            return val.split('–')[-1].strip()
        elif '—' in val:  # Handle em dash
            return val.split('—')[-1].strip()
        return val  # Return as is if no dash is present


    # Apply the transformation to the Level column
    final_df['Levels'] = final_df['Levels'].apply(fix_level)

    # Convert the Level column to numeric, setting invalid values to NaN
    final_df['Levels'] = pd.to_numeric(final_df['Levels'], errors='coerce').fillna(5)

# 4. Modify Location column
def fix_locations(val):
    if '/location/johto-' in val:
        val = val.split('/location/johto-')[1]
    val = val.replace('-','_')
    return val

if 'Location' in final_df.columns:
    final_df['Location'] = final_df['Location'].apply(fix_locations)
if 'location' in final_df.columns:
    final_df['location'] = final_df['location'].apply(fix_locations)

# Drop the specified columns if they exist
columns_to_drop = ['Games', 'Games.1', 'Games.2', 'Times', 'Rarity']
final_df = final_df.drop(columns=[col for col in columns_to_drop if col in final_df.columns])

# Rename columns to the desired format
column_rename_map = {
    'Pokemon': 'pokemon',
    'Levels': 'wild_level',
    'Details': 'details',
    'Method': 'wild_method',
    'Location': 'wild_location'
}

# Apply the renaming if the columns exist
final_df = final_df.rename(columns={k: v for k, v in column_rename_map.items() if k in final_df.columns})

# Load stages data
stages_df = pd.read_csv('../data_raw/data_raw_gen_2/stages_gen_2.csv')  # columns: location, location_stage, etc.
# Standardize location
stages_df['location'] = (
    stages_df['location']
    .str.lower()
    .str.replace(' ', '_', regex=False)
    .str.replace('\'', '', regex=False) # if apostrophes appear
)

# Join stages onto final_df
# final_df already has wild_location. Ensure wild_location is already standardized.
# If not standardized, do the same transformations:
final_df['wild_location'] = final_df['wild_location'].str.lower().str.replace(' ', '_', regex=False)
# Merge location_stage onto final_df
final_df = final_df.merge(stages_df[['location', 'location_stage']],
                          how='left', left_on='wild_location', right_on='location')

final_df = final_df.drop(columns='location')  # We don't need the duplicate column after merge

# Incorporate Unique Data
unique_df = pd.read_csv('../data_raw/data_raw_gen_2/wild_locations_unique_gen_2_raw.csv')  # columns: pokemon, location, level, method
# Standardize location
unique_df['location'] = (
    unique_df['location']
    .str.lower()
    .str.replace(' ', '_', regex=False)
    .str.replace('\'', '', regex=False)
)

# Rename columns in unique_df to match final_df schema before appending
# final_df columns before rename (assuming final): Pokemon, Levels, Details, Method, Location, location_stage
# unique_df columns: pokemon, location, level, method
# We must rename them to match final_df's pre-final naming:
unique_rename_map = {
    'pokemon': 'Pokemon',
    'location': 'wild_location',
    'level': 'Levels',
    'method': 'Method'
}
unique_df = unique_df.rename(columns=unique_rename_map)

# Add missing columns if needed
if 'Details' not in unique_df.columns:
    unique_df['Details'] = ''

# Merge location_stage for unique_df
unique_df = unique_df.merge(stages_df[['location', 'location_stage']],
                            how='left', left_on='wild_location', right_on='location')
unique_df = unique_df.drop(columns='location')

# Append unique data to final_df
final_df = pd.concat([final_df, unique_df], ignore_index=True)

# Incorporate Trades Data
trades_df = pd.read_csv('../data_raw/data_raw_gen_2/wild_locations_trades_gen_2_raw.csv') # columns: location, pokemon_out, pokemon_in
# Standardize location
trades_df['location'] = (
    trades_df['location']
    .str.lower()
    .str.replace(' ', '_', regex=False)
    .str.replace('\'', '', regex=False)
)

# For each trade row, determine stage:
trade_rows = []
for idx, row in trades_df.iterrows():
    trade_loc = row['location']
    pokemon_out = row['pokemon_out']
    pokemon_in = row['pokemon_in']

    # Find location_stage for this trade location
    loc_stage = stages_df.loc[stages_df['location'] == trade_loc, 'location_stage']
    if loc_stage.empty:
        # If no stage found, assume 0 or handle accordingly
        loc_stage = 0
    else:
        loc_stage = loc_stage.iloc[0]

    # Find earliest stage at which pokemon_in is available
    # Check final_df for pokemon_in entries
    matches = final_df.loc[final_df['Pokemon'].str.lower() == pokemon_in.lower(), 'location_stage']
    if matches.empty:
        # If the pokemon_in not found, assume stage 0 or the location stage
        pokemon_in_stage = loc_stage
    else:
        pokemon_in_stage = matches.min()

    # final stage = max of location_stage and pokemon_in_stage
    final_stage = max(loc_stage, pokemon_in_stage)

    # Construct a row similar to final_df
    # We have: Pokemon, Levels, Details, Method, Location, location_stage
    # For a trade, you might set Method = 'Trade', Levels = 5 (or some default?), Details = '',
    # and Pokemon = pokemon_in since that's what you get
    # wild_location = trade_loc (already standardized)
    trade_row = {
        'Pokemon': pokemon_in,        # You get pokemon_in from the trade
        'Levels': 5,                  # arbitrary default if not provided
        'Details': '',                # no details
        'Method': 'Trade',            # indicates it's a trade
        'wild_location': trade_loc,
        'location_stage': final_stage
    }
    trade_rows.append(trade_row)

# Append trade rows to final_df
if trade_rows:
    trade_df = pd.DataFrame(trade_rows)
    final_df = pd.concat([final_df, trade_df], ignore_index=True)

# Now perform final column renaming
column_rename_map = {
    'Pokemon': 'pokemon',
    'Levels': 'wild_level',
    'Details': 'details',
    'Method': 'wild_method',
    'wild_location': 'wild_location',  # already correct, but here for clarity
    'location_stage': 'wild_location_stage'
}

final_df = final_df.rename(columns=column_rename_map)

stats_df = pd.read_csv('../data_raw/data_raw_gen_2/stats_gen_2.csv')
final_df = pd.merge(final_df, stats_df[['pokemon', 'evo_id']], on='pokemon', how='left')


# Save the final dataframe
final_df.to_csv('../data_curated/data_curated_gen_2/wild_locations_gen_2.csv', index=False)
print("Data saved to wild_locations_gen_2.csv")