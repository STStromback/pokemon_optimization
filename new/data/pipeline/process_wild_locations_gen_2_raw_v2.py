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

# DEBUG
# johto_urls = [johto_urls[0]]

all_tables = []
n = 0

for loc_url in johto_urls:
    n += 1
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
                    # Parse the table into a DataFrame
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

# Post-processing steps
if 'Pokemon' in final_df.columns:
    final_df = final_df.dropna(subset=['Pokemon'])
    final_df = final_df[final_df['Pokemon'] != 'Swarm']

if 'Pokémon' in final_df.columns:
    final_df.rename(columns={'Pokémon': 'Pokemon'}, inplace=True)

if 'Pokemon' in final_df.columns:
    final_df['Pokemon'] = (
        final_df['Pokemon']
        .str.replace('é', 'e', regex=False)
        .str.replace('♀', '_f', regex=False)
        .str.replace('♂', '_m', regex=False)
    )

if 'Levels' in final_df.columns:
    def fix_level(val):
        val = str(val).strip()
        if '-' in val:
            return val.split('-')[-1].strip()
        return val

    final_df['Levels'] = final_df['Levels'].apply(fix_level)
    final_df['Levels'] = pd.to_numeric(final_df['Levels'], errors='coerce').fillna(5)

def fix_locations(val):
    if '/location/johto-' in val:
        val = val.split('/location/johto-')[1]
    return val.replace('-', '_')

if 'Location' in final_df.columns:
    final_df['Location'] = final_df['Location'].apply(fix_locations)

columns_to_drop = ['Games', 'Games.1', 'Games.2', 'Times', 'Rarity']
final_df = final_df.drop(columns=[col for col in columns_to_drop if col in final_df.columns])

column_rename_map = {
    'Pokemon': 'pokemon',
    'Levels': 'wild_level',
    'Details': 'details',
    'Method': 'wild_method',
    'Location': 'wild_location'
}
final_df = final_df.rename(columns={k: v for k, v in column_rename_map.items() if k in final_df.columns})

# Load stages data
stages_df = pd.read_csv('../data_raw/data_raw_gen_2/stages_gen_2.csv')
stages_df['location'] = stages_df['location'].str.lower().str.replace(' ', '_', regex=False)

final_df['wild_location'] = final_df['wild_location'].str.lower()
final_df = final_df.merge(stages_df[['location', 'location_stage']],
                          how='left', left_on='wild_location', right_on='location')
final_df.drop(columns='location', inplace=True)

# Incorporate Unique Data
unique_df = pd.read_csv('../data_raw/data_raw_gen_2/wild_locations_unique_gen_2_raw.csv')
unique_df['location'] = unique_df['location'].str.lower().str.replace(' ', '_', regex=False)
# unique_df = unique_df.rename(columns={
#     'pokemon': 'Pokemon',
#     'location': 'wild_location',
#     'level': 'Levels',
#     'method': 'Method'
# })
unique_df = unique_df.rename(columns={
    'level': 'wild_level',
    'method': 'wild_method'
})
unique_df['details'] = ''
unique_df = unique_df.merge(stages_df[['location', 'location_stage']],
                            how='left', on='location')
unique_df.drop(columns='location', inplace=True)
# unique_df = unique_df[['pokemon', 'level', 'details', 'method', 'location_stage']]
unique_df = unique_df[['pokemon', 'wild_level', 'details', 'wild_method', 'location_stage']]
final_df = pd.concat([final_df, unique_df], ignore_index=True)

# Incorporate Trades Data
trades_df = pd.read_csv('../data_raw/data_raw_gen_2/wild_locations_trades_gen_2_raw.csv')
trades_df['location'] = trades_df['location'].str.lower().str.replace(' ', '_', regex=False)

trade_rows = []
for _, row in trades_df.iterrows():
    trade_loc = row['location']
    loc_stage = stages_df.loc[stages_df['location'] == trade_loc, 'location_stage']
    loc_stage = loc_stage.iloc[0] if not loc_stage.empty else 0

    matches = final_df.loc[final_df['pokemon'].str.lower() == row['pokemon_in'].lower(), 'location_stage']
    pokemon_in_stage = matches.min() if not matches.empty else 0
    final_stage = max(loc_stage, pokemon_in_stage)

    trade_rows.append({
        'Pokemon': row['pokemon_in'],
        'Levels': 5,
        'Details': '',
        'Method': 'Trade',
        'wild_location': trade_loc,
        'location_stage': final_stage
    })

if trade_rows:
    trade_df = pd.DataFrame(trade_rows)
    trade_df = trade_df[['Pokemon', 'Levels', 'Details', 'Method', 'wild_location', 'location_stage']]#.rename(columns={'Pokemon':'pokemon','Levels':'level',''})
    trade_df = trade_df.rename(columns={'Pokemon': 'pokemon', 'Levels':'wild_level', 'Details':'details', 'Method':'wild_method'})
    final_df = pd.concat([final_df, trade_df], ignore_index=True)

# final_df.rename(columns={
#     'Pokemon': 'pokemon',
#     'Levels': 'wild_level',
#     'Details': 'details',
#     'Method': 'wild_method',
#     'wild_location': 'wild_location'
# }, inplace=True)

# Wild method (items) logic
final_df['wild_method'].replace({'Headbutt (Special)':'Heatbutt','Surfing':'Surf'})
temp_stages = stages_df[['key_items','location_stage']].rename(columns={'location_stage':'location_stage_item'})
temp_stages = temp_stages[temp_stages.key_items.notna()].drop_duplicates()
final_df = final_df.merge(temp_stages, left_on='wild_method', right_on='key_items', how='left')
final_df['location_stage_item'] = final_df['location_stage_item'].fillna(0)
final_df['location_stage'] = final_df[['location_stage','location_stage_item']].max()
final_df.drop('location_stage_item', axis=1, inplace=True)

final_df.to_csv('../data_curated/data_curated_gen_2/wild_locations_gen_2.csv', index=False)
print("Data saved to wild_locations_gen_2.csv")
