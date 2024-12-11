import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

df_gen1_stats = pd.read_csv('../gen_1/gen_1_data_raw/stats_gen_1.csv')
df_gen2_stats = pd.read_csv('../gen_2/stats_gen_2.csv')
df_gen3_stats = pd.read_csv('../gen_3/stats_gen_3.csv')

# Define physical move types for Gens 1-3
physical_types = ["Normal", "Fighting", "Flying", "Ground", "Rock", "Bug", "Ghost", "Poison", "Steel"]


def scrape_pokemon_moves(pokemon_name, generation):
    temp_name = pokemon_name
    # if temp_name == 'Farfetch\'d':
    #     temp_name = 'Farfetchd'
    if temp_name == 'Mr_Mime':
        temp_name = 'Mr-Mime'
    if temp_name == 'Nidoran_f':
        temp_name = 'Nidoran-f'
    if temp_name == 'Nidoran_m':
        temp_name = 'Nidoran-m'
    if temp_name == 'Ho_Oh':
        temp_name = 'Ho-Oh'
    if temp_name in ['Deoxys (Normal Forme)', 'Deoxys (Speed Forme)', 'Deoxys (Attack Forme)',
                     'Deoxys (Defense Forme)']:
        temp_name = 'Deoxys'

    url = f'https://pokemondb.net/pokedex/{temp_name}/moves/{generation}'
    # Set a user-agent to mimic a regular web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    tables = soup.find_all('table', {'class': 'data-table'})

    crit_moves = ["Aeroblast", "Air Cutter", "Blaze Kick", "Crabhammer",
                  "Cross Chop", "Karate Chop", "Leaf Blade", "Night Slash",
                  "Poison Tail", "Psycho Cut", "Razor Leaf", "Razor Wind",
                  "Sky Attack", "Slash", "Stone Edge"]

    moves = []
    for table in tables[:5]:
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) == 6 and 'Method' not in str(table.find_all('tr')[0]):
                if 'Lv.' in str(table.find_all('th')[0]):
                    move_lvl = int(cols[0].text.strip())
                else:
                    move_lvl = None
                move_name = cols[1].text.strip()
                move_type = cols[2].text.strip()
                move_power_text = cols[4].text.strip()
                move_accuracy_text = cols[5].text.strip()
            # else:
            #     move_lvl = None
            #     move_name = cols[0].text.strip()
            #     move_type = cols[1].text.strip()
            #     move_power_text = cols[3].text.strip()
            #     move_accuracy_text = cols[4].text.strip()

            if move_name in crit_moves:
                crit = True
            else:
                crit = False

            move_power = 0
            if move_power_text.isnumeric():
                move_power = int(move_power_text)

            move_accuracy = int(move_accuracy_text) if move_accuracy_text.isnumeric() else 0

            if move_power > 0 or move_name == "Return":  # Include only moves with non-zero power
                # if not any(chr.isdigit() for chr in move_type):  # Filter out any error values with integers in the type field
                #     moves.append([pokemon_name, move_name, move_type, move_power, move_accuracy])
                moves.append([pokemon_name, move_lvl, move_name, move_type, move_power, move_accuracy, crit])

    df = pd.DataFrame(moves, columns=['pokemon', 'lvl', 'move', 'type', 'power', 'accuracy', 'crit'])

    def correct_power(row):
        if row > 999:
            sp = str(row)
            splt = int(len(sp) / 2)
            row = int(sp[:splt])

        return row

    df['power'] = df['power'].apply(correct_power)

    def correct_accuracy(row):
        if row == 0:
            row = 100
        return row / 100

    df['accuracy'] = df['accuracy'].apply(correct_accuracy)

    # Remove duplicate moves
    df = df.drop_duplicates(subset=['pokemon', 'move'])

    # Custom modifications to the power of specific moves

    custom_powers = {
        'Hyper Beam': 75,
        'SolarBeam': 0,
        'Razor Wind': 0,
        'Skull Bash': 0,
        'Sky Attack': 0,
        'Return': 102,
        'Selfdestruct': 0,
        'Explosion': 0,
        'Double-Edge': 0,
        'Submission': 0,
        'Take Down': 0,
        'Dream Eater': 0,
        'Focus Punch': 0,
        'Double Kick': 60,
        'Bonemerang': 100,
        'Twinneedle': 50,
        'Arm Thrust': 45,
        'Barrage': 45,
        'Bone Rush': 75,
        'Bullet Seed': 30,
        'Comet Punch': 54,
        'Double Slap': 45,
        'Fury Attack': 45,
        'Fury Swipes': 54,
        'Pin Missile': 42,
        'Rock Blast': 75,
        'Spike Cannon': 60,
        'Triple Kick': 47
    }

    # Accounting for special use cases (hyper beam is effectively 150 with truant, and groudon has sun passive)
    if pokemon_name == 'Slaking':
        custom_powers['Hyper Beam'] = 150
    if pokemon_name == 'Groudon':
        custom_powers['Solar Beam'] = 120
    if pokemon_name == 'Kyogre':
        custom_powers['Thunder'] = 171.42857

    for move, power in custom_powers.items():
        df.loc[df['move'] == move, 'power'] = power

    df = df[df['power'] > 0]

    if df.empty:
        df.loc[-1] = [pokemon_name, 0, 'Struggle', 'Normal', 1, 1, False]

    # if pokemon_name == "Farfetch'd":
    #     pokemon_name = 'Farfetchd'
    if pokemon_name == 'Mr-Mime':
        df['pokemon'] = 'Mr_Mime'
    if pokemon_name == 'Nidoran-f':
        df['pokemon'] = 'Nidoran_f'
    if pokemon_name == 'Nidoran-m':
        df['pokemon'] = 'Nidoran_m'
    if pokemon_name == 'Ho-Oh':
        df['pokemon'] = 'Ho_Oh'

    return df


gen1_names_list = df_gen1_stats['pokemon']
gen2_names_list = df_gen2_stats['pokemon']
gen3_names_list = df_gen3_stats['pokemon']

gen_stats_lists = [df_gen1_stats, df_gen2_stats, df_gen3_stats]

gen_rom_num = ['1', '2', '3']
gen_ints = ['1', '2', '3']

gen_lists = [gen1_names_list, gen2_names_list, gen3_names_list]

gens = zip(gen_lists, gen_stats_lists, gen_rom_num, gen_ints)

# List of HM moves
hm_moves = ['Cut', 'Fly', 'Surf', 'Strength', 'Rock Smash', 'Whirlpool', 'Waterfall', 'Dive']

for gen_list, gen_stats, rom_num, gen_int in gens:
    print(f'\nStart Gen {gen_int}')
    df_f = None

    for name in gen_list:
        time.sleep(4)
        print(name)

        df = scrape_pokemon_moves(name, rom_num)

        if df_f is None:
            df_f = df
        else:
            df_f = pd.concat([df_f, df])

    df_f.to_csv(
        f'moves_gen_{gen_int}_new.csv',
        index=False)
