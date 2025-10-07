# Pokemon Learnset Scraper using PokeAPI
"""
Pokemon Learnset Scraper using PokeAPI

This script fetches Pokemon learnset data from the PokeAPI (pokeapi.co) instead of 
web scraping Bulbapedia. It generates CSV files in the same format as the original
scraper for compatibility with the existing optimization pipeline.

Format: pokemon,move_name,move_level,tm
- pokemon: Pokemon name
- move_name: Move name  
- move_level: Level learned (0 for TMs/HMs/tutors, actual level for level-up moves)
- tm: TM/HM code (e.g. "TM01", "HM05") or "tutor" for tutor moves, empty for level-up moves
"""

import requests
import pandas as pd
import time
import json
from typing import Dict, List, Set, Optional, Tuple
import re
import os

# Generation to version group mapping for PokeAPI
GENERATION_VERSION_GROUPS = {
    1: ['red-blue', 'yellow'],
    2: ['gold-silver', 'crystal'], 
    3: ['ruby-sapphire', 'emerald', 'firered-leafgreen']
}

# Total Pokemon available in each generation's games
GENERATION_POKEMON_COUNTS = {
    1: 151,  # Gen 1 games: Bulbasaur to Mew (1-151)
    2: 251,  # Gen 2 games: All Gen 1-2 Pokemon (1-251)  
    3: 386   # Gen 3 games: All Gen 1-3 Pokemon (1-386)
}

# Learn methods that should be excluded (event-only, special distributions, etc.)
EXCLUDED_LEARN_METHODS = {
    'stadium-surfing-pikachu',  # Pokemon Stadium special Pikachu
    'colosseum-purification',   # Pokemon Colosseum special
    'xd-shadow',                # Pokemon XD special
    'xd-purification',          # Pokemon XD special
    'form-change',              # Form changes, not actual learning
    'light-ball-egg',           # Special egg move requirements
}

# Standard learn methods that we process normally
# Note: 'egg' moves (breeding moves) are explicitly excluded - only level-up, TM/HM, and tutor moves are valid
STANDARD_LEARN_METHODS = {'level-up', 'machine', 'tutor'}

def load_move_tutor_adjustments() -> pd.DataFrame:
    """Load move tutor adjustments from CSV file."""
    adjustments_path = os.path.join('data', 'gen_all', 'move_tutor_adjustments.csv')
    if os.path.exists(adjustments_path):
        return pd.read_csv(adjustments_path)
    else:
        print(f"Warning: Move tutor adjustments file not found at {adjustments_path}")
        return pd.DataFrame(columns=['gen', 'move_name', 'exclude', 'new_location'])

def check_tutor_move_adjustment(move_name: str, generation: int, tutor_adjustments: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Check if a tutor move should be excluded or have its location adjusted.
    
    Returns:
        Tuple[bool, Optional[str]]: (should_exclude, new_location)
            - should_exclude: True if move should be completely excluded
            - new_location: Location string if move should be treated as TM at specific location, None otherwise
    """
    if tutor_adjustments.empty:
        return False, None
    
    # Normalize move name for comparison (handle different formats)
    move_name_normalized = move_name.replace(' ', '-').replace('_', '-').lower()
    
    # Filter adjustments for this generation and move
    matching_adjustments = tutor_adjustments[
        (tutor_adjustments['gen'] == generation) &
        (tutor_adjustments['move_name'].str.replace(' ', '-').str.replace('_', '-').str.lower() == move_name_normalized)
    ]
    
    if matching_adjustments.empty:
        return False, None
    
    # Get the first matching adjustment
    adjustment = matching_adjustments.iloc[0]
    
    # Check if move should be excluded
    if pd.notna(adjustment['exclude']) and str(adjustment['exclude']).lower() in ['true', '1', 'yes']:
        return True, None
    
    # Check if move has a new location
    if pd.notna(adjustment['new_location']) and adjustment['new_location'].strip():
        return False, adjustment['new_location'].strip()
    
    return False, None

def get_pokemon_list_for_generation(generation: int) -> List[int]:
    """Get the list of Pokemon IDs available in games of a specific generation."""
    if generation == 1:
        return list(range(1, 152))  # 1-151 (Gen 1 only)
    elif generation == 2:
        return list(range(1, 252))  # 1-251 (Gen 1 + Gen 2 Pokemon available in GSC)
    elif generation == 3:
        return list(range(1, 387))  # 1-386 (Gen 1 + Gen 2 + Gen 3 Pokemon available in RSE/FRLG)
    else:
        raise ValueError(f"Generation {generation} not supported")

def fetch_pokemon_data(pokemon_id: int, max_retries: int = 3) -> Optional[Dict]:
    """Fetch Pokemon data from PokeAPI with retry logic."""
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for Pokemon {pokemon_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch data for Pokemon {pokemon_id} after {max_retries} attempts")
                return None

def extract_machine_number(move_learn_method_name: str, version_group_name: str, move_name: str) -> Optional[str]:
    """
    Extract TM/HM number for machine-learned moves.
    
    Note: PokeAPI doesn't directly provide TM numbers in the Pokemon endpoint.
    This is a limitation - we'll need to make additional API calls to get TM numbers
    or use a mapping. For now, we'll mark machine moves as "machine".
    """
    if move_learn_method_name == "machine":
        # TODO: Could make additional API calls to get actual TM/HM numbers
        # For now, we'll use a generic "machine" identifier
        return "machine"
    elif move_learn_method_name == "tutor":
        return "tutor"
    return None

def normalize_move_name(move_name: str) -> str:
    """Normalize move names to match existing format."""
    # Convert from API format (lowercase with hyphens) to display format
    # Examples: "mega-punch" -> "Mega Punch", "solar-beam" -> "SolarBeam"
    
    # Special cases that need specific formatting
    special_cases = {
        "solar-beam": "SolarBeam",
        "bubble-beam": "BubbleBeam", 
        "double-edge": "Double-Edge",
        "thunder-shock": "ThunderShock",
        "thunder-punch": "ThunderPunch",
        "poison-powder": "PoisonPowder",
        "sleep-powder": "Sleep Powder",
        "stun-spore": "Stun Spore",
        "razor-leaf": "Razor Leaf",
        "self-destruct": "Selfdestruct",
        "double-slap": "DoubleSlap",
        "sonic-boom": "SonicBoom",
        "vine-whip": "Vine Whip",
        "take-down": "Take Down",
        "body-slam": "Body Slam",
        "skull-bash": "Skull Bash",
        "mega-punch": "Mega Punch",
        "mega-kick": "Mega Kick",
        "ice-beam": "Ice Beam",
        "fire-blast": "Fire Blast",
        "pin-missile": "Pin Missile",
        "dragon-rage": "Dragon Rage",
        "thunder-wave": "Thunder Wave",
        "tri-attack": "Tri Attack",
        "rock-slide": "Rock Slide",
        "mega-drain": "Mega Drain",
        "confuse-ray": "Confuse Ray",
        "night-shade": "Night Shade",
        "dream-eater": "Dream Eater",
        "horn-drill": "Horn Drill",
        "pay-day": "Pay Day",
        "seismic-toss": "Seismic Toss",
        "sky-attack": "Sky Attack",
        "egg-bomb": "Egg Bomb",
        "soft-boiled": "Softboiled",
        "fire-spin": "Fire Spin",
        "light-screen": "Light Screen",
        "defense-curl": "Defense Curl",
        "double-team": "Double Team",
        "focus-energy": "Focus Energy",
        "fury-attack": "Fury Attack",
        "horn-attack": "Horn Attack",
        "poison-sting": "Poison Sting",
        "rock-throw": "Rock Throw",
        "sand-attack": "Sand-Attack",
        "spike-cannon": "Spike Cannon",
        "tail-whip": "Tail Whip",
        "wing-attack": "Wing Attack",
        "quick-attack": "Quick Attack",
        "water-gun": "Water Gun",
        "drill-peck": "Drill Peck",
        "bone-club": "Bone Club",
        "hyper-beam": "Hyper Beam",
        "petal-dance": "Petal Dance",
        "leech-life": "Leech Life",
        "aurora-beam": "Aurora Beam",
        "mirror-move": "Mirror Move"
    }
    
    if move_name in special_cases:
        return special_cases[move_name]
    
    # Default: Title case with spaces
    return move_name.replace('-', ' ').title()

def normalize_pokemon_name(pokemon_name: str) -> str:
    """Normalize Pokemon names to match existing format and apply final cleanup."""
    # Convert from API format to display format
    # Most Pokemon names just need title case, but some have special cases
    
    special_cases = {
        "farfetchd": "Farfetch'd",
        "mr-mime": "Mr. Mime",
        "nidoran-f": "Nidoran♀",
        "nidoran-m": "Nidoran♂"
    }
    
    if pokemon_name in special_cases:
        name = special_cases[pokemon_name]
    else:
        name = pokemon_name.title()
    
    # Apply normalization steps:
    # 1. Replace - with _
    name = name.replace('-', '_')
    
    # 2. Replace ♀ with _f and ♂ with _m
    name = name.replace('♀', '_f')
    name = name.replace('♂', '_m')
    
    # 3. Remove other special characters (apostrophes, periods, etc.)
    import re
    name = re.sub(r"[^\w\s_]", "", name)
    
    # 4. Replace spaces with _
    name = name.replace(' ', '_')
    
    return name

def extract_moves_for_generation(pokemon_data: Dict, generation: int, tutor_adjustments: pd.DataFrame = None) -> List[Dict]:
    """Extract moves for a specific generation from Pokemon data."""
    moves_data = []
    version_groups = GENERATION_VERSION_GROUPS[generation]
    pokemon_name = normalize_pokemon_name(pokemon_data['name'])
    
    # Load tutor adjustments if not provided
    if tutor_adjustments is None:
        tutor_adjustments = load_move_tutor_adjustments()
    
    for move_entry in pokemon_data['moves']:
        move_name = normalize_move_name(move_entry['move']['name'])
        
        # First pass: Collect all learn methods for this move in this generation
        # and check if any tutor/special method should exclude the entire move
        matching_details = []
        should_skip_move = False
        
        for version_detail in move_entry['version_group_details']:
            version_group = version_detail['version_group']['name']
            
            if version_group in version_groups:
                learn_method = version_detail['move_learn_method']['name']
                
                # Skip excluded event-only learn methods
                if learn_method in EXCLUDED_LEARN_METHODS:
                    continue  # Don't add to matching_details
                
                # Skip egg moves (breeding moves) - they should never be included
                if learn_method == 'egg':
                    continue  # Don't add to matching_details
                
                # Check if this is a tutor or special move that should be excluded via adjustments
                # This includes both "tutor" methods and any non-standard methods
                if learn_method == "tutor" or learn_method not in STANDARD_LEARN_METHODS:
                    should_exclude, new_location = check_tutor_move_adjustment(move_name, generation, tutor_adjustments)
                    if should_exclude:
                        should_skip_move = True
                        break  # No need to check further
                
                matching_details.append(version_detail)
        
        # Skip this move entirely if it should be excluded
        if should_skip_move:
            continue
        
        # If no matching details found, skip this move
        if not matching_details:
            continue
        
        # Second pass: Process the move using the first matching detail
        # (prioritize level-up, then machine, then tutor)
        # Sort by priority: level-up > machine > tutor
        # Note: egg moves are already filtered out above
        learn_method_priority = {'level-up': 0, 'machine': 1, 'tutor': 2}
        matching_details.sort(key=lambda d: learn_method_priority.get(d['move_learn_method']['name'], 99))
        
        version_detail = matching_details[0]
        level_learned = version_detail['level_learned_at']
        learn_method = version_detail['move_learn_method']['name']
        version_group = version_detail['version_group']['name']
        
        # Determine TM/machine identifier
        tm_identifier = extract_machine_number(learn_method, version_group, move_name)
        
        # For level-up moves, tm should be empty (NaN in pandas)
        if learn_method == "level-up":
            tm_value = pd.NA
            move_level = level_learned
        else:
            # Check for tutor move adjustments (for location changes)
            if learn_method == "tutor":
                should_exclude, new_location = check_tutor_move_adjustment(move_name, generation, tutor_adjustments)
                
                if new_location:
                    # Treat as TM at specific location
                    tm_value = new_location
                else:
                    # Regular tutor move
                    tm_value = tm_identifier if tm_identifier else pd.NA
            else:
                # Regular machine or egg move
                tm_value = tm_identifier if tm_identifier else pd.NA
            
            move_level = 0  # Non-level moves use 0
        
        move_data = {
            'pokemon': pokemon_name,
            'move_name': move_name,
            'move_level': move_level,
            'tm': tm_value
        }
        
        moves_data.append(move_data)
    
    return moves_data

def scrape_generation_pokeapi(generation: int, limit: Optional[int] = None) -> pd.DataFrame:
    """Scrape Pokemon learnsets for a specific generation using PokeAPI."""
    print(f"Scraping Generation {generation} using PokeAPI...")
    
    # Load tutor adjustments once
    tutor_adjustments = load_move_tutor_adjustments()
    print(f"Loaded {len(tutor_adjustments)} move tutor adjustments")
    
    pokemon_ids = get_pokemon_list_for_generation(generation)
    if limit:
        pokemon_ids = pokemon_ids[:limit]
    
    all_moves = []
    failed_pokemon = []
    
    for i, pokemon_id in enumerate(pokemon_ids, 1):
        print(f"Processing Pokemon {pokemon_id} ({i}/{len(pokemon_ids)})...")
        
        pokemon_data = fetch_pokemon_data(pokemon_id)
        if pokemon_data is None:
            failed_pokemon.append(pokemon_id)
            continue
        
        try:
            moves = extract_moves_for_generation(pokemon_data, generation, tutor_adjustments)
            all_moves.extend(moves)
            
            # Rate limiting - be respectful to the API
            time.sleep(0.1)  # 100ms delay between requests
            
        except Exception as e:
            print(f"Error processing Pokemon {pokemon_id} ({pokemon_data.get('name', 'unknown')}): {e}")
            failed_pokemon.append(pokemon_id)
    
    if failed_pokemon:
        print(f"Warning: Failed to process {len(failed_pokemon)} Pokemon: {failed_pokemon}")
    
    # Create DataFrame and remove duplicates
    df = pd.DataFrame(all_moves, columns=['pokemon', 'move_name', 'move_level', 'tm'])
    df = df.drop_duplicates()
    
    print(f"Generation {generation} completed: {len(df)} total moves")
    return df

def test_specific_pokemon(generation: int, pokemon_names: List[str]) -> pd.DataFrame:
    """Test the scraper with specific Pokemon."""
    print(f"Testing specific Pokemon for Generation {generation}: {pokemon_names}")
    
    # Load tutor adjustments
    tutor_adjustments = load_move_tutor_adjustments()
    
    all_moves = []
    
    for pokemon_name in pokemon_names:
        # Convert name to lowercase for API
        api_name = pokemon_name.lower().replace("'", "").replace("♀", "-f").replace("♂", "-m")
        
        print(f"\nProcessing {pokemon_name} ({api_name})...")
        
        try:
            # Fetch by name instead of ID for testing
            url = f"https://pokeapi.co/api/v2/pokemon/{api_name}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            pokemon_data = response.json()
            
            moves = extract_moves_for_generation(pokemon_data, generation, tutor_adjustments)
            all_moves.extend(moves)
            
            # Show sample moves
            pokemon_moves = [m for m in moves if m['pokemon'] == pokemon_name]
            levelup_moves = [m for m in pokemon_moves if pd.isna(m['tm'])]
            levelup_moves.sort(key=lambda x: x['move_level'])
            
            print(f"Found {len(levelup_moves)} level-up moves:")
            for move in levelup_moves[:10]:  # Show first 10
                print(f"  Level {move['move_level']:3}: {move['move_name']}")
                
        except Exception as e:
            print(f"Error processing {pokemon_name}: {e}")
    
    return pd.DataFrame(all_moves, columns=['pokemon', 'move_name', 'move_level', 'tm'])

if __name__ == "__main__":
    # First test Pokemon name normalization
    print("Testing Pokemon name normalization...")
    test_names = ["gengar", "farfetchd", "mr-mime", "nidoran-f", "nidoran-m"]
    for name in test_names:
        normalized = normalize_pokemon_name(name)
        print(f"  {name} -> {normalized}")
    
    # Test with Gengar first to verify the approach
    print("\nTesting with Gengar to verify PokeAPI approach...")
    df_test = test_specific_pokemon(1, ["Gengar"])
    
    # Check for some expected moves
    print("\nVerifying expected moves for Gengar:")
    gengar_moves = df_test[df_test.pokemon == "Gengar"]
    levelup_moves = gengar_moves[gengar_moves.tm.isna()]
    
    expected_moves = ['Lick', 'Confuse Ray', 'Night Shade', 'Hypnosis', 'Dream Eater']
    all_found = True
    for move_name in expected_moves:
        found = levelup_moves[levelup_moves.move_name == move_name]
        if not found.empty:
            level = found.iloc[0]['move_level']
            print(f"  ✓ {move_name} at level {level}")
        else:
            print(f"  ✗ {move_name} - NOT FOUND!")
            all_found = False
    
    if all_found:
        print("\n✓ Test successful! Running full scrape for all generations...")
        
        # Run full scraping for all generations
        for gen in [1, 2, 3]:
            print(f"\n{'='*50}")
            print(f"GENERATION {gen}")
            print(f"{'='*50}")
            
            df_gen = scrape_generation_pokeapi(gen)
            
            # Save to data directory
            output_path = f"data/gen_{gen}/learnsets_gen_{gen}.csv"
            df_gen.to_csv(output_path, index=False)
            print(f"Gen {gen} completed: {len(df_gen)} total moves saved to {output_path}")
        
        print("\n✓ SCRAPING COMPLETE - All learnsets updated successfully using PokeAPI!")
    else:
        print("\n✗ Test failed - not all expected moves were found.")
        print("This might be due to move name formatting differences.")
        print("Please review the normalize_move_name function and test data.")
