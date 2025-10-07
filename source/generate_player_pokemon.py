"""
Pokemon Player Pokemon Generator

This program generates player pokemon data by combining stats and stages data
from the Pokemon optimization project.
"""
import pandas as pd
import json
import os
from typing import Union, List

def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    _keep = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
    def _norm(s):
        s = str(s).lower().replace("-", " ").replace(" ", "_")
        # Handle special Nidoran characters
        s = s.replace("♀", "_f").replace("♂", "_m")
        return "".join(ch for ch in s if ch in _keep).replace("__", "_")
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        df[col] = df[col].map(_norm)
    return df

def generate_player_pokemon(config_path: str = None, output_dir: str = None) -> pd.DataFrame:
    """
    Generate player pokemon dataframe from stats and stages data.
    
    Args:
        config_path: Path to config.json file. If None, uses default path.
        output_dir: Directory to save output files. If None, uses intermediate_files.
        
    Returns:
        pandas.DataFrame: Generated player pokemon dataframe
    """
    # Set default paths if not provided
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'config', 'config.json')
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'intermediate_files')
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get generation(s) from config
    generations = config.get('gen')
    if isinstance(generations, int):
        generations = [generations]
    elif not isinstance(generations, list):
        raise ValueError(f"Invalid generation format in config: {generations}")
    
    print(f"Processing generations: {generations}")
    
    # Combine data from all generations
    combined_df = pd.DataFrame()
    
    for gen in generations:
        print(f"Processing generation {gen}...")
        
        # Load stats and stages data
        gen_df = process_generation(gen, config_path)
        combined_df = pd.concat([combined_df, gen_df], ignore_index=True)
    
    # Save combined results
    for gen in generations:
        gen_data = combined_df[combined_df['generation'] == gen].copy()
        if not gen_data.empty:
            # Drop the generation column before saving since it's redundant for individual gen files
            gen_data_to_save = gen_data.drop('generation', axis=1)
            output_path = os.path.join(output_dir, f'player_pokemon_gen{gen}.csv')
            gen_data_to_save.to_csv(output_path, index=False)
            print(f"Saved generation {gen} data to: {output_path}")
    
    return combined_df


def process_generation(gen: int, config_path: str) -> pd.DataFrame:
    """
    Process a single generation's data.
    
    Args:
        gen: Generation number
        config_path: Path to config file
        
    Returns:
        pandas.DataFrame: Processed generation data
    """
    # Determine project root from config path
    project_root = os.path.dirname(os.path.dirname(config_path))
    data_dir = os.path.join(project_root, 'data', f'gen_{gen}')
    
    # Load stats and stages files
    stats_path = os.path.join(data_dir, f'stats_gen_{gen}.csv')
    stages_path = os.path.join(data_dir, f'stages_gen_{gen}.csv')
    
    print(f"Loading stats from: {stats_path}")
    print(f"Loading stages from: {stages_path}")
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not os.path.exists(stages_path):
        raise FileNotFoundError(f"Stages file not found: {stages_path}")
    
    stats_df = pd.read_csv(stats_path)
    stages_df = pd.read_csv(stages_path)
    
    # Load config for filtering rules
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Process the data
    result_df = process_pokemon_data(stats_df, stages_df, gen, config)

    return result_df


def process_pokemon_data(stats_df: pd.DataFrame, stages_df: pd.DataFrame, gen: int, config: dict) -> pd.DataFrame:
    """
    Process pokemon stats and stages data into the required format.
    
    Args:
        stats_df: Stats dataframe
        stages_df: Stages dataframe  
        gen: Generation number
        config: Configuration dictionary with filtering rules
        
    Returns:
        pandas.DataFrame: Processed dataframe
    """
    result_rows = []
    
    # Load pokemon availability data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    availability_path = os.path.join(project_root, 'data', 'gen_all', 'pokemon_availability.csv')
    
    if os.path.exists(availability_path):
        availability_df = pd.read_csv(availability_path)
        # Normalize pokemon names in availability data to match stats data format
        availability_df = normalize_text_columns(availability_df)
    else:
        availability_df = pd.DataFrame()  # Empty dataframe if file doesn't exist
    
    # Define column mappings for different generations
    # Generation 3 uses different column names
    column_mappings = get_column_mappings(stats_df.columns)
    
    # Group by evo_id to calculate evo_index_pp
    for evo_id, group in stats_df.groupby('evo_id'):
        # Sort by evolution level, then by whether it has an evolution item (stone evolutions come last), then by pokemon name
        group['has_evo_item'] = group['evo_item'].fillna('').astype(str) != ''
        group = group.sort_values(['evo_lvl', 'has_evo_item', 'pokemon']).reset_index(drop=True)
        
        for idx, row in group.iterrows():
            # Calculate evo_index_pp (1-based index within evolution line)
            evo_index_pp = idx + 1
            
            # Parse types
            types = row['types'].split('/')
            type_1_pp = types[0].lower() if len(types) > 0 else None
            type_2_pp = types[1].lower() if len(types) > 1 else None
            
            # Calculate evo_item_stage_pp
            evo_item_stage_pp = calculate_evo_item_stage(row.get('evo_item', ''), stages_df)
            
            # Calculate stage_available_pp
            normalized_pokemon_name = row['pokemon'].lower().replace("-", " ").replace(" ", "_")
            stage_available_pp = calculate_stage_available_pp(normalized_pokemon_name, gen, availability_df)
            
            # Create the result row using mapped column names
            result_row = {
                'evo_id_pp': evo_id,
                'evo_index_pp': evo_index_pp,
                'pokemon_pp': normalized_pokemon_name,
                'hp_pp_base': row['hp'],
                'attack_pp_base': row[column_mappings['attack']],
                'defense_pp_base': row[column_mappings['defense']],
                'sp_attack_pp_base': row[column_mappings['sp_attack']],
                'sp_defense_pp_base': row[column_mappings['sp_defense']],
                'speed_pp_base': row[column_mappings['speed']],
                'type_1_pp': type_1_pp,
                'type_2_pp': type_2_pp,
                'evo_lvl_pp': row['evo_lvl'],
                'evo_item_stage_pp': evo_item_stage_pp,
                'stage_available_pp': stage_available_pp,
                'legendary_pp': row['pokemon'] in ['Zapdos', 'Moltres', 'Articuno','Mewtwo','Mew','Celebi','Raikou','Entei','Suicune','Lugia','Ho_Oh','Kyogre','Groudon','Rayquaza','Jirachi','Deoxys','Regice','Regirock','Registeel','Latias','Latios'],
                'generation': gen  # Add generation column for filtering
            }
            
            result_rows.append(result_row)
    
    # Convert to DataFrame for easier manipulation
    result_df = pd.DataFrame(result_rows)

    # For each evo_id, find the minimum stage_available_pp and apply it to all Pokemon in that evolution line
    for evo_id in result_df['evo_id_pp'].unique():
        evo_group = result_df[result_df['evo_id_pp'] == evo_id]
        
        # Find minimum stage_available_pp, ignoring NaN values
        min_stage = evo_group['stage_available_pp'].min()
        
        # If we found a valid minimum (not NaN), apply it to all Pokemon in this evolution line
        if pd.notna(min_stage):
            result_df.loc[result_df['evo_id_pp'] == evo_id, 'stage_available_pp'] = min_stage
    
    # Apply config-based filtering and modifications
    result_df = apply_config_filters(result_df, config)

    # Default missing available stage to 999 and evo item stage to 0
    # Then filter dataframe for Pokemon that are available at any point or have an evo item
    max_stage = result_df['stage_available_pp'].max()
    result_df = result_df[result_df['stage_available_pp'] != 999]
    result_df['stage_available_pp'] = result_df['stage_available_pp'].fillna(999)
    result_df['evo_item_stage_pp'] = result_df['evo_item_stage_pp'].fillna(0)
    result_df = result_df[(result_df['stage_available_pp'] <= max_stage) | (result_df['evo_item_stage_pp'] != 0)]

    return result_df


def apply_config_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply config-based filtering and modifications to the pokemon dataframe.
    
    Args:
        df: Pokemon dataframe
        config: Configuration dictionary with filtering rules
        
    Returns:
        pandas.DataFrame: Filtered and modified dataframe
    """
    df = df.copy()
    
    # Rule 1: Filter out legendaries if config "legendaries" = "n"
    if config.get('legendaries', 'y').lower() == 'n':
        print("Filtering out legendaries...")
        df = df[df['legendary_pp'] != True]
    
    # Rule 2: Trade evolutions are now handled in calculate_player_pokemon.py
    # We no longer filter them out here to maintain all evo_id_pp options
    if config.get('trade_evolutions', 'y').lower() == 'n':
        print("Note: Trade evolution restrictions will be applied during Pokemon calculation phase")

    # Rule 3: Set stage_available_pp to 0 for all rows if config "all_starters" = "y"``
    if config.get('all_starters', 'n').lower() == 'y':
        print("Setting stage_available_pp to 0 for all Pokemon (all starters available)...")
        df['stage_available_pp'] = 0
    
    # Rule 4: Filter out excluded evo_id_pp values if "exclusions" list is provided
    exclusions = config.get('exclusions', [])
    if exclusions and isinstance(exclusions, list):
        print(f"Filtering out excluded evo_id_pp values: {exclusions}")
        df = df[~df['evo_id_pp'].isin(exclusions)]
    
    return df


def get_column_mappings(columns):
    """
    Get column name mappings based on available columns.
    Different generations use different column names.
    
    Args:
        columns: List of column names from the dataframe
        
    Returns:
        dict: Mapping of standardized names to actual column names
    """
    # Default mappings (Gen 1 & 2)
    mappings = {
        'attack': 'attack',
        'defense': 'defense', 
        'sp_attack': 'sp_attack',
        'sp_defense': 'sp_defense',
        'speed': 'speed'
    }
    
    # Gen 3 uses different column names
    if 'atk' in columns:
        mappings.update({
            'attack': 'atk',
            'defense': 'def',
            'sp_attack': 'spatk',
            'sp_defense': 'spdef',
            'speed': 'spd'
        })
    
    return mappings


def calculate_stage_available_pp(pokemon_name: str, gen: int, availability_df: pd.DataFrame) -> Union[int, None]:
    """
    Calculate the minimum stage_available for a pokemon in a specific generation.
    
    Args:
        pokemon_name: Normalized pokemon name
        gen: Generation number
        availability_df: Pokemon availability dataframe
        
    Returns:
        int or None: Minimum stage_available for the pokemon in the generation
    """
    if availability_df.empty:
        return None
    
    # Filter for matching pokemon and generation
    filtered_df = availability_df[
        (availability_df['pokemon'] == pokemon_name) & 
        (availability_df['gen'] == gen)
    ]
    
    if filtered_df.empty:
        return None
    
    # Return the minimum stage_available
    return int(filtered_df['stage_available'].min())

def calculate_evo_item_stage(evo_item: str, stages_df: pd.DataFrame) -> Union[int, None]:
    """
    Calculate the evolution item stage by matching evo_item with stages data.
    
    Args:
        evo_item: Evolution item from stats data
        stages_df: Stages dataframe
        
    Returns:
        int or None: Location stage where the evolution item is available
    """
    if pd.isna(evo_item) or evo_item == '' or evo_item is None:
        return None
    
    evo_item = str(evo_item).strip()
    
    # Search through stages for matching evo_items
    for _, stage_row in stages_df.iterrows():
        stage_evo_items = stage_row.get('evo_items', '')
        
        if pd.isna(stage_evo_items) or stage_evo_items == '':
            continue
        
        # Handle comma-separated evolution items
        stage_items = [item.strip() for item in str(stage_evo_items).split(',')]
        
        if evo_item in stage_items:
            return stage_row['location_stage']
    
    return None


def main():
    """
    Main function to run when script is executed directly.
    """
    try:
        print("Starting Pokemon Player Pokemon generation...")
        
        # Generate the dataframe
        result_df = generate_player_pokemon()
        
        print(f"\nGenerated dataframe with {len(result_df)} Pokemon entries")
        print("\nFirst few rows:")
        print(result_df.head())
        
        print("\nDataframe columns:")
        for col in result_df.columns:
            print(f"  - {col}")
        
        print("\nGeneration complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()