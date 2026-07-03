"""
Pokemon Player Pokemon Generator

This program generates player pokemon data by combining stats and stages data
from the Pokemon optimization project.
"""
import pandas as pd
from pathlib import Path
from typing import Union

from common import paths
from common.config import load_config
from common.generations import is_legendary
from common.logging_setup import get_logger
from common.text_utils import normalize_text, normalize_text_columns

logger = get_logger(__name__)

_STAT_COLUMN_NAMES_GEN12 = {
    "attack": "attack",
    "defense": "defense",
    "sp_attack": "sp_attack",
    "sp_defense": "sp_defense",
    "speed": "speed",
}

_STAT_COLUMN_NAMES_GEN3 = {
    "attack": "atk",
    "defense": "def",
    "sp_attack": "spatk",
    "sp_defense": "spdef",
    "speed": "spd",
}


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
        config_path = str(paths.CONFIG_FILE)

    if output_dir is None:
        output_dir = str(paths.intermediate_dir())

    # Load configuration
    logger.info("Loading configuration from: %s", config_path)
    config = load_config(config_path)
    
    # Get generation(s) from config
    generations = config.get('gen')
    if isinstance(generations, int):
        generations = [generations]
    elif not isinstance(generations, list):
        raise ValueError(f"Invalid generation format in config: {generations}")
    
    logger.info("Processing generations: %s", generations)
    
    # Combine data from all generations
    combined_df = pd.DataFrame()
    
    for gen in generations:
        logger.info("Processing generation %s...", gen)
        
        # Load stats and stages data
        gen_df = process_generation(gen, config_path)
        combined_df = pd.concat([combined_df, gen_df], ignore_index=True)
    
    # Save combined results
    for gen in generations:
        gen_data = combined_df[combined_df['generation'] == gen].copy()
        if not gen_data.empty:
            # Drop the generation column before saving since it's redundant for individual gen files
            gen_data_to_save = gen_data.drop('generation', axis=1)
            output_path = Path(output_dir) / f'player_pokemon_gen{gen}.csv'
            gen_data_to_save.to_csv(output_path, index=False)
            logger.info("Saved generation %s data to: %s", gen, output_path)
    
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
    data_dir = paths.gen_data_dir(gen)
    
    # Load stats and stages files
    stats_path = data_dir / f'stats_gen_{gen}.csv'
    stages_path = data_dir / f'stages_gen_{gen}.csv'
    
    logger.info("Loading stats from: %s", stats_path)
    logger.info("Loading stages from: %s", stages_path)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not stages_path.exists():
        raise FileNotFoundError(f"Stages file not found: {stages_path}")
    
    stats_df = pd.read_csv(stats_path)
    stages_df = pd.read_csv(stages_path)
    
    # Load config for filtering rules
    config = load_config(config_path)
    
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
    availability_path = paths.GEN_ALL_DIR / 'pokemon_availability.csv'
    
    if availability_path.exists():
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
            normalized_pokemon_name = normalize_text(row['pokemon'])
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
                'legendary_pp': is_legendary(row['pokemon']),
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
        logger.info("Filtering out legendaries...")
        df = df[df['legendary_pp'] != True]
    
    # Rule 2: Trade evolutions are now handled in calculate_player_pokemon.py
    # We no longer filter them out here to maintain all evo_id_pp options
    if config.get('trade_evolutions', 'y').lower() == 'n':
        logger.info("Note: Trade evolution restrictions will be applied during Pokemon calculation phase")

    # Rule 3: Set stage_available_pp to 0 for all rows if config "all_starters" = "y"``
    if config.get('all_starters', 'n').lower() == 'y':
        logger.info("Setting stage_available_pp to 0 for all Pokemon (all starters available)...")
        df['stage_available_pp'] = 0
    
    # Rule 4: Filter out excluded evo_id_pp values if "exclusions" list is provided
    exclusions = config.get('exclusions', [])
    if exclusions and isinstance(exclusions, list):
        logger.info("Filtering out excluded evo_id_pp values: %s", exclusions)
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
    if 'atk' in columns:
        return dict(_STAT_COLUMN_NAMES_GEN3)
    return dict(_STAT_COLUMN_NAMES_GEN12)


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
        logger.info("Starting Pokemon Player Pokemon generation...")
        
        # Generate the dataframe
        result_df = generate_player_pokemon()
        
        logger.info(f"Generated dataframe with {len(result_df)} Pokemon entries")
        logger.debug("First few rows:")
        logger.debug(result_df.head())
        
        logger.debug("Dataframe columns:")
        for col in result_df.columns:
            logger.debug(f"  - {col}")
        
        logger.info("Generation complete!")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()