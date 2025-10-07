import pandas as pd
from pathlib import Path

def generate_player_pokemon_x_encounters(gen: int, player_pokemon_df: pd.DataFrame, encounters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cartesian product of player_pokemon and encounters for a specific generation.
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        player_pokemon_df (pd.DataFrame): player_pokemon dataframe from generate_player_pokemon()
        encounters_df (pd.DataFrame): Encounters dataframe from generate_encounters()
        
    Returns:
        pd.DataFrame: Cartesian product of player_pokemon and encounters
    """
    out = player_pokemon_df.merge(encounters_df, how="cross")  # Cartesian product
    return out

# Optional: Test the function
if __name__ == "__main__":
    for g in (1, 2, 3):
        print(f'Calculating cartesian product of player_pokemon and encounters for gen {g}')
        # Load from CSV files for testing - look in intermediate_files directory
        intermediate_files_dir = Path(__file__).parent.parent / "intermediate_files"
        
        # Look for input files in intermediate_files directory (where they are saved when run manually)
        player_pokemon_path = intermediate_files_dir / f"player_pokemon_gen{g}.csv"
        encounters_path = intermediate_files_dir / f"encounters_gen{g}.csv"
        
        try:
            player_pokemon = pd.read_csv(player_pokemon_path)
            encounters = pd.read_csv(encounters_path)
            df_combo = generate_player_pokemon_x_encounters(g, player_pokemon, encounters)
            
            # Save to intermediate_files directory when run manually for debugging
            output_path = intermediate_files_dir / f"player_pokemon_encounter_pairs_gen{g}.csv"
            df_combo.to_csv(output_path, index=False)
            print(f"\n*** DEBUG MODE: Gen {g} player_pokemon-encounter pairs saved to {output_path} ***")
            
        except FileNotFoundError as e:
            print(f"Error: Missing input file for gen {g}")
            print(f"Expected player_pokemon file: {player_pokemon_path}")
            print(f"Expected encounters file: {encounters_path}")
            print("Please run generate_player_pokemon.py and generate_encounters.py first to create the required input files.")
            break