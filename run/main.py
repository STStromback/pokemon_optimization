"""
Main execution script for Pokemon optimization pipeline.
Executes the four main functions sequentially, passing dataframes between them.
"""

import sys
import json
from pathlib import Path

# Add functions and source directories to path for imports
sys.path.append(str(Path(__file__).parent / "functions"))
sys.path.append(str(Path(__file__).parent.parent / "source"))

from generate_encounters import generate_encounters
from generate_player_pokemon import generate_player_pokemon
from generate_pp_x_enc import generate_player_pokemon_x_encounters
from calculate_player_pokemon import calculate_player_pokemons
from calculate_availability import PokemonAvailabilityCalculator
from simulate_battles import simulate_battles
from calculate_best_party_v2 import calculate_best_party_milp
import scrape_learnsets_pokeapi
import scrape_availability

def load_config():
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using defaults.")
        return {"gen": "all"}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using defaults.")
        return {"gen": "all"}

def get_generations_to_process(config_gen):
    """Determine which generations to process based on config."""
    if config_gen == "all":
        return [1, 2, 3]
    elif isinstance(config_gen, list):
        # Handle list input (e.g., [2, 1] from config.json)
        valid_gens = [gen for gen in config_gen if gen in [1, 2, 3]]
        if valid_gens:
            return valid_gens
        else:
            print(f"Warning: No valid generations found in list '{config_gen}'. Processing all generations.")
            return [1, 2, 3]
    elif str(config_gen) in ["1", "2", "3"]:
        return [int(config_gen)]
    else:
        print(f"Warning: Invalid gen value '{config_gen}' in config. Processing all generations.")
        return [1, 2, 3]

def main():
    """Execute the complete Pokemon optimization pipeline."""
    
    # Load configuration
    config = load_config()
    generations = get_generations_to_process(config.get("gen", "all"))
    
    print(f"Processing generations: {generations}")
    
    # Optional Step: Initialize repository data if flag is set
    if config.get("initialize_repo", "n").lower() == "y":
        print("\n=== Repository Initialization: Scraping Data ===\n")
        
        # Step -2: Scrape learnsets from PokeAPI
        print("Step -2: Scraping Pokemon learnsets from PokeAPI...")
        try:
            scrape_learnsets_pokeapi.main()
            print("Learnset scraping complete!\n")
        except Exception as e:
            print(f"Error during learnset scraping: {e}")
            print("Continuing with pipeline...\n")
        
        # Step -1: Scrape availability data from PokemonDB
        print("Step -1: Scraping Pokemon availability from PokemonDB...")
        try:
            scrape_availability.main()
            print("Availability scraping complete!\n")
        except Exception as e:
            print(f"Error during availability scraping: {e}")
            print("Continuing with pipeline...\n")
    else:
        print("Skipping repository initialization (initialize_repo != 'y')\n")
    
    # Step 0: Calculate Pokemon availability (runs once for all generations)
    print("=== Step 0: Calculating Pokemon Availability ===")
    calculator = PokemonAvailabilityCalculator()
    
    try:
        # Set up file paths
        base_path = Path(__file__).parent.parent
        input_path = base_path / 'data' / 'gen_all' / 'pokemon_availability_raw.csv'
        output_path = base_path / 'data' / 'gen_all' / 'pokemon_availability.csv'
        
        # Load raw data
        print(f"Loading raw availability data from {input_path}")
        raw_data = calculator.load_raw_data(str(input_path))
        
        # Process the data
        print("Processing availability data with configuration rules...")
        processed_data = calculator.process_availability_data(raw_data)
        
        # Save processed data
        print(f"Saving processed availability data to {output_path}")
        calculator.save_processed_data(processed_data, str(output_path))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run scrape_availability.py first to generate the raw data")
        return
    except Exception as e:
        print(f"Unexpected error in availability calculation: {e}")
        return
    
    print("Pokemon availability calculation complete!\n")
    
    for gen in generations:
        print(f"\n=== Processing Generation {gen} ===")
        
        # Step 1: Generate encounters
        print(f"Step 1: Generating encounters for gen {gen}...")
        encounters_df = generate_encounters(gen)
        print(f"Generated {len(encounters_df)} encounters")
        
        # Step 2: Generate player_pokemon
        print(f"Step 2: Generating player_pokemon for gen {gen}...")
        config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        player_pokemon_df = generate_player_pokemon(str(config_path))
        # Filter for current generation
        player_pokemon_df = player_pokemon_df[player_pokemon_df['generation'] == gen]
        print(f"Generated {len(player_pokemon_df)} player_pokemon")
        
        # Step 3: Create cartesian product of player_pokemon and encounters
        print(f"Step 3: Creating player_pokemon-encounter pairs for gen {gen}...")
        player_pokemon_encounter_pairs_df = generate_player_pokemon_x_encounters(gen, player_pokemon_df, encounters_df)
        print(f"Generated {len(player_pokemon_encounter_pairs_df)} player_pokemon-encounter pairs")
        
        # Step 4: Calculate player_pokemon
        print(f"Step 4: Calculating player_pokemon for gen {gen}...")
        final_df = calculate_player_pokemons(gen, player_pokemon_encounter_pairs_df, config=config)
        # DEBUG: SAVE CALCULATE_POKEMON OUTPUT TO CSV
        final_df.to_csv(f"intermediate_files/player_pokemon_{gen}_MAIN_DEBUG.csv", index=False)
        print(f"Final dataset has {len(final_df)} records")
        
        # Step 5: Simulate battles
        print(f"Step 5: Simulating battles for gen {gen}...")
        battle_results = simulate_battles(gen, final_df)
        # DEBUG: SAVE BATTLE RESULTS TO CSV
        battle_results.to_csv(f"intermediate_files/battle_results_{gen}_MAIN_DEBUG.csv", index=True)
        print(f"Battle simulation complete! Results shape: {battle_results.shape}")
        
        # Note: No intermediate files are saved - all data passed directly in memory
        
        # Step 6: Calculate best party (MILP optimization)
        print(f"Step 6: Calculating optimal party for gen {gen} using MILP optimization...")
        config_path = Path(__file__).parent.parent / 'config' / 'config.json'
        
        try:
            optimization_results = calculate_best_party_milp(
                battle_results_df=battle_results,
                config_path=str(config_path),
                player_pokemon_df=final_df,
                time_limit=300,
                apply_dominance_filter=True,
                verbose=True
            )
            
            print(f"\nOptimization Results for Generation {gen}:")
            print(f"Best Fitness Score: {optimization_results['best_fitness']:.4f}")
            print(f"Best Party (evo_id_pp): {optimization_results['best_evo_id_pp']}")
            print(f"Solver Status: {optimization_results.get('solver_status', 'N/A')}")
            print(f"Original Search Space: {optimization_results.get('original_search_space', 0):,}")
            print(f"Method: {optimization_results['method']}")
            
            # MILP finds the global optimum directly with mathematical guarantee
            
        except Exception as e:
            print(f"Error during party optimization: {e}")
            print("Continuing with pipeline...")
        
        print(f"Generation {gen} processing complete!")

if __name__ == "__main__":
    main()