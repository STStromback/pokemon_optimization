"""Main execution script for the Pokemon optimization pipeline.

Runs the stages sequentially, passing DataFrames in memory:

    availability -> encounters -> player Pokemon -> pairing
    -> damage/stat calculation -> battle simulation -> MILP optimization

Usage:
    python run/main.py [--gen 1 2 | --gen all] [--config PATH] [--debug] [--initialize]
"""

import argparse
import sys
from pathlib import Path

# Bootstrap: make the source/ package root importable (single place, no per-module hacks).
SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# Windows consoles default to cp1252; force UTF-8 so progress/output symbols are safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from common import paths
from common.config import get_generations, load_config
from pipeline.generate_encounters import generate_encounters
from pipeline.generate_player_pokemon import generate_player_pokemon
from pipeline.pair_player_pokemon_encounters import generate_player_pokemon_x_encounters
from pipeline.calculate_player_pokemon import calculate_player_pokemons
from pipeline.calculate_availability import PokemonAvailabilityCalculator
from pipeline.simulate_battles import simulate_battles
from optimize.milp import calculate_best_party_milp
from scrapers import scrape_availability, scrape_learnsets_pokeapi

MILP_TIME_LIMIT = 300


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pokemon party optimization pipeline."
    )
    parser.add_argument(
        "--gen", nargs="+", default=None,
        help="Generations to process (e.g. --gen 1 2) or 'all'. Defaults to config.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a config JSON file (defaults to config/config.json).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save intermediate DataFrames to intermediate_files/ for inspection.",
    )
    parser.add_argument(
        "--initialize", action="store_true",
        help="Scrape fresh source data before running (overrides config).",
    )
    return parser.parse_args(argv)


def resolve_generations(args: argparse.Namespace, config: dict) -> list:
    """Determine which generations to process from CLI args, falling back to config."""
    if args.gen:
        if len(args.gen) == 1 and args.gen[0].lower() == "all":
            return get_generations({"gen": "all"})
        return get_generations({"gen": [int(g) for g in args.gen]})
    return get_generations(config)


def initialize_repository() -> None:
    """Scrape fresh learnset and availability data from external sources."""
    print("\n=== Repository Initialization: Scraping Data ===\n")

    print("Scraping Pokemon learnsets from PokeAPI...")
    try:
        scrape_learnsets_pokeapi.main()
        print("Learnset scraping complete!\n")
    except Exception as e:
        print(f"Error during learnset scraping: {e}\nContinuing with pipeline...\n")

    print("Scraping Pokemon availability from PokemonDB...")
    try:
        scrape_availability.main()
        print("Availability scraping complete!\n")
    except Exception as e:
        print(f"Error during availability scraping: {e}\nContinuing with pipeline...\n")


def calculate_availability() -> bool:
    """Process raw availability data into ``data/gen_all/pokemon_availability.csv``.

    Returns True on success, False if the pipeline should abort.
    """
    print("=== Step 0: Calculating Pokemon Availability ===")
    calculator = PokemonAvailabilityCalculator()
    input_path = paths.GEN_ALL_DIR / "pokemon_availability_raw.csv"
    output_path = paths.GEN_ALL_DIR / "pokemon_availability.csv"
    try:
        print(f"Loading raw availability data from {input_path}")
        raw_data = calculator.load_raw_data(str(input_path))
        print("Processing availability data with configuration rules...")
        processed_data = calculator.process_availability_data(raw_data)
        print(f"Saving processed availability data to {output_path}")
        calculator.save_processed_data(processed_data, str(output_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the availability scraper first (use --initialize) to generate the raw data.")
        return False
    except Exception as e:
        print(f"Unexpected error in availability calculation: {e}")
        return False
    print("Pokemon availability calculation complete!\n")
    return True


def process_generation(gen: int, config: dict, config_path: str, debug: bool) -> None:
    """Run the full optimization pipeline for a single generation."""
    print(f"\n=== Processing Generation {gen} ===")

    print(f"Step 1: Generating encounters for gen {gen}...")
    encounters_df = generate_encounters(gen)
    print(f"Generated {len(encounters_df)} encounters")

    print(f"Step 2: Generating player_pokemon for gen {gen}...")
    player_pokemon_df = generate_player_pokemon(config_path)
    player_pokemon_df = player_pokemon_df[player_pokemon_df["generation"] == gen]
    print(f"Generated {len(player_pokemon_df)} player_pokemon")

    print(f"Step 3: Creating player_pokemon-encounter pairs for gen {gen}...")
    pairs_df = generate_player_pokemon_x_encounters(gen, player_pokemon_df, encounters_df)
    print(f"Generated {len(pairs_df)} player_pokemon-encounter pairs")

    print(f"Step 4: Calculating player_pokemon for gen {gen}...")
    final_df = calculate_player_pokemons(gen, pairs_df, config=config)
    print(f"Final dataset has {len(final_df)} records")
    if debug:
        debug_path = paths.intermediate_dir() / f"player_pokemon_gen{gen}.csv"
        final_df.to_csv(debug_path, index=False)
        print(f"[debug] Saved {debug_path}")

    print(f"Step 5: Simulating battles for gen {gen}...")
    battle_results = simulate_battles(gen, final_df)
    print(f"Battle simulation complete! Results shape: {battle_results.shape}")
    if debug:
        debug_path = paths.intermediate_dir() / f"battle_results_gen{gen}.csv"
        battle_results.to_csv(debug_path, index=True)
        print(f"[debug] Saved {debug_path}")

    print(f"Step 6: Calculating optimal party for gen {gen} using MILP optimization...")
    try:
        results = calculate_best_party_milp(
            battle_results_df=battle_results,
            config_path=config_path,
            player_pokemon_df=final_df,
            time_limit=MILP_TIME_LIMIT,
            apply_dominance_filter=True,
            verbose=True,
        )
        print(f"\nOptimization Results for Generation {gen}:")
        print(f"Best Fitness Score: {results['best_fitness']:.4f}")
        print(f"Best Party (evo_id_pp): {results['best_evo_id_pp']}")
        print(f"Solver Status: {results.get('solver_status', 'N/A')}")
        print(f"Original Search Space: {results.get('original_search_space', 0):,}")
        print(f"Method: {results['method']}")
    except Exception as e:
        print(f"Error during party optimization: {e}\nContinuing with pipeline...")

    print(f"Generation {gen} processing complete!")


def main(argv=None) -> None:
    """Execute the complete Pokemon optimization pipeline."""
    args = parse_args(argv)
    config_path = args.config if args.config else str(paths.CONFIG_FILE)
    config = load_config(config_path)
    generations = resolve_generations(args, config)
    print(f"Processing generations: {generations}")

    if args.initialize or str(config.get("initialize_repo", "n")).lower() == "y":
        initialize_repository()
    else:
        print("Skipping repository initialization (use --initialize to scrape fresh data)\n")

    if not calculate_availability():
        return

    for gen in generations:
        process_generation(gen, config, config_path, args.debug)


if __name__ == "__main__":
    main()
