"""Main execution script for the Pokemon optimization pipeline.

Runs the stages sequentially, passing DataFrames in memory:

    availability -> encounters -> player Pokemon -> pairing
    -> damage/stat calculation -> battle simulation -> MILP optimization

Usage:
    python run/main.py [--gen 1 2 | --gen all] [--config PATH] [--debug] [--initialize]
"""

import argparse
import math
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
from common.config import get_generations, get_party_size_range, load_config
from pipeline.generate_encounters import generate_encounters
from pipeline.generate_player_pokemon import generate_player_pokemon
from pipeline.pair_player_pokemon_encounters import generate_player_pokemon_x_encounters
from pipeline.calculate_player_pokemon import calculate_player_pokemons
from pipeline.calculate_availability import PokemonAvailabilityCalculator
from pipeline.simulate_battles import simulate_battles
from optimize.milp import calculate_best_party_milp, save_results_to_file
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

    if 'party_size' not in final_df.columns:
        print("Warning: 'party_size' column missing from final_df; defaulting entire dataset to party_size=6")
        final_df = final_df.copy()
        final_df['party_size'] = 6

    party_sizes = get_party_size_range(config)
    print(f"Step 5/6: Evaluating party sizes {party_sizes} for gen {gen}...")

    party_size_outcomes = []  # list of dicts: {party_size, fitness, results}
    for ps in party_sizes:
        print(f"\n--- Generation {gen} / Party Size {ps} ---")
        ps_df = final_df[final_df['party_size'] == ps]
        if ps_df.empty:
            print(f"No rows for party_size {ps}; skipping.")
            continue

        print(f"Step 5 (ps={ps}): Simulating battles for gen {gen}...")
        battle_results = simulate_battles(gen, ps_df, config=config, party_size=ps)
        print(f"Battle simulation complete! Results shape: {battle_results.shape}")
        if debug:
            debug_path = paths.intermediate_dir() / f"battle_results_gen{gen}_ps{ps}.csv"
            battle_results.to_csv(debug_path, index=True)
            print(f"[debug] Saved {debug_path}")

        print(f"Step 6 (ps={ps}): Calculating optimal party of {ps} for gen {gen} using MILP optimization...")
        try:
            results = calculate_best_party_milp(
                battle_results_df=battle_results,
                config_path=config_path,
                player_pokemon_df=ps_df,
                time_limit=MILP_TIME_LIMIT,
                party_size=ps,
                skip_prefilter=True,
                save_results=False,
                verbose=True,
            )
            print(f"\nOptimization Results for Generation {gen}, Party Size {ps}:")
            print(f"Best Fitness Score: {results['best_fitness']:.4f}")
            print(f"Best Party (evo_id_pp): {results['best_evo_id_pp']}")
            print(f"Solver Status: {results.get('solver_status', 'N/A')}")
            print(f"Original Search Space: {results.get('original_search_space', 0):,}")
            print(f"Method: {results['method']}")
            party_size_outcomes.append({'party_size': ps, 'fitness': results['best_fitness'], 'results': results, 'player_pokemon_df': ps_df})
        except Exception as e:
            print(f"Error during party optimization for party_size {ps}: {e}\nContinuing with other party sizes...")
            party_size_outcomes.append({'party_size': ps, 'fitness': float('inf'), 'results': None, 'player_pokemon_df': None})

    winner = _summarize_party_size_outcomes(gen, party_size_outcomes)

    if winner is not None:
        ps = winner['party_size']
        print(f"\nSaving full results (report, visualizations, encounter CSVs) for winning party size {ps}...")
        save_results_to_file(
            winner['results'],
            config_path,
            winner['results']['battle_df'],
            winner['player_pokemon_df'],
            party_size=ps,
        )
    else:
        print(f"\nNo party size produced an optimal solution for generation {gen}; nothing to save.")

    print(f"Generation {gen} processing complete!")


def _summarize_party_size_outcomes(gen: int, outcomes: list) -> dict:
    """Print/save a summary comparing all evaluated party sizes for this generation, write a
    separate report listing every party size that tied for the optimal fitness value, and return
    the single outcome selected to have its full results (report/visualizations/CSVs) saved
    (lowest fitness, ties broken by smallest party size). Returns None if no party size produced
    an optimal solution.
    """
    if not outcomes:
        print(f"\nNo party-size results to summarize for generation {gen}.")
        return None

    # Primary objective: lowest fitness (rounded to avoid float-noise ties).
    # Secondary objective: smallest party size on ties.
    FITNESS_ROUND_DECIMALS = 6
    finite_outcomes = [o for o in outcomes if math.isfinite(o['fitness'])]

    lines = [f"Party Size Comparison for Generation {gen}", "=" * 60, ""]
    for o in sorted(outcomes, key=lambda x: x['party_size']):
        status = f"{o['fitness']:.6f}" if math.isfinite(o['fitness']) else "FAILED / NO OPTIMAL SOLUTION"
        lines.append(f"Party Size {o['party_size']}: fitness = {status}")

    winner = None
    if not finite_outcomes:
        lines.append("")
        lines.append("No party size produced an optimal solution.")
        print("\n" + "\n".join(lines))
    else:
        best_rounded = min(round(o['fitness'], FITNESS_ROUND_DECIMALS) for o in finite_outcomes)
        tied_outcomes = [o for o in finite_outcomes if round(o['fitness'], FITNESS_ROUND_DECIMALS) == best_rounded]
        winner = min(tied_outcomes, key=lambda o: o['party_size'])

        lines.append("")
        lines.append(
            f"WINNER: Party Size {winner['party_size']} "
            f"(fitness = {winner['fitness']:.6f})"
        )
        lines.append(f"Best Party (evo_id_pp): {winner['results']['best_evo_id_pp']}")
        if len(tied_outcomes) > 1:
            tied_sizes = sorted(o['party_size'] for o in tied_outcomes)
            lines.append(f"NOTE: {len(tied_outcomes)} party sizes tied for the optimal fitness value: {tied_sizes}")
            lines.append("See optimal_party_solutions file for details on all tied solutions.")
        print("\n" + "\n".join(lines))

        _write_optimal_solutions_report(gen, tied_outcomes, winner, best_rounded)

    summary_path = paths.intermediate_dir() / f"party_size_summary_gen{gen}.txt"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"[summary] Saved {summary_path}")
    except Exception as e:
        print(f"Warning: could not write party size summary file: {e}")

    return winner


def _write_optimal_solutions_report(gen: int, tied_outcomes: list, winner: dict, best_rounded: float) -> None:
    """Write a report listing every party size that tied for the optimal fitness value (there
    will always be at least one entry: the winner). Only the winner's full results are saved to
    results/; this report is the record of any other equally-optimal party compositions.
    """
    lines = [f"Optimal Party Solutions for Generation {gen} (fitness = {best_rounded:.6f})", "=" * 60, ""]
    if len(tied_outcomes) > 1:
        lines.append(f"{len(tied_outcomes)} party sizes tied for the optimal fitness value.")
    else:
        lines.append("Single optimal solution found (no ties).")
    lines.append("")

    for o in sorted(tied_outcomes, key=lambda x: x['party_size']):
        marker = "  <-- SELECTED (full report/visualizations/CSVs saved to results/)" if o['party_size'] == winner['party_size'] else ""
        lines.append(f"Party Size {o['party_size']}{marker}")
        lines.append(f"  Best Party (evo_id_pp): {o['results']['best_evo_id_pp']}")
        lines.append("")

    report_path = paths.intermediate_dir() / f"optimal_party_solutions_gen{gen}.txt"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(f"[optimal solutions] Saved {report_path}")
    except Exception as e:
        print(f"Warning: could not write optimal solutions file: {e}")


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
