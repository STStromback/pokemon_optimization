"""Capture and diff pipeline outputs to prove a refactor does not change results.

This is a developer/verification tool (not part of the shipped pipeline). It runs
the in-memory pipeline stages that are being refactored -- encounter generation,
player-Pokemon generation, pairing, damage/stat calculation, and battle
simulation -- and writes the resulting DataFrames to CSV so two runs can be
compared byte/numerically.

Usage:
    python scripts/pipeline_snapshot.py capture <out_dir> [--gen 1 2 3]
    python scripts/pipeline_snapshot.py diff <baseline_dir> <new_dir> [--gen 1 2 3]

The MILP stage is intentionally excluded: it is deterministic given
``battle_results`` and is not modified by this refactor, so an identical
``battle_results`` snapshot is sufficient to prove end-to-end parity.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the source/ package root importable (mirrors tests/conftest.py).
SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from common import paths
from common.config import load_config
from pipeline.generate_encounters import generate_encounters
from pipeline.generate_player_pokemon import generate_player_pokemon
from pipeline.pair_player_pokemon_encounters import generate_player_pokemon_x_encounters
from pipeline.calculate_player_pokemon import calculate_player_pokemons
from pipeline.simulate_battles import simulate_battles

ARTIFACTS = ("encounters", "player_pokemon", "calculated", "battle_results")


def capture_generation(gen: int, out_dir: Path, config: dict, config_path: str) -> None:
    """Run the stages for one generation and save each artifact to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    encounters_df = generate_encounters(gen)
    encounters_df.to_csv(out_dir / f"encounters_gen{gen}.csv", index=False)

    player_pokemon_df = generate_player_pokemon(config_path)
    player_pokemon_df = player_pokemon_df[player_pokemon_df["generation"] == gen]
    player_pokemon_df.to_csv(out_dir / f"player_pokemon_gen{gen}.csv", index=False)

    pairs_df = generate_player_pokemon_x_encounters(gen, player_pokemon_df, encounters_df)

    calculated_df = calculate_player_pokemons(gen, pairs_df, config=config)
    calculated_df.to_csv(out_dir / f"calculated_gen{gen}.csv", index=False)

    battle_results = simulate_battles(gen, calculated_df)
    battle_results.to_csv(out_dir / f"battle_results_gen{gen}.csv")

    print(f"[capture] gen {gen}: saved {len(ARTIFACTS)} artifacts to {out_dir}")


def diff_csv(baseline: Path, new: Path) -> bool:
    """Return True if the two CSVs are equal (with float tolerance ~0)."""
    if not baseline.exists() or not new.exists():
        print(f"  MISSING: {baseline.name} (baseline={baseline.exists()} new={new.exists()})")
        return False
    a = pd.read_csv(baseline)
    b = pd.read_csv(new)
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_like=False, rtol=0, atol=0)
        print(f"  OK: {baseline.name}  ({a.shape[0]} rows x {a.shape[1]} cols)")
        return True
    except AssertionError as exc:
        first_line = str(exc).strip().splitlines()[0]
        print(f"  DIFF: {baseline.name} -> {first_line}")
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Run stages and save artifacts.")
    cap.add_argument("out_dir")
    cap.add_argument("--gen", nargs="+", type=int, default=[1, 2, 3])

    dff = sub.add_parser("diff", help="Compare two artifact directories.")
    dff.add_argument("baseline_dir")
    dff.add_argument("new_dir")
    dff.add_argument("--gen", nargs="+", type=int, default=[1, 2, 3])

    args = parser.parse_args(argv)
    config_path = str(paths.CONFIG_FILE)
    config = load_config(config_path)

    if args.command == "capture":
        out_dir = Path(args.out_dir)
        for gen in args.gen:
            capture_generation(gen, out_dir, config, config_path)
        return 0

    # diff
    baseline_dir = Path(args.baseline_dir)
    new_dir = Path(args.new_dir)
    all_ok = True
    for gen in args.gen:
        print(f"=== Generation {gen} ===")
        for name in ARTIFACTS:
            fname = f"{name}_gen{gen}.csv"
            ok = diff_csv(baseline_dir / fname, new_dir / fname)
            all_ok = all_ok and ok
    print("\nRESULT:", "ALL IDENTICAL" if all_ok else "DIFFERENCES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
