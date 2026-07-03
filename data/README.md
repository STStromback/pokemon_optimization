# Data Directory

Input data is organized by generation (`gen_1`, `gen_2`, `gen_3`) plus shared,
cross-generation data in `gen_all`. Files fall into three provenance categories:

- **Raw** – scraped or externally-sourced, unprocessed inputs.
- **Curated** – hand-maintained reference tables consumed directly by the pipeline.
- **Generated** – produced by the pipeline from raw inputs (regenerated on each run).

> Pipeline *outputs* (battle results, optimal parties, plots) are **not** stored
> here. They go to `results/` (final) and `intermediate_files/` (debug only).

## Per-generation files (`gen_1`, `gen_2`, `gen_3`)

| File | Category | Used by | Notes |
|------|----------|---------|-------|
| `TrainerData*Raw.{txt,csv}` | raw | `pipeline.generate_encounters` | Trainer rosters per game. Gen 3 uses the `.csv`. |
| `stats_gen_*.csv` | curated | encounters, player-Pokemon, optimizer | Base stats / types / evolution data. |
| `stages_gen_*.csv` | curated | availability, encounters, player-Pokemon | Location -> progression stage mapping. |
| `typechart_gen_*.csv` | curated | `pipeline.calculate_player_pokemon` | Type effectiveness. |
| `learnsets_gen_*.csv` | raw (scraped) | `pipeline.calculate_player_pokemon` | From PokeAPI via `scrapers/scrape_learnsets_pokeapi.py`. |
| `unique_moves_alt_gen_*.csv` | curated | encounters, player-Pokemon | Move power/type/category used by the engine. |
| `trainer_dvs.csv` (gen 2) | curated | `pipeline.generate_encounters` | Per-trainer DVs for Gen 2 stat calc. |
| `alternate_moves_gen_1.csv` (gen 1) | curated | `pipeline.generate_encounters` | Per-trainer move overrides. |
| `pokemon_data_alt.txt` (gen 1) | raw | `pipeline.generate_encounters` | Gen 1 learnset source. |

## Cross-generation files (`gen_all`)

| File | Category | Used by |
|------|----------|---------|
| `pokemon_availability_raw.csv` | raw (scraped) | availability step (input) |
| `pokemon_availability.csv` | **generated** | player-Pokemon, optimizer (produced by the availability step from the raw file) |
| `pokemon_availability_adjustments.csv` | curated | `pipeline.calculate_availability` |
| `TrainerDataRawAdjustments.csv` | curated | `pipeline.generate_encounters` |
| `exp_table.csv`, `exp_types.csv` | curated | encounters, player-Pokemon |
| `move_power_adjustments.csv` | curated | `pipeline.generate_encounters` |
| `move_tutor_adjustments.csv` | curated | `scrapers/scrape_learnsets_pokeapi.py` |
| `physical_move_types.txt`, `special_move_types.txt`, `crit_moves.txt` | curated | `pipeline.calculate_player_pokemon` |

## Apparently unused (reference/source only)

These files are **not referenced by the current pipeline code**. They are kept as
source/reference material for manual data curation and are safe to archive if
confirmed unneeded:

- `moves_gen_*_backup.csv` (all generations)
- `unique_moves_gen_2.csv`, `unique_moves_gen_3.csv` (the engine uses the `_alt` variants)
- `abilities_gen_3.csv` (ability effects are implemented in code, not loaded from CSV)
- `TrainerDataGen3Raw.txt` (the pipeline reads `TrainerDataGen3Raw.csv`)
- `wild_locations_*_raw*.{txt,csv}` (source material behind the curated availability data)

## Data flow summary

```
scrapers/  ->  data/*/ (raw)  ->  availability step  ->  pokemon_availability.csv (generated)
                                                            |
            curated reference tables  ------------------>  pipeline  ->  results/
```
