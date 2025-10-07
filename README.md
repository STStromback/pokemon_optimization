# Pokemon In-Game Party Optimization (Generations 1-3)

[![Tests](https://img.shields.io/badge/tests-148%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Generation Coverage](https://img.shields.io/badge/gens-1--3-orange)]()

A sophisticated computational pipeline that determines the **mathematically optimal** 6-Pokemon party for completing Pokemon Red, Crystal, and Emerald versions. Uses advanced battle simulation, damage calculation, and Mixed-Integer Linear Programming (MILP) optimization to identify the best possible team composition.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Optimization Results](#optimization-results)
- [Testing](#testing)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Technical Details](#technical-details)
- [License](#license)

---

## Overview

This project tackles the age-old Pokemon question: **"What is the best possible party?"** — but with mathematical rigor instead of subjective debate. 

By simulating all possible encounters, calculating precise damage values using authentic game mechanics, and leveraging advanced optimization algorithms, this pipeline identifies the **global optimum** party composition for each generation.

### Key Innovations

- **Authentic Game Mechanics**: Implements generation-specific damage formulas, stat calculations, critical hits, abilities, STAB, type effectiveness, badge boosts, and more
- **Comprehensive Simulation**: Evaluates every player Pokemon variant against every trainer encounter across the entire game
- **Provably Optimal Solutions**: Uses MILP solvers to guarantee mathematically optimal results (not heuristic approximations)
- **Deterministic Results**: Fully reproducible runs with stable sorting and hash-based tie-breaking
- **Flexible Configuration**: Supports restrictions (starter limits, trade evolution toggles, legendary exclusions)

---

## Key Features

### Battle Simulation Engine
- **Complete Damage Calculation**: Generation-specific formulas for HP, Attack, Defense, Speed, Special stats
- **Type System**: Full type effectiveness charts for all three generations
- **Advanced Mechanics**:
  - Same Type Attack Bonus (STAB)
  - Critical hit rates (generation-specific)
  - Badge boosts (12.5% Gen 1/2, 10% Gen 3)
  - Item type boosts (10% Gen 2/3)
  - Generation 3 abilities (Wonder Guard, Levitate, Intimidate, Pure Power, Thick Fat, etc.)
  - Weather effects (Drizzle, Drought)
  - Special cases (Truant + Hyper Beam, Shedinja HP=1)
- **Move Selection**: Automatic optimal move selection considering TMs, HMs, level-up moves, and evolution chain inheritance

### Optimization Algorithms
1. **Pareto Dominance Filtering**: Eliminates strictly inferior Pokemon before optimization
2. **Mixed-Integer Linear Programming (MILP)**: Provably optimal solutions with mathematical guarantees
3. **Branch-and-Bound** (deprecated): Early exhaustive search approach
4. **Genetic Algorithm** (deprecated): Non-deterministic heuristic approach

### Data Pipeline
- **Web Scraping**: Automated data collection from PokeAPI and PokemonDB
- **Encounter Generation**: Processes raw trainer data with proper DV (Individual Value) calculations
- **Availability Tracking**: Stage-gated Pokemon availability throughout game progression
- **Move Learning**: Complete learnset data with TM/HM acquisition stages

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/pokemon_optimization_revised_v2.git
cd pokemon_optimization_revised_v2

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt

# Run optimization for all generations
python run/main.py
```

**Output**: Optimal party results will be saved to the `results/` directory with visualization charts and detailed logs.

---

## Installation

### Requirements
- Python 3.12+ (3.10+ may work)
- ~500MB disk space for data files
- ~2GB RAM during optimization
- Internet connection (for initial data scraping if enabled)

### Dependencies

Install all dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computations
- `pulp>=2.7.0` - MILP solver interface
- `requests>=2.31.0` - HTTP library for web scraping
- `beautifulsoup4>=4.12.0` - HTML parsing for web scraping
- `lxml>=4.9.0` - XML/HTML parser

**Testing dependencies:**
- `pytest>=8.0.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting

**Optional but recommended:**
- `matplotlib>=3.7.0` - Visualization and chart generation
- `tqdm>=4.65.0` - Progress bars for long-running operations

### MILP Solver Setup

PuLP requires a MILP solver backend. The default `PULP_CBC_CMD` solver is included with PuLP, but you can install additional solvers for better performance:

**Option 1: Use default CBC solver (no additional setup required)**
- Automatically installed with PuLP
- Sufficient for this project

**Option 2: Install GLPK (optional, for better performance)**
```bash
# Windows (via conda)
conda install -c conda-forge glpk

# Ubuntu/Debian
sudo apt-get install glpk-utils

# macOS
brew install glpk
```

**Option 3: Install Gurobi (optional, requires license)**
- Free academic licenses available
- Significantly faster for large problems
- Visit https://www.gurobi.com for installation

---

## Usage

### Basic Execution

Run the complete pipeline for all three generations:

```bash
python run/main.py
```

### Configuration-Controlled Execution

Modify `config/config.json` to customize behavior:

```json
{
    "initialize_repo": "n",
    "gen": [1, 2, 3],
    "level_calc_method": "sequential_max",
    "trade_evolutions": "y",
    "legendaries": "y",
    "all_starters": "n",
    "restrictions": {
        "A": [1, 2, 3],
        "B": [68, 69, 70, 102, 103],
        "C": [82, 83, 84],
        "D": [139, 140, 141]
    },
    "exclusions": [155],
    "easy_dog_catch": "n",
    "drop_first_rival_encounter": "y"
}
```

### Running Individual Components

```bash
# Generate encounter data
python run/functions/generate_encounters.py

# Generate player Pokemon variants
python run/functions/generate_player_pokemon.py

# Calculate damage values
python source/calculate_player_pokemon.py

# Simulate battles
python source/simulate_battles.py

# Run optimization
python source/calculate_best_party_v2.py
```

---

## Configuration

### Config Options (`config/config.json`)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `initialize_repo` | string | Scrape fresh data from web sources ("y"/"n") | "n" |
| `gen` | list/int | Generations to process ([1,2,3] or single gen) | [1,2,3] |
| `level_calc_method` | string | Pokemon level calculation method | "sequential_max" |
| `trade_evolutions` | string | Allow trade evolution Pokemon ("y"/"n") | "y" |
| `legendaries` | string | Allow legendary Pokemon ("y"/"n") | "y" |
| `all_starters` | string | Allow all starters or restrict to one ("y"/"n") | "n" |
| `restrictions` | object | Pokemon restriction groups (max 1 per group) | See below |
| `exclusions` | list | Pokemon to exclude by evo_id | [155] |
| `easy_dog_catch` | string | Special case for legendary dogs in Gen 2 | "n" |
| `drop_first_rival_encounter` | string | Skip first rival battle | "y" |

### Restriction Groups

Prevents multiple Pokemon from the same group in the optimal party:

- **Group A**: Starter Pokemon [1, 2, 3] (Bulbasaur, Ivysaur, Venusaur)
- **Group B**: Machop/Exeggcute lines [68, 69, 70, 102, 103]
- **Group C**: Magnemite line [82, 83, 84]
- **Group D**: Fossil Pokemon [139, 140, 141] (Kabuto, Omanyte lines)

---

## Project Structure

```
pokemon_optimization_revised_v2/
├── config/
│   └── config.json                      # Configuration settings
├── data/
│   ├── gen_1/                           # Generation 1 data files
│   ├── gen_2/                           # Generation 2 data files
│   ├── gen_3/                           # Generation 3 data files
│   └── gen_all/                         # Cross-generation data
├── source/
│   ├── calculate_player_pokemon.py      # Damage calculation engine
│   ├── simulate_battles.py              # Battle simulation & scoring
│   ├── calculate_best_party_v2.py       # MILP optimization
│   ├── calculate_availability.py        # Pokemon availability filtering
│   └── ...                              # Other source modules
├── run/
│   ├── main.py                          # Main pipeline executor
│   └── functions/
│       ├── generate_encounters.py       # Encounter data generation
│       ├── generate_player_pokemon.py   # Player Pokemon generation
│       └── ...                          # Other pipeline functions
├── tests/
│   ├── test_damage_calculations.py      # Damage calculation tests
│   ├── test_simulate_battles.py         # Battle simulation tests
│   ├── test_calculate_best_party.py     # Optimization tests
│   └── test_calculate_availability.py   # Availability tests
├── results/                             # Optimization output files
├── intermediate_files/                  # Debug/intermediate CSVs
├── report/                              # Detailed analysis report
├── .gitignore                           # Git ignore rules
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## How It Works

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Step 0: Calculate Pokemon Availability                     │
│  → Processes availability data from PokemonDB               │
│  → Applies stage-gating and configuration rules             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Generate Encounters                                │
│  → Loads raw trainer data                                   │
│  → Calculates encounter Pokemon stats using DV values       │
│  → Creates encounters_genX.csv                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Generate Player Pokemon                            │
│  → Creates all valid Pokemon evolutionary chains            │
│  → Filters by config (trade evolutions, legendaries, etc.)  │
│  → Generates player_pokemon_genX.csv                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Create Cartesian Product                           │
│  → Pairs every player Pokemon with every encounter          │
│  → Creates player_pokemon_encounters_with_level_genX.csv    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Calculate Damage & Stats                           │
│  → Calculates HP, Speed, Damage for both Pokemon            │
│  → Selects optimal moves considering TMs, evolution chains  │
│  → Applies abilities, badge boosts, STAB, type effectiveness│
│  → Outputs player_pokemon_genX_MAIN_DEBUG.csv               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Simulate Battles                                   │
│  → Determines battle outcomes (Win/Tie/Loss/Unavailable)    │
│  → Calculates fitness scores using HP remaining             │
│  → Applies Pareto dominance filtering                       │
│  → Removes equivalent rows                                  │
│  → Creates pivot table: battle_results_genX.csv             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 6: MILP Optimization                                  │
│  → Formulates as Mixed-Integer Linear Program               │
│  → Applies restriction constraints                          │
│  → Solver finds provably optimal 6-Pokemon party            │
│  → Generates visualizations and result files                │
└─────────────────────────────────────────────────────────────┘
```

### Fitness Function

For each Pokemon-Encounter matchup, fitness is calculated as:

```
Fitness Score:
- Unavailable:  1,000,000,000  (Pokemon not yet obtainable)
- Loss:         1,000,000 + (opponent_hp_remaining_ratio)
- Tie:          1,000
- Win:          1 - (player_hp_remaining_ratio)
```

**Lower is better.** The objective is to minimize total fitness across all encounters.

### Battle Outcome Determination

```python
# Calculate hits to KO
player_hits_to_ko = ceil(opponent_hp / player_damage)
opponent_hits_to_ko = ceil(player_hp / opponent_damage)

# Determine winner based on speed and hits
if player_speed > opponent_speed:
    outcome = "Win" if player_hits_to_ko <= opponent_hits_to_ko else "Loss"
elif player_speed < opponent_speed:
    outcome = "Win" if player_hits_to_ko < opponent_hits_to_ko else "Loss"
else:  # Speed tie
    outcome = "Tie" if player_hits_to_ko == opponent_hits_to_ko else (outcome based on damage)
```

---

## Optimization Results

### Generation 1: Pokemon Red

**Optimal Party (Fitness: 12.584)**

1. **Blastoise** (Squirtle) - Starter, strong early game
2. **Nidoking** (Nidoran♂) - Early evolution, great coverage
3. **Alakazam** (Abra) - Psychic powerhouse
4. **Dugtrio** (Diglett) - Fast Ground-type
5. **Gengar** (Gastly) - Ghost immunity
6. **Jolteon** (Eevee) - Electric coverage

### Generation 2: Pokemon Crystal

**Optimal Party (Fitness: 12.550)**

1. **Meganium** (Chikorita) - Grass starter
2. **Pidgeot** (Pidgey) - Early Normal/Flying
3. **Gengar** (Gastly) - Ghost immunity
4. **Golem** (Geodude) - Rock/Ground tank
5. **Fearow** (Spearow) - Fast Normal/Flying
6. **Alakazam** (Abra) - Psychic powerhouse

### Generation 3: Pokemon Emerald

**Optimal Party (Fitness: 19.591)**

1. **Blaziken** (Torchic) - Fire/Fighting starter
2. **Ludicolo** (Lotad) - Water/Grass coverage
3. **Swellow** (Taillow) - Fast Normal/Flying
4. **Alakazam** (Abra) - Psychic powerhouse
5. **Golem** (Geodude) - Rock/Ground tank
6. **Rayquaza** - Legendary late-game addition

### Key Insights

- **Alakazam** appears in all three optimal parties (best overall Pokemon!)
- **Early evolution** Pokemon are heavily favored
- **Trade evolutions** (Alakazam, Gengar, Golem) appear frequently
- **Type coverage** and **speed** are critical factors
- Gen 3 is notably harder (higher fitness score)

---

## Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_damage_calculations.py -v

# Run with coverage
pytest tests/ --cov=source --cov-report=html
```

### Test Coverage

- ✅ **148 tests passing** (100% pass rate)
- `test_damage_calculations.py`: 61 tests
- `test_simulate_battles.py`: 33 tests
- `test_calculate_best_party.py`: 29 tests
- `test_calculate_availability.py`: 26 tests

### Tested Components

- Pokemon stat calculations (Gen 1/2/3 formulas)
- Critical hit rates (generation-specific)
- Ability modifiers (Gen 3)
- Badge boosts
- Type effectiveness
- Battle outcome determination
- Pareto dominance filtering
- Restriction handling
- Text normalization

---

## Methodology

### Approach: Simplified Simulation + MILP

This project uses a **deductive** approach (as opposed to heuristic):

1. **Deductive**: Simulates actual game mechanics as accurately as possible
2. **Deterministic**: Same inputs always produce same outputs
3. **Provably Optimal**: MILP solver guarantees global optimum

### Why Not Full Reinforcement Learning?

- **Cost**: Would require weeks of GPU time ($1000s in cloud costs)
- **Complexity**: Extremely difficult to train a model to beat the game
- **Interpretability**: Black-box results are hard to analyze

### Why Simplified Simulation?

- **Feasible**: Runs in minutes to hours on consumer hardware
- **Accurate**: Captures core battle mechanics that matter most
- **Flexible**: Easy to modify assumptions and constraints
- **Verifiable**: Results can be validated against actual gameplay

---

## Limitations

### Major Simplifications

1. **No Non-Damaging Moves**: Excludes status moves (Sleep, Paralysis), stat modifiers (Swords Dance, Calm Mind), and healing moves
2. **No Consumable Items**: Potions, X-items, berries are not considered
3. **Perfect Information Assumption**: Player always knows optimal Pokemon to use
4. **Limited Move Variety**: Each Pokemon restricted to 4 move types max
5. **No Switching Strategy**: Assumes best Pokemon starts each battle
6. **No Catch/Flee Mechanics**: All wild encounters assumed won
7. **Deterministic Damage**: Uses expected values (no damage roll variance)

### Biases

- **Against status-focused Pokemon**: Parasect (Spore), Hypno (Hypnosis)
- **Against setup sweepers**: Belly Drum, Dragon Dance strategies
- **Against stalling tactics**: Toxic + Protect combos
- **Toward high-speed sweepers**: Fast Pokemon favored for one-shot potential

### Known Issues

- Some edge cases with ability interactions may not be perfect
- Move power adjustments may not cover all special cases
- Trainer AI is assumed to be optimal (not realistic)

---

## Technical Details

### Damage Formula (Simplified Gen 3)

```
Damage = ((2 * Level / 5 + 2) * Power * Attack / Defense / 50 + 2)
         × STAB × Type1 × Type2 × Critical × Random × Other
```

**Modifiers:**
- **STAB**: 1.5× if move type matches Pokemon type
- **Type Effectiveness**: 0×, 0.25×, 0.5×, 1×, 2×, 4×
- **Critical**: 2× (Gen 1/2) or 1.5× (Gen 3)
- **Random**: 0.85-1.0× (this analysis uses expected value 0.925×)

### MILP Formulation

**Decision Variables:**
- `x_i ∈ {0,1}`: Select Pokemon i for party
- `z_ij ∈ [0,1]`: Pokemon i covers encounter j

**Objective:**
```
Minimize: Σ_j Σ_i (fitness_ij × z_ij)
```

**Constraints:**
```
Σ_i x_i = 6                           # Exactly 6 Pokemon
Σ_i z_ij = 1  ∀j                      # Each encounter covered by exactly 1 Pokemon
z_ij ≤ x_i    ∀i,j                    # Only selected Pokemon can cover
Σ_(i∈Group_k) x_i ≤ 1  ∀k             # At most 1 per restriction group
```

---

## Contributing

Contributions welcome! Areas for improvement:

- Add Generation 4-9 support
- Implement non-damaging move simulation
- Add real damage roll variance
- Improve ability/item interaction coverage
- Optimize computational performance
- Expand test coverage

---

## License

This project is for educational and research purposes. Pokemon and related trademarks are property of Nintendo, Game Freak, and The Pokemon Company.

---

## Citation

If you use this work in research, please cite:

```
Stromback, S. (2025). Pokemon In-Game Party Optimization: Generations 1-3.
GitHub repository: https://github.com/yourusername/pokemon_optimization_revised_v2
```

---

## Acknowledgments

- **Data Sources**: PokeAPI, PokemonDB, Bulbapedia, GameFAQs
- **Optimization**: PuLP library and CBC/GLPK solvers
- **Inspiration**: Tommy Odland's "The Best Pokemon Party" blog post

---

**For detailed methodology and analysis, see the [full report](report/pokemon_optimization_report.md).**

**For test documentation, see [tests/README_TEST.md](tests/README_TEST.md).**
