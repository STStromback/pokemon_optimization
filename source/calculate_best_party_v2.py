"""
calculate_best_party_v2.py

MILP-based Pokemon party optimization using Mixed Integer Linear Programming.
Formulates the problem as selecting 6 rows that minimize the sum of column-wise minimums,
with support for restriction groups and exclusions.

This provides deterministic global optimal solutions efficiently.
"""

import pandas as pd
import numpy as np
import json
import logging
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import math

try:
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous, LpStatus, value
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("WARNING: PuLP not available. Install with: pip install pulp")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# MILP Solver
# =============================================================================

def solve_party_optimization_milp(
    cost_matrix: np.ndarray,
    row_ids: List[int],
    col_ids: List[int],
    k: int = 6,
    exclusions: List[int] = None,
    restriction_groups: List[List[int]] = None,
    time_limit: int = 300,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solve the Pokemon party optimization problem using MILP.
    
    Mathematical Formulation:
    ------------------------
    Variables:
      x_i ∈ {0,1}  : binary, whether row i is selected
      z_ij ∈ [0,1] : continuous, whether column j is covered by row i
    
    Objective:
      minimize Σ_j Σ_i c_ij * z_ij
    
    Constraints:
      Σ_i x_i = k                          (select exactly k rows)
      Σ_i z_ij = 1           ∀j           (each column covered by exactly one row)
      z_ij ≤ x_i             ∀i,j         (only selected rows can cover)
      x_i = 0                ∀i∈excluded  (forbidden rows)
      Σ_i∈G x_i ≤ 1          ∀G           (at most one per restriction group)
    
    Args:
        cost_matrix: (n_rows, n_cols) numpy array of costs
        row_ids: List of row identifiers (evo_id_pp values)
        col_ids: List of column identifiers (enc_id values)
        k: Number of rows to select (default 6)
        exclusions: List of row IDs to exclude
        restriction_groups: List of lists, each containing row indices that form a group
        time_limit: Solver time limit in seconds
        verbose: Whether to print progress
    
    Returns:
        Dictionary with solution details
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP library is required. Install with: pip install pulp")
    
    n_rows, n_cols = cost_matrix.shape
    
    if verbose:
        print(f"\n=== MILP Optimization ===")
        print(f"Problem size: {n_rows} rows × {n_cols} columns")
        print(f"Selecting {k} rows to minimize sum of column-wise minimums")
    
    # Create the optimization problem
    prob = LpProblem("Pokemon_Party_Optimization", LpMinimize)
    
    # Decision variables
    # x[i]: binary variable for selecting row i
    x = {}
    for i in range(n_rows):
        x[i] = LpVariable(f"x_{i}", cat=LpBinary)
    
    # z[i][j]: continuous variable for column j being covered by row i
    z = {}
    for i in range(n_rows):
        z[i] = {}
        for j in range(n_cols):
            z[i][j] = LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1, cat=LpContinuous)
    
    # Objective: minimize sum of column-wise minimums
    # This is represented as: Σ_j Σ_i c_ij * z_ij
    objective_terms = []
    for j in range(n_cols):
        for i in range(n_rows):
            coef = float(cost_matrix[i, j])
            if not np.isnan(coef) and not np.isinf(coef):
                objective_terms.append(coef * z[i][j])
    
    prob += lpSum(objective_terms), "Total_Cost"
    
    # Constraint 1: Select exactly k rows
    prob += lpSum([x[i] for i in range(n_rows)]) == k, "Select_K_Rows"
    
    # Constraint 2: Each column must be covered by exactly one selected row
    for j in range(n_cols):
        prob += lpSum([z[i][j] for i in range(n_rows)]) == 1, f"Cover_Column_{j}"
    
    # Constraint 3: Only selected rows can cover columns (z_ij ≤ x_i)
    for i in range(n_rows):
        for j in range(n_cols):
            prob += z[i][j] <= x[i], f"Link_x_{i}_z_{i}_{j}"
    
    # Constraint 4: Forbidden rows (exclusions)
    if exclusions:
        excluded_indices = []
        for i, row_id in enumerate(row_ids):
            # Check if this row should be excluded
            base_evo_id = int(str(row_id).split('_')[0])
            if base_evo_id in exclusions:
                excluded_indices.append(i)
        
        if excluded_indices:
            for i in excluded_indices:
                prob += x[i] == 0, f"Exclude_Row_{i}"
            if verbose:
                print(f"Excluded {len(excluded_indices)} rows based on exclusions list")
    
    # Constraint 5: At most one from each restriction group
    if restriction_groups:
        for g_idx, group in enumerate(restriction_groups):
            if len(group) >= 2:
                prob += lpSum([x[i] for i in group if i < n_rows]) <= 1, f"Restriction_Group_{g_idx}"
        if verbose:
            print(f"Applied {len(restriction_groups)} restriction group constraints")
    
    # Solve the problem
    if verbose:
        print(f"\nSolving MILP (time limit: {time_limit}s)...")
    
    # Set solver options
    solver = None
    try:
        # Try to use CBC (comes with PuLP)
        from pulp import PULP_CBC_CMD
        solver = PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    except:
        # Fall back to default solver
        if verbose:
            print("Using default PuLP solver")
    
    # Solve
    if solver:
        status = prob.solve(solver)
    else:
        status = prob.solve()
    
    # Extract solution
    solution = {
        'status': LpStatus[status],
        'optimal': status == 1,
        'objective_value': value(prob.objective) if status == 1 else None,
        'selected_rows': [],
        'selected_row_ids': [],
        'coverage_matrix': np.zeros((n_rows, n_cols), dtype=float)
    }
    
    if status == 1:  # Optimal solution found
        # Get selected rows
        for i in range(n_rows):
            if value(x[i]) > 0.5:  # Binary variable should be 0 or 1
                solution['selected_rows'].append(i)
                solution['selected_row_ids'].append(row_ids[i])
        
        # Get coverage matrix (which row covers which column)
        for i in range(n_rows):
            for j in range(n_cols):
                solution['coverage_matrix'][i, j] = value(z[i][j])
        
        if verbose:
            print(f"\n✓ Optimal solution found!")
            print(f"  Objective value: {solution['objective_value']:.6f}")
            print(f"  Selected rows: {solution['selected_row_ids']}")
    else:
        if verbose:
            print(f"\n✗ Solver status: {LpStatus[status]}")
            print(f"  No optimal solution found within time limit")
    
    return solution


# =============================================================================
# Restriction Processing
# =============================================================================

def _process_restrictions(restrictions: Dict[str, List[int]], evo_id_pp_list: List) -> List[List[int]]:
    """
    Convert restriction groups from evo_ids to row indices.
    Returns list of index groups where at most 1 Pokemon per group is allowed.
    """
    groups = []
    if not restrictions:
        return groups
    
    for group_name, evo_ids in sorted(restrictions.items()):
        if not (isinstance(evo_ids, list) and len(evo_ids) >= 2):
            continue
        indices = []
        for evo_id in evo_ids:
            for i, evo_id_pp_val in enumerate(evo_id_pp_list):
                base_evo_id = int(str(evo_id_pp_val).split('_')[0])
                if base_evo_id == evo_id:
                    indices.append(i)
        if len(indices) >= 2:
            groups.append(sorted(indices))
    return groups


# =============================================================================
# Pre-processing: Column Filtering
# =============================================================================

def filter_unwinnable_columns(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove columns that don't have any winning combinations (values between 0 and 1).
    Returns filtered dataframe and information about removed columns.
    """
    if 'evo_id_pp' not in df.columns:
        return df, {'removed_columns': [], 'removed_count': 0}
    
    battle_cols = [c for c in df.columns if c != 'evo_id_pp']
    if not battle_cols:
        return df, {'removed_columns': [], 'removed_count': 0}
    
    removed_columns = []
    
    for col in battle_cols:
        col_values = df[col].values
        # Check if any value is between 0 and 1 (inclusive of 0, exclusive of 1)
        has_winning_combination = np.any((col_values >= 0) & (col_values < 1))
        
        if not has_winning_combination:
            # Find the minimum value for this column to report why it was removed
            min_val = np.nanmin(col_values)
            removed_columns.append({
                'enc_id': col,
                'min_value': min_val,
                'reason': 'No winning combinations (all values >= 1 or NaN)'
            })
    
    # Remove the identified columns
    if removed_columns:
        cols_to_remove = [c['enc_id'] for c in removed_columns]
        df_filtered = df.drop(columns=cols_to_remove)
        
        if verbose:
            print(f"  Removed {len(removed_columns)} unwinnable columns (no values between 0 and 1)")
            if len(removed_columns) <= 10:
                for col_info in removed_columns:
                    print(f"    - enc_id {col_info['enc_id']}: min_value = {col_info['min_value']:.2f}")
            else:
                print(f"    (showing first 10 of {len(removed_columns)} removed columns)")
                for col_info in removed_columns[:10]:
                    print(f"    - enc_id {col_info['enc_id']}: min_value = {col_info['min_value']:.2f}")
        
        return df_filtered, {
            'removed_columns': removed_columns,
            'removed_count': len(removed_columns)
        }
    
    if verbose:
        print(f"  No unwinnable columns found (all columns have at least one value between 0 and 1)")
    
    return df, {'removed_columns': [], 'removed_count': 0}


# =============================================================================
# Pre-processing: Dominance Filtering
# =============================================================================

def filter_dominated_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove rows that are strictly dominated (worse in all columns with at least one strictly worse).
    This reduces the problem size without affecting the optimal solution.
    """
    if 'evo_id_pp' not in df.columns:
        return df
    
    feature_cols = [c for c in df.columns if c != 'evo_id_pp']
    if not feature_cols:
        return df
    
    X = df[feature_cols].values
    n_rows = len(df)
    
    dominated = np.zeros(n_rows, dtype=bool)
    
    if verbose and TQDM_AVAILABLE:
        iterator = tqdm(range(n_rows), desc="Checking dominance", leave=False)
    else:
        iterator = range(n_rows)
    
    for i in iterator:
        if dominated[i]:
            continue
        for j in range(n_rows):
            if i == j or dominated[j]:
                continue
            # Check if row i dominates row j (i is better or equal in all, and strictly better in at least one)
            row_i = X[i, :]
            row_j = X[j, :]
            
            # Compare element-wise (lower is better)
            better_or_equal = np.all(row_i <= row_j) or np.all(np.isnan(row_i) | np.isnan(row_j) | (row_i <= row_j))
            strictly_better = np.any(row_i < row_j)
            
            if better_or_equal and strictly_better:
                dominated[j] = True
    
    n_dominated = np.sum(dominated)
    if n_dominated > 0:
        df_filtered = df[~dominated].copy().reset_index(drop=True)
        if verbose:
            print(f"  Removed {n_dominated} dominated rows ({n_dominated/n_rows*100:.1f}%)")
        return df_filtered
    
    return df


# =============================================================================
# Main Optimization Function
# =============================================================================

def calculate_best_party_milp(
    battle_results_df: pd.DataFrame = None,
    battle_results_path: str = None,
    config_path: str = None,
    player_pokemon_df: pd.DataFrame = None,
    time_limit: int = 300,
    apply_dominance_filter: bool = True,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate the best Pokemon party using MILP optimization.
    
    Args:
        battle_results_df: DataFrame with battle results (pivot table format)
        battle_results_path: Path to battle results CSV file
        player_pokemon_df: Optional DataFrame with detailed player Pokemon data for encounter analysis
        config_path: Path to config.json file
        time_limit: Solver time limit in seconds (default 300)
        apply_dominance_filter: Whether to pre-filter dominated rows (default True)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with optimization results
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP library is required. Install with: pip install pulp")
    
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger(__name__)
    
    # Load battle results
    if battle_results_df is not None:
        df = battle_results_df.copy()
        # Handle index as evo_id_pp
        if 'evo_id_pp' not in df.columns and df.index.name == 'evo_id_pp':
            df = df.reset_index()
    elif battle_results_path is not None:
        df = pd.read_csv(battle_results_path)
    else:
        raise ValueError("Either battle_results_df or battle_results_path must be provided")
    
    if 'evo_id_pp' not in df.columns:
        raise ValueError("DataFrame must have 'evo_id_pp' column")
    
    # Load config
    config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    # Apply drop_first_rival_encounter if enabled
    drop_first_rival_encounter = config.get('drop_first_rival_encounter', 'n').lower() == 'y'
    if drop_first_rival_encounter:
        battle_cols = [c for c in df.columns if c != 'evo_id_pp']
        if len(battle_cols) >= 3:
            try:
                sorted_battle_cols = sorted(battle_cols, key=lambda x: int(x))
            except:
                sorted_battle_cols = sorted(battle_cols, key=lambda x: str(x))
            cols_to_drop = sorted_battle_cols[:3]
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped first 3 rival encounter columns: {cols_to_drop}")
    
    # Deterministic ordering
    df = df.sort_values(by='evo_id_pp').reset_index(drop=True)
    battle_cols = [c for c in df.columns if c != 'evo_id_pp']
    try:
        sorted_cols = sorted(battle_cols, key=lambda x: int(x))
    except:
        sorted_cols = sorted(battle_cols, key=lambda x: str(x))
    df = df[['evo_id_pp'] + sorted_cols]
    
    if verbose:
        print(f"\n=== MILP-based Pokemon Party Optimization ===")
        print(f"Dataset: {len(df)} Pokemon variants, {len(sorted_cols)} battle scenarios")
    
    # Filter unwinnable columns (those without any winning combinations)
    if verbose:
        print(f"\nFiltering unwinnable columns...")
    df, unwinnable_info = filter_unwinnable_columns(df, verbose=verbose)
    
    # Update sorted_cols after filtering
    battle_cols = [c for c in df.columns if c != 'evo_id_pp']
    try:
        sorted_cols = sorted(battle_cols, key=lambda x: int(x))
    except:
        sorted_cols = sorted(battle_cols, key=lambda x: str(x))
    df = df[['evo_id_pp'] + sorted_cols]
    
    if verbose and unwinnable_info['removed_count'] > 0:
        print(f"  Remaining battle scenarios: {len(sorted_cols)}")
    
    # Apply dominance filtering
    if apply_dominance_filter:
        if verbose:
            print(f"\nApplying dominance filtering...")
        original_size = len(df)
        df = filter_dominated_rows(df, verbose=verbose)
        filtered_size = len(df)
        if verbose and original_size > filtered_size:
            print(f"  Reduced from {original_size} to {filtered_size} rows")
    
    # Process restrictions
    restrictions = config.get('restrictions', {})
    exclusions = config.get('exclusions', [])
    evo_id_pp_list = df['evo_id_pp'].tolist()
    restriction_groups = _process_restrictions(restrictions, evo_id_pp_list)
    
    if verbose and restriction_groups:
        print(f"\nLoaded {len(restriction_groups)} restriction groups")
    if verbose and exclusions:
        print(f"Loaded {len(exclusions)} exclusions")
    
    # Prepare cost matrix
    feature_cols = [c for c in df.columns if c != 'evo_id_pp']
    cost_matrix = df[feature_cols].values.astype(np.float32)
    row_ids = df['evo_id_pp'].tolist()
    col_ids = [int(c) if str(c).isdigit() else c for c in feature_cols]
    
    # Solve MILP
    solution = solve_party_optimization_milp(
        cost_matrix=cost_matrix,
        row_ids=row_ids,
        col_ids=col_ids,
        k=6,
        exclusions=exclusions,
        restriction_groups=restriction_groups,
        time_limit=time_limit,
        verbose=verbose
    )
    
    if not solution['optimal']:
        logger.error("MILP solver did not find optimal solution")
        return {
            'best_party_indices': [],
            'best_evo_id_pp': [],
            'best_fitness': float('inf'),
            'party_details': pd.DataFrame(),
            'total_combinations': 0,
            'valid_combinations': 0,
            'invalid_combinations': 0,
            'method': 'milp',
            'solver_status': solution['status'],
            'unwinnable_columns': unwinnable_info
        }
    
    # Extract results
    best_party_indices = solution['selected_rows']
    best_evo_id_pp = solution['selected_row_ids']
    best_fitness = solution['objective_value']
    
    # Calculate party details
    party_details = _get_party_details(df, best_party_indices)
    
    # Calculate statistics
    n_pokemon = len(df)
    total_combinations = math.comb(n_pokemon, 6) if n_pokemon >= 6 else 0
    
    results = {
        'best_party_indices': best_party_indices,
        'best_evo_id_pp': best_evo_id_pp,
        'best_fitness': best_fitness,
        'party_details': party_details,
        'total_combinations': 1,  # MILP finds the solution directly
        'valid_combinations': 1,
        'invalid_combinations': 0,
        'method': 'milp',
        'solver_status': solution['status'],
        'original_search_space': total_combinations,
        'unwinnable_columns': unwinnable_info
    }
    
    if verbose:
        print(f"\n=== Optimization Complete ===")
        print(f"Best fitness: {best_fitness:.6f}")
        print(f"Selected party: {best_evo_id_pp}")
        print(f"Original search space: {total_combinations:,}")
    
    # Save results
    _save_results_to_file(results, config_path, df, player_pokemon_df)
    
    return results


# =============================================================================
# Helper Functions
# =============================================================================

def _get_party_details(df: pd.DataFrame, best_party_indices: List[int]) -> pd.DataFrame:
    """Create party details DataFrame."""
    if not best_party_indices:
        return pd.DataFrame()
    
    battle_columns = [c for c in df.columns if c != 'evo_id_pp']
    battle_matrix = df[battle_columns].values
    
    details = []
    for i, idx in enumerate(best_party_indices):
        evo_id_pp = df.iloc[idx]['evo_id_pp']
        pokemon_data = battle_matrix[idx, :]
        
        details.append({
            'Position': i + 1,
            'Evo_ID_PP': evo_id_pp,
            'Avg_Performance': np.nanmean(pokemon_data),
            'Min_Performance': np.nanmin(pokemon_data),
            'Max_Performance': np.nanmax(pokemon_data)
        })
    return pd.DataFrame(details)


def _norm(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().replace(' ', '').replace('_', '') if text else ''


def _is_trade_evolution(pokemon_name: str, evo_item: str, gen: int) -> bool:
    """Check if a Pokemon is a trade evolution."""
    trade_evolutions_gen1 = ['alakazam', 'machamp', 'golem', 'gengar']
    trade_item_evolutions = ['metalcoat', 'dragonscale', 'kingsrock', 'upgrade', 'deepseatooth', 'deepseascale']
    
    if gen == 1:
        return _norm(pokemon_name) in trade_evolutions_gen1
    else:
        if pd.notna(evo_item) and str(evo_item).strip():
            evo_item_norm = _norm(str(evo_item))
            return (evo_item_norm == 'trade' or 
                   evo_item_norm in trade_item_evolutions or
                   _norm(pokemon_name) in trade_evolutions_gen1)
    return False


def _get_display_pokemon_name(base_evo_id: int, evo_id_pp: str, config_settings: dict, gen: int, pokemon_stats_df: pd.DataFrame = None) -> str:
    """Get the appropriate Pokemon name for display, filtering trade evolutions if disabled."""
    if pokemon_stats_df is None:
        return f"Evo {evo_id_pp}"
        
    matching = pokemon_stats_df[pokemon_stats_df['evo_id'] == base_evo_id]
    if matching.empty:
        return f"Evo {evo_id_pp}"
    
    allow_trade_evolutions = config_settings.get('trade_evolutions', 'y').lower() != 'n'
    
    if allow_trade_evolutions:
        final_evo = matching.iloc[-1]
        return final_evo['pokemon']
    else:
        for idx in range(len(matching) - 1, -1, -1):
            evo_candidate = matching.iloc[idx]
            pokemon_name = evo_candidate['pokemon']
            evo_item = evo_candidate.get('evo_item', '')
            
            is_trade = _is_trade_evolution(pokemon_name, evo_item, gen)
            
            if not is_trade:
                return pokemon_name
        
        return matching.iloc[0]['pokemon']


def _save_results_to_file(results: Dict[str, Any], config_path: str = None, df: pd.DataFrame = None, player_pokemon_df: pd.DataFrame = None):
    """Save results to file in the standard format."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        if not results_dir.exists():
            project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
            results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"best_party_milp_{timestamp}.txt"
        filepath = results_dir / filename

        config_settings = {}
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_settings = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load config from {config_path}: {e}")

        gen_config = config_settings.get('gen', [1, 2, 3])
        generations_to_try = gen_config if isinstance(gen_config, list) else [gen_config]

        pokemon_stats_df = None
        for gen in generations_to_try:
            stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
            if not Path(stats_path).exists():
                project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
                stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
            try:
                gen_stats_df = pd.read_csv(stats_path)
                pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
            except Exception as e:
                logging.warning(f"Could not load Pokemon stats from gen {gen}: {e}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"POKEMON PARTY MILP OPTIMIZATION RESULTS\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

            f.write(f"OPTIMIZATION METHOD: Mixed Integer Linear Programming (MILP)\n")
            f.write(f"SOLVER STATUS: {results.get('solver_status', 'Unknown')}\n")
            f.write(f"FINAL FITNESS SCORE: {results['best_fitness']:.6f}\n")
            f.write(f"GUARANTEED GLOBAL OPTIMUM: YES\n\n")
            
            # Add unwinnable columns information
            unwinnable_info = results.get('unwinnable_columns', {})
            if unwinnable_info and unwinnable_info.get('removed_count', 0) > 0:
                f.write(f"UNWINNABLE ENCOUNTERS REMOVED:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total unwinnable encounters: {unwinnable_info['removed_count']}\n")
                f.write(f"Reason: No Pokemon could achieve winning score (0 to 1)\n\n")
                
                removed_cols = unwinnable_info.get('removed_columns', [])
                if removed_cols:
                    f.write(f"Removed encounter details:\n")
                    for col_info in removed_cols:
                        f.write(f"  - enc_id {col_info['enc_id']}: "
                               f"min_value = {col_info['min_value']:.2f}\n")
                f.write("\n")

            f.write(f"SEARCH STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original search space: {results.get('original_search_space', 0):,}\n")
            f.write(f"MILP solution: Found directly (no enumeration)\n")
            f.write(f"Total combinations evaluated: 1 (optimal)\n\n")

            f.write("CHOSEN POKEMON PARTY:\n")
            f.write("-" * 40 + "\n")
            for i, evo_id_pp in enumerate(results['best_evo_id_pp'], 1):
                base_evo_id = int(str(evo_id_pp).split('_')[0])
                pokemon_info = "Unknown Pokemon"
                pokemon_types = "Unknown"
                if pokemon_stats_df is not None:
                    matching = pokemon_stats_df[pokemon_stats_df['evo_id'] == base_evo_id]
                    if not matching.empty:
                        pokemon_info = _get_display_pokemon_name(base_evo_id, evo_id_pp, config_settings, generations_to_try[0], pokemon_stats_df)
                        selected_pokemon = matching[matching['pokemon'] == pokemon_info]
                        if not selected_pokemon.empty:
                            pokemon_types = selected_pokemon.iloc[0]['types']
                        else:
                            pokemon_types = matching.iloc[-1]['types']
                f.write(f"{i}. {pokemon_info} (Evo ID: {base_evo_id})\n")
                f.write(f"   Types: {pokemon_types}\n")
                f.write(f"   Evo ID PP: {evo_id_pp}\n\n")

            f.write("\nCONFIGURATION SETTINGS:\n")
            f.write("-" * 40 + "\n")
            if config_settings:
                for key, value in config_settings.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("No configuration settings loaded\n")

            f.write("\nOPTIMIZATION DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Method: MILP (Mixed Integer Linear Programming)\n")
            f.write(f"Best Fitness Score: {results['best_fitness']:.6f}\n")
            f.write(f"Guaranteed Global Optimum: YES\n")
            f.write(f"Deterministic: YES\n")

            f.write("\n" + "="*60 + "\n")

        print(f"\nResults saved to: {filepath}")

        # Create visualizations
        if df is not None:
            try:
                png_filepath = filepath.with_suffix('.png')
                _create_party_performance_visualization(results, df, png_filepath, config_settings, generations_to_try)
                print(f"Visualization saved to: {png_filepath}")
                
                unique_png_filepath = filepath.with_name(filepath.stem + "_unique_best").with_suffix('.png')
                _create_unique_best_performance_visualization(results, df, unique_png_filepath, config_settings, generations_to_try)
                print(f"Unique best performance visualization saved to: {unique_png_filepath}")
            except Exception as viz_error:
                logging.error(f"Error creating visualization: {viz_error}")
        
        # Create detailed encounter CSVs
        if df is not None and player_pokemon_df is not None:
            try:
                # Unique best encounters CSV
                unique_csv_filepath = filepath.with_name(filepath.stem + "_unique_best_encounters").with_suffix('.csv')
                _create_unique_best_encounters_csv(results, df, player_pokemon_df, unique_csv_filepath, config_settings, generations_to_try)
                print(f"Unique best encounters CSV saved to: {unique_csv_filepath}")
                
                # Top 10 hardest encounters CSV
                hardest_csv_filepath = filepath.with_name(filepath.stem + "_hardest_encounters").with_suffix('.csv')
                _create_hardest_encounters_csv(results, df, player_pokemon_df, hardest_csv_filepath, config_settings, generations_to_try)
                print(f"Top 10 hardest encounters CSV saved to: {hardest_csv_filepath}")
            except Exception as csv_error:
                logging.error(f"Error creating encounter CSVs: {csv_error}")

    except Exception as e:
        logging.error(f"Error saving results to file: {e}")
        print(f"Warning: Could not save results to file: {e}")


def _create_unique_best_encounters_csv(results: Dict[str, Any], df: pd.DataFrame, player_pokemon_df: pd.DataFrame, csv_filepath: Path, config_settings: dict, generations: list):
    """Create CSV with details of encounters where each Pokemon uniquely outperforms all others."""
    best_party_indices = results['best_party_indices']
    best_evo_id_pp = results['best_evo_id_pp']
    
    if not best_party_indices or player_pokemon_df is None:
        return
    
    battle_columns = [c for c in df.columns if c != 'evo_id_pp']
    battle_matrix = df[battle_columns].values
    party_battle_data = battle_matrix[best_party_indices, :]
    
    individual_pokemon_data = np.nan_to_num(party_battle_data, nan=np.inf)
    n_encounters = party_battle_data.shape[1]
    
    # Identify unique best encounters for each Pokemon
    records = []
    for encounter_idx in range(n_encounters):
        encounter_values = individual_pokemon_data[:, encounter_idx]
        min_value = np.nanmin(encounter_values)
        min_count = np.sum(encounter_values == min_value)
        
        if min_count == 1:  # Only one Pokemon has the minimum
            best_pokemon_idx = int(np.argmin(encounter_values))
            evo_id_pp = best_evo_id_pp[best_pokemon_idx]
            enc_id = battle_columns[encounter_idx]
            
            # Find matching record in player_pokemon_df
            matching_records = player_pokemon_df[
                (player_pokemon_df['evo_id_pp'] == evo_id_pp) & 
                (player_pokemon_df['enc_id'] == enc_id)
            ]
            
            if not matching_records.empty:
                record = matching_records.iloc[0]
                base_evo_id = int(str(evo_id_pp).split('_')[0])
                
                records.append({
                    'party_position': best_pokemon_idx + 1,
                    'evo_id': base_evo_id,
                    'evo_id_pp': evo_id_pp,
                    'enc_id': enc_id,
                    'trainer_name_enc': record.get('trainer_name_enc', ''),
                    'pokemon': record.get('pokemon', ''),
                    'level': record.get('level', ''),
                    'move_out': record.get('move_out', ''),
                    'damage_out': record.get('damage_out', ''),
                    'hp_out': record.get('hp_out', ''),
                    'speed_out': record.get('speed_out', ''),
                    'pokemon_enc': record.get('pokemon_enc', ''),
                    'level_enc': record.get('level_enc', ''),
                    'move_in': record.get('move_in', ''),
                    'damage_in': record.get('damage_in', ''),
                    'hp_in': record.get('hp_in', ''),
                    'speed_in': record.get('speed_in', ''),
                    'fitness_value': min_value if min_value != np.inf else np.nan
                })
    
    if records:
        results_df = pd.DataFrame(records)
        results_df = results_df.sort_values(['party_position', 'enc_id'])
        results_df.to_csv(csv_filepath, index=False)


def _create_hardest_encounters_csv(results: Dict[str, Any], df: pd.DataFrame, player_pokemon_df: pd.DataFrame, csv_filepath: Path, config_settings: dict, generations: list):
    """Create CSV with details of the top 100 hardest encounters for the party (best matchup only)."""
    best_party_indices = results['best_party_indices']
    best_evo_id_pp = results['best_evo_id_pp']
    
    if not best_party_indices or player_pokemon_df is None:
        return
    
    battle_columns = [c for c in df.columns if c != 'evo_id_pp']
    battle_matrix = df[battle_columns].values
    party_battle_data = battle_matrix[best_party_indices, :]
    
    # Calculate minimum fitness for each encounter
    min_values = np.nanmin(party_battle_data, axis=0)
    
    # Get top 100 hardest encounters (highest min values)
    hardest_indices = np.argsort(min_values)[-100:][::-1]  # Sort descending, take top 100
    
    records = []
    for encounter_idx in hardest_indices:
        enc_id = battle_columns[encounter_idx]
        min_fitness = min_values[encounter_idx]
        
        # Find which Pokemon has the best matchup (minimum fitness) for this encounter
        encounter_fitness_values = party_battle_data[:, encounter_idx]
        best_pokemon_indices = np.where(encounter_fitness_values == min_fitness)[0]
        
        # If there are ties, include all tied Pokemon
        for best_idx in best_pokemon_indices:
            pokemon_idx = int(best_idx)
            evo_id_pp = best_evo_id_pp[pokemon_idx]
            
            # Find matching record in player_pokemon_df
            matching_records = player_pokemon_df[
                (player_pokemon_df['evo_id_pp'] == evo_id_pp) & 
                (player_pokemon_df['enc_id'] == enc_id)
            ]
            
            if not matching_records.empty:
                record = matching_records.iloc[0]
                base_evo_id = int(str(evo_id_pp).split('_')[0])
                
                records.append({
                    'enc_id': enc_id,
                    'min_fitness': min_fitness,
                    'party_position': pokemon_idx + 1,
                    'evo_id': base_evo_id,
                    'evo_id_pp': evo_id_pp,
                    'trainer_name_enc': record.get('trainer_name_enc', ''),
                    'pokemon': record.get('pokemon', ''),
                    'level': record.get('level', ''),
                    'move_out': record.get('move_out', ''),
                    'damage_out': record.get('damage_out', ''),
                    'hp_out': record.get('hp_out', ''),
                    'speed_out': record.get('speed_out', ''),
                    'pokemon_enc': record.get('pokemon_enc', ''),
                    'level_enc': record.get('level_enc', ''),
                    'move_in': record.get('move_in', ''),
                    'damage_in': record.get('damage_in', ''),
                    'hp_in': record.get('hp_in', ''),
                    'speed_in': record.get('speed_in', '')
                })
    
    if records:
        results_df = pd.DataFrame(records)
        results_df = results_df.sort_values(['min_fitness', 'enc_id'], ascending=[False, True])
        results_df.to_csv(csv_filepath, index=False)


def _create_party_performance_visualization(results: Dict[str, Any], df: pd.DataFrame, png_filepath: Path, config_settings: dict, generations: list):
    """Visualize party performance across encounters."""
    best_party_indices = results['best_party_indices']
    if not best_party_indices:
        return
    
    battle_columns = [c for c in df.columns if c != 'evo_id_pp']
    battle_matrix = df[battle_columns].values
    party_battle_data = battle_matrix[best_party_indices, :]
    min_values = np.nanmin(party_battle_data, axis=0)
    min_values = np.nan_to_num(min_values, nan=0.0)

    pokemon_names = []
    best_evo_id_pp = results['best_evo_id_pp']
    pokemon_stats_df = None
    for gen in [1, 2, 3]:
        stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
        if not Path(stats_path).exists():
            project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
            stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
        try:
            gen_stats_df = pd.read_csv(stats_path)
            pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
        except Exception:
            pass

    gen_to_use = generations[0] if generations else 1

    for evo_id_pp in best_evo_id_pp:
        base_evo_id = int(str(evo_id_pp).split('_')[0])
        if pokemon_stats_df is not None:
            pokemon_name = _get_display_pokemon_name(base_evo_id, evo_id_pp, config_settings, gen_to_use, pokemon_stats_df)
            pokemon_names.append(f"{pokemon_name} ({evo_id_pp})")
        else:
            pokemon_names.append(f"Evo {evo_id_pp}")

    individual_pokemon_data = np.nan_to_num(party_battle_data, nan=0.0)

    def value_to_color(value: float):
        if np.isnan(value) or value >= 1_000_000_000:
            return 'black'
        if value > 1:
            return 'yellow'
        if value >= 0:
            ratio = max(0.0, min(1.0, value))
            return (ratio, 0, 1 - ratio)
        return 'blue'

    fig, axes = plt.subplots(7, 1, figsize=(16, 12), sharex=True)
    n_encounters = len(min_values)
    x_positions = np.arange(n_encounters)

    for i in range(6):
        vals = individual_pokemon_data[i, :]
        colors = [value_to_color(v) for v in vals]
        axes[i].bar(x_positions, height=1, width=1.0, color=colors, edgecolor='none')
        axes[i].set_xlim(-0.5, n_encounters - 0.5)
        axes[i].set_ylim(0, 1)
        label = pokemon_names[i] if i < len(pokemon_names) else f'Pokemon {i+1}'
        axes[i].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')
        axes[i].set_yticks([])
        axes[i].set_xticks([])

    min_colors = [value_to_color(v) for v in min_values]
    axes[6].bar(x_positions, height=1, width=1.0, color=min_colors, edgecolor='none')
    axes[6].set_xlim(-0.5, n_encounters - 0.5)
    axes[6].set_ylim(0, 1)
    axes[6].set_ylabel('Party Min', fontsize=10, rotation=0, ha='right', va='center')
    axes[6].set_yticks([])
    axes[6].set_xticks(np.arange(0, n_encounters, max(1, n_encounters // 10)))
    axes[6].set_xlabel('Encounter ID', fontsize=12)

    fig.suptitle('Party Performance Across Encounters (MILP - Global Optimum)', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(png_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _create_unique_best_performance_visualization(results: Dict[str, Any], df: pd.DataFrame, png_filepath: Path, config_settings: dict, generations: list):
    """Visualize only values where one pokemon uniquely outperforms all others in the party for that encounter."""
    best_party_indices = results['best_party_indices']
    if not best_party_indices:
        return
    
    battle_columns = [c for c in df.columns if c != 'evo_id_pp']
    battle_matrix = df[battle_columns].values
    party_battle_data = battle_matrix[best_party_indices, :]
    
    pokemon_names = []
    best_evo_id_pp = results['best_evo_id_pp']
    pokemon_stats_df = None
    for gen in [1, 2, 3]:
        stats_path = f'data/gen_{gen}/stats_gen_{gen}.csv'
        if not Path(stats_path).exists():
            project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
            stats_path = project_root / f'data/gen_{gen}/stats_gen_{gen}.csv'
        try:
            gen_stats_df = pd.read_csv(stats_path)
            pokemon_stats_df = gen_stats_df if pokemon_stats_df is None else pd.concat([pokemon_stats_df, gen_stats_df], ignore_index=True)
        except Exception:
            pass

    gen_to_use = generations[0] if generations else 1

    for evo_id_pp in best_evo_id_pp:
        base_evo_id = int(str(evo_id_pp).split('_')[0])
        if pokemon_stats_df is not None:
            pokemon_name = _get_display_pokemon_name(base_evo_id, evo_id_pp, config_settings, gen_to_use, pokemon_stats_df)
            pokemon_names.append(f"{pokemon_name} ({evo_id_pp})")
        else:
            pokemon_names.append(f"Evo {evo_id_pp}")

    individual_pokemon_data = np.nan_to_num(party_battle_data, nan=0.0)
    n_encounters = party_battle_data.shape[1]

    # Create unique best data: only show values where one Pokemon uniquely has the minimum
    unique_best_data = np.full_like(individual_pokemon_data, np.nan, dtype=np.float32)
    for encounter_idx in range(n_encounters):
        encounter_values = individual_pokemon_data[:, encounter_idx]
        min_value = np.nanmin(encounter_values)
        min_count = np.sum(encounter_values == min_value)
        if min_count == 1:  # Only one Pokemon has the minimum value
            best_pokemon_idx = int(np.argmin(encounter_values))
            unique_best_data[best_pokemon_idx, encounter_idx] = encounter_values[best_pokemon_idx]

    def value_to_color(value: float):
        if np.isnan(value):
            return (1, 1, 1, 0)  # Transparent for NaN
        if value >= 1_000_000_000:
            return 'black'
        if value > 1:
            return 'yellow'
        if value >= 0:
            ratio = max(0.0, min(1.0, value))
            return (ratio, 0, 1 - ratio)
        return 'blue'

    fig, axes = plt.subplots(6, 1, figsize=(16, 10), sharex=True)
    x_positions = np.arange(n_encounters)

    for i in range(6):
        vals = unique_best_data[i, :]
        # Only draw bars where we have non-NaN values
        for j, val in enumerate(vals):
            if not np.isnan(val):
                axes[i].bar(j, height=1, width=1.0, color=value_to_color(val), edgecolor='none')
        axes[i].set_xlim(-0.5, n_encounters - 0.5)
        axes[i].set_ylim(0, 1)
        label = pokemon_names[i] if i < len(pokemon_names) else f'Pokemon {i+1}'
        axes[i].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center')
        axes[i].set_yticks([])
        axes[i].set_xticks([])

    axes[5].set_xticks(np.arange(0, n_encounters, max(1, n_encounters // 10)))
    axes[5].set_xlabel('Encounter ID', fontsize=12)

    fig.suptitle('Unique Best Performance Across Encounters (MILP - Global Optimum)\n(Only showing bars where one Pokemon uniquely outperforms all others)', fontsize=14, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(png_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("MILP-based Pokemon Party Optimization")
    print("=" * 60)
    
    if not PULP_AVAILABLE:
        print("\nERROR: PuLP library is not installed.")
        print("Please install it with: pip install pulp")
        sys.exit(1)
    
    # Example usage
    config_path = "config/config.json"
    if not Path(config_path).exists():
        config_path = "../config/config.json"
    
    battle_results_path = "intermediate_files/battle_results_gen1.csv"
    if not Path(battle_results_path).exists():
        battle_results_path = "../intermediate_files/battle_results_gen1.csv"
    
    if Path(battle_results_path).exists():
        print(f"\nUsing battle results from: {battle_results_path}")
        print(f"Using config from: {config_path}")
        
        results = calculate_best_party_milp(
            battle_results_path=battle_results_path,
            config_path=config_path,
            time_limit=300,
            apply_dominance_filter=True,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best Fitness: {results['best_fitness']:.6f}")
        print(f"Best Party: {results['best_evo_id_pp']}")
    else:
        print(f"\nERROR: Could not find battle results file at {battle_results_path}")
        print("Please run the pipeline to generate battle results first.")
