"""
Plot exp_enc (experience encounter) values over time.

This program:
1. Generates encounters dataframe following the same steps as main.py
2. Visualizes both original exp_enc and sequential_max exp_enc values
3. X-axis: enc_id
4. Y-axis: exp values
5. Marks first enc_id for stages with "gym", "league", or "plateau" in location_enc
6. Saves plots to results folder with timestamp
7. Processes each generation defined in config.json
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add source directory to path for imports
sys.path.append(str(Path(__file__).parent))

from generate_encounters import (
    load_trainer_data,
    apply_location_adjustments,
    add_stages,
    calc_enc_stats,
    calc_enc_moves,
    add_encounter_moves,
    add_exp_enc,
    add_enc_id,
    add_evo_info_enc,
    _apply_power_adjustments,
    normalize_text_columns
)


def load_config():
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using defaults.")
        return {"gen": [1, 2, 3], "level_calc_method (sequential_max/independent)": "sequential_max"}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using defaults.")
        return {"gen": [1, 2, 3], "level_calc_method (sequential_max/independent)": "sequential_max"}


def get_generations_to_process(config_gen):
    """Determine which generations to process based on config."""
    if config_gen == "all":
        return [1, 2, 3]
    elif isinstance(config_gen, list):
        # Handle list input (e.g., [1, 2, 3] from config.json)
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


def generate_encounters_with_original_exp(gen: int) -> pd.DataFrame:
    """
    Generate encounter data with ORIGINAL exp_enc values (before sequential_max).
    
    Args:
        gen (int): Pokemon generation (1, 2, or 3)
        
    Returns:
        pd.DataFrame: Encounter data with original exp_enc values
    """
    print(f'  - Generating encounters with original exp_enc for gen {gen}')
    
    df = load_trainer_data(gen)
    df = apply_location_adjustments(df)
    df = add_stages(gen, df)
    df = calc_enc_stats(gen, df)
    enc_moves = calc_enc_moves(gen, df)
    df = add_encounter_moves(gen, df, enc_moves)
    df = _apply_power_adjustments(df)
    df = add_exp_enc(df)
    
    # Do NOT apply sequential_max here - we want original values
    df = add_enc_id(df)
    df = add_evo_info_enc(gen, df)
    
    # Filter out duplicates
    duplicate_columns = [
        'pokemon_enc', 'stage_enc', 'level_enc', 
        'move_name_1_enc', 'move_name_2_enc', 'move_name_3_enc', 'move_name_4_enc'
    ]
    df = df.drop_duplicates(subset=duplicate_columns, keep='first')
    
    return df


def apply_sequential_max_to_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sequential_max logic to encounters dataframe by modifying exp_enc values.
    For each row, exp_enc becomes the maximum of all exp_enc values at or before that stage.
    
    Args:
        df: DataFrame with original exp_enc values
        
    Returns:
        DataFrame with sequential_max exp_enc values
    """
    print("  - Applying sequential_max logic...")
    
    df = df.copy()
    
    # Sort by stage_enc and exp_enc to ensure proper order
    df = df.sort_values(['stage_enc', 'exp_enc'], ascending=[True, True]).reset_index(drop=True)
    
    # Apply sequential max logic: each exp_enc becomes max of all previous exp_enc at or before that stage
    max_exp_enc = 0
    for idx in df.index:
        current_exp_enc = df.loc[idx, 'exp_enc']
        if pd.notna(current_exp_enc):
            # Take max of previous max_exp_enc vs current exp_enc
            max_exp_enc = max(max_exp_enc, current_exp_enc)
            # Update exp_enc to reflect the sequential max
            df.loc[idx, 'exp_enc'] = max_exp_enc
    
    return df


def identify_milestone_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify first enc_id for stages with "gym", "league", or "plateau" in location_enc,
    and special trainer encounters (Red, rivals) in trainer_name_enc.
    
    Args:
        df: Encounters dataframe
        
    Returns:
        DataFrame of milestone encounters with enc_id, label, stage_enc, and exp_enc
    """
    # Filter for milestone locations
    location_keywords = ['gym', 'league', 'plateau']
    
    def is_location_milestone(location):
        if pd.isna(location):
            return False
        location_str = str(location).lower()
        return any(keyword in location_str for keyword in location_keywords)
    
    def get_trainer_label(trainer_name):
        """Check if trainer is a special encounter and return label."""
        if pd.isna(trainer_name):
            return None
        trainer_str = str(trainer_name).lower()
        
        # Check for Red
        if 'pokemon_trainer_red' in trainer_str:
            return 'Red'
        
        # Check for rival (green or rival)
        if 'green' in trainer_str or 'rival' in trainer_str:
            return 'rival'
        
        return None
    
    # Add columns for milestone detection
    df = df.copy()
    df['is_location_milestone'] = df['location_enc'].apply(is_location_milestone)
    df['trainer_label'] = df['trainer_name_enc'].apply(get_trainer_label)
    df['is_trainer_milestone'] = df['trainer_label'].notna()
    
    # Combine both types of milestones
    df['is_milestone'] = df['is_location_milestone'] | df['is_trainer_milestone']
    
    # Create label column: use trainer_label if available, otherwise use location_enc
    df['milestone_label'] = df.apply(
        lambda row: row['trainer_label'] if pd.notna(row['trainer_label']) else row['location_enc'],
        axis=1
    )
    
    milestone_df = df[df['is_milestone']].copy()
    
    if len(milestone_df) == 0:
        print("  - No milestone encounters found")
        return pd.DataFrame(columns=['enc_id', 'milestone_label', 'stage_enc', 'exp_enc'])
    
    # Get first enc_id for each stage that has milestones
    milestone_df = milestone_df.sort_values(['stage_enc', 'enc_id'])
    milestone_df = milestone_df.groupby('stage_enc').first().reset_index()
    
    print(f"  - Found {len(milestone_df)} milestone encounters")
    return milestone_df[['enc_id', 'milestone_label', 'stage_enc', 'exp_enc']]


def plot_exp_enc_over_time(gen: int, df_original: pd.DataFrame, df_sequential_max: pd.DataFrame, 
                           milestone_df: pd.DataFrame, output_path: Path):
    """
    Create visualization of exp_enc values over enc_id.
    
    Args:
        gen: Pokemon generation
        df_original: DataFrame with original exp_enc values
        df_sequential_max: DataFrame with sequential_max exp_enc values
        milestone_df: DataFrame of milestone encounters
        output_path: Path to save the plot
    """
    print(f"  - Creating visualization for gen {gen}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot original exp_enc
    ax.plot(df_original['enc_id'], df_original['exp_enc'], 
            label='Original exp_enc', color='steelblue', linewidth=1.5, alpha=0.7)
    
    # Plot sequential_max exp_enc
    ax.plot(df_sequential_max['enc_id'], df_sequential_max['exp_enc'], 
            label='Sequential Max exp_enc', color='darkred', linewidth=2, alpha=0.8)
    
    # Mark milestone encounters
    if len(milestone_df) > 0:
        for _, row in milestone_df.iterrows():
            enc_id = row['enc_id']
            label = row['milestone_label']
            exp_value = row['exp_enc']
            
            # Draw vertical line
            ax.axvline(x=enc_id, color='green', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add label at the top of the plot
            ax.text(enc_id, ax.get_ylim()[1], label, 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, color='green', alpha=0.7)
    
    # Labels and title
    ax.set_ylabel('Experience Value (exp_enc)', fontsize=12)
    ax.set_title(f'Pokemon Generation {gen}: Experience Progression Over Encounters',
                fontsize=14, fontweight='bold')
    
    # Remove x-axis tick labels
    ax.tick_params(axis='x', labelbottom=False)
    
    # Legend (positioned halfway down the left side)
    ax.legend(loc='center left', fontsize=10, bbox_to_anchor=(0, 0.5))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  - Saved plot to: {output_path}")


def plot_exp_enc_over_time_log(gen: int, df_original: pd.DataFrame, df_sequential_max: pd.DataFrame, 
                               milestone_df: pd.DataFrame, output_path: Path):
    """
    Create visualization of exp_enc values over enc_id with LOG SCALE on y-axis.
    
    Args:
        gen: Pokemon generation
        df_original: DataFrame with original exp_enc values
        df_sequential_max: DataFrame with sequential_max exp_enc values
        milestone_df: DataFrame of milestone encounters
        output_path: Path to save the plot
    """
    print(f"  - Creating log-scale visualization for gen {gen}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot original exp_enc
    ax.plot(df_original['enc_id'], df_original['exp_enc'], 
            label='Original exp_enc', color='steelblue', linewidth=1.5, alpha=0.7)
    
    # Plot sequential_max exp_enc
    ax.plot(df_sequential_max['enc_id'], df_sequential_max['exp_enc'], 
            label='Sequential Max exp_enc', color='darkred', linewidth=2, alpha=0.8)
    
    # Mark milestone encounters
    if len(milestone_df) > 0:
        for _, row in milestone_df.iterrows():
            enc_id = row['enc_id']
            label = row['milestone_label']
            exp_value = row['exp_enc']
            
            # Draw vertical line
            ax.axvline(x=enc_id, color='green', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add label at the top of the plot
            ax.text(enc_id, ax.get_ylim()[1], label, 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, color='green', alpha=0.7)
    
    # Labels and title
    ax.set_ylabel('Experience Value (exp_enc) - Log Scale', fontsize=12)
    ax.set_title(f'Pokemon Generation {gen}: Experience Progression Over Encounters (Log Scale)',
                fontsize=14, fontweight='bold')
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Remove x-axis tick labels
    ax.tick_params(axis='x', labelbottom=False)
    
    # Legend (positioned in bottom right for log scale)
    ax.legend(loc='lower right', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  - Saved log-scale plot to: {output_path}")


def plot_exp_enc_difference(gen: int, df_original: pd.DataFrame, df_sequential_max: pd.DataFrame, 
                           milestone_df: pd.DataFrame, output_path: Path):
    """
    Create visualization of the DIFFERENCE between sequential_max and original exp_enc.
    
    Args:
        gen: Pokemon generation
        df_original: DataFrame with original exp_enc values
        df_sequential_max: DataFrame with sequential_max exp_enc values
        milestone_df: DataFrame of milestone encounters
        output_path: Path to save the plot
    """
    print(f"  - Creating difference plot for gen {gen}...")
    
    # Calculate difference - ensure proper alignment by enc_id
    df_diff = df_original[['enc_id', 'exp_enc']].copy()
    df_diff = df_diff.rename(columns={'exp_enc': 'exp_original'})
    df_diff['exp_sequential_max'] = df_sequential_max['exp_enc'].values
    df_diff['exp_diff'] = df_diff['exp_sequential_max'] - df_diff['exp_original']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot difference
    ax.plot(df_diff['enc_id'], df_diff['exp_diff'], 
            label='Sequential Max - Original', color='purple', linewidth=2, alpha=0.8)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Mark milestone encounters
    if len(milestone_df) > 0:
        for _, row in milestone_df.iterrows():
            enc_id = row['enc_id']
            label = row['milestone_label']
            
            # Draw vertical line
            ax.axvline(x=enc_id, color='green', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add label at the top of the plot
            ax.text(enc_id, ax.get_ylim()[1], label, 
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, color='green', alpha=0.7)
    
    # Labels and title
    ax.set_ylabel('Experience Difference (Sequential Max - Original)', fontsize=12)
    ax.set_title(f'Pokemon Generation {gen}: Difference between Sequential Max and Original exp_enc',
                fontsize=14, fontweight='bold')
    
    # Invert y-axis so larger values go down
    ax.invert_yaxis()
    
    # Remove x-axis tick labels
    ax.tick_params(axis='x', labelbottom=False)
    
    # Legend (positioned halfway down the left side)
    ax.legend(loc='center left', fontsize=10, bbox_to_anchor=(0, 0.5))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  - Saved difference plot to: {output_path}")


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    generations = get_generations_to_process(config.get("gen", [1, 2, 3]))
    
    print(f"=== Generating Experience Progression Plots ===")
    print(f"Processing generations: {generations}\n")
    
    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    for gen in generations:
        print(f"Processing Generation {gen}...")
        
        # Step 1: Generate encounters with original exp_enc
        df_original = generate_encounters_with_original_exp(gen)
        print(f"  - Generated {len(df_original)} encounters with original exp_enc")
        
        # Step 2: Apply sequential_max to get modified exp_enc
        df_sequential_max = apply_sequential_max_to_encounters(df_original.copy())
        print(f"  - Applied sequential_max logic")
        
        # Step 3: Identify milestone encounters
        milestone_df = identify_milestone_encounters(df_original)
        
        # Step 4: Create visualizations (3 plots per generation)
        
        # Plot 1: Linear scale
        output_filename = f"exp_enc_progression_gen{gen}_{timestamp}.png"
        output_path = results_dir / output_filename
        plot_exp_enc_over_time(gen, df_original, df_sequential_max, milestone_df, output_path)
        
        # Plot 2: Log scale
        output_filename_log = f"exp_enc_progression_gen{gen}_{timestamp}_log.png"
        output_path_log = results_dir / output_filename_log
        plot_exp_enc_over_time_log(gen, df_original, df_sequential_max, milestone_df, output_path_log)
        
        # Plot 3: Difference
        output_filename_diff = f"exp_enc_progression_gen{gen}_{timestamp}_difference.png"
        output_path_diff = results_dir / output_filename_diff
        plot_exp_enc_difference(gen, df_original, df_sequential_max, milestone_df, output_path_diff)
        
        # Print statistics
        print(f"\n  Statistics for Generation {gen}:")
        print(f"  - Total encounters: {len(df_original)}")
        print(f"  - Original exp_enc range: {df_original['exp_enc'].min():,.0f} - {df_original['exp_enc'].max():,.0f}")
        print(f"  - Sequential max exp_enc range: {df_sequential_max['exp_enc'].min():,.0f} - {df_sequential_max['exp_enc'].max():,.0f}")
        print(f"  - Milestone encounters marked: {len(milestone_df)}")
        print()
    
    print(f"=== All generations processed successfully! ===")
    print(f"Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
