import pandas as pd
import numpy as np
import argparse


def read_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Ensure all columns except 'variant_id' are numeric
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with a large number (since larger values are worse)
    df.fillna(df.max().max() + 1, inplace=True)
    return df


def calculate_min_sum(selected_rows):
    # Calculate the element-wise minima across the selected rows
    min_values = np.min(selected_rows, axis=0)
    # Sum the minima to get the overall value
    total_min_sum = np.sum(min_values)
    return total_min_sum, min_values


def greedy_selection(df, n):
    # Initialize selected indices and remaining indices
    selected_indices = []
    remaining_indices = list(df.index)
    # Exclude 'variant_id' column for calculations
    data = df.iloc[:, 1:].values
    total_columns = data.shape[1]

    # Initialize current minima as infinity
    current_minima = np.full(data.shape[1], np.inf)

    for _ in range(n):
        best_reduction = -np.inf
        best_index = None

        for idx in remaining_indices:
            # Calculate new minima if this row is added
            new_minima = np.minimum(current_minima, data[idx])
            # Calculate reduction in sum of minima
            reduction = np.sum(current_minima) - np.sum(new_minima)
            if reduction > best_reduction:
                best_reduction = reduction
                best_index = idx

        if best_index is not None:
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
            # Update current minima
            current_minima = np.minimum(current_minima, data[best_index])
        else:
            # No improvement possible
            break

    # Return the selected rows
    return df.iloc[selected_indices]


def get_final_evolution_names(stats_file, variants_file):
    # Read stats and variants files
    stats_df = pd.read_csv(stats_file)
    variants_df = pd.read_csv(variants_file)

    # Find the last Pokémon name for each evo_id
    last_evolutions = stats_df.groupby('evo_id').last().reset_index()

    # Merge with variants to get variant_id
    last_evolutions = last_evolutions.merge(variants_df[['evo_id', 'variant_id']], on='evo_id', how='left')
    return last_evolutions[['variant_id', 'pokemon']]


def main(file_path='ehl_pivot.csv', stats_file='stats_gen_1.csv', variants_file='variants_gen_1.csv', n=6):
    df = read_data(file_path)
    selected_rows = greedy_selection(df, n)

    # Sort by variant_id
    selected_rows = selected_rows.sort_values(by='variant_id').reset_index(drop=True)

    # Get the team value and min values
    team_value, _ = calculate_min_sum(selected_rows.iloc[:, 1:])

    # Get final evolution names
    final_evolutions = get_final_evolution_names(stats_file, variants_file)

    # Merge the selected rows with final evolution names
    result = selected_rows.merge(final_evolutions, on='variant_id', how='left')

    # Prepare output text
    output_lines = ["Team Composition:"]
    for _, row in result.iterrows():
        output_lines.append(f"Variant ID: {row['variant_id']}, Pokemon: {row['pokemon']}")
    output_lines.append(f"\nTeam Value (Sum of Minimum Values): {team_value}")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to txt file
    with open('../gen_1_output/final_summary_gen_1.txt', 'w') as f:
        f.write("\n".join(output_lines))

    print("\nSummary saved to 'team_composition_output.txt'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the best combination of variant_id options.')
    parser.add_argument('--file_path', type=str, default='ehl_pivot.csv', help='Path to the input CSV file.')
    parser.add_argument('--stats_file', type=str, default='stats_gen_1.csv', help='Path to the stats CSV file.')
    parser.add_argument('--variants_file', type=str, default='variants_gen_1.csv',
                        help='Path to the variants CSV file.')
    parser.add_argument('--n', type=int, default=6, help='Number of variant_id options to select.')
    args = parser.parse_args()

    main(file_path=args.file_path, stats_file=args.stats_file, variants_file=args.variants_file, n=args.n)
