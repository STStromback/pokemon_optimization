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


def read_restricted_combinations(restriction_file):
    """
    Reads a configuration file for restricted variant_id combinations.
    Each line in the file should list restricted variant_id values separated by commas.
    For example:
    1,2,3
    4,5
    """
    restricted_combinations = []
    with open(restriction_file, 'r') as f:
        for line in f:
            restricted_combinations.append(set(map(int, line.strip().split(','))))
    return restricted_combinations


def is_valid_selection(selected_ids, restricted_combinations):
    """
    Checks if the selected variant_id values are valid based on the restricted combinations.
    """
    for restriction in restricted_combinations:
        if len(selected_ids.intersection(restriction)) > 1:
            return False
    return True


def greedy_selection_with_restrictions(df, n, restricted_combinations):
    """
    Greedy selection with restrictions on variant_id combinations.
    """
    selected_indices = []
    remaining_indices = list(df.index)
    data = df.iloc[:, 1:].values
    current_minima = np.full(data.shape[1], np.inf)
    selected_ids = set()

    for _ in range(n):
        best_reduction = -np.inf
        best_index = None

        for idx in remaining_indices:
            candidate_id = df.loc[idx, 'variant_id']
            candidate_set = selected_ids.union({candidate_id})

            # Skip this candidate if it violates restrictions
            if not is_valid_selection(candidate_set, restricted_combinations):
                continue

            new_minima = np.minimum(current_minima, data[idx])
            reduction = np.sum(current_minima) - np.sum(new_minima)
            if reduction > best_reduction:
                best_reduction = reduction
                best_index = idx

        if best_index is not None:
            selected_indices.append(best_index)
            remaining_indices.remove(best_index)
            current_minima = np.minimum(current_minima, data[best_index])
            selected_ids.add(df.loc[best_index, 'variant_id'])
        else:
            # No valid selections remaining
            break

    return df.iloc[selected_indices]


def calculate_min_sum(selected_rows):
    # Calculate the element-wise minima across the selected rows
    min_values = np.min(selected_rows, axis=0)
    # Sum the minima to get the overall value
    total_min_sum = np.sum(min_values)
    return total_min_sum, min_values


def get_final_evolution_names(stats_file, variants_file):
    # Read stats and variants files
    stats_df = pd.read_csv(stats_file)
    variants_df = pd.read_csv(variants_file)

    # Find the last Pokémon name for each evo_id
    last_evolutions = stats_df.groupby('evo_id').last().reset_index()

    # Merge with variants to get variant_id
    last_evolutions = last_evolutions.merge(variants_df[['evo_id', 'variant_id']], on='evo_id', how='left')
    return last_evolutions[['variant_id', 'pokemon']]


def main(file_path='ehl_pivot.csv', stats_file='stats_gen_1.csv', variants_file='variants_gen_1.csv',
         restriction_file='restrictions_gen_1.txt', n=6):
    df = read_data(file_path)
    restricted_combinations = read_restricted_combinations(restriction_file) if restriction_file else []

    selected_rows = greedy_selection_with_restrictions(df, n, restricted_combinations)
    selected_rows = selected_rows.sort_values(by='variant_id').reset_index(drop=True)

    team_value, _ = calculate_min_sum(selected_rows.iloc[:, 1:])
    final_evolutions = get_final_evolution_names(stats_file, variants_file)
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
    print("\nSummary saved to 'final_summary_gen_1.txt'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the best combination of variant_id options with restrictions.')
    parser.add_argument('--file_path', type=str, default='ehl_pivot.csv', help='Path to the input CSV file.')
    parser.add_argument('--stats_file', type=str, default='stats_gen_1.csv', help='Path to the stats CSV file.')
    parser.add_argument('--variants_file', type=str, default='variants_gen_1.csv',
                        help='Path to the variants CSV file.')
    parser.add_argument('--restriction_file', type=str, default=None,
                        help='Path to the restriction configuration file.')
    parser.add_argument('--n', type=int, default=6, help='Number of variant_id options to select.')
    args = parser.parse_args()

    main(file_path=args.file_path, stats_file=args.stats_file, variants_file=args.variants_file,
         restriction_file=args.restriction_file, n=args.n)
