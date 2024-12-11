import pandas as pd
import numpy as np
import random
import argparse
from deap import base, creator, tools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, ListedColormap
import os


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
            restriction = set(map(int, line.strip().split(',')))
            restricted_combinations.append(restriction)
    return restricted_combinations


def is_invalid_combination(individual, invalid_combinations):
    individual_set = set(individual)
    for invalid_set in invalid_combinations:
        if len(individual_set.intersection(invalid_set)) > 1:
            return True
    return False


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
         restriction_file=None, n=6):
    # Load data
    filtered_df = read_data(file_path)
    # Filter out first three encounters (optional battle)
    filtered_df.drop(['enc_id_1','enc_id_2','enc_id_3'], axis=1, inplace=True)
    variant_ids = filtered_df['variant_id'].tolist()

    # Load variants file for mapping evo_id to variant_id
    variants_df = pd.read_csv(variants_file)

    # Define invalid combinations
    invalid_evo_ids = []
    if restriction_file:
        invalid_evo_ids = read_restricted_combinations(restriction_file)

    # Map evo_id to variant_id
    evo_id_to_variant_ids = variants_df.groupby('evo_id')['variant_id'].apply(set).to_dict()
    invalid_combinations = [set().union(*(evo_id_to_variant_ids.get(evo_id, set()) for evo_id in invalid_set)) for
                            invalid_set in invalid_evo_ids]

    # Define fitness and individual classes using DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_pokemon", random.choice, variant_ids)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pokemon, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness caching to avoid redundant evaluations
    fitness_cache = {}

    def evaluate(individual):
        individual_tuple = tuple(sorted(individual))
        if individual_tuple in fitness_cache:
            return fitness_cache[individual_tuple]

        if is_invalid_combination(individual, invalid_combinations):
            fitness_cache[individual_tuple] = (float('inf'),)  # Assign a high penalty value for invalid combinations
            return fitness_cache[individual_tuple]

        team_data = filtered_df[filtered_df['variant_id'].isin(individual)].iloc[:, 1:]
        if team_data.empty or len(team_data) < n:
            # Handle cases where the individual does not correspond to valid data
            fitness_cache[individual_tuple] = (float('inf'),)
            return fitness_cache[individual_tuple]

        min_values = team_data.min(axis=0)
        team_value = min_values.sum()

        fitness_cache[individual_tuple] = (team_value,)
        return fitness_cache[individual_tuple]

    def custom_mutate(individual):
        mutation_rate = 0.2  # Adjust mutation rate as needed
        if random.random() < mutation_rate:
            index_to_replace = random.randrange(len(individual))
            individual[index_to_replace] = toolbox.attr_pokemon()
        return individual,

    def custom_mate(ind1, ind2):
        tools.cxUniform(ind1, ind2, indpb=0.5)
        return ind1, ind2

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", custom_mate)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def run_genetic_algorithm():
        population_size = 1000  # Adjust population size as needed
        generations = 100  # Adjust the number of generations as needed
        mutation_rate = 0.2  # Adjust mutation rate as needed

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        elite_size = int(0.05 * population_size)  # Percentage of elite individuals

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.mean([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))
        stats.register("min", lambda x: np.min([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))
        stats.register("max", lambda x: np.max([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields

        # Evaluate the initial population's fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for gen in range(generations):
            offspring = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update population
            pop[:] = tools.selBest(pop, elite_size) + offspring
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if gen % 10 == 0:
                print(logbook.stream)

        best_team = tools.selBest(pop, 1)[0]
        best_team_fitness = evaluate(best_team)
        return best_team, best_team_fitness, logbook

    # Run the genetic algorithm
    best_team, best_team_fitness, logbook = run_genetic_algorithm()

    # Get the selected rows
    best_variant_ids = best_team
    selected_rows = filtered_df[filtered_df['variant_id'].isin(best_variant_ids)].sort_values(
        by='variant_id').reset_index(drop=True)

    # Save best_variant_ids to a CSV file for use in the heatmap generation
    best_variant_ids_df = pd.DataFrame({'variant_id': best_variant_ids})
    best_variant_ids_df.to_csv('best_variant_ids.csv', index=False)

    # Get final evolution names
    final_evolutions = get_final_evolution_names(stats_file, variants_file)

    # Merge the selected rows with final evolution names
    result = selected_rows.merge(final_evolutions, on='variant_id', how='left')

    # Prepare output text
    output_lines = ["Team Composition:"]
    for _, row in result.iterrows():
        output_lines.append(f"Variant ID: {row['variant_id']}, Pokémon: {row['pokemon']}")
    output_lines.append(f"\nTeam Value (Sum of Minimum Values): {best_team_fitness[0]}")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to txt file
    with open('../gen_1_output/final_output_gen_1.txt', 'w') as f:
        f.write("\n".join(output_lines))
    print("\nSummary saved to 'final_output_gen_1.txt'.")

    # Plot the results
    plot_results(logbook)

    # Generate additional outputs (heatmaps and worst encounters)
    generate_additional_outputs(filtered_df, best_variant_ids, stats_file, variants_file)


def plot_results(logbook):
    gen = logbook.select("gen")
    min_fits = logbook.select("min")
    avg_fits = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, min_fits, label="Minimum Fitness", marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("fitness_over_generations.png")
    plt.show()


def generate_additional_outputs(filtered_df, best_variant_ids, stats_file, variants_file):
    # Generate heatmaps and identify worst encounters
    print("\nGenerating additional outputs...")

    # Load 'ehl_pivot.csv' to get the list of 'enc_id's
    ehl_pivot_df = pd.read_csv('../gen_1_data_curated/ehl_pivot.csv')
    # Drop first three columns (optional battles)
    ehl_pivot_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)


    # Extract 'enc_id' columns
    enc_id_columns = [col for col in ehl_pivot_df.columns if col.startswith('enc_id_')]

    # Extract 'enc_id's from the column names
    enc_id_order = [int(col.replace('enc_id_', '')) for col in enc_id_columns]

    # Sort the 'enc_id_columns' based on 'enc_id_order'
    enc_id_columns_sorted = ['enc_id_' + str(enc_id) for enc_id in sorted(enc_id_order)]

    # Load the matrix file (ehl_pivot.csv)
    matrix_df = pd.read_csv('../gen_1_data_curated/ehl_pivot.csv')
    matrix_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)

    # Ensure 'variant_id' is included in sorted columns
    sorted_columns = ['variant_id'] + enc_id_columns_sorted

    # Filter the matrix DataFrame to only include sorted columns
    sorted_matrix_df = matrix_df[sorted_columns]

    # Save the sorted matrix to a new CSV file
    sorted_matrix_df.to_csv('sorted_matrix1.csv', index=False)

    print("The columns of the matrix have been sorted and saved to 'sorted_matrix1.csv'.")

    # Load the sorted matrix and filter for best variant_ids
    matrix = pd.read_csv('../gen_1_data_curated/sorted_matrix1.csv')
    matrix = matrix[matrix['variant_id'].isin(best_variant_ids)]

    # Load variants and stats files
    variants_df = pd.read_csv(variants_file)
    stats_df = pd.read_csv(stats_file)

    # Map 'variant_id' to 'evo_id' using variants_df
    variant_to_evo = variants_df.set_index('variant_id')['evo_id'].to_dict()

    # Map 'evo_id' to Pokémon name using stats_df
    evo_to_pokemon = {}
    stats_df['evo_lvl'].fillna(0, inplace=True)
    stats_df['evo_stage'].fillna(0, inplace=True)
    for evo_id, group in stats_df.groupby('evo_id'):
        # Choose the Pokémon with the highest evo_lvl <= 100, or highest evo_stage if no evo_lvl
        group = group[group['evo_lvl'] <= 100]
        if len(group) == 1:
            pokemon = group.sort_values(by='evo_lvl', ascending=False).iloc[0]['pokemon']
        elif group.evo_lvl.value_counts()[group['evo_lvl'].max()] > 1:
            if group.evo_stage.value_counts()[group['evo_stage'].max()] > 1:
                pokemon = group.iloc[-1]['pokemon']
            else:
                pokemon = group.sort_values(by='evo_stage', ascending=False).iloc[0]['pokemon']
        else:
            pokemon = group.sort_values(by='evo_lvl', ascending=False).iloc[0]['pokemon']
        evo_to_pokemon[evo_id] = pokemon

    # Convert variant_ids to Pokémon names
    matrix['variant_id'] = matrix['variant_id'].map(lambda x: evo_to_pokemon.get(variant_to_evo.get(x, ''), 'Unknown'))

    # Set the variant_id as the index for better visualization in the heatmap
    matrix.set_index('variant_id', inplace=True)

    # Normalize the values such that any value greater than 1 is set to 1
    matrix_normalized = matrix.applymap(lambda x: 1 if x >= 1 else x)

    # Calculate the minimum value out of the rows for each encounter
    min_values = matrix_normalized.min(axis=0)
    min_values.name = 'Min Value'

    # Append the minimum values row to the matrix
    matrix_normalized = matrix_normalized._append(min_values)

    # Create a colormap for the normalized values with red for values exactly 1
    cmap = sns.color_palette("viridis", as_cmap=True)
    norm = Normalize(vmin=0, vmax=1)
    colors = cmap(norm(np.linspace(0, 1, cmap.N)))
    colors[255] = [1, 0, 0, 1]  # Red color for the value 1
    new_cmap = ListedColormap(colors)

    # Define key encounter points
    key_encounters = {
        17: 'Brock',
        88: 'Misty',
        186: 'Lt. Surge',
        367: 'Erika',
        662: 'Koga',
        494: 'Sabrina',
        776: 'Blaine',
        795: 'Giovanni',
        836: 'Elite Four'
    }
    key_enc_ids = list(key_encounters.keys())
    key_labels = list(key_encounters.values())

    # Create the original heatmap
    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust the figure size as needed
    sns.heatmap(matrix_normalized, cmap=new_cmap, cbar=True, annot=False, xticklabels=False, ax=ax)
    plt.title('Heatmap of Encounter Values for Selected Pokémon Variants')
    plt.xlabel('Encounters')
    plt.ylabel('Pokémon')

    # Add custom x-axis labels for key encounters
    xticks_positions = []
    for enc_id in key_enc_ids:
        col_name = f'enc_id_{enc_id}'
        if col_name in matrix_normalized.columns:
            xticks_positions.append(matrix_normalized.columns.get_loc(col_name))
        else:
            continue
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')

    # Add a colorbar with a label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized EHL Values')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the original heatmap
    plt.savefig('pokemon_encounter_heatmap_normalized_with_min_gen1.png')
    plt.show()

    # Create a DataFrame to store unique minimum values
    unique_min_matrix = matrix.copy()

    # Filter for unique minimum values
    for col in matrix.columns:
        min_val = matrix[col].min()
        if (matrix[col] == min_val).sum() == 1:
            unique_min_matrix[col] = np.where(matrix[col] == min_val, matrix[col], np.nan)
        else:
            unique_min_matrix[col] = np.nan

    # Create the unique minimum heatmap
    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust the figure size as needed
    sns.heatmap(unique_min_matrix, cmap=new_cmap, cbar=True, annot=False, xticklabels=False, ax=ax, norm=norm)
    plt.title('Heatmap of Unique Minimum Encounter Values for Selected Pokémon Variants')
    plt.xlabel('Encounters')
    plt.ylabel('Pokémon')

    # Add custom x-axis labels for key encounters
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')

    # Add a colorbar with a label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized EHL Values')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the unique minimum heatmap
    plt.savefig('pokemon_unique_min_encounter_heatmap_gen1.png')
    plt.show()

    # Create a DataFrame to store all minimum values, including those shared by multiple Pokémon
    all_min_matrix = matrix.copy()

    # Filter for all minimum values
    for col in matrix.columns:
        min_val = matrix[col].min()
        all_min_matrix[col] = np.where(matrix[col] == min_val, matrix[col], np.nan)

    # Create the all minimum heatmap
    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust the figure size as needed
    sns.heatmap(all_min_matrix, cmap=new_cmap, cbar=True, annot=False, xticklabels=False, ax=ax, norm=norm)
    plt.title('Heatmap of All Minimum Encounter Values for Selected Pokémon Variants')
    plt.xlabel('Encounters')
    plt.ylabel('Pokémon')

    # Add custom x-axis labels for key encounters
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')

    # Add a colorbar with a label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized EHL Values')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Save the all minimum heatmap
    plt.savefig('pokemon_all_min_encounter_heatmap_gen1.png')
    plt.show()

    # Identify the top 100 worst encounters based on the minimum EHL values
    min_values_no_min = min_values.drop('Min Value', errors='ignore')  # Remove 'Min Value' if present
    worst_encounters = min_values_no_min.nlargest(100)

    # Output the worst encounters to a CSV file
    worst_encounters_df = worst_encounters.reset_index()
    worst_encounters_df.columns = ['enc_id', 'min_ehl']
    worst_encounters_df.to_csv('worst_encounters_gen1.csv', index=False)

    # Display the worst encounters
    print("Top 100 worst encounters based on minimum EHL values:")
    print(worst_encounters_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find the best combination of variant_id options using a genetic algorithm.')
    parser.add_argument('--file_path', type=str, default='ehl_pivot.csv', help='Path to the input CSV file.')
    parser.add_argument('--stats_file', type=str, default='../gen_1_data_raw/stats_gen_1.csv', help='Path to the stats CSV file.')
    parser.add_argument('--variants_file', type=str, default='variants_gen_1.csv',
                        help='Path to the variants CSV file.')
    parser.add_argument('--restriction_file', type=str, default='../gen_1_config/restrictions_gen_1.txt',
                        help='Path to the restriction configuration file.')
    parser.add_argument('--n', type=int, default=6, help='Number of variant_id options to select.')
    args = parser.parse_args()

    main(file_path=args.file_path, stats_file=args.stats_file, variants_file=args.variants_file,
         restriction_file=args.restriction_file, n=args.n)
