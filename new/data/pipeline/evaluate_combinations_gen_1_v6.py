import pandas as pd
import numpy as np
import random
import argparse
from deap import base, creator, tools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, ListedColormap
import os
import matplotlib

matplotlib.use('Agg')

config = pd.read_csv('../config/config.csv')
gen = int(config[config.rule == 'gen'].value.values[0])

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

    config = pd.read_csv('../config/config.csv')
    restrictions_flag = True if config[config.rule == 'restrictions'].value.values[0].lower() == 'y' else False

    if restrictions_flag:
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
    # Read stats and variants files with correct encoding
    stats_df = pd.read_csv(stats_file, encoding='utf-8')
    variants_df = pd.read_csv(variants_file, encoding='utf-8')

    # Find the last Pokémon for each evo_id
    last_evolutions = stats_df.groupby('evo_id').last().reset_index()

    # Merge with variants to get variant_id and move types
    last_evolutions = last_evolutions.merge(
        variants_df[['evo_id', 'variant_id', 'move_type_1', 'move_type_2', 'move_type_3', 'move_type_4']],
        on='evo_id', how='left'
    )

    # Select the necessary columns
    return last_evolutions[['variant_id', 'pokemon', 'move_type_1', 'move_type_2', 'move_type_3', 'move_type_4']]


def main(file_path=f'../data_curated/data_curated_gen_{gen}/ehl_pivot_gen_{gen}.csv', stats_file=f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv',
         variants_file=f'../data_curated/data_curated_gen_{gen}/variants_gen_{gen}.csv', restriction_file=None):
    # Check if the output file exists and clear its contents
    final_output_file = f'../output/output_gen_{gen}/final_output_gen_{gen}.txt'
    if os.path.exists(final_output_file):
        with open(final_output_file, 'w', encoding='utf-8') as f:
            f.write("")  # Clear the file contents
        print(f"Cleared contents of '{final_output_file}'.")
    else:
        print(f"'{final_output_file}' does not exist. A new file will be created.")

    # Load evo_id to Pokémon mapping
    stats_df = pd.read_csv(stats_file, encoding='utf-8')
    variants_df = pd.read_csv(variants_file, encoding='utf-8')

    # Get the last Pokémon name for each evo_id
    last_pokemon_names = stats_df.groupby('evo_id').last().reset_index()[['evo_id', 'pokemon']]

    # Merge with variants to map variant_id to Pokémon name
    variant_to_pokemon = variants_df.merge(last_pokemon_names, on='evo_id', how='left')

    # Create a dictionary for fast lookup
    variant_id_to_pokemon = dict(zip(variant_to_pokemon['variant_id'], variant_to_pokemon['pokemon']))

    overall_results = []  # To store results for the new plot

    for party_size in range(6, 0, -1):  # Loop over party sizes from 6 to 1
        print(f"Running genetic algorithm for party size {party_size}...")

        # Run the genetic algorithm for the current party size
        best_team, best_team_fitness, logbook = run_genetic_algorithm(
            file_path, stats_file, variants_file, restriction_file, n=party_size)

        # Get the selected rows for the best team
        best_variant_ids = best_team
        filtered_df = read_data(file_path)
        selected_rows = filtered_df[filtered_df['variant_id'].isin(best_variant_ids)].sort_values(
            by='variant_id').reset_index(drop=True)

        # Save results for the party size
        overall_results.append((party_size, best_team_fitness[0]))

        # Generate and save heatmaps and fitness plots for the current party size
        generate_additional_outputs(filtered_df, best_variant_ids, stats_file, variants_file,
                                    suffix=f"_party_{party_size}")
        plot_results(logbook, suffix=f"_party_{party_size}")

        # Prepare output lines for the results
        output_lines = [
            f"Results for Party Size {party_size}:",
            f"Best Team Fitness (EHL): {best_team_fitness[0]}"
        ]
        for _, row in selected_rows.iterrows():
            pokemon_name = variant_id_to_pokemon.get(row['variant_id'], "Unknown")
            output_lines.append(f"Variant ID: {row['variant_id']} ({pokemon_name}), Encounter Value: {row.iloc[1:].sum()}")

        # Append the results to the file
        with open(final_output_file, 'a', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")

    # Plot the change in overall team value (EHL) for each party size
    plot_team_value_over_party_sizes(overall_results)




def run_genetic_algorithm(file_path, stats_file, variants_file, restriction_file, n):
    # Load data
    filtered_df = read_data(file_path)
    variant_ids = filtered_df['variant_id'].tolist()

    # Load variants file for mapping evo_id to variant_id
    variants_df = pd.read_csv(variants_file)

    # Define invalid combinations
    invalid_evo_ids = []
    if restriction_file:
        invalid_evo_ids = read_restricted_combinations(restriction_file)

    # Map evo_id to variant_id
    evo_id_to_variant_ids = variants_df.groupby('evo_id')['variant_id'].apply(set).to_dict()
    invalid_combinations = [set().union(*(evo_id_to_variant_ids.get(evo_id, set()) for evo_id in invalid_set))
                            for invalid_set in invalid_evo_ids]

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

    # Run the genetic algorithm
    population_size = 5000  # Adjust population size as needed
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



def plot_results(logbook, suffix=""):
    genn = logbook.select("gen")
    min_fits = logbook.select("min")
    avg_fits = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(genn, min_fits, label="Minimum Fitness", marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../output/output_gen_{gen}/fitness_over_generations{suffix}_gen_{gen}.png")
    plt.show()

def plot_team_value_over_party_sizes(results):
    party_sizes, team_values = zip(*results)

    plt.figure(figsize=(8, 5))
    plt.plot(party_sizes, team_values, marker='o', linestyle='-', label="Team Value (EHL)")
    plt.xlabel("Party Size")
    plt.ylabel("Team Value (EHL)")
    plt.title("Change in Team Value Across Party Sizes")
    plt.xticks(party_sizes)  # Ensure party sizes are on the x-axis
    plt.yscale('log')
    plt.gca().invert_xaxis()  # Reverse the x-axis for decreasing party sizes
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../output/output_gen_{gen}/team_value_across_party_sizes_gen_{gen}.png")
    plt.show()

def generate_additional_outputs(filtered_df, best_variant_ids, stats_file, variants_file, suffix=""):
    # Generate heatmaps and identify worst encounters
    print("\nGenerating additional outputs...")

    # Load 'ehl_pivot.csv' to get the list of 'enc_id's
    ehl_pivot_df = pd.read_csv(f'../data_curated/data_curated_gen_{gen}/ehl_pivot_gen_{gen}.csv')
    # Drop first three columns (optional battles)
    ehl_pivot_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)

    # Extract 'enc_id' columns
    enc_id_columns = [col for col in ehl_pivot_df.columns if col.startswith('enc_id_')]

    # Extract 'enc_id's from the column names
    enc_id_order = [int(col.replace('enc_id_', '')) for col in enc_id_columns]

    # Sort the 'enc_id_columns' based on 'enc_id_order'
    enc_id_columns_sorted = ['enc_id_' + str(enc_id) for enc_id in sorted(enc_id_order)]

    # Load the matrix file (ehl_pivot.csv)
    matrix_df = pd.read_csv(f'../data_curated/data_curated_gen_{gen}/ehl_pivot_gen_{gen}.csv')
    matrix_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)

    # Ensure 'variant_id' is included in sorted columns
    sorted_columns = ['variant_id'] + enc_id_columns_sorted

    # Filter the matrix DataFrame to only include sorted columns
    sorted_matrix_df = matrix_df[sorted_columns]

    # Save the sorted matrix to a new CSV file
    # NOTE: rename this file? Do we even need it?
    sorted_matrix_file = f'../output/output_gen_{gen}/sorted_matrix1{suffix}_gen_{gen}.csv'
    sorted_matrix_df.to_csv(sorted_matrix_file, index=False)

    print(f"The columns of the matrix have been sorted and saved to '{sorted_matrix_file}'.")

    # Load the sorted matrix and filter for best variant_ids
    matrix = pd.read_csv(sorted_matrix_file)
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

    # Define key encounter points
    if gen == 1:
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
    elif gen == 2:
        key_encounters = {
            30: 'Falkner',
            81: 'Bugsy',
            115: 'Whitney',
            178: 'Morty',
            278: 'Chuck/Jasmine',
            338: 'Pryce',
            431: 'Clair',
            526: 'Elite Four',
            558: 'Lt. Surge',
            571: 'Sabrina',
            626: 'Misty',
            639: 'Erika',
            665: 'Janine',
            732: 'Brock',
            770: 'Blaine',
            785: 'Blue',
            791: 'Red'
        }
    else:
        key_counters = {}
    key_enc_ids = list(key_encounters.keys())
    key_labels = list(key_encounters.values())

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(20, 5))  # Adjust the figure size as needed
    sns.heatmap(matrix_normalized, cmap="viridis", cbar=True, annot=False, xticklabels=False, ax=ax)
    plt.title(f'Heatmap of Encounter Values for Selected Pokémon Variants {suffix}')
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

    # Save the heatmap
    heatmap_file = f'../output/output_gen_{gen}/pokemon_encounter_heatmap{suffix}_gen_{gen}.png'
    plt.savefig(heatmap_file)
    plt.show()

    print(f"Heatmap saved to '{heatmap_file}'.")



if __name__ == '__main__':
    config = pd.read_csv('../config/config.csv')
    gen = int(config[config.rule == 'gen'].value.values[0])
    parser = argparse.ArgumentParser(
        description='Find the best combination of variant_id options using a genetic algorithm.')
    parser.add_argument('--file_path', type=str, default=f'../data_curated/data_curated_gen_{gen}/ehl_pivot_gen_{gen}.csv', help='Path to the input CSV file.')
    parser.add_argument('--stats_file', type=str, default=f'../data_raw/data_raw_gen_{gen}/stats_gen_{gen}.csv',
                        help='Path to the stats CSV file.')
    parser.add_argument('--variants_file', type=str, default=f'../data_curated/data_curated_gen_{gen}/variants_gen_{gen}.csv',
                        help='Path to the variants CSV file.')
    parser.add_argument('--restriction_file', type=str, default='../config/restrictions.txt',
                        help='Path to the restriction configuration file.')
    parser.add_argument('--n', type=int, default=6, help='Number of variant_id options to select.')
    args = parser.parse_args()

    main(file_path=args.file_path, stats_file=args.stats_file, variants_file=args.variants_file,
         restriction_file=args.restriction_file)
