import pandas as pd
import numpy as np
import random
import argparse
from deap import base, creator, tools
import matplotlib.pyplot as plt
import os

def read_data(file_path):
    df = pd.read_csv(file_path)
    # Ensure all columns except 'variant_id' are numeric
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with a large number (since larger values are worse)
    df.fillna(df.max().max() + 1, inplace=True)
    return df

def read_restricted_combinations(restriction_file):
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
    stats_df = pd.read_csv(stats_file, encoding='utf-8')
    variants_df = pd.read_csv(variants_file, encoding='utf-8')

    # Find the last Pokémon for each evo_id
    last_evolutions = stats_df.groupby('evo_id').last().reset_index()

    # Merge with variants to get variant_id and move types
    last_evolutions = last_evolutions.merge(
        variants_df[['evo_id', 'variant_id', 'move_type_1', 'move_type_2', 'move_type_3', 'move_type_4']],
        on='evo_id', how='left'
    )

    return last_evolutions[['variant_id', 'pokemon', 'move_type_1', 'move_type_2', 'move_type_3', 'move_type_4']]

def main(file_path='ehl_pivot.csv', stats_file='stats_gen_1.csv', variants_file='variants_gen_1.csv',
         restriction_file=None, n=6):
    # Load data from ehl_pivot.csv
    filtered_df = read_data(file_path)
    # Remove first three encounters
    filtered_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)
    # Only consider variant_ids from filtered_df
    variant_ids = filtered_df['variant_id'].tolist()

    # Load variants for invalid combinations logic
    variants_df = pd.read_csv(variants_file)

    # Define invalid combinations
    invalid_evo_ids = []
    if restriction_file:
        invalid_evo_ids = read_restricted_combinations(restriction_file)

    evo_id_to_variant_ids = variants_df.groupby('evo_id')['variant_id'].apply(set).to_dict()
    # Build invalid combinations from evo_ids, but may include variants not in variant_ids
    invalid_combinations = [set().union(*(evo_id_to_variant_ids.get(evo_id, set()) for evo_id in invalid_set))
                            for invalid_set in invalid_evo_ids]

    # Intersect invalid combinations with the set of variant_ids from ehl_pivot.csv
    variant_ids_set = set(variant_ids)
    invalid_combinations = [ic & variant_ids_set for ic in invalid_combinations]

    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # GA will only pick variants from variant_ids derived from ehl_pivot.csv
    toolbox.register("attr_pokemon", random.choice, variant_ids)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pokemon, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    fitness_cache = {}

    def evaluate(individual):
        individual_tuple = tuple(sorted(individual))
        if individual_tuple in fitness_cache:
            return fitness_cache[individual_tuple]

        if is_invalid_combination(individual, invalid_combinations):
            fitness_cache[individual_tuple] = (float('inf'),)
            return fitness_cache[individual_tuple]

        team_data = filtered_df[filtered_df['variant_id'].isin(individual)].iloc[:, 1:]
        if team_data.empty or len(team_data) < n:
            fitness_cache[individual_tuple] = (float('inf'),)
            return fitness_cache[individual_tuple]

        min_values = team_data.min(axis=0)
        team_value = min_values.sum()

        fitness_cache[individual_tuple] = (team_value,)
        return fitness_cache[individual_tuple]

    def custom_mutate(individual):
        mutation_rate = 0.2
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
        population_size = 1000
        generations = 100
        mutation_rate = 0.2

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        elite_size = int(0.05 * population_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.mean([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))
        stats.register("min", lambda x: np.min([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))
        stats.register("max", lambda x: np.max([val[0] for val in x if len(val) > 0 and val[0] != float('inf')]))

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields

        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        for gen in range(generations):
            offspring = toolbox.select(pop, len(pop) - elite_size)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
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

    best_team, best_team_fitness, logbook = run_genetic_algorithm()

    # Ensure uniqueness
    best_variant_ids = list(dict.fromkeys(best_team))

    selected_rows = filtered_df[filtered_df['variant_id'].isin(best_variant_ids)].sort_values(
        by='variant_id').reset_index(drop=True)

    print("Best team variant IDs:", best_variant_ids)
    print("Selected rows shape:", selected_rows.shape)
    print("Selected rows variant_ids:", selected_rows['variant_id'].unique())

    final_evolutions = get_final_evolution_names(stats_file, variants_file)
    result = selected_rows.merge(final_evolutions, on='variant_id', how='left')

    print("Resulting merged rows shape:", result.shape)
    print("Result Pokémon names:", result['pokemon'].unique())

    # Prepare output text
    output_lines = ["Team Composition:"]
    for _, row in result.iterrows():
        types = ', '.join([str(row[t]) for t in ['move_type_1', 'move_type_2', 'move_type_3', 'move_type_4']
                           if pd.notna(row[t]) and row[t] != ''])
        output_lines.append(f"Variant ID: {row['variant_id']}, Pokémon: {row['pokemon']}, Move Types: {types}")
    output_lines.append(f"\nTeam Value (Sum of Minimum Values): {best_team_fitness[0]}")

    for line in output_lines:
        print(line)

    with open('../gen_1_output/final_output_gen_1.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print("\nSummary saved to 'final_output_gen_1.txt'.")

    # Plot results
    plot_results(logbook)

    # Generate additional outputs
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
    print("\nGenerating additional outputs...")

    ehl_pivot_df = pd.read_csv('../gen_1_data_curated/ehl_pivot.csv')
    ehl_pivot_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)

    enc_id_columns = [col for col in ehl_pivot_df.columns if col.startswith('enc_id_')]
    enc_id_order = [int(col.replace('enc_id_', '')) for col in enc_id_columns]
    enc_id_columns_sorted = ['enc_id_' + str(enc_id) for enc_id in sorted(enc_id_order)]

    matrix_df = pd.read_csv('../gen_1_data_curated/ehl_pivot.csv')
    matrix_df.drop(['enc_id_1', 'enc_id_2', 'enc_id_3'], axis=1, inplace=True)
    sorted_columns = ['variant_id'] + enc_id_columns_sorted
    sorted_matrix_df = matrix_df[sorted_columns]
    sorted_matrix_df.to_csv('sorted_matrix1.csv', index=False)
    print("The columns of the matrix have been sorted and saved to 'sorted_matrix1.csv'.")

    # Filter the matrix for best_variant_ids
    matrix = pd.read_csv('../gen_1_pipeline/sorted_matrix1.csv')
    matrix = matrix[matrix['variant_id'].isin(best_variant_ids)]

    variants_df = pd.read_csv(variants_file)
    stats_df = pd.read_csv(stats_file)

    variant_to_evo = variants_df.set_index('variant_id')['evo_id'].to_dict()

    stats_df['evo_lvl'].fillna(0, inplace=True)
    stats_df['evo_stage'].fillna(0, inplace=True)
    evo_to_pokemon = {}
    for evo_id, group in stats_df.groupby('evo_id'):
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

    matrix['variant_id'] = matrix['variant_id'].map(lambda x: evo_to_pokemon.get(variant_to_evo.get(x, ''), 'Unknown'))
    matrix.set_index('variant_id', inplace=True)

    matrix_normalized = matrix.applymap(lambda x: 1 if x >= 1 else x)
    min_values = matrix_normalized.min(axis=0)
    matrix_normalized.loc['Min Value'] = min_values

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
    xticks_positions = []
    for enc_id in key_enc_ids:
        col_name = f'enc_id_{enc_id}'
        if col_name in matrix_normalized.columns:
            xticks_positions.append(matrix_normalized.columns.get_loc(col_name))

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(matrix_normalized.values, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title('Heatmap of Encounter Values for Selected Pokémon Variants')
    ax.set_xlabel('Encounters')
    ax.set_ylabel('Pokémon')
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(matrix_normalized.shape[0]))
    ax.set_yticklabels(matrix_normalized.index)
    ax.set_ylim(-0.5, matrix_normalized.shape[0] - 0.5)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized EHL Values')
    plt.savefig('pokemon_encounter_heatmap_normalized_with_min_gen1.png')
    plt.show()

    # Unique minimum matrix
    unique_min_matrix = matrix.copy()
    for col in matrix.columns:
        min_val = matrix[col].min()
        if (matrix[col] == min_val).sum() == 1:
            unique_min_matrix[col] = np.where(matrix[col] == min_val, matrix[col], np.nan)
        else:
            unique_min_matrix[col] = np.nan

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(unique_min_matrix, aspect='auto', interpolation='nearest')
    ax.set_title('Heatmap of Unique Minimum Encounter Values')
    ax.set_xlabel('Encounters')
    ax.set_ylabel('Pokémon')

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(unique_min_matrix.shape[0]))
    ax.set_yticklabels(unique_min_matrix.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized EHL Values')

    plt.tight_layout()
    plt.savefig('pokemon_unique_min_encounter_heatmap_gen1.png')
    plt.show()

    # All minimum matrix
    all_min_matrix = matrix.copy()
    for col in matrix.columns:
        min_val = matrix[col].min()
        all_min_matrix[col] = np.where(matrix[col] == min_val, matrix[col], np.nan)

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(all_min_matrix, aspect='auto', interpolation='nearest')
    ax.set_title('Heatmap of All Minimum Encounter Values')
    ax.set_xlabel('Encounters')
    ax.set_ylabel('Pokémon')

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(key_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(all_min_matrix.shape[0]))
    ax.set_yticklabels(all_min_matrix.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized EHL Values')

    plt.tight_layout()
    plt.savefig('pokemon_all_min_encounter_heatmap_gen1.png')
    plt.show()

    # Worst encounters
    min_values_no_min = min_values.drop('Min Value', errors='ignore')
    worst_encounters = min_values_no_min.nlargest(100)
    worst_encounters_df = worst_encounters.reset_index()
    worst_encounters_df.columns = ['enc_id', 'min_ehl']
    worst_encounters_df.to_csv('worst_encounters_gen1.csv', index=False)

    print("Top 100 worst encounters based on minimum EHL values:")
    print(worst_encounters_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find the best combination of variant_id options using a genetic algorithm.')
    parser.add_argument('--file_path', type=str, default='ehl_pivot.csv', help='Path to the input CSV file.')
    parser.add_argument('--stats_file', type=str, default='../gen_1_data_raw/stats_gen_1.csv',
                        help='Path to the stats CSV file.')
    parser.add_argument('--variants_file', type=str, default='variants_gen_1.csv',
                        help='Path to the variants CSV file.')
    parser.add_argument('--restriction_file', type=str, default='../gen_1_config/restrictions_gen_1.txt',
                        help='Path to the restriction configuration file.')
    parser.add_argument('--n', type=int, default=6, help='Number of variant_id options to select.')
    args = parser.parse_args()

    main(file_path=args.file_path, stats_file=args.stats_file, variants_file=args.variants_file,
         restriction_file=args.restriction_file, n=args.n)
