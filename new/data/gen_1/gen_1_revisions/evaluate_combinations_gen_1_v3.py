import pandas as pd
import numpy as np
import random
import argparse
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


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
    variant_ids = filtered_df['variant_id'].tolist()

    # Load variants file for mapping evo_id to variant_id
    variants_df = pd.read_csv(variants_file)

    # Define invalid combinations
    invalid_evo_ids = []
    if restriction_file:
        invalid_evo_ids = read_restricted_combinations(restriction_file)

    # Map evo_id to variant_id
    evo_id_to_variant_ids = variants_df.groupby('evo_id')['variant_id'].apply(set).to_dict()
    invalid_combinations = [
        set().union(*(evo_id_to_variant_ids[evo_id] for evo_id in invalid_set if evo_id in evo_id_to_variant_ids)) for
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
        population_size = 10000  # Adjust population size as needed
        generations = 200  # Adjust the number of generations as needed
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

        # **Evaluate the initial population's fitness**
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

    # Get final evolution names
    final_evolutions = get_final_evolution_names(stats_file, variants_file)

    # Merge the selected rows with final evolution names
    result = selected_rows.merge(final_evolutions, on='variant_id', how='left')

    # Prepare output text
    output_lines = ["Team Composition:"]
    for _, row in result.iterrows():
        output_lines.append(f"Variant ID: {row['variant_id']}, Pokemon: {row['pokemon']}")
    output_lines.append(f"\nTeam Value (Sum of Minimum Values): {best_team_fitness[0]}")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to txt file
    with open('../gen_1_output/final_summary_gen_1.txt', 'w') as f:
        f.write("\n".join(output_lines))
    print("\nSummary saved to 'final_summary_gen_1.txt'.")

    # Optionally, plot the results
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

    plot_results(logbook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find the best combination of variant_id options using a genetic algorithm.')
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
