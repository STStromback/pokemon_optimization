import pandas as pd
import numpy as np
import random

# Parameters
party_size = 6
population_size = 1000
generations = 100
mutation_rate = 0.1
tournament_size = 5

# Read the main data
ehl_df = pd.read_csv("../gen_1_data_curated/ehl_pivot.csv")
# Identify the enc_id columns
enc_columns = [col for col in ehl_df.columns if col.startswith("enc_id_")]

# Read variants data
variants_df = pd.read_csv("../gen_1_data_curated/variants_gen_1.csv")

# Read stats data
stats_df = pd.read_csv("../gen_1_data_raw/stats_gen_1.csv")

# We need to pick the last row for each evo_id to get the final evolutionary stage name.
# Group by evo_id and pick the last row
final_stage_stats = stats_df.groupby("evo_id").tail(1).copy()

# Merge variants with final stage stats to have a single lookup
merged = pd.merge(variants_df, final_stage_stats, on='evo_id', how='left')

# Create a quick lookup dictionary from variant_id to (Pokemon, move_types)
variant_info = {}
for _, row in merged.iterrows():
    variant_id = row['variant_id']
    pokemon_name = row['pokemon']
    move_types = [row['move_type_1'], row['move_type_2'], row['move_type_3'], row['move_type_4']]
    variant_info[variant_id] = (pokemon_name, move_types)

# Convert variant_id list for quick reference
all_variants = ehl_df['variant_id'].unique()
all_variants = sorted(all_variants)

# Objective function: For a given combination of variant IDs, calculate sum of minima across enc_id columns
def objective_function(team):
    # team is a list of variant_ids
    subset = ehl_df[ehl_df['variant_id'].isin(team)]
    # Take the min of each enc_id column across the chosen variants, then sum
    mins = subset[enc_columns].min(axis=0)
    return mins.sum()

# GA Functions

def create_individual():
    # Create a random team of `party_size` distinct variant_ids
    return random.sample(list(all_variants), party_size)

def mutate(individual):
    # With probability mutation_rate, replace one member of the team with another random variant
    if random.random() < mutation_rate:
        idx = random.randint(0, party_size-1)
        # Replace with a variant not currently in the team
        possible_variants = set(all_variants) - set(individual)
        if not possible_variants:
            return individual  # No mutation if no alternatives
        new_variant = random.choice(list(possible_variants))
        individual[idx] = new_variant
    return individual

def crossover(parent1, parent2):
    # One-point crossover
    point = random.randint(1, party_size-1)
    child1 = parent1[:point] + [v for v in parent2 if v not in parent1[:point]]
    # Ensure child1 is exactly party_size in length
    # If duplicates arise or length issues, we handle them:
    # Fill missing slots if any
    if len(child1) < party_size:
        missing = set(all_variants) - set(child1)
        while len(child1) < party_size and missing:
            child1.append(missing.pop())
    # If still duplicates or length mismatch (unlikely with careful selection), just ensure uniqueness
    child1 = child1[:party_size]
    # Similarly for child2
    child2 = parent2[:point] + [v for v in parent1 if v not in parent2[:point]]
    if len(child2) < party_size:
        missing = set(all_variants) - set(child2)
        while len(child2) < party_size and missing:
            child2.append(missing.pop())
    child2 = child2[:party_size]
    return child1, child2

def tournament_selection(population, scores):
    # Tournament selection: pick a few individuals and return the best
    competitors = random.sample(list(zip(population, scores)), tournament_size)
    competitors.sort(key=lambda x: x[1], reverse=True)
    return competitors[0][0]

# Initialize population
population = [create_individual() for _ in range(population_size)]
scores = [objective_function(ind) for ind in population]

# Genetic algorithm main loop
for gen in range(generations):
    new_population = []
    # Elite: carry forward the best individual
    best_idx = np.argmin(scores)
    best_ind = population[best_idx]
    best_score = scores[best_idx]
    new_population.append(best_ind)

    # Fill the rest of the population
    while len(new_population) < population_size:
        parent1 = tournament_selection(population, scores)
        parent2 = tournament_selection(population, scores)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        new_population.append(child1)
        if len(new_population) < population_size:
            new_population.append(child2)

    population = new_population
    scores = [objective_function(ind) for ind in population]

# After GA completion
best_idx = np.argmin(scores)
best_team = population[best_idx]
best_value = scores[best_idx]

# Sort the best team for consistent output
best_team = sorted(best_team)

# Prepare output
lines = []
lines.append("Team Composition:")
for variant_id in best_team:
    pokemon_name, move_types = variant_info.get(variant_id, ("Unknown", ["none","none","none","none"]))
    # The example output shows only 4 types. If some are 'none', just print them as is
    types_str = ", ".join(move_types)
    lines.append(f"Variant ID: {variant_id}, Pokemon: {pokemon_name}, Types: {types_str}")
lines.append(f"\nTeam Value (Sum of Minimum Values): {best_value}")

# Write to txt
with open("../gen_1_output/best_team.txt", "w") as f:
    f.write("\n".join(lines))

print("Best team composition saved to best_team.txt")
