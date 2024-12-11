import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, ListedColormap
import seaborn as sns

from deap import base, creator, tools

# Parameters
PARTY_SIZE = 6
POP_SIZE = 1000
N_GEN = 100
MUT_PB = 0.1
CXPB = 0.8
TOURN_SIZE = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ==================
# Data Loading
# ==================
ehl_df = pd.read_csv("../gen_1_data_curated/ehl_pivot.csv")
enc_columns = [col for col in ehl_df.columns if col.startswith("enc_id_")]

variants_df = pd.read_csv("../gen_1_data_curated/variants_gen_1.csv")
stats_df = pd.read_csv("../gen_1_data_raw/stats_gen_1.csv")

# Get final stage stats by grouping by evo_id and taking the last row
final_stage_stats = stats_df.groupby("evo_id").tail(1).copy()

# Merge variants with final stage stats to get pokemon name and move types
merged = pd.merge(variants_df, final_stage_stats, on='evo_id', how='left')

# Create a lookup from variant_id -> (pokemon_name, [move_types])
variant_info = {}
for _, row in merged.iterrows():
    v_id = row['variant_id']
    pokemon_name = row['pokemon']
    move_types = [row['move_type_1'], row['move_type_2'], row['move_type_3'], row['move_type_4']]
    variant_info[v_id] = (pokemon_name, move_types)

all_variants = sorted(ehl_df['variant_id'].unique())

# Objective function: given a team (list of variant_ids)
def objective_function(team):
    subset = ehl_df[ehl_df['variant_id'].isin(team)]
    mins = subset[enc_columns].min(axis=0)
    return (mins.sum(),)  # DEAP expects a tuple for fitness

# Repair individual to ensure uniqueness
def repair_individual(ind, all_variants, party_size):
    # If duplicates exist, replace them with random unused variants
    unique_set = set()
    duplicates = []
    for i, v in enumerate(ind):
        if v in unique_set:
            duplicates.append(i)
        else:
            unique_set.add(v)
    if duplicates:
        # Missing variants to fill from
        missing = list(set(all_variants) - unique_set)
        random.shuffle(missing)
        for i in duplicates:
            if not missing:
                break
            ind[i] = missing.pop()
    return ind

# ==================
# DEAP Setup
# ==================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute: generate a random team of unique variants
toolbox.register("team", random.sample, all_variants, PARTY_SIZE)

# Individual and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.team)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation
toolbox.register("evaluate", objective_function)

# Selection
toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)

# Crossover
def cx_one_point_unique(ind1, ind2):
    # One-point crossover
    cx_point = random.randint(1, PARTY_SIZE - 1)
    child1 = ind1[:cx_point] + ind2[cx_point:]
    child2 = ind2[:cx_point] + ind1[cx_point:]

    # Repair uniqueness
    child1 = repair_individual(child1, all_variants, PARTY_SIZE)
    child2 = repair_individual(child2, all_variants, PARTY_SIZE)

    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2

toolbox.register("mate", cx_one_point_unique)

# Mutation: random reset mutation with uniqueness repair
def mut_individual(ind):
    if random.random() < MUT_PB:
        # Choose a position to mutate
        pos = random.randint(0, PARTY_SIZE-1)
        # Replace with a variant not currently in the individual
        current_set = set(ind)
        choices = list(set(all_variants) - current_set)
        if choices:
            ind[pos] = random.choice(choices)
    return (ind,)

toolbox.register("mutate", mut_individual)

# ==================
# Run the GA
# ==================
# Tracking the best fitness value in each generation
best_fitness_per_gen = []

pop = toolbox.population(n=POP_SIZE)

# Evaluate initial population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for gen in range(N_GEN):
    # Select the next generation
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover
    for i in range(0, len(offspring)-1, 2):
        if random.random() < CXPB:
            toolbox.mate(offspring[i], offspring[i+1])
            # Evaluate children
            del offspring[i].fitness.values
            del offspring[i+1].fitness.values

    # Apply mutation
    for i in range(len(offspring)):
        if random.random() < MUT_PB:
            toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    # Re-evaluate fitness of modified individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Track the best fitness in this generation
    best_fitness = min(ind.fitness.values[0] for ind in pop)
    best_fitness_per_gen.append(best_fitness)

# ==================
# Find best individual
# ==================
best_ind = tools.selBest(pop, 1)[0]
best_value = best_ind.fitness.values[0]
best_team = sorted(best_ind)

# ==================
# Output Results
# ==================
lines = []
lines.append("Team Composition:")
for variant_id in best_team:
    pokemon_name, move_types = variant_info.get(variant_id, ("Unknown", ["none","none","none","none"]))
    types_str = ", ".join(move_types)
    lines.append(f"Variant ID: {variant_id}, Pokemon: {pokemon_name}, Types: {types_str}")

lines.append(f"\nTeam Value (Sum of Minimum Values): {best_value}")

with open("../gen_1_output/best_team.txt", "w") as f:
    f.write("\n".join(lines))

print("Best team composition saved to best_team.txt")
print("\n".join(lines))

# Plot the progression of the genetic algorithm
plt.figure(figsize=(10, 6))
plt.plot(range(N_GEN), best_fitness_per_gen, marker='o', linestyle='-', label="Best Fitness")
plt.xlabel("Generation")
plt.ylabel("Best Fitness (Sum of Min Values)")
plt.title("Genetic Algorithm Progression")
plt.legend()
plt.grid()
plt.show()


# Extract the data for the best party
best_party_df = ehl_df[ehl_df["variant_id"].isin(best_ind)]

# Normalize the values for the color scale
values = best_party_df[enc_columns].values
colors = []

# Define color normalization and colormap
norm = mcolors.Normalize(vmin=0, vmax=1)
cmap_yellow_blue = mcolors.LinearSegmentedColormap.from_list("YellowBlue", ["blue", "yellow"])
cmap_red = mcolors.LinearSegmentedColormap.from_list("Red", ["red", "red"])

# Generate colors for each value
for row in values:
    row_colors = []
    for value in row:
        if value > 1:
            row_colors.append(cmap_red(1))  # Red for values > 1
        else:
            row_colors.append(cmap_yellow_blue(norm(value)))  # Yellow to Blue
    colors.append(row_colors)

# Plot the horizontal bar plot
# Use logarithmic normalization for the full range of values
max_value = ehl_df[enc_columns].max().max()
norm = Normalize(vmin=1, vmax=max_value)

# Create a colormap with red for values exactly equal to 1
cmap = sns.color_palette("viridis", as_cmap=True)
colors = cmap(np.linspace(0, 1, 256))
colors[128] = [1, 0, 0, 1]  # Red for the value 1 (midpoint of the range)
new_cmap = ListedColormap(colors)

# Generate colors for each value
colors = []
for row in values:
    row_colors = []
    for value in row:
        if value == 1:
            row_colors.append([1, 0, 0, 1])  # Explicitly red for value 1
        else:
            normalized_value = norm(value)  # Normalize using the updated range
            row_colors.append(new_cmap(normalized_value))
    colors.append(row_colors)

# Re-plot the horizontal bar plot with the updated colormap
fig, ax = plt.subplots(figsize=(10, 6))
y_positions = range(len(best_party_df))

for i, row_colors in enumerate(colors):
    for j, color in enumerate(row_colors):
        ax.barh(y_positions[i], 1, left=j, color=color, edgecolor="black")

# Set y-axis labels to the Pokémon names (simulated)
y_labels = [f"Pokemon_{variant_id}" for variant_id in best_party_df["variant_id"]]
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels)

# Set x-axis labels to the enc_id columns
ax.set_xticks(range(len(enc_columns)))
ax.set_xticklabels(enc_columns)

# Add plot title and labels
ax.set_title("Best Party's Encounter Values (Custom Colormap with Log Scale)")
ax.set_xlabel("enc_id Columns")
ax.set_ylabel("Pokémon in Best Party")

plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()