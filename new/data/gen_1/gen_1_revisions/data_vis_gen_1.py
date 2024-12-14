import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, ListedColormap

# Load the encounter table
enc_table = pd.read_csv('enc_table_v3.csv')

# Extract the order of enc_id
enc_id_order = enc_table['enc_id'].tolist()

# Load the matrix file
matrix_df = pd.read_csv('matrix_filtered.csv')

# Create a mapping of enc_id to column names in matrix_df
enc_id_columns = [f'enc_id_{enc_id}' for enc_id in enc_id_order if f'enc_id_{enc_id}' in matrix_df.columns]

# Sort the columns of the matrix DataFrame
sorted_columns = ['variant_id'] + enc_id_columns
sorted_matrix_df = matrix_df[sorted_columns]

# Save the sorted matrix to a new CSV file
sorted_matrix_df.to_csv('sorted_matrix1.csv', index=False)

print("The columns of the matrix have been sorted and saved to 'sorted_matrix1.csv'.")

# Load the sorted matrix and filter for specific variant_ids
matrix = pd.read_csv('../gen_1_data_curated/sorted_matrix1.csv')

# Load best team variant_ids
best_team = pd.read_csv('best_team_with_ehl.csv')['variant_id'].tolist()
matrix = matrix[matrix['variant_id'].isin(best_team)]

# Load gen1_variants.csv and gen1_stats_v3.csv
gen1_variants = pd.read_csv('gen1_variants.csv')
gen1_stats = pd.read_csv('../gen1_stats_v3.csv')

# Map variant_id to evo_id using gen1_variants
variant_to_evo = gen1_variants.set_index('variant_id')['evo_id'].to_dict()

# Map evo_id to Pokémon name using gen1_stats
evo_to_pokemon = {}
gen1_stats['evo_lvl'].fillna(0, inplace=True)
gen1_stats['evo_stage'].fillna(0, inplace=True)
for evo_id, group in gen1_stats.groupby('evo_id'):
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
matrix['variant_id'] = matrix['variant_id'].map(lambda x: evo_to_pokemon[variant_to_evo[x]])

# Set the variant_id as the index for better visualization in the heatmap
matrix.set_index('variant_id', inplace=True)

# Normalize the values such that any value greater than 1 is set to 1
matrix_normalized = matrix.applymap(lambda x: 1 if x >= 1 else x)

# Calculate the minimum value out of the 6 rows for each encounter
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
    752: 'Brock',
    754: 'Misty',
    756: 'Lt. Surge',
    759: 'Erika',
    762: 'Koga',
    770: 'Sabrina',
    766: 'Blaine',
    610: 'Giovanni',
    821: 'Elite Four'
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
ax.set_xticks([enc_id_columns.index(f'enc_id_{enc_id}') for enc_id in key_enc_ids])
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
ax.set_xticks([enc_id_columns.index(f'enc_id_{enc_id}') for enc_id in key_enc_ids])
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
ax.set_xticks([enc_id_columns.index(f'enc_id_{enc_id}') for enc_id in key_enc_ids])
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
worst_encounters = min_values.nlargest(100)

# Output the worst encounters to a CSV file
worst_encounters_df = worst_encounters.reset_index()
worst_encounters_df.columns = ['enc_id', 'min_ehl']
worst_encounters_df.to_csv('worst_encounters_gen1.csv', index=False)

# Display the worst encounters
print("Top 100 worst encounters based on minimum EHL values:")
print(worst_encounters_df)
