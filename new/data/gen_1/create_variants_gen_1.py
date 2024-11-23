import pandas as pd
from itertools import combinations

def generate_variants(types):
    # Generate all unique combinations of up to four types
    variants = set()
    types = sorted(set(types))  # Sort and remove duplicates
    for r in range(1, 5):  # From 1 type to 4 types
        for combo in combinations(types, r):
            # Pad with 'none' to ensure all variants have four elements
            full_combo = list(combo) + ['none'] * (4 - len(combo))
            variants.add(tuple(full_combo))
    return variants

def is_subset(current, variants):
    # Check if current variant is a subset of any other variant
    current_set = set(current) - {'none'}
    for variant in variants:
        if current_set < set(variant) - {'none'}:
            return True
    return False

def main():
    # Load data
    moves_data = pd.read_csv('moves_gen_1.csv')
    moves_data = moves_data.dropna(subset=['move_type'])  # Ensure all entries have a type

    # Load mapping
    stage_map = pd.read_csv('stages_gen_1.csv')

    # Add evo_id
    stats = pd.read_csv('stats_gen_1.csv')
    evo_ids = stats[['evo_id','pokemon']]
    moves_data = pd.merge(left=moves_data, right=stats, on='pokemon', how='left')

    # Remap move_stage column to integer form
    moves_data['move_stage'] = moves_data['move_stage'].map(dict(zip(stage_map.location,stage_map.location_stage)))

    # Process each pokemon ID
    grouped = moves_data.groupby('evo_id')
    unique_variants_list = []

    for pokemon_id, group in grouped:
        types = group['move_type'].tolist()
        all_variants = generate_variants(types)
        unique_variants = [v for v in all_variants if not is_subset(v, all_variants)]

        for variant in unique_variants:
            unique_variants_list.append({
                'evo_id': pokemon_id,
                'move_type_1': variant[0],
                'move_type_2': variant[1],
                'move_type_3': variant[2],
                'move_type_4': variant[3]
            })

    # Convert list to DataFrame
    variants_df = pd.DataFrame(unique_variants_list)

    # Assign a unique identifier for each variant combination within the same evo_id
    # variants_df['variant_id'] = variants_df.groupby(['evo_id']).cumcount()
    variants_df['variant_id'] = variants_df.index + 1

    variants_df['evo_id'] = variants_df['evo_id'].astype(int)
    
    # Save to CSV
    variants_df.to_csv('variants_gen_1.csv', index=False)


if __name__ == '__main__':
    main()
