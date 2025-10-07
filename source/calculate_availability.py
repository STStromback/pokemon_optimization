#!/usr/bin/env python3
"""
Pokemon Availability Calculator
Processes raw Pokemon location data and applies game configuration rules.
"""

import pandas as pd
import logging
import os
import json
from typing import Dict
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PokemonAvailabilityCalculator:
    def __init__(self):
        # Load config.json
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load stages data for location mapping
        self.stages_data = self._load_stages_data()

    def _load_stages_data(self) -> Dict[str, pd.DataFrame]:
        """Load stages data for all generations."""
        stages_data = {}
        # Go from source/ to project root (1 level up)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for gen in ['1', '2', '3']:
            stages_path = os.path.join(base_path, 'data', f'gen_{gen}', f'stages_gen_{gen}.csv')
            try:
                stages_data[gen] = pd.read_csv(stages_path)
                logger.info(f"Loaded stages data for generation {gen}: {len(stages_data[gen])} locations")
            except Exception as e:
                logger.error(f"Could not load stages data for generation {gen}: {e}")
                logger.error(f"Attempted path: {stages_path}")
                stages_data[gen] = pd.DataFrame()
        
        return stages_data
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by making lowercase, spaces to _, and removing non-alphanumeric characters."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase and replace spaces with underscores
        normalized = str(text).lower().replace(' ', '_')
        # Handle Unicode characters - replace é with e
        normalized = normalized.replace('é', 'e')
        # Keep only letters, numbers, and underscores
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
    
    def _normalize_location_name(self, location_name: str) -> str:
        """Normalize location name to match stages data format."""
        # Convert to lowercase and replace spaces with underscores
        _KEEP_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_/")
        normalized = location_name.lower().replace(' ', '_')
        # Handle Unicode characters - replace é with e
        normalized = normalized.replace('é', 'e')
        # Remove any characters not in _KEEP_CHARS
        normalized = ''.join(c for c in normalized if c in _KEEP_CHARS)
        
        # Handle common location name variations
        location_mappings = {
            # Gen 1 mappings
            'route_1': 'route_1',
            'route_2': 'route_2', 
            'route_3': 'route_3',
            'route_4': 'route_4',
            'route_5': 'route_5',
            'route_6': 'route_6',
            'route_7': 'route_7',
            'route_8': 'route_8',
            'route_9': 'route_9',
            'route_10': 'route_10',
            'route_11': 'route_11',
            'route_12': 'route_12',
            'route_13': 'route_13',
            'route_14': 'route_14',
            'route_15': 'route_15',
            'route_16': 'route_16',
            'route_17': 'route_17',
            'route_18': 'route_18',
            'route_19': 'route_19',
            'route_20': 'route_20',
            'route_21': 'route_21',
            'route_22': 'route_22',
            'route_23': 'route_23',
            'route_24': 'route_24',
            'route_25': 'route_25',
            'pallet_town': 'pallet_town',
            'viridian_city': 'viridian_city',
            'viridian_forest': 'viridian_forest',
            'pewter_city': 'pewter_city',
            'cerulean_city': 'cerulean_city',
            'cerulean_cave': 'cerulean_cave',
            'vermilion_city': 'vermilion_city',
            'lavender_town': 'lavender_town',
            'celadon_city': 'celadon_city',
            'fuchsia_city': 'fuchsia_city',
            'saffron_city': 'saffron_city',
            'cinnabar_island': 'cinnabar_island',
            'indigo_plateau': 'indigo_plateau',
            'mt_moon': 'mt_moon',
            'rock_tunnel': 'rock_tunnel',
            'power_plant': 'power_plant',
            'safari_zone': 'safari_zone',
            'seafoam_islands': 'seafoam_islands',
            'pokémon_mansion': 'pokemon_mansion',
            'victory_road': 'victory_road',
            "digletts_cave": 'digletts_cave',
            'pokemon_tower': 'pokemon_tower',
            'silph_co': 'silph_co',
            'underground_path_56': 'route_5',
            
            # Gen 2 mappings  
            'new_bark_town': 'new_bark_town',
            'route_29': 'route_29',
            'route_30': 'route_30',
            'route_31': 'route_31',
            'route_32': 'route_32',
            'route_33': 'route_33',
            'route_34': 'route_34',
            'route_35': 'route_35',
            'route_36': 'route_36',
            'route_37': 'route_37',
            'route_38': 'route_38',
            'route_39': 'route_39',
            'route_40': 'route_40',
            'route_41': 'route_41',
            'route_42': 'route_42',
            'route_43': 'route_43',
            'route_44': 'route_44',
            'route_45': 'route_45',
            'route_46': 'route_46',
            'cherrygrove_city': 'cherrygrove_city',
            'violet_city': 'violet_city',
            'azalea_town': 'azalea_town',
            'goldenrod_city': 'goldenrod_city',
            'ecruteak_city': 'ecruteak_city',
            'olivine_city': 'olivine_city',
            'cianwood_city': 'cianwood_city',
            'mahogany_town': 'mahogany_town',
            'blackthorn_city': 'blackthorn_city',
            'lake_of_rage': 'lake_of_rage',
            'whirl_islands': 'whirl_islands',
            'mt_silver': 'mt_silver',
            'ice_path': 'ice_path',
            'dragons_den': 'dragons_den',
            'tohjo_falls': 'tohjo_falls',
            'slowpoke_well': 'slowpoke_well',
            'ilex_forest': 'ilex_forest',
            'union_cave': 'union_cave',
            'ruins_of_alph': 'ruins_of_alph',
            'national_park': 'national_park',
            'tin_tower': 'tin_tower',
            'team_rocket_hq': 'team_rocket_hq',
            'radio_tower': 'radio_tower',
            'lighthouse': 'lighthouse',
            'dark_cave': 'dark_cave',
            'sprout_tower': 'sprout_tower',
            'mt_mortar': 'mt_mortar',
            'roaming_johto': 'burned_tower',
            'seafoam_islands': 'cinnabar_island',
            'fuchsia_city': 'fuchsia_city',
            
            # Gen 3 mappings (normalized to match stages_gen_3.csv format)
            'littleroot_town': 'littleroot_city',
            'route_101': 'route_101',
            'route_102': 'route_102', 
            'route_103': 'route_103',
            'route_104': 'route_104_south',  # South variant by default
            'route_105': 'route_105',
            'route_106': 'route_106',
            'route_107': 'route_107',
            'route_108': 'route_108',
            'route_109': 'route_109',
            'route_110': 'route_110',
            'route_111': 'route_111',
            'route_112': 'route_112',
            'route_113': 'route_113',
            'route_114': 'route_114',
            'route_115': 'route_115',
            'route_116': 'route_116',
            'route_117': 'route_117',
            'route_118': 'route_118',
            'route_119': 'route_119',
            'route_120': 'route_120',
            'route_121': 'route_121',
            'route_124': 'route_124',
            'route_125': 'route_125',
            'route_126': 'route_126',
            'route_127': 'route_127',
            'route_128': 'route_128',
            'route_129': 'route_129',
            'route_130': 'route_130',
            'route_131': 'route_131',
            'route_132': 'route_132',
            'route_133': 'route_133',
            'route_134': 'route_134',
            'oldale_town': 'oldale_town',
            'petalburg_city': 'petalburg_city',
            'rustboro_city': 'rustboro_city',
            'dewford_town': 'dewford_town',
            'slateport_city': 'slateport_city',
            'mauville_city': 'mauville_city',
            'verdanturf_town': 'verdanturf_town',
            'fallarbor_town': 'fallarbor_town',
            'lavaridge_town': 'lavaridge_town',
            'fortree_city': 'fortree_city',
            'lilycove_city': 'lilycove_city',
            'mossdeep_city': 'mossdeep_city',
            'sootopolis_city': 'sootopolis_city',
            'pacifidlog_town': 'pacifidlog_town',
            'ever_grande_city': 'ever_grande_city',
            'petalburg_woods': 'petalburg_woods',
            'granite_cave': 'granite_cave',
            'mr_chimney': 'mt_chimney',
            'jagged_pass': 'jagged_pass',
            'fiery_path': 'fiery_path',
            'meteor_falls': 'meteor_falls',
            'new_mauville': 'new_mauville',
            'seafloor_cavern': 'seafloor_cavern',
            'cave_of_origin': 'cave_of_origin',
            'victory_road': 'victory_road',
            'safari_zone': 'safari_zone',
            'sky_pillar': 'sky_pillar',
            'shoal_cave': 'shoal_cave',
            'weather_institute': 'weather_institute',
            'abandoned_ship': 'abandonded_ship',
            'desert_ruins': 'desert_ruins',
            'island_cave': 'island_cave',
            'ancient_tomb': 'ancient_tomb',
            'rusturf_tunnel': 'rusturf_tunnel_first',
            'altering_cave': 'postgame',
            'artisan_cave': 'postgame',
            'battle_tower': 'postgame',
            'cave_of_origin': 'postgame',
            'desert_underpass': 'postgame',
            'ever_grande_city': 'pokemon_league',
            'mirage_island': 'postgame',
            'mirage_tower': 'postgame',
            'roaming_hoenn': 'postgame',
            'southern_island': 'postgame',
            'team_magmaaqua_hideout': 'postgame',
            'mt_pyre': 'mt_pyre',
            'birth_island': 'postgame',
            'faraway_island': 'postgame',
            'team_magma/aqua_hideout': 'magma_hideout'
        }
        
        return location_mappings.get(normalized, normalized)
    
    def _get_method_stage(self, method: str, generation: str) -> str:
        """Get method stage by matching with key_items in stages data."""
        if not method or generation not in self.stages_data:
            return ''
        
        stages_df = self.stages_data[generation]
        if stages_df.empty or 'key_items' not in stages_df.columns:
            return ''
        
        # Normalize method for matching
        normalized_method = method.lower().replace(' ', '_')
        
        # Search for method in key_items column
        for _, row in stages_df.iterrows():
            key_items = str(row['key_items']).lower() if pd.notna(row['key_items']) else ''
            if key_items and normalized_method in key_items.split(','):
                return str(row['location_stage']) if pd.notna(row['location_stage']) else ''
        
        # If method not found, log a warning but don't error
        if normalized_method in ['surf', 'old_rod', 'good_rod', 'super_rod', 'headbutt', 'rock_smash']:
            logger.warning(f"Key method '{normalized_method}' not found in stages data for generation {generation}")
        
        return ''

    def _load_adjustments_data(self) -> pd.DataFrame:
        """Load Pokemon availability adjustments from CSV."""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adjustments_path = os.path.join(base_path, 'data', 'gen_all', 'pokemon_availability_adjustments.csv')
        
        if not os.path.exists(adjustments_path):
            logger.warning(f"Adjustments file not found: {adjustments_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(adjustments_path)
            logger.info(f"Loaded {len(df)} adjustment rules")
            return df
        except Exception as e:
            logger.error(f"Error loading adjustments file: {e}")
            return pd.DataFrame()
    
    def _apply_manual_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply manual adjustments from pokemon_availability_adjustments.csv."""
        adjustments_df = self._load_adjustments_data()
        
        if adjustments_df.empty:
            logger.info("No adjustments to apply")
            return df
        
        processed_df = df.copy()
        adjustments_applied = 0
        
        # Normalize text columns in main dataframe for matching
        processed_df['pokemon_norm'] = processed_df['pokemon'].apply(self._normalize_text)
        processed_df['location_norm'] = processed_df['location'].apply(self._normalize_text)
        processed_df['sub_location_norm'] = processed_df['sub_location'].apply(self._normalize_text)
        processed_df['method_norm'] = processed_df['method'].apply(self._normalize_text)
        
        for _, adj_row in adjustments_df.iterrows():
            # Normalize adjustment criteria
            adj_pokemon = self._normalize_text(adj_row['pokemon']) if pd.notna(adj_row['pokemon']) else ''
            adj_location = self._normalize_text(adj_row['location']) if pd.notna(adj_row['location']) else ''
            adj_sub_location = self._normalize_text(adj_row['sub_location']) if pd.notna(adj_row['sub_location']) else ''
            adj_method = self._normalize_text(adj_row['method']) if pd.notna(adj_row['method']) else ''
            adj_new_location = self._normalize_text(adj_row['new_location']) if pd.notna(adj_row['new_location']) else ''
            
            # Create match conditions (empty string means match any value)
            conditions = []
            
            if adj_pokemon:
                conditions.append(processed_df['pokemon_norm'] == adj_pokemon)
            
            if adj_location:
                conditions.append(processed_df['location_norm'] == adj_location)
            
            if adj_sub_location:
                conditions.append(processed_df['sub_location_norm'] == adj_sub_location)
            
            # Find matching rows
            if conditions:
                mask = conditions[0]
                for condition in conditions[1:]:
                    mask = mask & condition
            else:
                # If no conditions, match all rows (shouldn't happen but handle gracefully)
                mask = processed_df.index >= 0
            
            matching_rows = processed_df[mask]
            
            if len(matching_rows) > 0:
                # Apply adjustments - either method or new_location should have a value
                if adj_method:
                    # Replace method with new method value
                    processed_df.loc[mask, 'method'] = adj_row['method']  # Use original (non-normalized) value
                    
                    # CRITICAL FIX: When changing method, we need to clear method_stage so it gets recalculated
                    # This fixes the Ho-Oh availability bug where rainbow_feather method wasn't properly staged
                    processed_df.loc[mask, 'method_stage'] = pd.NA
                    
                    adjustments_applied += len(matching_rows)
                    logger.info(f"Applied method adjustment '{adj_row['method']}' to {len(matching_rows)} rows (method_stage will be recalculated)")
                
                elif adj_new_location:
                    # Replace location with new_location value
                    processed_df.loc[mask, 'location'] = adj_row['new_location']  # Use original (non-normalized) value
                    adjustments_applied += len(matching_rows)
                    logger.info(f"Applied location adjustment '{adj_row['new_location']}' to {len(matching_rows)} rows")
                
                else:
                    logger.warning(f"Adjustment row has neither method nor new_location value: {adj_row}")
            else:
                logger.debug(f"No matches found for adjustment: {adj_row}")
        
        # Remove temporary normalization columns
        processed_df = processed_df.drop(columns=['pokemon_norm', 'location_norm', 'sub_location_norm', 'method_norm'])
        
        logger.info(f"Applied {adjustments_applied} total adjustments")
        return processed_df
    
    def load_raw_data(self, input_path: str) -> pd.DataFrame:
        """Load raw Pokemon availability data from CSV."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Raw data file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded raw data with {len(df)} records")
        return df

    def process_availability_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw data and apply configuration rules."""
        # Start with a copy of the input data
        processed_df = df.copy()
        
        # Apply manual adjustments first
        processed_df = self._apply_manual_adjustments(processed_df)
        
        # Add location_stage and method_stage columns
        processed_df['location_stage'] = ''
        processed_df['method_stage'] = ''
        
        for gen in ['1', '2', '3']:
            gen_mask = processed_df['gen'] == int(gen)
            if not gen_mask.any():
                continue
                
            stages_df = self.stages_data.get(gen, pd.DataFrame())
            if stages_df.empty:
                continue
                
            # Map location names and add location_stage
            for idx, row in processed_df[gen_mask].iterrows():
                location_match = None
                
                # Normalize both raw location and stages locations for matching
                normalized_raw_location = self._normalize_location_name(row['location'])
                
                # Create normalized stages locations for comparison
                stages_df_normalized = stages_df.copy()
                stages_df_normalized['location_normalized'] = stages_df_normalized['location'].apply(self._normalize_location_name)
                
                # Try normalized matching
                location_match = stages_df_normalized[stages_df_normalized['location_normalized'] == normalized_raw_location]
                
                if not location_match.empty:
                    processed_df.at[idx, 'location_stage'] = str(location_match.iloc[0]['location_stage'])
                else:
                    # Log unmapped locations for manual review
                    normalized_location = self._normalize_location_name(row['location'])
                    logger.warning(f"Unmapped location for Gen {gen}: '{row['location']}' -> '{normalized_location}'")
                
                # Add method_stage
                if 'method' in row and row['method']:
                    method_stage = self._get_method_stage(row['method'], gen)
                    processed_df.at[idx, 'method_stage'] = method_stage
        
        # Apply configuration-based modifications
        processed_df = self._apply_config_rules(processed_df)
        
        return processed_df

    def _apply_config_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configuration rules to the data."""
        # Manually set suicune in gen 2 to location tin_tower_1f and location_stage to 34
        suicune_mask = (df['gen'] == '2') & (df['pokemon'] == 'Suicune')
        if suicune_mask.any():
            df.loc[suicune_mask, 'location_stage'] = '34'

        # If easy_dog_catch config = "n", manually set method_stage for raikou, entei and suicune in gen 2 to 39
        if self.config['easy_dog_catch'] == 'n':
            legendary_dogs = ['Raikou', 'Entei', 'Suicune']
            for dog in legendary_dogs:
                dog_mask = (df['gen'] == '2') & (df['pokemon'] == dog)
                if dog_mask.any():
                    df.loc[dog_mask, 'method_stage'] = '39'
        
        # If all_starters config = "y", manually set location_stage and method_stage for all pokemon to 0
        if self.config['all_starters'] == 'y':
            df['location_stage'] = '0'
            df['method_stage'] = '0'

        # If legendaries config = "n", manually set all legendary pokemon in all gens to enc_stage = 999
        # Legendaries = zapdos, moltres, articuno, mew, mewtwo, raikou, entei, suicune, lugia, ho-oh, regirock, regice, registeel, latios, latias, kyogre, groudon, rayquaza, celebi, jirachi, deoxys
        if self.config['legendaries'] == 'n':
            legendary_pokemon = ['zapdos', 'moltres', 'articuno', 'mew', 'mewtwo', 'raikou', 'entei', 'suicune', 'lugia', 'ho-oh', 'regirock', 'regice', 'registeel', 'latios', 'latias', 'kyogre', 'groudon', 'rayquaza', 'celebi', 'jirachi', 'deoxys']
            legendary_mask = df['pokemon'].isin(legendary_pokemon)
            if legendary_mask.any():
                df.loc[legendary_mask, 'enc_stage'] = '999'
        
        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to CSV file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Sort dataframe
        df['location_stage'] = df['location_stage'].fillna(0).replace('',0).astype(int)
        df['method_stage'] = df['method_stage'].fillna(0).replace('',0).astype(int)
        df = df.sort_values(by=['gen', 'location_stage', 'location', 'sub_location', 'method'])
        df['stage_available'] = df[['location_stage', 'method_stage']].max(axis=1)

        # Fix Ho-oh name to Ho_oh
        df.loc[df['pokemon'] == 'Ho-oh', 'pokemon'] = 'Ho_oh'
        
        # Convert gender symbols in pokemon names
        df['pokemon'] = df['pokemon'].str.replace('♀', '_F', regex=False)
        df['pokemon'] = df['pokemon'].str.replace('♂', '_M', regex=False)

        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Total unique records: {len(df)}")
        
        # Print summary statistics
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total records: {len(df)}")
        for gen in sorted(df['gen'].unique()):
            gen_data = df[df['gen'] == gen]
            version_name = {'1': 'Red', '2': 'Crystal', '3': 'Emerald'}[str(gen)]
            print(f"Generation {gen} ({version_name}): {len(gen_data)} records")
            print(f"  Unique Pokemon: {gen_data['pokemon'].nunique()}")
            print(f"  Unique Locations: {gen_data['location'].nunique()}")
            if 'method' in gen_data.columns:
                print(f"  Unique Methods: {gen_data['method'].nunique()}")

def main():
    """Main execution function."""
    calculator = PokemonAvailabilityCalculator()
    
    try:
        # Set up file paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(base_path, 'data', 'gen_all', 'pokemon_availability_raw.csv')
        
        # CRITICAL: Save to the location where calculate_variants.py expects to read from
        # This ensures proper file flow for both debug runs and main pipeline
        output_path = os.path.join(base_path, 'data', 'gen_all', 'pokemon_availability.csv')
        
        # Also save to test_outputs for debugging reference
        test_output_path = os.path.join(base_path, 'test_outputs', 'pokemon_availability.csv')
        
        # Load raw data
        raw_data = calculator.load_raw_data(input_path)
        
        # Process the data
        processed_data = calculator.process_availability_data(raw_data)
        
        # Save processed data to BOTH locations for proper file flow
        # 1. Save to production location where calculate_variants.py expects it
        calculator.save_processed_data(processed_data, output_path)
        print(f"*** PRODUCTION: Output saved to {output_path} ***")
        
        # 2. Save to test_outputs directory for debugging reference  
        calculator.save_processed_data(processed_data, test_output_path)
        print(f"*** DEBUG COPY: Output saved to {test_output_path} ***")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Make sure to run scrape_availability.py first to generate the raw data")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()