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

from common import paths
from common.config import load_config
from common.text_utils import normalize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_LOCATION_KEEP_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_/")

_LEGENDARY_DOGS = ['Raikou', 'Entei', 'Suicune']

class PokemonAvailabilityCalculator:
    def __init__(self):
        # Load config.json
        self.config = load_config()
        
        # Load stages data for location mapping
        self.stages_data = self._load_stages_data()
        
        # Load location name overrides from CSV
        self._location_overrides = self._load_location_overrides()
        
        # Load availability legendary list from CSV
        self._availability_legendaries = self._load_availability_legendaries()

    def _load_stages_data(self) -> Dict[str, pd.DataFrame]:
        """Load stages data for all generations."""
        stages_data = {}
        for gen in ['1', '2', '3']:
            stages_path = str(paths.gen_data_dir(gen) / f'stages_gen_{gen}.csv')
            try:
                stages_data[gen] = pd.read_csv(stages_path)
                logger.info(f"Loaded stages data for generation {gen}: {len(stages_data[gen])} locations")
            except Exception as e:
                logger.error(f"Could not load stages data for generation {gen}: {e}")
                logger.error(f"Attempted path: {stages_path}")
                stages_data[gen] = pd.DataFrame()
        
        return stages_data

    def _load_availability_legendaries(self) -> list:
        """Load the legendary pokemon list used for the legendaries=n config rule."""
        try:
            df = pd.read_csv(paths.GEN_ALL_DIR / "availability_legendaries.csv")
            return df["pokemon"].tolist()
        except Exception as e:
            logger.error(f"Could not load availability legendaries: {e}")
            return []

    def _load_location_overrides(self) -> dict:
        """Load location name overrides from data/gen_all/location_name_overrides.csv."""
        try:
            df = pd.read_csv(paths.GEN_ALL_DIR / "location_name_overrides.csv")
            return dict(zip(df["normalized_input"], df["mapped_output"]))
        except Exception as e:
            logger.error(f"Could not load location overrides: {e}")
            return {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text using the project's canonical normalization (see common.text_utils)."""
        return normalize_text(text)
    
    def _normalize_location_name(self, location_name: str) -> str:
        """Normalize location name to match stages data format."""
        normalized = location_name.lower().replace(' ', '_')
        normalized = normalized.replace('é', 'e')
        normalized = ''.join(c for c in normalized if c in _LOCATION_KEEP_CHARS)
        return self._location_overrides.get(normalized, normalized)
    
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
        adjustments_path = str(paths.GEN_ALL_DIR / 'pokemon_availability_adjustments.csv')
        
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
            for dog in _LEGENDARY_DOGS:
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
            legendary_mask = df['pokemon'].isin(self._availability_legendaries)
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
        logger.info("PROCESSING SUMMARY")
        logger.info(f"Total records: {len(df)}")
        for gen in sorted(df['gen'].unique()):
            gen_data = df[df['gen'] == gen]
            version_name = {'1': 'Red', '2': 'Crystal', '3': 'Emerald'}[str(gen)]
            logger.info(f"Generation {gen} ({version_name}): {len(gen_data)} records")
            logger.info(f"  Unique Pokemon: {gen_data['pokemon'].nunique()}")
            logger.info(f"  Unique Locations: {gen_data['location'].nunique()}")
            if 'method' in gen_data.columns:
                logger.info(f"  Unique Methods: {gen_data['method'].nunique()}")

def main():
    """Main execution function."""
    calculator = PokemonAvailabilityCalculator()
    
    try:
        # Set up file paths
        input_path = str(paths.GEN_ALL_DIR / 'pokemon_availability_raw.csv')
        output_path = str(paths.GEN_ALL_DIR / 'pokemon_availability.csv')
        
        # Load raw data
        raw_data = calculator.load_raw_data(input_path)
        
        # Process the data
        processed_data = calculator.process_availability_data(raw_data)
        
        # Save processed data
        calculator.save_processed_data(processed_data, output_path)
        logger.info(f"Output saved to {output_path}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Make sure to run scrape_availability.py first to generate the raw data")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()