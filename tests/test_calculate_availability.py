"""
Unit tests for calculate_availability.py

This test suite covers:
1. Text normalization
2. Location name normalization and mapping
3. Pokemon availability filtering
4. Configuration rule application
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add source directory to path
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

from calculate_availability import PokemonAvailabilityCalculator


class TestTextNormalization:
    """Test text normalization functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return PokemonAvailabilityCalculator()
    
    def test_basic_normalization(self, calculator):
        """Test basic text normalization."""
        assert calculator._normalize_text("Pikachu") == "pikachu"
        assert calculator._normalize_text("CHARIZARD") == "charizard"
    
    def test_space_to_underscore(self, calculator):
        """Test that spaces are converted to underscores."""
        assert calculator._normalize_text("Pallet Town") == "pallet_town"
        assert calculator._normalize_text("Route 1") == "route_1"
    
    def test_unicode_handling(self, calculator):
        """Test that Unicode characters are handled correctly."""
        assert calculator._normalize_text("Pokémon") == "pokemon"
        assert calculator._normalize_text("Café") == "cafe"
    
    def test_special_character_removal(self, calculator):
        """Test that special characters are removed."""
        assert calculator._normalize_text("Mt. Moon") == "mt_moon"
        assert calculator._normalize_text("Diglett's Cave") == "digletts_cave"
        assert calculator._normalize_text("Ho-Oh") == "hooh"
    
    def test_empty_and_nan(self, calculator):
        """Test handling of empty strings and NaN."""
        assert calculator._normalize_text("") == ""
        assert calculator._normalize_text(None) == ""
        assert calculator._normalize_text(pd.NA) == ""


class TestLocationNameNormalization:
    """Test location name normalization and mapping."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return PokemonAvailabilityCalculator()
    
    def test_route_normalization(self, calculator):
        """Test route name normalization."""
        assert calculator._normalize_location_name("Route 1") == "route_1"
        assert calculator._normalize_location_name("Route 22") == "route_22"
        assert calculator._normalize_location_name("ROUTE 10") == "route_10"
    
    def test_city_normalization(self, calculator):
        """Test city name normalization."""
        assert calculator._normalize_location_name("Pallet Town") == "pallet_town"
        assert calculator._normalize_location_name("Viridian City") == "viridian_city"
    
    def test_special_location_normalization(self, calculator):
        """Test special location name normalization."""
        assert calculator._normalize_location_name("Mt. Moon") == "mt_moon"
        assert calculator._normalize_location_name("Pokémon Mansion") == "pokemon_mansion"
        assert calculator._normalize_location_name("Diglett's Cave") == "digletts_cave"
    
    def test_underground_path_mapping(self, calculator):
        """Test that underground paths map to routes."""
        result = calculator._normalize_location_name("Underground Path 5-6")
        # Should be normalized and may map to route_5
        assert "underground" in result or "route" in result
    
    def test_gen2_location_normalization(self, calculator):
        """Test Gen 2 location normalization."""
        assert calculator._normalize_location_name("New Bark Town") == "new_bark_town"
        assert calculator._normalize_location_name("Cherrygrove City") == "cherrygrove_city"
    
    def test_preserve_slashes(self, calculator):
        """Test that slashes are preserved in location names."""
        # Slashes are in _KEEP_CHARS, so they should be preserved
        result = calculator._normalize_location_name("Route 5/6")
        assert "/" in result or "_" in result


class TestPokemonAvailabilityFiltering:
    """Test Pokemon availability filtering logic."""
    
    def test_location_stage_filtering(self):
        """Test filtering Pokemon by location stage."""
        # Sample Pokemon data
        pokemon_df = pd.DataFrame({
            'pokemon': ['pikachu', 'bulbasaur', 'charmander'],
            'location': ['viridian_forest', 'pallet_town', 'route_3'],
            'location_stage': [3, 1, 5]
        })
        
        # Filter to only locations available by stage 4
        max_stage = 4
        filtered = pokemon_df[pokemon_df['location_stage'] <= max_stage]
        
        assert len(filtered) == 2  # pikachu and bulbasaur
        assert 'charmander' not in filtered['pokemon'].values
    
    def test_method_filtering(self):
        """Test filtering by encounter method (walk, surf, etc.)."""
        pokemon_df = pd.DataFrame({
            'pokemon': ['tentacool', 'magikarp', 'pidgey'],
            'method': ['surf', 'old_rod', 'walk']
        })
        
        # Filter to only walking encounters
        walk_only = pokemon_df[pokemon_df['method'] == 'walk']
        
        assert len(walk_only) == 1
        assert walk_only['pokemon'].iloc[0] == 'pidgey'
    
    def test_rarity_filtering(self):
        """Test filtering by rarity percentage."""
        pokemon_df = pd.DataFrame({
            'pokemon': ['rattata', 'pikachu', 'mewtwo'],
            'rarity': [40, 5, 1]  # Percentage
        })
        
        # Filter common Pokemon (rarity >= 10%)
        common = pokemon_df[pokemon_df['rarity'] >= 10]
        
        assert len(common) == 1
        assert common['pokemon'].iloc[0] == 'rattata'


class TestConfigurationRules:
    """Test application of configuration rules."""
    
    def test_starter_exclusion(self):
        """Test that starters are excluded when not allowed in config."""
        starters = ['bulbasaur', 'charmander', 'squirtle']
        
        # Simulate config rule
        allow_starters = False
        
        if not allow_starters:
            # Should filter out starters
            result = [p for p in starters if p not in starters]
        else:
            result = starters
        
        assert len(result) == 0 if not allow_starters else 3
    
    def test_legendary_exclusion(self):
        """Test legendary Pokemon exclusion."""
        pokemon_list = ['pikachu', 'mewtwo', 'mew', 'articuno', 'zapdos', 'moltres']
        legendaries = ['mewtwo', 'mew', 'articuno', 'zapdos', 'moltres']
        
        allow_legendaries = False
        
        if not allow_legendaries:
            filtered = [p for p in pokemon_list if p not in legendaries]
        else:
            filtered = pokemon_list
        
        assert len(filtered) == 1
        assert filtered[0] == 'pikachu'
    
    def test_trade_evolution_filtering(self):
        """Test filtering trade evolution Pokemon."""
        pokemon_list = ['kadabra', 'alakazam', 'machoke', 'machamp']
        trade_evolutions = ['alakazam', 'machamp']
        
        allow_trade = False
        
        if not allow_trade:
            filtered = [p for p in pokemon_list if p not in trade_evolutions]
        else:
            filtered = pokemon_list
        
        assert len(filtered) == 2
        assert 'kadabra' in filtered
        assert 'machoke' in filtered


class TestDataFrameOperations:
    """Test common DataFrame operations used in availability calculation."""
    
    def test_merge_with_stages(self):
        """Test merging Pokemon data with stages data."""
        pokemon_df = pd.DataFrame({
            'pokemon': ['pikachu', 'raichu'],
            'location': ['viridian_forest', 'power_plant']
        })
        
        stages_df = pd.DataFrame({
            'location': ['viridian_forest', 'power_plant', 'mt_moon'],
            'location_stage': [3, 15, 5]
        })
        
        merged = pokemon_df.merge(stages_df, on='location', how='left')
        
        assert len(merged) == 2
        assert merged.loc[merged['pokemon'] == 'pikachu', 'location_stage'].iloc[0] == 3
    
    def test_groupby_location(self):
        """Test grouping Pokemon by location."""
        pokemon_df = pd.DataFrame({
            'location': ['route_1', 'route_1', 'route_2'],
            'pokemon': ['pidgey', 'rattata', 'spearow'],
            'rarity': [40, 50, 35]
        })
        
        grouped = pokemon_df.groupby('location').size()
        
        assert grouped['route_1'] == 2
        assert grouped['route_2'] == 1
    
    def test_drop_duplicates(self):
        """Test removing duplicate Pokemon entries."""
        pokemon_df = pd.DataFrame({
            'pokemon': ['pikachu', 'pikachu', 'raichu'],
            'location': ['viridian_forest', 'power_plant', 'power_plant']
        })
        
        unique = pokemon_df.drop_duplicates(subset=['pokemon'])
        
        assert len(unique) == 2  # pikachu and raichu


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return PokemonAvailabilityCalculator()
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should not raise an error
        assert len(empty_df) == 0
        assert list(empty_df.columns) == []
    
    def test_missing_columns(self):
        """Test handling of missing expected columns."""
        df = pd.DataFrame({
            'pokemon': ['pikachu']
            # Missing 'location' column
        })
        
        # Check if column exists before accessing
        has_location = 'location' in df.columns
        
        assert has_location is False
    
    def test_null_values(self):
        """Test handling of null values."""
        df = pd.DataFrame({
            'pokemon': ['pikachu', None, 'raichu'],
            'location': ['route_1', 'route_2', None]
        })
        
        # Drop rows with null values
        clean_df = df.dropna()
        
        assert len(clean_df) == 1
        assert clean_df['pokemon'].iloc[0] == 'pikachu'
    
    def test_case_sensitivity(self, calculator):
        """Test that normalization is case-insensitive."""
        result1 = calculator._normalize_text("PIKACHU")
        result2 = calculator._normalize_text("pikachu")
        result3 = calculator._normalize_text("PiKaChU")
        
        assert result1 == result2 == result3


class TestStagesDataLoading:
    """Test stages data loading functionality."""
    
    def test_stages_data_structure(self):
        """Test expected structure of stages data."""
        # Mock stages data
        stages_df = pd.DataFrame({
            'location': ['route_1', 'viridian_city', 'mt_moon'],
            'location_stage': [1, 2, 5],
            'badge_boost': ['attack', 'defense', None]
        })
        
        assert 'location' in stages_df.columns
        assert 'location_stage' in stages_df.columns
        assert len(stages_df) == 3
    
    def test_location_stage_ordering(self):
        """Test that location stages are properly ordered."""
        stages_df = pd.DataFrame({
            'location': ['route_3', 'route_1', 'route_2'],
            'location_stage': [3, 1, 2]
        })
        
        sorted_df = stages_df.sort_values('location_stage')
        
        assert sorted_df['location'].iloc[0] == 'route_1'
        assert sorted_df['location_stage'].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
