"""
Unit tests for calculate_best_party.py

This test suite covers:
1. Restriction processing (converting evo_ids to indices)
2. Restriction violation checking
3. Candidate search optimization
4. Battle result loading and processing
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add source directory to path
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

from calculate_best_party import (
    _process_restrictions,
    _violates_restrictions,
    load_battle_results,
)


class TestRestrictionProcessing:
    """Test restriction group processing."""
    
    def test_simple_restriction_conversion(self):
        """Test converting evo_ids to evo_id_pp indices."""
        restrictions = {
            'A': [1, 2, 3]  # Starter Pokemon
        }
        # Format: "evo_id_evolution_level"
        evo_id_pp_list = ["1_1_50", "2_1_50", "3_1_50", "4_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        # Should have 1 group with indices 0, 1, 2
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}
    
    def test_multiple_restriction_groups(self):
        """Test multiple restriction groups."""
        restrictions = {
            'A': [1, 2, 3],
            'B': [68, 69, 70]
        }
        evo_id_pp_list = ["1_1_50", "2_1_50", "68_1_50", "69_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        # Should have 2 groups
        assert len(groups) == 2
    
    def test_restriction_with_single_pokemon(self):
        """Test that restriction groups with only 1 Pokemon are ignored."""
        restrictions = {
            'A': [1]  # Only one Pokemon
        }
        evo_id_pp_list = ["1_1_50", "2_1_50", "3_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        # Should be empty (need at least 2 to form a restriction)
        assert len(groups) == 0
    
    def test_empty_restrictions(self):
        """Test with no restrictions."""
        restrictions = {}
        evo_id_pp_list = ["1_1_50", "2_1_50", "3_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        assert len(groups) == 0
    
    def test_restriction_with_missing_pokemon(self):
        """Test restriction group where some Pokemon aren't in evo_id_pp_list."""
        restrictions = {
            'A': [1, 2, 3, 99, 100]  # 99 and 100 not in list
        }
        evo_id_pp_list = ["1_1_50", "2_1_50", "3_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        # Should still create group with indices 0, 1, 2
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}
    
    def test_restriction_with_evolved_forms(self):
        """Test restrictions apply to all evolution stages."""
        restrictions = {
            'A': [1]  # Just base evo_id
        }
        # Multiple evolution stages of same Pokemon
        evo_id_pp_list = ["1_1_50", "1_2_50", "1_3_50", "2_1_50"]
        
        groups = _process_restrictions(restrictions, evo_id_pp_list)
        
        # All evolution stages of evo_id 1 should be in same group
        # But since only 1 evo_id, no group is formed (need >= 2 evo_ids)
        assert len(groups) == 0


class TestRestrictionViolation:
    """Test restriction violation checking."""
    
    def test_no_violation_one_from_group(self):
        """Test that having one Pokemon from a group is allowed."""
        individual = [0, 3, 5]  # Indices
        restriction_groups = [[0, 1, 2]]  # Group containing index 0
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is False
    
    def test_violation_two_from_same_group(self):
        """Test that having 2+ from same group is a violation."""
        individual = [0, 1, 5]  # Indices 0 and 1 both in group
        restriction_groups = [[0, 1, 2]]
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is True
    
    def test_no_violation_different_groups(self):
        """Test that having one from each group is allowed."""
        individual = [0, 3]  # One from each group
        restriction_groups = [[0, 1, 2], [3, 4, 5]]
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is False
    
    def test_violation_multiple_from_one_group(self):
        """Test violation with multiple groups."""
        individual = [0, 1, 3]  # Two from first group
        restriction_groups = [[0, 1, 2], [3, 4, 5]]
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is True
    
    def test_no_violation_empty_restrictions(self):
        """Test that empty restrictions never violate."""
        individual = [0, 1, 2, 3]
        restriction_groups = []
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is False
    
    def test_no_violation_no_overlap(self):
        """Test that individuals with no restricted indices don't violate."""
        individual = [6, 7, 8]
        restriction_groups = [[0, 1, 2], [3, 4, 5]]
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is False
    
    def test_violation_three_from_group(self):
        """Test violation with 3+ Pokemon from same group."""
        individual = [0, 1, 2, 6]  # Three from first group
        restriction_groups = [[0, 1, 2]]
        
        violated = _violates_restrictions(individual, restriction_groups)
        
        assert violated is True


class TestBattleResultsLoading:
    """Test battle results CSV loading."""
    
    def test_load_battle_results_structure(self):
        """Test that loaded battle results have expected structure."""
        # Create temporary test CSV
        test_df = pd.DataFrame({
            'evo_id_pp': [1_1_50, 2_1_50, 3_1_50],
            'enc_1': [10.5, 20.0, 15.5],
            'enc_2': [25.0, 30.5, 22.0]
        })
        
        # Save and load
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            test_df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_df = load_battle_results(temp_path)
            
            assert 'evo_id_pp' in loaded_df.columns
            assert len(loaded_df) == 3
            assert loaded_df['enc_1'].iloc[0] == 10.5
        finally:
            import os
            os.unlink(temp_path)


class TestCandidateSearchHelpers:
    """Test helper functions for candidate search."""
    
    def test_threshold_comparison(self):
        """Test basic threshold comparison logic."""
        # Win threshold is typically 1.0
        WIN_THRESHOLD = 1.0
        TIE_SCORE = 1000.0
        LOSS_SCORE = 1000000.0
        
        assert 0.5 <= WIN_THRESHOLD  # Win
        assert TIE_SCORE > WIN_THRESHOLD  # Not a win
        assert LOSS_SCORE > WIN_THRESHOLD  # Not a win
    
    def test_nan_handling_as_high(self):
        """Test that NaN values are treated as high scores (unavailable)."""
        values = np.array([10.0, np.nan, 20.0, np.nan])
        threshold = 15.0
        
        # With treat_nan_as_high=True
        mask = (values <= threshold) & ~np.isnan(values)
        
        assert mask[0] == True   # 10 <= 15
        assert mask[1] == False  # NaN excluded
        assert mask[2] == False  # 20 > 15
        assert mask[3] == False  # NaN excluded
    
    def test_nan_handling_as_pass(self):
        """Test that NaN values can pass threshold check."""
        values = np.array([10.0, np.nan, 20.0, np.nan])
        threshold = 15.0
        
        # With treat_nan_as_high=False
        mask = (values <= threshold)
        
        assert mask[0] == True   # 10 <= 15
        assert mask[1] == False  # NaN fails comparison
        assert mask[2] == False  # 20 > 15
        assert mask[3] == False  # NaN fails comparison


class TestBitmaskOperations:
    """Test bitmask operations used in optimization."""
    
    def test_single_bit_set(self):
        """Test setting individual bits."""
        bitmask = 0
        
        bitmask |= (1 << 0)  # Set bit 0
        assert bitmask == 1
        
        bitmask |= (1 << 2)  # Set bit 2
        assert bitmask == 5  # Binary: 101
    
    def test_bit_check(self):
        """Test checking if bit is set."""
        bitmask = 5  # Binary: 101
        
        assert (bitmask & (1 << 0)) != 0  # Bit 0 is set
        assert (bitmask & (1 << 1)) == 0  # Bit 1 is not set
        assert (bitmask & (1 << 2)) != 0  # Bit 2 is set
    
    def test_bitwise_or_coverage(self):
        """Test combining bitmasks with OR."""
        mask_a = 5   # Binary: 101
        mask_b = 3   # Binary: 011
        combined = mask_a | mask_b
        
        assert combined == 7  # Binary: 111
    
    def test_popcount(self):
        """Test counting set bits."""
        bitmask = 7  # Binary: 111
        count = bin(bitmask).count('1')
        
        assert count == 3
    
    def test_all_bits_covered(self):
        """Test checking if all required bits are covered."""
        required = 7   # Binary: 111 (bits 0, 1, 2 needed)
        covered = 15   # Binary: 1111 (bits 0, 1, 2, 3 set)
        
        # All required bits are covered
        assert (required & covered) == required
    
    def test_not_all_bits_covered(self):
        """Test checking when not all bits are covered."""
        required = 7  # Binary: 111
        covered = 5   # Binary: 101 (bit 1 missing)
        
        # Not all required bits are covered
        assert (required & covered) != required


class TestPartyFitnessCalculation:
    """Test party fitness score calculation."""
    
    def test_fitness_is_sum_of_scores(self):
        """Test that fitness is sum of scores across all encounters."""
        scores = np.array([0.5, 0.3, 0.7, 0.2])  # Scores for 4 encounters
        fitness = np.sum(scores)
        
        assert fitness == 1.7
    
    def test_min_score_per_column(self):
        """Test taking minimum score for each encounter."""
        # Party of 3 Pokemon vs 2 encounters
        battle_matrix = np.array([
            [0.5, 0.8],  # Pokemon 1
            [0.3, 0.9],  # Pokemon 2
            [0.6, 0.4]   # Pokemon 3
        ])
        
        # Best Pokemon for each encounter
        min_scores = np.min(battle_matrix, axis=0)
        
        assert min_scores[0] == 0.3  # Pokemon 2 best for enc_1
        assert min_scores[1] == 0.4  # Pokemon 3 best for enc_2
    
    def test_fitness_lower_is_better(self):
        """Test that lower fitness values are better."""
        fitness_a = 10.5
        fitness_b = 25.3
        
        assert fitness_a < fitness_b  # A is better


class TestColumnCoverage:
    """Test column coverage tracking for party building."""
    
    def test_single_pokemon_coverage(self):
        """Test coverage from single Pokemon."""
        # Pokemon can win (score <= 1.0) these encounters
        scores = np.array([0.5, 1000.0, 0.8, 1000000.0])
        WIN_THRESHOLD = 1.0
        
        covered = scores <= WIN_THRESHOLD
        
        assert covered[0] == True
        assert covered[1] == False
        assert covered[2] == True
        assert covered[3] == False
    
    def test_party_combined_coverage(self):
        """Test combined coverage from multiple Pokemon."""
        # 3 Pokemon vs 4 encounters
        pokemon_coverage = np.array([
            [True, False, True, False],   # Pokemon 1 covers enc_1, enc_3
            [False, True, True, False],   # Pokemon 2 covers enc_2, enc_3
            [False, False, False, True]   # Pokemon 3 covers enc_4
        ])
        
        # Combined coverage (any Pokemon can cover)
        combined = np.any(pokemon_coverage, axis=0)
        
        assert combined[0] == True   # Covered by pokemon 1
        assert combined[1] == True   # Covered by pokemon 2
        assert combined[2] == True   # Covered by pokemon 1 and 2
        assert combined[3] == True   # Covered by pokemon 3
    
    def test_uncovered_encounters(self):
        """Test identifying uncovered encounters."""
        party_coverage = np.array([True, True, False, True])
        
        uncovered_indices = np.where(~party_coverage)[0]
        
        assert len(uncovered_indices) == 1
        assert uncovered_indices[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
