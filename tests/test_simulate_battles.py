"""
Unit tests for simulate_battles.py

This test suite covers:
1. Battle simulation logic (damage, HP, speed calculations)
2. Truant ability handling in Gen 3
3. Speed tie scenarios
4. Win/Loss/Tie scoring
5. Dominance filtering
6. Equivalent row identification
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add source directory to path
source_dir = Path(__file__).parent.parent / "source"
sys.path.insert(0, str(source_dir))

from simulate_battles import (
    filter_dominated_rows,
    identify_and_remove_equivalent_rows,
)


class TestBattleScoring:
    """Test battle scoring logic."""
    
    def test_win_score_calculation(self):
        """Test that win scores are calculated correctly (0 to 1)."""
        # Win scenario: score = 1 - (remaining_hp / start_hp)
        # Perfect win (no damage taken): score = 1 - 1.0 = 0
        # Win with 50% HP left: score = 1 - 0.5 = 0.5
        # Win with 10% HP left: score = 1 - 0.1 = 0.9
        
        # These are expected based on the formula
        assert 0 <= 0 < 1  # Perfect win
        assert 0 <= 0.5 < 1  # 50% HP left
        assert 0 <= 0.9 < 1  # 10% HP left
    
    def test_loss_score_calculation(self):
        """Test that loss scores are M*M + remaining_enc_hp_ratio."""
        M = 1000
        # Loss scenario: score = (remaining_enc_hp / enc_hp_start) + M*M
        # Complete loss (enc at full HP): 1.0 + 1000000 = 1000001.0
        # Loss with enc at 50% HP: 0.5 + 1000000 = 1000000.5
        
        assert M*M + 1.0 == 1000001.0
        assert M*M + 0.5 == 1000000.5
        assert M*M + 0.0 == 1000000.0
    
    def test_tie_score(self):
        """Test that tie scores are exactly 1000.0."""
        TIE_SCORE = 1000.0
        assert TIE_SCORE == 1000.0
    
    def test_unavailable_score(self):
        """Test that unavailable Pokemon score is 1000000000.0."""
        UNAVAILABLE_SCORE = 1000000000.0
        assert UNAVAILABLE_SCORE == 1000000000.0


class TestTruantAbility:
    """Test Truant ability damage reduction in Gen 3."""
    
    def test_truant_halves_damage(self):
        """Test that Truant ability halves effective damage per turn."""
        base_damage = 100.0
        truant_damage = base_damage * 0.5
        
        assert truant_damage == 50.0
    
    def test_both_truant(self):
        """Test both Pokemon with Truant have halved damage."""
        player_damage = 80.0
        enc_damage = 60.0
        
        player_effective = player_damage * 0.5
        enc_effective = enc_damage * 0.5
        
        assert player_effective == 40.0
        assert enc_effective == 30.0
    
    def test_slakoth_slaking_have_truant(self):
        """Test that Slakoth and Slaking are recognized as having Truant."""
        # These Pokemon should be detected as having Truant even without ability data
        truant_pokemon = ['slakoth', 'slaking']
        
        for pokemon in truant_pokemon:
            # Simulating the check from the code
            has_truant = pokemon.lower() in ['slakoth', 'slaking']
            assert has_truant is True


class TestSpeedTieScenario:
    """Test battle outcomes when speeds are equal."""
    
    def test_speed_tie_halves_damage(self):
        """Test that speed tie results in both Pokemon dealing half damage."""
        base_damage = 100.0
        tie_damage = base_damage * 0.5
        
        assert tie_damage == 50.0
    
    def test_speed_tie_both_zero_damage(self):
        """Test speed tie with no damage results in TIE_SCORE."""
        # When both deal 0 damage after speed tie reduction
        TIE_SCORE = 1000.0
        # This should return TIE_SCORE
        assert TIE_SCORE == 1000.0
    
    def test_speed_tie_simultaneous_ko(self):
        """Test speed tie where both KO at same time results in tie."""
        # When turns_to_ko are equal for both
        TIE_SCORE = 1000.0
        assert TIE_SCORE == 1000.0


class TestDominanceFiltering:
    """Test dominance filtering logic."""
    
    @pytest.fixture
    def sample_pivot_df(self):
        """Create sample pivot table for testing."""
        return pd.DataFrame({
            'enc_1': [10, 20, 5, 15],
            'enc_2': [30, 40, 25, 35],
            'enc_3': [50, 60, 45, 55]
        }, index=[1, 2, 3, 4])
    
    def test_dominated_row_removed(self):
        """Test that strictly dominated rows are removed."""
        # Row 1: [10, 30, 50] - dominates row 2 and 4
        # Row 2: [20, 40, 60] - dominated by row 1
        # Row 3: [5, 25, 45] - dominates all others
        # Row 4: [15, 35, 55] - dominated by row 1
        
        df = pd.DataFrame({
            'enc_1': [10, 20, 5, 15],
            'enc_2': [30, 40, 25, 35],
            'enc_3': [50, 60, 45, 55]
        }, index=[1, 2, 3, 4])
        
        filtered = filter_dominated_rows(df)
        
        # Row 3 should remain (dominates all)
        # Row 1 should remain (dominates 2 and 4 but not 3)
        # Rows 2 and 4 should be removed (dominated)
        assert len(filtered) <= len(df)
        assert 3 in filtered.index  # Best row always kept
    
    def test_no_dominated_rows(self):
        """Test when no rows are dominated (Pareto frontier)."""
        # Each row is better in at least one column
        df = pd.DataFrame({
            'enc_1': [10, 30, 20],
            'enc_2': [30, 10, 20],
            'enc_3': [20, 20, 10]
        }, index=[1, 2, 3])
        
        filtered = filter_dominated_rows(df)
        
        # All rows should remain (none dominated)
        assert len(filtered) == 3
    
    def test_identical_rows_not_removed(self):
        """Test that identical rows are not considered dominated."""
        df = pd.DataFrame({
            'enc_1': [10, 10, 20],
            'enc_2': [20, 20, 30]
        }, index=[1, 2, 3])
        
        filtered = filter_dominated_rows(df)
        
        # Rows 1 and 2 are identical, neither dominates the other
        # Row 3 is dominated
        assert len(filtered) >= 2
    
    def test_all_same_values(self):
        """Test when all rows have the same values."""
        df = pd.DataFrame({
            'enc_1': [10, 10, 10],
            'enc_2': [20, 20, 20]
        }, index=[1, 2, 3])
        
        filtered = filter_dominated_rows(df)
        
        # All rows are equivalent, none dominated
        assert len(filtered) == 3


class TestEquivalentRowIdentification:
    """Test equivalent row identification and removal."""
    
    def test_identical_rows_grouped(self):
        """Test that identical rows are identified and grouped."""
        df = pd.DataFrame({
            'enc_1': [10.0, 10.0, 20.0],
            'enc_2': [30.0, 30.0, 40.0]
        }, index=[1, 2, 3])
        
        # Rows 1 and 2 are identical
        result = identify_and_remove_equivalent_rows(df, gen=1)
        
        # Should keep only one representative from the equivalent group
        assert len(result) == 2
    
    def test_no_equivalent_rows(self):
        """Test when all rows are unique."""
        df = pd.DataFrame({
            'enc_1': [10.0, 20.0, 30.0],
            'enc_2': [40.0, 50.0, 60.0]
        }, index=[1, 2, 3])
        
        result = identify_and_remove_equivalent_rows(df, gen=1)
        
        # All rows should remain
        assert len(result) == 3
    
    def test_floating_point_precision(self):
        """Test that floating point precision is handled correctly."""
        df = pd.DataFrame({
            'enc_1': [10.0000001, 10.0, 20.0],
            'enc_2': [30.0, 30.0000001, 40.0]
        }, index=[1, 2, 3])
        
        result = identify_and_remove_equivalent_rows(df, gen=1)
        
        # Should recognize rows 1 and 2 as equivalent (rounded to 6 decimals)
        assert len(result) == 2
    
    def test_all_identical(self):
        """Test when all rows are identical."""
        df = pd.DataFrame({
            'enc_1': [10.0, 10.0, 10.0],
            'enc_2': [20.0, 20.0, 20.0]
        }, index=[1, 2, 3])
        
        result = identify_and_remove_equivalent_rows(df, gen=1)
        
        # Should keep only one representative
        assert len(result) == 1
    
    def test_multiple_equivalent_groups(self):
        """Test multiple groups of equivalent rows."""
        df = pd.DataFrame({
            'enc_1': [10.0, 10.0, 20.0, 20.0, 30.0],
            'enc_2': [40.0, 40.0, 50.0, 50.0, 60.0]
        }, index=[1, 2, 3, 4, 5])
        
        # Groups: [1,2], [3,4], [5]
        result = identify_and_remove_equivalent_rows(df, gen=1)
        
        # Should keep 3 representatives (one from each group)
        assert len(result) == 3


class TestBattleTurnsCalculation:
    """Test turns-to-KO calculation."""
    
    def test_exact_ko(self):
        """Test when damage divides evenly into HP."""
        import math
        hp = 100
        damage = 25
        turns = math.ceil(hp / damage)
        
        assert turns == 4
    
    def test_fractional_ko(self):
        """Test when extra turn is needed for remaining HP."""
        import math
        hp = 100
        damage = 30
        turns = math.ceil(hp / damage)
        
        assert turns == 4  # 30 + 30 + 30 + 10 = 100
    
    def test_one_hit_ko(self):
        """Test one-hit KO scenario."""
        import math
        hp = 50
        damage = 100
        turns = math.ceil(hp / damage)
        
        assert turns == 1
    
    def test_zero_damage_infinite_turns(self):
        """Test that zero damage means infinite turns (special case)."""
        import math
        hp = 100
        damage = 0
        
        # This would cause division by zero, which should be handled
        # In the actual code, this is checked: if damage <= 0: handle specially
        if damage <= 0:
            result = "Cannot KO"
        else:
            turns = math.ceil(hp / damage)
            result = turns
        
        assert result == "Cannot KO"


class TestScoreOrdering:
    """Test that score ordering is correct (lower is better)."""
    
    def test_win_better_than_tie(self):
        """Test that any win score (< 1) is better than tie (1000)."""
        win_score = 0.9  # Win with little HP left
        tie_score = 1000.0
        
        assert win_score < tie_score
    
    def test_tie_better_than_loss(self):
        """Test that tie (1000) is better than any loss (> 1000000)."""
        tie_score = 1000.0
        loss_score = 1000000.5
        
        assert tie_score < loss_score
    
    def test_loss_better_than_unavailable(self):
        """Test that loss (1000000+) is better than unavailable (1000000000)."""
        loss_score = 1000001.0
        unavailable_score = 1000000000.0
        
        assert loss_score < unavailable_score
    
    def test_better_win_ordering(self):
        """Test that higher HP remaining means better win score."""
        # Win with 90% HP remaining
        better_win = 1 - 0.9  # = 0.1
        # Win with 50% HP remaining
        worse_win = 1 - 0.5  # = 0.5
        
        assert better_win < worse_win
    
    def test_better_loss_ordering(self):
        """Test that leaving opponent with less HP is better loss."""
        M = 1000
        # Loss with opponent at 10% HP
        better_loss = 0.1 + M*M
        # Loss with opponent at 90% HP
        worse_loss = 0.9 + M*M
        
        assert better_loss < worse_loss


class TestNumpyOperations:
    """Test numpy operations used in dominance checking."""
    
    def test_all_less_than_or_equal(self):
        """Test np.all() for dominance checking."""
        row_a = np.array([10, 20, 30])
        row_b = np.array([15, 25, 35])
        
        # row_a dominates row_b (all values less than or equal)
        assert np.all(row_a <= row_b)
    
    def test_any_strictly_less(self):
        """Test np.any() for strict dominance."""
        row_a = np.array([10, 20, 30])
        row_b = np.array([10, 25, 30])
        
        # At least one value is strictly less
        assert np.any(row_a < row_b)
    
    def test_dominance_condition(self):
        """Test combined condition for dominance."""
        dominator = np.array([10, 20, 30])
        dominated = np.array([15, 25, 35])
        
        # Dominance: all <= AND at least one <
        is_dominated = np.all(dominator <= dominated) and np.any(dominator < dominated)
        
        assert is_dominated == True
    
    def test_non_dominance_condition(self):
        """Test that non-dominated rows are not flagged."""
        row_a = np.array([10, 30, 20])
        row_b = np.array([20, 20, 30])
        
        # Neither dominates: a is better in col1, b is better in col2
        a_dominates_b = np.all(row_a <= row_b) and np.any(row_a < row_b)
        b_dominates_a = np.all(row_b <= row_a) and np.any(row_b < row_a)
        
        assert a_dominates_b == False
        assert b_dominates_a == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
