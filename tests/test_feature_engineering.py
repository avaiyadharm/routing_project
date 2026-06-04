"""
Unit tests for feature engineering pipeline.
Tests all temporal, meteorological, and event feature computations.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_engineering import FeatureEngineer


class TestTemporalFeatures:
    """Test temporal feature calculations."""

    @pytest.fixture
    def engineer(self):
        """Create feature engineer instance."""
        return FeatureEngineer(region='delaware')

    def test_holiday_state_regular_day(self, engineer):
        """Regular weekday should return state 0."""
        date = datetime(2026, 6, 15)  # Monday, not a holiday
        state = engineer.get_holiday_state(date)
        assert state == 0, "Regular day should have state 0"

    def test_holiday_state_actual_holiday(self, engineer):
        """Actual holiday should return state 2."""
        date = datetime(2026, 7, 4)  # Independence Day
        state = engineer.get_holiday_state(date)
        assert state == 2, "Holiday should have state 2"

    def test_holiday_state_eve_surge(self, engineer):
        """Day before holiday should trigger eve-of-holiday surge (state 1)."""
        date = datetime(2026, 7, 3)  # Day before Independence Day
        state = engineer.get_holiday_state(date)
        assert state == 1, "Day before holiday should have state 1 (eve surge)"

    def test_holiday_state_post_rebound(self, engineer):
        """Days after holiday should show rebound effect (state 4)."""
        date = datetime(2026, 7, 5)  # Day after Independence Day
        state = engineer.get_holiday_state(date)
        assert state == 4, "Day after holiday should have state 4 (post rebound)"

    def test_school_calendar_in_semester(self, engineer):
        """January should be in school semester (2025-2026 academic year)."""
        date = datetime(2026, 1, 15)  # January 2026 is in school semester
        season, phase = engineer.get_school_calendar_state(date)
        assert season == 1, "January should be in semester"
        assert -1 <= phase <= 1, "Phase should be in [-1, 1] range"

    def test_school_calendar_summer_break(self, engineer):
        """July-August should be summer break (between school years)."""
        date = datetime(2026, 7, 15)
        season, phase = engineer.get_school_calendar_state(date)
        assert season == 0, "July should be summer break (between academic years)"

    def test_school_hours_active(self, engineer):
        """School-influenced hours (7 AM - 3 PM) should be active when in semester."""
        date = datetime(2026, 9, 15)  # September, in semester
        season, _ = engineer.get_school_calendar_state(date)

        # Test morning hours
        assert engineer.get_school_hours_active(8, school_season=1) is True
        assert engineer.get_school_hours_active(14, school_season=1) is True
        assert engineer.get_school_hours_active(4, school_season=1) is False
        assert engineer.get_school_hours_active(20, school_season=1) is False

        # During break, should be inactive
        assert engineer.get_school_hours_active(8, school_season=0) is False

    def test_payday_cycle_near_1st(self, engineer):
        """Days near 1st should show strong payday effect."""
        date_0 = datetime(2026, 6, 30)  # Day before
        date_1 = datetime(2026, 7, 1)   # Payday
        date_2 = datetime(2026, 7, 2)   # Day after

        days_to_0, _ = engineer.get_payday_cycle(date_0)
        days_to_1, _ = engineer.get_payday_cycle(date_1)
        days_to_2, _ = engineer.get_payday_cycle(date_2)

        assert 0 <= days_to_0 <= 2, "Day before payday should be within 2 days"
        assert 0 <= days_to_1, "On payday"
        assert 0 <= days_to_2, "After payday"

    def test_cyclical_encoding_hour_wrap(self, engineer):
        """Hour cyclical encoding should show 23 and 0 as nearby but not identical."""
        features_23 = engineer.encode_cyclical_features(23, 182)
        features_0 = engineer.encode_cyclical_features(0, 182)
        features_12 = engineer.encode_cyclical_features(12, 182)

        # Hours 23 and 0 should be closer than 23 and 12
        hour_sin_diff_23_0 = abs(features_23['hour_sin'] - features_0['hour_sin'])
        hour_sin_diff_23_12 = abs(features_23['hour_sin'] - features_12['hour_sin'])

        assert hour_sin_diff_23_0 < hour_sin_diff_23_12, "Hour 23 and 0 should be closer than 23 and 12"

    def test_build_temporal_feature_dict(self, engineer):
        """Complete temporal feature dict should have all expected keys."""
        timestamp = datetime(2026, 9, 15, 14, 30)  # 2:30 PM
        features = engineer.build_temporal_feature_dict(timestamp)

        # Check all expected keys exist
        expected_keys = [
            'hour_of_day', 'day_of_week', 'month', 'day_of_month',
            'holiday_state_ordinal', 'school_season_state', 'school_phase',
            'school_hours_active', 'days_to_payday', 'payday_cycle_sin',
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
        ]

        for key in expected_keys:
            assert key in features, f"Missing feature key: {key}"

        # Validate value ranges
        assert 0 <= features['hour_of_day'] <= 23
        assert 0 <= features['day_of_week'] <= 6
        assert 1 <= features['month'] <= 12
        assert -1 <= features['hour_sin'] <= 1
        assert -1 <= features['hour_cos'] <= 1


class TestMeteorologicalFeatures:
    """Test meteorological feature calculations."""

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_precipitation_multiplier_dry(self, engineer):
        """Dry conditions should have 1.0x multiplier."""
        mult = engineer.get_precipitation_multiplier(0.0)
        assert mult == 1.0

    def test_precipitation_multiplier_light_drizzle(self, engineer):
        """Light drizzle (1 mm/hr) should have minimal effect."""
        mult = engineer.get_precipitation_multiplier(1.0)
        assert 1.0 <= mult < 1.1, "Light drizzle should have <10% penalty"

    def test_precipitation_multiplier_moderate_rain(self, engineer):
        """Moderate rain (10 mm/hr) should increase multiplier significantly."""
        mult = engineer.get_precipitation_multiplier(10.0)
        assert 1.2 <= mult <= 1.4, "Moderate rain should have 20-40% penalty"

    def test_precipitation_multiplier_heavy_rain(self, engineer):
        """Heavy rain (25 mm/hr) should have severe multiplier."""
        mult = engineer.get_precipitation_multiplier(25.0)
        assert mult > 1.5, "Heavy rain should have >50% penalty"

    def test_precipitation_multiplier_monotonic(self, engineer):
        """Higher precipitation should never decrease multiplier."""
        multipliers = [
            engineer.get_precipitation_multiplier(i) for i in [0, 2, 5, 10, 20, 50]
        ]
        # Check monotonic increase
        for i in range(len(multipliers) - 1):
            assert multipliers[i] <= multipliers[i + 1], "Multipliers should increase monotonically"

    def test_visibility_multiplier_clear(self, engineer):
        """Clear visibility (>10km) should have 1.0x multiplier."""
        mult = engineer.get_visibility_multiplier(15000)
        assert mult == 1.0

    def test_visibility_multiplier_fog(self, engineer):
        """Dense fog (<1km) should have >1.3x multiplier."""
        mult = engineer.get_visibility_multiplier(500)
        assert mult > 1.3, "Dense fog should have >30% penalty"

    def test_visibility_multiplier_moderate_fog(self, engineer):
        """Moderate fog (2km) should have 1.15-1.25x multiplier."""
        mult = engineer.get_visibility_multiplier(2000)
        assert 1.1 <= mult <= 1.3, "Moderate fog should have 10-30% penalty"

    def test_glare_risk_not_sunrise_sunset(self, engineer):
        """Glare risk should be 0 outside sunrise/sunset hours."""
        glare = engineer.compute_glare_risk(90, 90, 12)  # Noon, perpendicular roads
        assert glare == 0.0, "No glare at noon"

    def test_glare_risk_sunrise_perpendicular(self, engineer):
        """Perpendicular east-west roads during sunrise should have high glare."""
        glare = engineer.compute_glare_risk(0, 90, 7)  # East-facing road, sun in east, 7 AM
        assert glare > 0.5, "Perpendicular road during sunrise should have high glare"

    def test_glare_risk_parallel(self, engineer):
        """Parallel roads to sun should have minimal glare."""
        glare = engineer.compute_glare_risk(90, 90, 7)  # Road aligned with sun
        assert glare < 0.3, "Parallel road should have minimal glare"

    def test_road_surface_state_dry(self, engineer):
        """No rain should give dry surface state."""
        state = engineer.compute_road_surface_state(0.0)
        assert state == 0, "No rain should be dry"

    def test_road_surface_state_damp(self, engineer):
        """Light rain with high drainage should give damp state."""
        state = engineer.compute_road_surface_state(1.5, elevation_grade_percent=5, urban_density=0.1)
        assert state == 1, "Light rain with good drainage should be damp"

    def test_road_surface_state_wet(self, engineer):
        """Moderate rain with poor drainage should give wet state."""
        state = engineer.compute_road_surface_state(4.0, elevation_grade_percent=0, urban_density=0.9)
        assert state == 2, "Moderate rain with very poor drainage should be wet"

    def test_road_surface_state_hydroplaning(self, engineer):
        """Heavy rain should give hydroplaning risk state."""
        state = engineer.compute_road_surface_state(20.0)
        assert state == 3, "Heavy rain should show hydroplaning risk"


class TestEventFeatures:
    """Test event feature calculations."""

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_event_proximity_outside_radius(self, engineer):
        """Edge outside event radius should have 1.0x multiplier."""
        mult = engineer.get_event_proximity_decay(10.0, 5.0)  
        assert mult == 1.0

    def test_event_proximity_at_epicenter(self, engineer):
        """Edge at event epicenter should have maximum multiplier."""
        mult = engineer.get_event_proximity_decay(0.0, 5.0)
        assert mult > 1.4, "At epicenter should have maximum ~1.5x"
        assert mult <= 1.5

    def test_event_proximity_decay_smooth(self, engineer):
        """Event impact should decay smoothly with distance."""
        impacts = [
            engineer.get_event_proximity_decay(d, 5.0) for d in [0, 1, 2, 3, 4, 5]
        ]
        # Should be monotonically decreasing
        for i in range(len(impacts) - 1):
            assert impacts[i] >= impacts[i + 1], "Impact should decrease with distance"

    def test_event_phase_pre_event(self, engineer):
        """More than 2 hours before event should be pre_buffer phase."""
        now = datetime(2026, 6, 15, 10, 0)
        event_start = datetime(2026, 6, 15, 13, 0)  # 3 hours later
        phase = engineer.get_event_phase(now, event_start, event_start)
        assert phase == 0, "More than 2 hours before should be pre_buffer"

    def test_event_phase_early_arrivals(self, engineer):
        """30 minutes before event should be early_arrivals phase."""
        now = datetime(2026, 6, 15, 12, 30)
        event_start = datetime(2026, 6, 15, 13, 0)
        phase = engineer.get_event_phase(now, event_start, event_start)
        assert phase == 1, "Within 30 min before should be early_arrivals"

    def test_event_phase_egress(self, engineer):
        """Within 90 minutes after event should be egress phase."""
        now = datetime(2026, 6, 15, 13, 30)  # 30 min after start
        event_start = datetime(2026, 6, 15, 13, 0)
        phase = engineer.get_event_phase(now, event_start, event_start)
        assert phase == 3, "Within 90 min after should be egress"

    def test_event_phase_normalized(self, engineer):
        """Far from event should be normalized phase."""
        now = datetime(2026, 6, 15, 15, 0)
        event_start = datetime(2026, 6, 15, 13, 0)
        phase = engineer.get_event_phase(now, event_start, event_start)
        assert phase == 4, "Far from event should be normalized"


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_day_of_year_jan_1(self, engineer):
        """January 1 should be day 1 of year."""
        doy = engineer._to_day_of_year(1, 1)
        assert doy == 1

    def test_day_of_year_march_1(self, engineer):
        """March 1 should be day 60 of year."""
        doy = engineer._to_day_of_year(3, 1)
        assert doy == 60  # 31 (Jan) + 28 (Feb) + 1

    def test_day_of_year_dec_31(self, engineer):
        """December 31 should be day 365."""
        doy = engineer._to_day_of_year(12, 31)
        assert doy == 365

    def test_date_in_range_inside(self, engineer):
        """Date within range should return True."""
        result = engineer._date_in_range((6, 15), (6, 1), (6, 30))
        assert result is True

    def test_date_in_range_outside(self, engineer):
        """Date outside range should return False."""
        result = engineer._date_in_range((7, 15), (6, 1), (6, 30))
        assert result is False

    def test_date_in_range_boundary_start(self, engineer):
        """Start date of range should return True."""
        result = engineer._date_in_range((6, 1), (6, 1), (6, 30))
        assert result is True

    def test_date_in_range_boundary_end(self, engineer):
        """End date of range should return True."""
        result = engineer._date_in_range((6, 30), (6, 1), (6, 30))
        assert result is True



class TestInfrastructureFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_disruption_cost_multiplier(self, engineer):
        disruptions = [{'type': 'lane_closure', 'closed_lanes': 1, 'total_lanes': 2}]
        cost = engineer.get_disruption_cost_multiplier(disruptions)
        assert cost == 1.25

    def test_vsl_and_pricing_cost(self, engineer):
        cost = engineer.get_vsl_and_pricing_cost(30.0, 60.0, True, True)
        assert cost == 2.6  # (60/30) * 1.3

    def test_railway_crossing_cost(self, engineer):
        cost = engineer.get_railway_crossing_cost(True, True, True, True)
        assert cost == 10.0  # 2.5 * 4.0


class TestTopologicalFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_road_classification_cost(self, engineer):
        cost = engineer.get_road_classification_cost('motorway', 3)
        assert cost == 1.0

    def test_signalization_cost(self, engineer):
        cost = engineer.get_signalization_cost(5.0)
        assert cost == 1.225

    def test_turn_penalty(self, engineer):
        cost = engineer.get_turn_penalty(90, False, 10.0, False)
        # turn=90 -> U-turn (base 2.0). However in logic: -90 < angle <= -10 is left, angle >= 90 is U-turn.
        # radius_penalty for 10m = 1.2
        # total = 2.0 * 1.2 = 2.4
        assert cost == 2.4
class TestVehicleFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_dimension_compatibility(self, engineer):
        trav, pen = engineer.get_vehicle_dimension_compatibility(4.0, 10000, 4.5, 15000)
        assert trav is True
        assert pen == 1.25  # margin = 0.5, penalty = 1.0 + 0.5 * 0.5

    def test_type_performance(self, engineer):
        cost = engineer.get_vehicle_type_performance('heavy_truck', 10.0, 16.0, 10.0, 'residential', 0.5, 5.0)
        # grade_mult = 1.0 + 0.45 * 10 = 5.5
        # traverse_pen = inf since > 1.5 * 10
        assert cost == float('inf')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
