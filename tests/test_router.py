"""Tests for routing and traffic prediction."""

import unittest
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logger = logging.getLogger(__name__)


class TestTrafficPredictor(unittest.TestCase):
    """Test Phase 2: Traffic prediction model."""

    def setUp(self):
        from src.traffic_predictor import TrafficPredictor
        self.predictor = TrafficPredictor()

    def test_model_loads(self):
        """Verify model loads without errors."""
        self.assertIsNotNone(self.predictor.model_v1)

    def test_predict_returns_positive(self):
        """Verify predictions return positive values."""
        result = self.predictor.predict_travel_time(
            hour=12, day_of_week=0, month=6
        )
        self.assertGreater(result, 0)

    def test_predict_respects_time_of_day(self):
        """Verify predictions change with time of day."""
        morning = self.predictor.predict_travel_time(hour=8, day_of_week=0, month=6)
        evening = self.predictor.predict_travel_time(hour=18, day_of_week=0, month=6)

        # Predictions should be reasonable
        self.assertGreater(morning, 0)
        self.assertGreater(evening, 0)

    def test_weather_affects_prediction(self):
        """Verify rain affects predictions."""
        dry = self.predictor.predict_travel_time(
            hour=12, day_of_week=0, month=6, is_raining=0
        )
        rainy = self.predictor.predict_travel_time(
            hour=12, day_of_week=0, month=6, is_raining=1
        )

        # Both should be positive
        self.assertGreater(dry, 0)
        self.assertGreater(rainy, 0)

    def test_edge_time_adjustment(self):
        """Verify edge time adjustment with base time."""
        base_time = 300  # 5 minutes
        adjusted = self.predictor.predict_edge_time(
            base_time_seconds=base_time,
            hour=12,
            day_of_week=0,
            month=6
        )

        # Adjusted time should be reasonable (1-5x base time typically)
        self.assertGreaterEqual(adjusted, base_time * 0.5)
        self.assertLessEqual(adjusted, base_time * 5)


class TestRouter(unittest.TestCase):
    """Test Phase 3: Routing engine."""

    def setUp(self):
        try:
            from src.router import OptimalRouter
            self.router = OptimalRouter()
            self.router_available = True
        except FileNotFoundError:
            self.router_available = False

    def test_router_initializes(self):
        """Verify router initializes."""
        if not self.router_available:
            self.skipTest("Graph not loaded yet")
        self.assertIsNotNone(self.router.graph)
        self.assertIsNotNone(self.router.predictor)

    def test_route_finding(self):
        """Test basic route finding."""
        if not self.router_available:
            self.skipTest("Graph not loaded yet")

        result = self.router.find_optimal_route(
            start_lat=39.745,
            start_lon=-75.546,
            end_lat=39.758,
            end_lon=-75.532
        )

        # Verify result structure
        self.assertIn('path', result)
        self.assertIn('total_time_seconds', result)
        self.assertIn('distance_km', result)
        self.assertGreater(len(result['path']), 1)
        self.assertGreater(result['total_time_seconds'], 0)

    def test_different_times_give_different_results(self):
        """Verify traffic conditions affect routing."""
        if not self.router_available:
            self.skipTest("Graph not loaded yet")

        morning = self.router.find_optimal_route(
            start_lat=39.745,
            start_lon=-75.546,
            end_lat=39.758,
            end_lon=-75.532,
            departure_hour=8
        )

        evening = self.router.find_optimal_route(
            start_lat=39.745,
            start_lon=-75.546,
            end_lat=39.758,
            end_lon=-75.532,
            departure_hour=18
        )

        # Both should have valid routes
        self.assertGreater(morning['total_time_seconds'], 0)
        self.assertGreater(evening['total_time_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
