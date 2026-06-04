"""Phase 2: Load trained XGBoost model and predict traffic-adjusted travel times."""

import pickle
import numpy as np
import logging
from pathlib import Path
from config import MODEL_PATH, ML_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrafficPredictor:
    """Wrapper for ML model to predict travel times based on contextual features."""

    def __init__(self, model_path: Path = MODEL_PATH):
        """
        Load trained XGBoost model.

        Args:
            model_path: Path to pickled XGBoost model
        """
        logger.info(f"💾 Loading ML model from {model_path}...")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        logger.info(f"✅ Model loaded successfully")
        self.feature_names = ML_FEATURES

    def predict_travel_time(
        self,
        hour: int,
        day_of_week: int,
        month: int,
        is_raining: int = 0,
        is_festival_zone: int = 0
    ) -> float:
        """
        Predict travel time in seconds for given conditions.

        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            month: Month (1-12)
            is_raining: Binary (0 or 1)
            is_festival_zone: Binary (0 or 1)

        Returns:
            Predicted travel time in seconds
        """
        # Prepare feature vector in exact order used during training
        features = np.array([[hour, day_of_week, month, is_raining, is_festival_zone]], dtype=float)

        # Predict
        predicted_seconds = self.model.predict(features)[0]

        return max(predicted_seconds, 30)  # Minimum 30 seconds

    def predict_edge_time(
        self,
        base_time_seconds: float,
        hour: int,
        day_of_week: int,
        month: int,
        is_raining: int = 0,
        is_festival_zone: int = 0
    ) -> float:
        """
        Adjust base travel time for an edge using ML predictions.

        This takes the base physical travel time (from distance + speed limit)
        and applies a traffic multiplier based on contextual factors.

        Args:
            base_time_seconds: Base travel time from distance and speed
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            month: Month (1-12)
            is_raining: Binary (0 or 1)
            is_festival_zone: Binary (0 or 1)

        Returns:
            ML-adjusted travel time in seconds
        """
        # Get ML prediction for the conditions
        predicted_time = self.predict_travel_time(hour, day_of_week, month, is_raining, is_festival_zone)

        # Calculate multiplier: how much traffic delays this trip vs. baseline
        # If predicted > base, there's congestion; if predicted < base, road is clear
        if base_time_seconds > 0:
            multiplier = predicted_time / base_time_seconds
            # Clamp multiplier to reasonable range (0.5x to 3x)
            multiplier = max(0.5, min(multiplier, 3.0))
            adjusted_time = base_time_seconds * multiplier
        else:
            adjusted_time = predicted_time

        return max(adjusted_time, 30)  # Minimum 30 seconds

    def batch_predict(self, conditions_list):
        """
        Predict travel times for multiple conditions.

        Args:
            conditions_list: List of dicts with 'hour', 'day_of_week', 'month', 'is_raining', 'is_festival_zone'

        Returns:
            List of predicted travel times in seconds
        """
        results = []
        for cond in conditions_list:
            t = self.predict_travel_time(
                hour=cond.get('hour', 12),
                day_of_week=cond.get('day_of_week', 0),
                month=cond.get('month', 6),
                is_raining=cond.get('is_raining', 0),
                is_festival_zone=cond.get('is_festival_zone', 0)
            )
            results.append(t)
        return results


def test_predictor():
    """Quick test of the predictor."""
    logger.info("=" * 60)
    logger.info("🧪 Testing Traffic Predictor")
    logger.info("=" * 60)

    try:
        predictor = TrafficPredictor()

        # Test 1: Morning commute
        morning_time = predictor.predict_travel_time(hour=8, day_of_week=0, month=6)
        logger.info(f"Morning (8 AM): {morning_time:.1f} seconds")

        # Test 2: Evening rush
        evening_time = predictor.predict_travel_time(hour=18, day_of_week=0, month=6, is_raining=1)
        logger.info(f"Evening (6 PM) in rain: {evening_time:.1f} seconds")

        # Test 3: Weekend
        weekend_time = predictor.predict_travel_time(hour=12, day_of_week=5, month=6)
        logger.info(f"Weekend noon: {weekend_time:.1f} seconds")

        # Test 4: Edge adjustment
        base_time = 300  # 5 minutes base
        adjusted = predictor.predict_edge_time(base_time, hour=18, day_of_week=0, month=6)
        logger.info(f"Base: {base_time}s → Adjusted for peak traffic: {adjusted:.1f}s (multiplier: {adjusted/base_time:.2f}x)")

        logger.info("✅ Predictor tests passed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Predictor test failed: {e}")
        raise


if __name__ == '__main__':
    test_predictor()
