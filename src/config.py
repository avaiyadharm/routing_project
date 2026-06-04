"""Configuration constants for the routing system."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DELAWARE_OSM_FILE = PROJECT_ROOT / "delaware-latest.osm.pbf"
GRAPH_PICKLE_PATH = DATA_DIR / "delaware_graph.pkl"
SEGMENT_FEATURES_PATH = DATA_DIR / "segment_features.csv"

# Model paths
MODEL_PATH = PROJECT_ROOT / "traffic_xgb_model.pkl"
TRAINING_DATA_PATH = PROJECT_ROOT / "train.csv"

# ML Features (must match training features)
ML_FEATURES = ['hour_of_day', 'day_of_week', 'month', 'is_raining', 'is_festival_zone']

# Routing parameters
SPEED_LIMIT_DEFAULT = 30  # km/h when unknown
MAX_REASONABLE_SPEED = 130  # km/h for highways
MIN_REASONABLE_SPEED = 10   # km/h in dense areas

# Graph parameters
SIMPLIFY_GRAPH = True
RETAIN_ALL = False
CUSTOM_FILTER = '["highway"]["area"!~"yes"]["access"!~"private"]'

# Dijkstra search parameters
SEARCH_RADIUS = 0.05  # degrees (~5.5 km at equator)
MAX_RESULTS_NEAREST_NODE = 1

# ==================== FEATURE ENGINEERING CONFIGURATION ====================

# Feature category enable/disable flags
ENABLE_TEMPORAL_FEATURES = True
ENABLE_METEOROLOGICAL_FEATURES = True
ENABLE_EVENT_FEATURES = True
ENABLE_INFRASTRUCTURE_FEATURES = False  # Phase 3
ENABLE_TOPOLOGICAL_FEATURES = False     # Phase 3
ENABLE_VEHICLE_FEATURES = False         # Phase 3

# ML Models
MODEL_V1_PATH = MODEL_PATH  # Original 5-feature model
MODEL_V2_PATH = PROJECT_ROOT / "traffic_xgb_model_v2.pkl"  # Expanded 50+ features
USE_LEGACY_MODEL = True  # Switch to False when v2 is trained

# Enhanced ML Features (Phase 1: Temporal)
ML_FEATURES_V2_TEMPORAL = [
    'hour_of_day', 'day_of_week', 'month', 'day_of_month',
    'holiday_state_ordinal', 'school_season_state', 'school_phase',
    'school_hours_active', 'days_to_payday', 'payday_cycle_sin',
    'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
]

# Enhanced ML Features (Phase 2: Meteorological + Events)
ML_FEATURES_V2_WEATHER = [
    'precipitation_intensity_mmhr', 'precip_cost_multiplier',
    'visibility_meters', 'visibility_cost_multiplier',
    'glare_risk_indicator', 'road_surface_state',
]

ML_FEATURES_V2_EVENTS = [
    'event_distance_km', 'event_capacity_scale', 'event_phase_ordinal',
]

# Full feature list (for future use)
ML_FEATURES_V2_FULL = (
    ML_FEATURES_V2_TEMPORAL +
    ML_FEATURES_V2_WEATHER +
    ML_FEATURES_V2_EVENTS
)

# Regional holiday calendars (2026)
HOLIDAYS_US_2026 = [
    (1, 1),    # New Year
    (1, 19),   # MLK Jr. Day
    (2, 16),   # Presidents Day
    (5, 25),   # Memorial Day
    (7, 4),    # Independence Day
    (9, 7),    # Labor Day
    (10, 12),  # Columbus Day
    (11, 26),  # Thanksgiving
    (11, 27),  # Day After Thanksgiving
    (12, 25),  # Christmas
]

# School calendar (Delaware 2025-2026 academic year)
SCHOOL_CALENDAR_DELAWARE = {
    'semester_start': (8, 15),
    'semester_end': (6, 15),
    'breaks': [
        ((11, 24), (11, 28)),   # Thanksgiving break
        ((12, 22), (1, 5)),     # Winter break
        ((3, 16), (3, 20)),     # Spring break
    ]
}

# Weather and precipitation thresholds
MIN_PRECIPITATION_FOR_EFFECT = 0.5  # mm/hr
MAX_VISIBILITY_CLEAR = 10000  # meters
MIN_VISIBILITY_HAZARD = 1000   # meters

# Event database configuration
EVENT_RADIUS_MAPPING = {
    'festival_100k': 3.0,        # 100K+ capacity
    'festival_50k': 2.5,         # 50K-100K capacity
    'stadium': 2.0,              # 30K-50K capacity
    'arena': 1.5,                # 10K-30K capacity
    'concert_hall': 1.0,         # 5K-10K capacity
}

# Payday configuration
PAYDAY_DAYS = [1, 15]  # 1st and 15th of each month
PAYDAY_IMPACT_WINDOW = 2  # days before/after payday
