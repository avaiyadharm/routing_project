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
