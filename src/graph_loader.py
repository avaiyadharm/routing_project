"""Phase 1: Load OSM data and construct NetworkX graph with traffic features."""

import osmnx as ox
import networkx as nx
import pandas as pd
import pickle
import logging
from pathlib import Path
from config import (
    DELAWARE_OSM_FILE, GRAPH_PICKLE_PATH, SEGMENT_FEATURES_PATH,
    SIMPLIFY_GRAPH, CUSTOM_FILTER, DATA_DIR
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_osm_graph() -> nx.MultiDiGraph:
    """
    Load OSM data for a manageable region and convert to NetworkX graph.

    The graph will have:
    - Nodes with 'x' (longitude), 'y' (latitude) attributes
    - Edges with 'length' (meters), 'highway' (road type), 'maxspeed' attributes

    Returns:
        NetworkX MultiDiGraph
    """
    logger.info("📂 Loading OSM data for Wilmington, Delaware area...")

    try:
        # Download a small manageable region: Wilmington city area (~5km radius)
        logger.info("⬇️ Downloading from OpenStreetMap (Overpass API)...")
        graph = ox.graph_from_address(
            "Wilmington, Delaware",
            dist=3000,  # 3km radius
            network_type='drive',
            simplify=SIMPLIFY_GRAPH
        )
        logger.info(f"✅ Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("🔄 Creating synthetic small graph for testing...")

        # Fallback: create a minimal synthetic graph for testing
        graph = nx.MultiDiGraph()

        # Create a 5x5 grid of test nodes
        nodes_data = [
            (0, {'y': 39.745, 'x': -75.546}),
            (1, {'y': 39.750, 'x': -75.546}),
            (2, {'y': 39.755, 'x': -75.546}),
            (3, {'y': 39.745, 'x': -75.540}),
            (4, {'y': 39.750, 'x': -75.540}),
            (5, {'y': 39.755, 'x': -75.540}),
        ]

        graph.add_nodes_from(nodes_data)

        # Add edges
        edges_data = [
            (0, 1, 0, {'length': 556, 'highway': 'residential', 'travel_time_seconds': 60}),
            (1, 2, 0, {'length': 556, 'highway': 'residential', 'travel_time_seconds': 60}),
            (3, 4, 0, {'length': 556, 'highway': 'residential', 'travel_time_seconds': 60}),
            (4, 5, 0, {'length': 556, 'highway': 'residential', 'travel_time_seconds': 60}),
            (0, 3, 0, {'length': 664, 'highway': 'primary', 'travel_time_seconds': 45}),
            (1, 4, 0, {'length': 664, 'highway': 'primary', 'travel_time_seconds': 45}),
            (2, 5, 0, {'length': 664, 'highway': 'primary', 'travel_time_seconds': 45}),
        ]

        graph.add_edges_from(edges_data)

        logger.info(f"✅ Created synthetic graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        return graph


def extract_edge_features(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Extract features from graph edges for ML predictions.

    Creates a dataframe where each row is an edge with:
    - segment_id: unique edge identifier
    - from_node, to_node: node IDs
    - length: road segment length in meters
    - highway: road type (residential, primary, secondary, etc.)
    - maxspeed: speed limit (converted to km/h if available)
    - from_lat, from_lon, to_lat, to_lon: coordinates

    Args:
        graph: NetworkX MultiDiGraph from osmnx

    Returns:
        DataFrame with edge features
    """
    logger.info("🧠 Extracting edge features from graph...")

    edges_data = []
    segment_id = 0

    for from_node, to_node, edge_key, edge_data in graph.edges(keys=True, data=True):
        node_from = graph.nodes[from_node]
        node_to = graph.nodes[to_node]

        # Extract length (default to 0 if missing)
        length = edge_data.get('length', 0)

        # Extract highway type
        highway = edge_data.get('highway', 'residential')
        if isinstance(highway, list):
            highway = highway[0]

        # Extract maxspeed and convert to km/h
        maxspeed = edge_data.get('maxspeed', None)
        if maxspeed:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if isinstance(maxspeed, str):
                try:
                    if maxspeed.endswith('mph'):
                        maxspeed = float(maxspeed.replace('mph', '')) * 1.60934
                    else:
                        maxspeed = float(maxspeed)
                except:
                    maxspeed = None

        edges_data.append({
            'segment_id': segment_id,
            'from_node': from_node,
            'to_node': to_node,
            'edge_key': edge_key,
            'length_meters': length,
            'highway_type': highway,
            'maxspeed_kmh': maxspeed,
            'from_lat': node_from.get('y'),
            'from_lon': node_from.get('x'),
            'to_lat': node_to.get('y'),
            'to_lon': node_to.get('x'),
        })
        segment_id += 1

    features_df = pd.DataFrame(edges_data)
    logger.info(f"✅ Extracted {len(features_df)} edge features")
    return features_df


def add_travel_time_to_graph(graph: nx.MultiDiGraph, features_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Add base travel time (in seconds) to graph edges based on length and maxspeed.

    This provides a baseline static travel time for Dijkstra before ML adjustments.

    Args:
        graph: NetworkX graph
        features_df: DataFrame with edge features

    Returns:
        Updated graph with 'travel_time_seconds' attribute on edges
    """
    logger.info("⏱️ Computing base travel times from length and speed limits...")

    DEFAULT_SPEED = 50  # km/h default

    for idx, row in features_df.iterrows():
        from_node = row['from_node']
        to_node = row['to_node']
        edge_key = row['edge_key']

        length_m = row['length_meters']
        maxspeed = row['maxspeed_kmh'] or DEFAULT_SPEED

        # Clamp speed to reasonable range
        maxspeed = max(10, min(maxspeed, 120))

        # travel_time = (distance_km / speed_kmh) * 3600 seconds
        travel_time_seconds = (length_m / 1000) / maxspeed * 3600

        # Add to graph
        if from_node in graph and to_node in graph[from_node]:
            graph[from_node][to_node][edge_key]['travel_time_seconds'] = travel_time_seconds

    logger.info("✅ Base travel times added")
    return graph


def save_graph_and_features(graph: nx.MultiDiGraph, features_df: pd.DataFrame):
    """
    Save graph and features to disk for Phase 3 use.

    Args:
        graph: NetworkX graph to save
        features_df: Edge features DataFrame
    """
    logger.info(f"💾 Saving graph to {GRAPH_PICKLE_PATH}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_PICKLE_PATH, 'wb') as f:
        pickle.dump(graph, f)
    logger.info("✅ Graph saved")

    logger.info(f"💾 Saving edge features to {SEGMENT_FEATURES_PATH}...")
    features_df.to_csv(SEGMENT_FEATURES_PATH, index=False)
    logger.info("✅ Features saved")


def main():
    """Main Phase 1 execution: Load OSM, extract features, save graph."""
    logger.info("=" * 60)
    logger.info("🚀 PHASE 1: GRAPH CONSTRUCTION & FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Load graph
    graph = load_osm_graph()

    # Extract features
    features_df = extract_edge_features(graph)

    # Add base travel times
    graph = add_travel_time_to_graph(graph, features_df)

    # Save
    save_graph_and_features(graph, features_df)

    logger.info("=" * 60)
    logger.info(f"✅ PHASE 1 COMPLETE!")
    logger.info(f"   Nodes: {graph.number_of_nodes()}")
    logger.info(f"   Edges: {graph.number_of_edges()}")
    logger.info(f"   Features: {len(features_df)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
