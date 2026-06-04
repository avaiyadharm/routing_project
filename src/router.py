"""Phase 3: Dynamic routing engine with ML-predicted edge weights using Dijkstra's algorithm."""

import pickle
import heapq
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import networkx as nx

from config import GRAPH_PICKLE_PATH
from traffic_predictor import TrafficPredictor
from utils import find_nearest_node, haversine_distance, format_route_for_display

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimalRouter:
    """Dijkstra-based router with ML-predicted dynamic edge weights."""

    def __init__(self, graph_path: Path = GRAPH_PICKLE_PATH):
        """
        Initialize router with graph and predictor.

        Args:
            graph_path: Path to pickled NetworkX graph
        """
        logger.info(f"📂 Loading graph from {graph_path}...")

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found at {graph_path}. Run graph_loader.py first.")

        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

        logger.info(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        # Initialize traffic predictor
        self.predictor = TrafficPredictor()

        # Cache for node coordinates
        self._node_coords_cache = {}

    def _get_node_coords(self, node: int) -> Tuple[float, float]:
        """Get (lat, lon) for a node."""
        if node not in self._node_coords_cache:
            node_data = self.graph.nodes[node]
            self._node_coords_cache[node] = (node_data['y'], node_data['x'])
        return self._node_coords_cache[node]

    def find_optimal_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        departure_hour: int = 12,
        departure_day: int = 0,
        departure_month: int = 6,
        is_raining: int = 0,
        is_festival_zone: int = 0
    ) -> Dict:
        """
        Find optimal route from source to destination using Dijkstra with ML-predicted weights.

        Args:
            start_lat, start_lon: Source coordinates
            end_lat, end_lon: Destination coordinates
            departure_hour: Hour of day for prediction (0-23)
            departure_day: Day of week for prediction (0=Monday, 6=Sunday)
            departure_month: Month for prediction (1-12)
            is_raining: Binary flag
            is_festival_zone: Binary flag

        Returns:
            Dict with keys:
            - 'path': List of (lat, lon) tuples along the route
            - 'node_ids': List of node IDs
            - 'total_time_seconds': Predicted trip duration
            - 'distance_km': Approximate distance
            - 'num_turns': Number of segments
        """
        logger.info(f"🔍 Finding optimal route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})...")
        logger.info(f"   Departure: Hour={departure_hour}, Day={departure_day}, Month={departure_month}")
        logger.info(f"   Conditions: Raining={is_raining}, Festival={is_festival_zone}")

        # Find nearest start and end nodes
        start_node = find_nearest_node(self.graph, start_lat, start_lon)
        end_node = find_nearest_node(self.graph, end_lat, end_lon)

        logger.info(f"   Start node: {start_node}, End node: {end_node}")

        # Run Dijkstra with ML-adjusted weights
        try:
            path = self._dijkstra_shortest_path(
                start_node,
                end_node,
                departure_hour,
                departure_day,
                departure_month,
                is_raining,
                is_festival_zone
            )
        except nx.NetworkXNoPath:
            logger.error("❌ No path found between source and destination")
            raise

        if not path or len(path) < 2:
            logger.error("❌ Invalid path returned")
            raise ValueError("No valid path found")

        # Calculate route metrics
        total_time_seconds = 0
        total_distance_km = 0

        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]

            # Get base travel time
            edge_data = self.graph[from_node][to_node]
            if isinstance(edge_data, dict):
                base_time = edge_data.get('travel_time_seconds', 0)
            else:
                # MultiDiGraph: take first edge
                base_time = list(edge_data.values())[0].get('travel_time_seconds', 0)

            # Get ML-adjusted time
            ml_time = self.predictor.predict_edge_time(
                base_time,
                departure_hour,
                departure_day,
                departure_month,
                is_raining,
                is_festival_zone
            )
            total_time_seconds += ml_time

            # Distance
            lat1, lon1 = self._get_node_coords(from_node)
            lat2, lon2 = self._get_node_coords(to_node)
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            total_distance_km += dist

        # Convert nodes to coordinates
        route_coords = [self._get_node_coords(node) for node in path]

        result = {
            'path': route_coords,
            'node_ids': path,
            'total_time_seconds': total_time_seconds,
            'total_time_minutes': total_time_seconds / 60,
            'distance_km': total_distance_km,
            'num_turns': len(path),
            'start_node': start_node,
            'end_node': end_node
        }

        logger.info(f"✅ Route found!")
        logger.info(f"   Distance: {result['distance_km']:.2f} km")
        logger.info(f"   Predicted time: {result['total_time_minutes']:.1f} minutes ({result['total_time_seconds']:.0f}s)")
        logger.info(f"   Segments: {result['num_turns']}")

        return result

    def _dijkstra_shortest_path(
        self,
        start: int,
        end: int,
        hour: int,
        day: int,
        month: int,
        is_raining: int,
        is_festival_zone: int
    ) -> List[int]:
        """
        Dijkstra's algorithm with ML-predicted dynamic edge weights.

        Args:
            start, end: Start and end node IDs
            hour, day, month: Temporal context for ML predictions
            is_raining, is_festival_zone: Environmental context

        Returns:
            List of node IDs representing the shortest path
        """
        # Priority queue: (distance, node, path)
        pq = [(0, start, [start])]
        visited = set()
        distances = {start: 0}

        while pq:
            current_dist, current_node, path = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == end:
                return path

            # Get neighbors
            for next_node in self.graph.successors(current_node):
                if next_node in visited:
                    continue

                # Get base edge weight
                edge_data = self.graph[current_node][next_node]

                # Handle MultiDiGraph
                if isinstance(edge_data, dict):
                    base_time = edge_data.get('travel_time_seconds', 0)
                else:
                    base_time = list(edge_data.values())[0].get('travel_time_seconds', 0)

                if base_time <= 0:
                    base_time = 300  # Default 5 minutes

                # Get ML-adjusted weight
                ml_adjusted_time = self.predictor.predict_edge_time(
                    base_time, hour, day, month, is_raining, is_festival_zone
                )

                new_distance = current_dist + ml_adjusted_time

                # Only process if we found a better path
                if next_node not in distances or new_distance < distances[next_node]:
                    distances[next_node] = new_distance
                    new_path = path + [next_node]
                    heapq.heappush(pq, (new_distance, next_node, new_path))

        # No path found
        raise nx.NetworkXNoPath(f"No path between {start} and {end}")


def test_router():
    """Quick test of the router."""
    logger.info("=" * 60)
    logger.info("🧪 Testing Optimal Router")
    logger.info("=" * 60)

    try:
        router = OptimalRouter()
        logger.info("✅ Router initialized successfully")

        # Test with Wilmington, Delaware coordinates (approximation)
        result = router.find_optimal_route(
            start_lat=39.745,
            start_lon=-75.546,
            end_lat=39.758,
            end_lon=-75.532,
            departure_hour=9,
            departure_day=0,
            departure_month=6,
            is_raining=0,
            is_festival_zone=0
        )

        logger.info(f"📍 Route from {result['start_node']} to {result['end_node']}")
        logger.info(f"   Distance: {result['distance_km']:.2f} km")
        logger.info(f"   Time: {result['total_time_minutes']:.1f} min ({result['total_time_seconds']:.0f}s)")
        logger.info(f"   First 5 waypoints: {result['path'][:5]}")

        logger.info("✅ Router tests passed!")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.warning(f"⚠️ Cannot run router test yet: {e}")
        logger.info("   This is expected if graph_loader.py hasn't finished yet")


if __name__ == '__main__':
    test_router()
