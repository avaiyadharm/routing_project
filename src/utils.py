"""Utility functions for routing system."""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in kilometers.

    Args:
        lat1, lon1: First point latitude/longitude in degrees
        lat2, lon2: Second point latitude/longitude in degrees

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def find_nearest_node(graph, lat: float, lon: float) -> int:
    """
    Find the nearest node in graph to given coordinates.

    Args:
        graph: NetworkX graph with 'y' (lat) and 'x' (lon) attributes on nodes
        lat, lon: Target coordinates in degrees

    Returns:
        Node ID of nearest node
    """
    nodes = list(graph.nodes(data=True))
    coords = np.array([[node[1].get('y', 0), node[1].get('x', 0)] for node in nodes])

    tree = cKDTree(coords)
    distance, idx = tree.query([lat, lon])

    return nodes[idx][0]


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate geographic coordinates are in valid range."""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def format_route_for_display(route_nodes: List[int], node_coords: dict) -> List[Tuple[float, float]]:
    """
    Convert list of node IDs to list of (lat, lon) coordinates.

    Args:
        route_nodes: List of node IDs in route
        node_coords: Dict mapping node ID to (lat, lon)

    Returns:
        List of (lat, lon) tuples
    """
    return [node_coords[node] for node in route_nodes]
