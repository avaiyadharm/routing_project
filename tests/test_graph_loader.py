"""Tests for graph loading and edge features."""

import unittest
from pathlib import Path
import logging

from src.config import GRAPH_PICKLE_PATH, SEGMENT_FEATURES_PATH

logger = logging.getLogger(__name__)


class TestGraphLoader(unittest.TestCase):
    """Test Phase 1: Graph loading and feature extraction."""

    def test_graph_file_exists(self):
        """Verify graph pickle file was created."""
        self.assertTrue(
            GRAPH_PICKLE_PATH.exists(),
            f"Graph file not found at {GRAPH_PICKLE_PATH}. Run: python src/graph_loader.py"
        )

    def test_segment_features_exist(self):
        """Verify segment features CSV was created."""
        self.assertTrue(
            SEGMENT_FEATURES_PATH.exists(),
            f"Features file not found at {SEGMENT_FEATURES_PATH}. Run: python src/graph_loader.py"
        )

    def test_graph_structure(self):
        """Verify graph has reasonable structure."""
        import pickle
        import networkx as nx

        if not GRAPH_PICKLE_PATH.exists():
            self.skipTest("Graph not loaded yet")

        with open(GRAPH_PICKLE_PATH, 'rb') as f:
            graph = pickle.load(f)

        # Check it's a graph
        self.assertIsInstance(graph, (nx.DiGraph, nx.MultiDiGraph))

        # Check has nodes
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)

        # Check nodes have coordinates
        for node in list(graph.nodes())[:5]:
            node_data = graph.nodes[node]
            self.assertIn('x', node_data)  # longitude
            self.assertIn('y', node_data)  # latitude

    def test_segment_features_format(self):
        """Verify segment features have expected columns."""
        import pandas as pd

        if not SEGMENT_FEATURES_PATH.exists():
            self.skipTest("Features not loaded yet")

        df = pd.read_csv(SEGMENT_FEATURES_PATH)

        # Check expected columns
        expected_cols = ['segment_id', 'from_node', 'to_node', 'length_meters', 'highway_type']
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertEqual(df['segment_id'].dtype, int)
        self.assertGreater(len(df), 0)


if __name__ == '__main__':
    unittest.main()
