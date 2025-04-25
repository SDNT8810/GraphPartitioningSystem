import unittest
import numpy as np
from ..utils import graph_metrics

class DummyPartition:
    def __init__(self, nodes, conductance=0.0):
        self.nodes = set(nodes)
        self.conductance = conductance
    def __len__(self):
        return len(self.nodes)

class DummyGraph:
    def __init__(self, edges):
        self.edges = edges
    def get_neighbors(self, node):
        return [v for u, v in self.edges if u == node] + [u for u, v in self.edges if v == node]

class TestGraphMetrics(unittest.TestCase):
    def test_cut_size(self):
        # Two partitions, one edge between them
        graph = DummyGraph(edges=[(0, 1), (1, 2), (2, 3)])
        partitions = [DummyPartition([0, 1]), DummyPartition([2, 3])]
        cut = graph_metrics.compute_cut_size(graph, partitions)
        self.assertEqual(cut, 1)
    def test_balance(self):
        partitions = [DummyPartition([0, 1]), DummyPartition([2, 3, 4])]
        balance = graph_metrics.compute_balance(partitions)
        self.assertAlmostEqual(balance, 2/3)
    def test_conductance(self):
        graph = DummyGraph(edges=[(0, 1), (1, 2), (2, 3)])
        partitions = [DummyPartition([0, 1]), DummyPartition([2, 3])]
        conductance = graph_metrics.compute_conductance(graph, partitions)
        # One edge between partitions, one internal edge in each partition
        # New formula: conductance = external_edges / (internal_edges + external_edges)
        # For each partition:
        # Partition 1: 1 external edge, 1 internal edge -> 1/(1+1) = 0.5
        # Partition 2: 1 external edge, 1 internal edge -> 1/(1+1) = 0.5
        # Mean conductance = 0.5
        self.assertAlmostEqual(conductance, 0.5)

if __name__ == '__main__':
    unittest.main()
