import unittest
import torch
from ..core.graph import Graph, Partition

class TestGraphValidation(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(num_nodes=4, edge_probability=1.0)
        self.graph.node_features = torch.randn(4, 4)
        p0 = Partition(id=0, nodes={0, 1})
        p1 = Partition(id=1, nodes={2, 3})
        self.graph.partitions = {0: p0, 1: p1}

    def test_valid_graph(self):
        self.assertTrue(self.graph.validate())

    def test_orphan_node(self):
        self.graph.partitions[0].nodes.remove(1)
        with self.assertRaises(ValueError) as ctx:
            self.graph.validate()
        self.assertIn("missing", str(ctx.exception))

    def test_duplicate_node(self):
        self.graph.partitions[1].nodes.add(0)
        with self.assertRaises(ValueError) as ctx:
            self.graph.validate()
        self.assertIn("multiple", str(ctx.exception))

    def test_asymmetric_adjacency(self):
        self.graph.adj_matrix[0, 1] = 0  # break symmetry
        with self.assertRaises(ValueError) as ctx:
            self.graph.validate()
        self.assertIn("symmetric", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
