import unittest
import torch
from ..core.graph import Graph, Partition
import os

class TestGraphSerialization(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(num_nodes=4, edge_probability=1.0)
        self.graph.node_features = torch.arange(16).reshape(4,4).float()
        # Create partitions
        p0 = Partition(id=0, nodes={0,1}, density=0.5, conductance=0.1)
        p1 = Partition(id=1, nodes={2,3}, density=0.7, conductance=0.2)
        self.graph.partitions = {0: p0, 1: p1}
        self.path = 'test_graph.json'

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_to_dict_and_from_dict(self):
        d = self.graph.to_dict()
        g2 = Graph.from_dict(d)
        self.assertEqual(g2.num_nodes, self.graph.num_nodes)
        self.assertTrue(torch.equal(g2.adj_matrix, self.graph.adj_matrix))
        self.assertTrue(torch.equal(g2.node_features, self.graph.node_features))
        self.assertEqual(set(g2.partitions.keys()), set(self.graph.partitions.keys()))
        for pid in g2.partitions:
            self.assertEqual(g2.partitions[pid].nodes, self.graph.partitions[pid].nodes)
            self.assertAlmostEqual(g2.partitions[pid].density, self.graph.partitions[pid].density)
            self.assertAlmostEqual(g2.partitions[pid].conductance, self.graph.partitions[pid].conductance)

    def test_save_and_load(self):
        self.graph.save(self.path)
        loaded = Graph.load(self.path)
        self.assertEqual(loaded.num_nodes, self.graph.num_nodes)
        self.assertTrue(torch.equal(loaded.adj_matrix, self.graph.adj_matrix))
        self.assertTrue(torch.equal(loaded.node_features, self.graph.node_features))
        self.assertEqual(set(loaded.partitions.keys()), set(self.graph.partitions.keys()))
        for pid in loaded.partitions:
            self.assertEqual(loaded.partitions[pid].nodes, self.graph.partitions[pid].nodes)
            self.assertAlmostEqual(loaded.partitions[pid].density, self.graph.partitions[pid].density)
            self.assertAlmostEqual(loaded.partitions[pid].conductance, self.graph.partitions[pid].conductance)

if __name__ == '__main__':
    unittest.main()
