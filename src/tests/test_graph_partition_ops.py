import unittest
import torch
from ..core.graph import Graph, Partition

class TestGraphPartitionOps(unittest.TestCase):
    def setUp(self):
        self.graph = Graph(num_nodes=6, edge_probability=1.0)
        self.graph.node_features = torch.randn(6, 4)
        # Start with 3 partitions
        self.graph.partitions = {
            0: Partition(id=0, nodes={0, 1}),
            1: Partition(id=1, nodes={2, 3}),
            2: Partition(id=2, nodes={4, 5})
        }

    def test_is_balanced(self):
        self.assertTrue(self.graph.is_balanced())
        # Unbalance
        self.graph.partitions[0].nodes.add(2)
        self.graph.partitions[1].nodes.remove(2)
        self.assertFalse(self.graph.is_balanced())

    def test_balance_partitions(self):
        self.graph.partitions[0].nodes.add(2)
        self.graph.partitions[1].nodes.remove(2)
        self.graph.balance_partitions()
        sizes = [len(p.nodes) for p in self.graph.partitions.values()]
        self.assertTrue(max(sizes) - min(sizes) <= 1)
        self.assertTrue(self.graph.is_balanced())

    def test_merge_partitions(self):
        self.graph.merge_partitions(0, 1)
        self.assertIn(0, self.graph.partitions)
        self.assertNotIn(1, self.graph.partitions)
        self.assertEqual(len(self.graph.partitions[0].nodes), 4)

    def test_split_partition(self):
        self.graph.merge_partitions(0, 1)  # Now partition 0 has 4 nodes
        self.graph.split_partition(0)
        sizes = [len(p.nodes) for p in self.graph.partitions.values()]
        self.assertIn(2, sizes)
        self.assertIn(2, sizes)
        self.assertEqual(sum(sizes), 6)

if __name__ == '__main__':
    unittest.main()
