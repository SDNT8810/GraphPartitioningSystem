import unittest
import torch
import numpy as np
from ..core.graph import Graph, Partition
from ..strategies.spectral import SpectralPartitioningStrategy
from ..strategies.dynamic_partitioning import DynamicPartitioning
from ..strategies.hybrid import HybridPartitioningStrategy
from ..config.system_config import PartitionConfig, AgentConfig
from ..utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance

class TestStrategies(unittest.TestCase):
    def setUp(self):
        # Create configs
        self.config = PartitionConfig(
            num_partitions=2,  # Set to 2 partitions
            balance_weight=0.5,
            cut_size_weight=0.3,
            conductance_weight=0.2
        )
        self.agent_config = AgentConfig(
            learning_rate=0.001,
            epsilon=0.1,
            num_episodes=10,
            max_steps=10
        )
        
        # Create a small test graph
        self.graph = Graph(num_nodes=10, edge_probability=0.3)
        # Initialize graph with some edges
        self.graph.adj_matrix = torch.randint(0, 2, (10, 10))
        # Make symmetric
        self.graph.adj_matrix = ((self.graph.adj_matrix + self.graph.adj_matrix.t()) > 0).float()
        # Zero diagonal
        self.graph.adj_matrix.fill_diagonal_(0)
        # Initialize node features
        self.graph.node_features = torch.randn(10, 4)  # 4 features per node
        # Initialize partitions
        self.graph.partitions = {}
        for i in range(2):  # Start with 2 partitions
            partition = Partition(i)
            for node in range(i*5, (i+1)*5):  # 5 nodes per partition
                partition.add_node(node)
            self.graph.partitions[i] = partition
        
    def test_spectral_strategy(self):
        strategy = SpectralPartitioningStrategy(self.config)
        partitions = strategy.partition(self.graph)
        # Check basic properties
        self.assertIsInstance(partitions, dict)
        self.assertTrue(all(isinstance(nodes, set) for nodes in partitions.values()))
        # Check that all nodes are assigned
        all_nodes = set().union(*partitions.values())
        self.assertEqual(len(all_nodes), self.graph.num_nodes)
        
    def test_dynamic_strategy(self):
        strategy = DynamicPartitioning(self.config)
        # Initialize graph and agent for dynamic strategy
        strategy.initialize(self.graph, self.agent_config)
        partitions = strategy.partition(self.graph)
        self.assertIsNotNone(partitions)
        self.assertEqual(len(partitions), 2)
        self.assertGreater(compute_balance(partitions), 0.5)
        
    def test_hybrid_strategy(self):
        strategy = HybridPartitioningStrategy(self.config)
        # Initialize the dynamic partitioning component
        strategy.rl = DynamicPartitioning(self.config)
        strategy.rl.initialize(self.graph, self.agent_config)
        partitions = strategy.partition(self.graph)
        self.assertIsNotNone(partitions)
        self.assertEqual(len(partitions), 2)
        self.assertGreater(compute_balance(partitions), 0.5)
        
    def test_spectral_strategy(self):
        strategy = SpectralPartitioningStrategy(self.config)
        partitions = strategy.partition(self.graph)
        self.assertIsNotNone(partitions)
        self.assertEqual(len(partitions), 2)  # Should match num_partitions in config
        self.assertGreater(compute_balance(partitions), 0.5)
        
    def test_strategy_comparison(self):
        strategies = [
            SpectralPartitioningStrategy(self.config),
            DynamicPartitioning(self.config),
            HybridPartitioningStrategy(self.config)
        ]
        
        results = []
        for strategy in strategies:
            if isinstance(strategy, DynamicPartitioning):
                strategy.initialize(self.graph, self.agent_config)
            elif isinstance(strategy, HybridPartitioningStrategy):
                strategy.rl = DynamicPartitioning(self.config)
                strategy.rl.initialize(self.graph, self.agent_config)
                
            partitions = strategy.partition(self.graph)
            results.append({
                'cut_size': compute_cut_size(self.graph, partitions),
                'balance': compute_balance(partitions),
                'conductance': compute_conductance(self.graph, partitions)
            })
        
        # Compare results
        self.assertEqual(len(results), 3)
        balances = [r['balance'] for r in results]
        self.assertTrue(all(b > 0.5 for b in balances))

if __name__ == '__main__':
    unittest.main()
