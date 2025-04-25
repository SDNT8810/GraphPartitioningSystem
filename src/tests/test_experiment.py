import unittest
import os
import tempfile
import yaml
import torch
from pathlib import Path
from ..core.graph import Graph, Partition
from ..config.system_config import PartitionConfig, AgentConfig, GraphConfig, get_configs, load_config
from ..strategies.spectral import SpectralPartitioningStrategy
from ..strategies.dynamic_partitioning import DynamicPartitioning
from ..strategies.hybrid import HybridPartitioningStrategy
from ..utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance

def init_test_graph(graph_config):
    """Initialize a test graph with partitions."""
    # Create graph
    num_nodes = graph_config.num_nodes
    edge_probability = graph_config.edge_probability
    graph = Graph(num_nodes, edge_probability)
    
    # Initialize graph with some edges
    graph.adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes))
    # Make symmetric
    graph.adj_matrix = ((graph.adj_matrix + graph.adj_matrix.t()) > 0).float()
    # Zero diagonal
    graph.adj_matrix.fill_diagonal_(0)
    # Initialize node features
    graph.node_features = torch.randn(num_nodes, 4)  # 4 features per node
    
    # Initialize partitions
    num_partitions = 2  # Start with 2 partitions
    num_nodes_per_partition = num_nodes // num_partitions
    graph.partitions = {}
    for i in range(num_partitions):
        partition = Partition(i)
        for node in range(i*num_nodes_per_partition, (i+1)*num_nodes_per_partition):
            partition.add_node(node)
        graph.partitions[i] = partition
    
    return graph

def run_single_experiment(args):
    """Run a single experiment with the given arguments."""
    try:
        strategy_name = args['strategy']
        config = args['config']
        graph = args['graph']
        agent_config = args.get('agent_config')
        
        # Create strategy instance
        if strategy_name == 'spectral':
            strategy = SpectralPartitioningStrategy(config)
        elif strategy_name == 'dynamic':
            strategy = DynamicPartitioning(config)
            if agent_config:
                strategy.initialize(graph, agent_config)
        elif strategy_name == 'hybrid':
            strategy = HybridPartitioningStrategy(config)
            if agent_config:
                strategy.rl = DynamicPartitioning(config)
                strategy.rl.initialize(graph, agent_config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        # Run partitioning
        partitions = strategy.partition(graph)
        
        # Compute metrics
        metrics = {
            'cut_size': compute_cut_size(graph, partitions),
            'balance': compute_balance(partitions),
            'conductance': compute_conductance(graph, partitions)
        }
        
        return metrics
    except Exception as e:
        print(f"Error in run: {str(e)}")
        return None

def run_experiment(args):
    """Run multiple experiments and return results."""
    results = []
    for i in range(args['num_runs']):
        try:
            result = run_single_experiment(args)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f'Error in run {i}: {e}')
    return results

class TestExperiment(unittest.TestCase):
    def setUp(self):
        # Load test configuration
        config_path = str(Path(__file__).parent.parent.parent / 'configs' / 'test_config.yaml')
        self.graph_config, self.partition_config, self.agent_config = get_configs(config_path)
        self.test_config = load_config(config_path)

    def test_spectral_experiment(self):
        # Create test graph
        graph = init_test_graph(self.graph_config)
        
        # Run experiment
        args = {
            'strategy': 'spectral',
            'config': self.partition_config,
            'graph': graph,
            'num_runs': self.test_config['test']['num_runs']
        }
        results = run_experiment(args)
        
        # Check results
        self.assertGreater(len(results), 0)  # At least one successful run
        for run_results in results:
            self.assertIsInstance(run_results, dict)
            self.assertIn('cut_size', run_results)
            self.assertIn('balance', run_results)
            self.assertIn('conductance', run_results)
    
    def test_dynamic_experiment(self):
        # Create test graph
        graph = init_test_graph(self.graph_config)
        
        # Run experiment
        args = {
            'strategy': 'dynamic',
            'config': self.partition_config,
            'graph': graph,
            'agent_config': self.agent_config,
            'num_runs': self.test_config['test']['num_runs']
        }
        results = run_experiment(args)
        
        # Check results
        self.assertGreater(len(results), 0)  # At least one successful run
        for run_results in results:
            self.assertIsInstance(run_results, dict)
            self.assertIn('cut_size', run_results)
            self.assertIn('balance', run_results)
            self.assertIn('conductance', run_results)
    
    def test_hybrid_experiment(self):
        # Create test graph
        graph = init_test_graph(self.graph_config)
        
        # Run experiment
        args = {
            'strategy': 'hybrid',
            'config': self.partition_config,
            'graph': graph,
            'agent_config': self.agent_config,
            'num_runs': self.test_config['test']['num_runs']
        }
        results = run_experiment(args)
        
        # Check results
        self.assertGreater(len(results), 0)  # At least one successful run
        for run_results in results:
            self.assertIsInstance(run_results, dict)
            self.assertIn('cut_size', run_results)
            self.assertIn('balance', run_results)
            self.assertIn('conductance', run_results)
    
    def test_invalid_strategy(self):
        # Create test graph
        graph = init_test_graph(self.graph_config)
        
        # Run experiment with invalid strategy
        args = {
            'strategy': 'invalid',
            'config': self.partition_config,
            'graph': graph,
            'num_runs': self.test_config['test']['num_runs']
        }
        results = run_experiment(args)
        
        # Check results
        self.assertEqual(len(results), 0)  # No successful runs
    
    def test_config_variations(self):
        # Test different partition configurations
        variations = [
            PartitionConfig(
                num_partitions=2,
                balance_weight=0.8,
                cut_size_weight=0.1,
                conductance_weight=0.1,
                use_laplacian=True
            ),
            PartitionConfig(
                num_partitions=3,
                balance_weight=0.3,
                cut_size_weight=0.6,
                conductance_weight=0.1,
                use_laplacian=True
            )
        ]
        
        # Create test graph
        graph = init_test_graph(self.graph_config)
        
        for config in variations:
            # Run experiment
            args = {
                'strategy': 'spectral',
                'config': config,
                'graph': graph,
                'num_runs': 1  # Single run for each variation
            }
            results = run_experiment(args)
            
            # Check results
            self.assertGreater(len(results), 0)  # At least one successful run
            for run_results in results:
                self.assertIsInstance(run_results, dict)
                self.assertIn('cut_size', run_results)
                self.assertIn('balance', run_results)
                self.assertIn('conductance', run_results)

if __name__ == '__main__':
    unittest.main()
