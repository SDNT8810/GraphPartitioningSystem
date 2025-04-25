import torch
import numpy as np
import networkx as nx
import pytest
from ..core.graph import Graph, Partition
from ..strategies.spectral import SpectralPartitioningStrategy
from ..strategies.dynamic_partitioning import DynamicPartitioning
from ..strategies.hybrid import HybridPartitioningStrategy
from ..config.system_config import PartitionConfig, AgentConfig
from ..utils.graph_metrics import compute_cut_size, compute_balance, compute_conductance

@pytest.fixture
def test_config():
    return PartitionConfig(
        num_partitions=2,  # Set to 2 partitions
        balance_weight=0.5,
        cut_size_weight=0.3,
        conductance_weight=0.2
    )

@pytest.fixture
def test_agent_config():
    return AgentConfig(
        learning_rate=0.001,
        epsilon=0.1,
        num_episodes=10,
        max_steps=10
    )

@pytest.fixture
def test_graph():
    graph = Graph(num_nodes=10, edge_probability=0.3)
    # Initialize graph with some edges
    graph.adj_matrix = torch.randint(0, 2, (10, 10))
    # Make symmetric
    graph.adj_matrix = ((graph.adj_matrix + graph.adj_matrix.t()) > 0).float()
    # Zero diagonal
    graph.adj_matrix.fill_diagonal_(0)
    # Initialize node features
    graph.node_features = torch.randn(10, 4)  # 4 features per node
    # Initialize partitions
    graph.partitions = {}
    for i in range(2):  # Start with 2 partitions
        partition = Partition(i)
        for node in range(i*5, (i+1)*5):  # 5 nodes per partition
            partition.add_node(node)
        graph.partitions[i] = partition
    return graph
        
def test_spectral_strategy(test_config, test_graph):
    strategy = SpectralPartitioningStrategy(test_config)
    partitions = strategy.partition(test_graph)
    # Check basic properties
    assert isinstance(partitions, dict)
    assert all(isinstance(p, Partition) for p in partitions.values())
    # Check that all nodes are assigned
    all_nodes = set().union(*(p.nodes for p in partitions.values()))
    assert len(all_nodes) == test_graph.num_nodes

def test_dynamic_strategy(test_config, test_agent_config, test_graph):
    strategy = DynamicPartitioning(test_config)
    # Initialize graph and agent for dynamic strategy
    strategy.initialize(test_graph, test_agent_config)
    partitions = strategy.partition(test_graph)
    assert partitions is not None
    assert len(partitions) == 2
    assert compute_balance(partitions) > 0.5

def test_hybrid_strategy(test_config, test_agent_config, test_graph):
    strategy = HybridPartitioningStrategy(test_config)
    # Initialize the dynamic partitioning component
    strategy.rl = DynamicPartitioning(test_config)
    strategy.rl.initialize(test_graph, test_agent_config)
    partitions = strategy.partition(test_graph)
    assert partitions is not None
    assert len(partitions) == 2
    assert compute_balance(partitions) > 0.5

def test_strategy_comparison(test_config, test_agent_config, test_graph):
    strategies = [
        SpectralPartitioningStrategy(test_config),
        DynamicPartitioning(test_config),
        HybridPartitioningStrategy(test_config)
    ]
    
    results = []
    for strategy in strategies:
        if isinstance(strategy, DynamicPartitioning):
            strategy.initialize(test_graph, test_agent_config)
        elif isinstance(strategy, HybridPartitioningStrategy):
            strategy.rl = DynamicPartitioning(test_config)
            strategy.rl.initialize(test_graph, test_agent_config)
            
        partitions = strategy.partition(test_graph)
        results.append({
            'cut_size': compute_cut_size(test_graph, partitions),
            'balance': compute_balance(partitions),
            'conductance': compute_conductance(test_graph, partitions)
        })
    
    # Compare results
    assert len(results) == 3
    balances = [r['balance'] for r in results]
    assert all(b > 0.5 for b in balances)

@pytest.mark.benchmark(group="strategies")
def test_benchmark_strategies(benchmark, test_config, test_agent_config):
    """Benchmark different partitioning strategies."""
    # Create a small test graph
    num_nodes = 20  # Reduced from 100
    edge_probability = 0.2  # Increased from 0.1 for better connectivity
    graph = nx.erdos_renyi_graph(num_nodes, edge_probability)
    adj_matrix = nx.adjacency_matrix(graph).todense()
    
    # Convert to our Graph format
    test_graph = Graph(num_nodes=num_nodes)
    test_graph.adj_matrix = torch.tensor(adj_matrix).float()
    test_graph.node_features = torch.randn(num_nodes, 4)  # 4 features per node
    
    # Modify agent config for faster testing
    test_agent_config.num_episodes = 2  # Reduced from 10
    test_agent_config.max_steps = 5     # Reduced from 10
    
    # Initialize strategies
    strategies = [
        SpectralPartitioningStrategy(test_config),
        DynamicPartitioning(test_config),
        HybridPartitioningStrategy(test_config)
    ]
    
    def run_partition():
        for strategy in strategies:
            if isinstance(strategy, DynamicPartitioning):
                strategy.initialize(test_graph, test_agent_config)
            elif isinstance(strategy, HybridPartitioningStrategy):
                strategy.rl = DynamicPartitioning(test_config)
                strategy.rl.initialize(test_graph, test_agent_config)
            strategy.partition(test_graph)
    
    # Run the benchmark
    benchmark(run_partition)

