from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple
import torch
import numpy as np
from ..core.graph import Graph, Partition
from ..config.system_config import PartitionConfig

class BasePartitioningStrategy(ABC):
    """Base class for all partitioning strategies."""
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        
    @abstractmethod
    def partition(self, graph: Graph) -> Dict[int, Set[int]]:
        """Partition the graph into subgraphs.
        
        Args:
            graph: The graph to partition
            
        Returns:
            Dictionary mapping partition IDs to sets of node IDs
        """
        pass
        
    @abstractmethod
    def evaluate(self, graph: Graph, partitions: Dict[int, Set[int]]) -> Dict[str, float]:
        """Evaluate the quality of a partitioning.
        
        Args:
            graph: The original graph
            partitions: The partitioning to evaluate
            
        Returns:
            Dictionary of metrics (balance, conductance, etc.)
        """
        pass
        
    def _calculate_balance(self, partitions: Dict[int, Set[int]], total_nodes: int) -> float:
        """Calculate the balance score of a partitioning."""
        sizes = [len(nodes) for nodes in partitions.values()]
        if not sizes:
            return 0.0
        return 1 - (max(sizes) - min(sizes)) / total_nodes
        
    def _calculate_conductance(self, graph: Graph, partition: Set[int]) -> float:
        """Calculate the conductance of a partition."""
        if not partition:
            return 0.0
            
        cut_edges = 0
        total_edges = 0
        
        for node in partition:
            for neighbor in graph.get_neighbors(node):
                if neighbor not in partition:
                    cut_edges += 1
                total_edges += 1
                
        return cut_edges / total_edges if total_edges > 0 else 0.0
        
    def _calculate_density(self, graph: Graph, partition: Set[int]) -> float:
        """Calculate the density of a partition."""
        if not partition:
            return 0.0
            
        internal_edges = 0
        possible_edges = len(partition) * (len(partition) - 1) / 2
        
        for i, u in enumerate(partition):
            for v in list(partition)[i+1:]:
                if graph.has_edge(u, v):
                    internal_edges += 1
                    
        return internal_edges / possible_edges if possible_edges > 0 else 0.0
        
    def _calculate_cut_size(self, graph: Graph, partitions: Dict[int, Set[int]]) -> int:
        """Calculate the total cut size of a partitioning."""
        cut_size = 0
        for partition_id, nodes in partitions.items():
            for node in nodes:
                for neighbor in graph.get_neighbors(node):
                    if neighbor not in nodes:
                        cut_size += 1
        return cut_size // 2  # Each edge is counted twice 