import torch
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from ..config.system_config import PartitionConfig

@dataclass
class Partition:
    # ... existing code ...
    def to_dict(self):
        return {
            'id': self.id,
            'nodes': list(self.nodes),
            'density': self.density,
            'conductance': self.conductance,
        }

    @staticmethod
    def from_dict(data):
        return Partition(
            id=data['id'],
            nodes=set(data['nodes']),
            density=data.get('density', 0.0),
            conductance=data.get('conductance', 0.0),
        )

    """Represents a partition of nodes in the graph."""
    id: int
    nodes: Set[int] = field(default_factory=set)
    density: float = 0.0
    conductance: float = 0.0
    
    def add_node(self, node: int) -> None:
        """Add a node to the partition."""
        self.nodes.add(node)
        
    def remove_node(self, node: int) -> None:
        """Remove a node from the partition."""
        self.nodes.discard(node)
        
    def __add__(self, other: 'Partition') -> 'Partition':
        """Support addition of partitions."""
        result = Partition(id=self.id)
        result.nodes = self.nodes.union(other.nodes)
        return result
        
    def __truediv__(self, scalar: int) -> 'Partition':
        """Support division by scalar."""
        result = Partition(id=self.id)
        # For division, we'll take a subset of nodes
        node_list = list(self.nodes)
        subset_size = len(node_list) // scalar
        result.nodes = set(node_list[:subset_size])
        return result
        
    def __str__(self) -> str:
        """String representation of the partition."""
        return f"Partition(id={self.id}, nodes={len(self.nodes)}, density={self.density:.4f}, conductance={self.conductance:.4f})"
        
    def __format__(self, format_spec: str) -> str:
        """Support string formatting."""
        return self.__str__()
        
    def update_density(self, graph: 'Graph') -> None:
        """Update the partition's density based on internal edges."""
        if not self.nodes:
            self.density = 0.0
            return
            
        internal_edges = 0
        possible_edges = len(self.nodes) * (len(self.nodes) - 1) / 2
        
        for u in self.nodes:
            for v in self.nodes:
                if u < v and graph.has_edge(u, v):
                    internal_edges += 1
                    
        self.density = internal_edges / possible_edges if possible_edges > 0 else 0.0
        
    def update_conductance(self, graph: 'Graph') -> None:
        """Update the partition's conductance."""
        if not self.nodes:
            self.conductance = 0.0
            return
            
        cut_edges = 0
        total_edges = 0
        
        for u in self.nodes:
            for v in graph.get_neighbors(u):
                if v not in self.nodes:
                    cut_edges += 1
                total_edges += 1
                
        self.conductance = cut_edges / total_edges if total_edges > 0 else 0.0
        
    def __len__(self) -> int:
        """Return the number of nodes in the partition."""
        return len(self.nodes)

import json

class Graph:
    # ... existing code ...
    def is_balanced(self, tolerance: float = 0.1) -> bool:
        """Return True if all partitions are within tolerance of the mean size."""
        sizes = [len(p.nodes) for p in self.partitions.values()]
        if not sizes:
            return True
        mean_size = sum(sizes) / len(sizes)
        return all(abs(s - mean_size) / mean_size <= tolerance for s in sizes)

    def balance_partitions(self, tolerance: float = 0.1):
        """Redistribute nodes to minimize size imbalance across partitions."""
        sizes = {pid: len(p.nodes) for pid, p in self.partitions.items()}
        mean_size = sum(sizes.values()) / len(sizes) if sizes else 0
        # Flatten all nodes
        all_nodes = [n for p in self.partitions.values() for n in p.nodes]
        # Reassign nodes round-robin
        sorted_nodes = sorted(all_nodes)
        pids = list(self.partitions.keys())
        for p in self.partitions.values():
            p.nodes.clear()
        for i, n in enumerate(sorted_nodes):
            self.partitions[pids[i % len(pids)]].nodes.add(n)
        # Update metrics
        for p in self.partitions.values():
            p.update_density(self)
            p.update_conductance(self)

    def merge_partitions(self, pid1: int, pid2: int):
        """Merge two partitions into one (pid1 keeps the merged nodes, pid2 is removed)."""
        if pid1 not in self.partitions or pid2 not in self.partitions:
            raise ValueError("Invalid partition IDs")
        self.partitions[pid1].nodes.update(self.partitions[pid2].nodes)
        del self.partitions[pid2]
        self.partitions[pid1].update_density(self)
        self.partitions[pid1].update_conductance(self)

    def split_partition(self, pid: int):
        """Split a partition into two (simple even/random split)."""
        import random
        if pid not in self.partitions:
            raise ValueError("Invalid partition ID")
        nodes = list(self.partitions[pid].nodes)
        if len(nodes) < 2:
            raise ValueError("Partition too small to split")
        random.shuffle(nodes)
        half = len(nodes) // 2
        nodes1, nodes2 = set(nodes[:half]), set(nodes[half:])
        self.partitions[pid].nodes = nodes1
        new_pid = max(self.partitions.keys()) + 1
        self.partitions[new_pid] = Partition(id=new_pid, nodes=nodes2)
        self.partitions[pid].update_density(self)
        self.partitions[pid].update_conductance(self)
        self.partitions[new_pid].update_density(self)
        self.partitions[new_pid].update_conductance(self)

    # ... existing code ...
    def validate_structure(self):
        """Check for orphan nodes, symmetric adjacency, and valid node indices."""
        orphan_nodes = []
        for i in range(self.num_nodes):
            neighbors = self.get_neighbors(i)
            if isinstance(neighbors, int):  # handle 0-degree nodes
                neighbors = []
            if len(neighbors) == 0:
                orphan_nodes.append(i)
        if orphan_nodes:
            raise ValueError(f"Orphan nodes detected: {orphan_nodes}")
        # Check adjacency symmetry
        if not torch.allclose(self.adj_matrix, self.adj_matrix.T):
            raise ValueError("Adjacency matrix is not symmetric (graph must be undirected)")
        # Check node indices
        if self.adj_matrix.shape[0] != self.num_nodes:
            raise ValueError(f"Adjacency matrix shape {self.adj_matrix.shape} does not match num_nodes {self.num_nodes}")

    def validate_partitions(self):
        """Check that all nodes are assigned to exactly one partition, and no node is missing or duplicated."""
        assigned = set()
        for p in self.partitions.values():
            for n in p.nodes:
                if n in assigned:
                    raise ValueError(f"Node {n} assigned to multiple partitions.")
                assigned.add(n)
        missing = set(range(self.num_nodes)) - assigned
        if missing:
            raise ValueError(f"Nodes missing from partitions: {missing}")

    def validate(self):
        """Run all validation checks."""
        self.validate_structure()
        self.validate_partitions()
        return True

    """Represents a graph with node and edge management."""
    
    def __init__(self, num_nodes: int, edge_probability: float = 0.3, 
                 weight_range: Tuple[float, float] = (0.1, 1.0), config=None):
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability
        self.weight_range = weight_range
        self.config = config
        
        # Initialize adjacency matrix
        self.adj_matrix = torch.zeros((num_nodes, num_nodes))
        self._generate_random_graph()
        
        # Initialize node features
        self.node_features = None  # Will be initialized when needed based on config
        
        # Initialize partitions
        self.partitions: Dict[int, Partition] = {}
        
    def _generate_random_graph(self) -> None:
        """Generate a random graph with given edge probability."""
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if torch.rand(1) < self.edge_probability:
                    weight = torch.rand(1) * (self.weight_range[1] - self.weight_range[0]) + self.weight_range[0]
                    self.adj_matrix[i, j] = weight
                    self.adj_matrix[j, i] = weight
                    
    def get_edge_index(self) -> torch.Tensor:
        """Get edge indices in COO format."""
        edge_index = torch.nonzero(self.adj_matrix).t()
        return edge_index
        
    def get_edge_weights(self) -> torch.Tensor:
        """Get edge weights."""
        return self.adj_matrix[self.adj_matrix != 0]
        
    def get_node_features(self) -> torch.Tensor:
        """Get node features."""
        if self.node_features is None:
            base_feature_dim = 24  # Fixed base dimension for node features
            self.node_features = torch.randn(self.num_nodes, base_feature_dim)
        return self.node_features
        
    def has_edge(self, u: int, v: int) -> bool:
        """Check if an edge exists between nodes u and v."""
        return self.adj_matrix[u, v] > 0
        
    def get_neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        neighbors = torch.nonzero(self.adj_matrix[node]).squeeze()
        if neighbors.dim() == 0:
            return [int(neighbors.item())] if neighbors.numel() > 0 else []
        return [int(x) for x in neighbors.tolist()]
        
    def add_partition(self, partition_id: int) -> Partition:
        """Add a new empty partition."""
        if partition_id in self.partitions:
            raise ValueError(f"Partition {partition_id} already exists")
            
        partition = Partition(id=partition_id)
        self.partitions[partition_id] = partition
        return partition
        
    def move_node(self, node: int, from_partition: Optional[int], to_partition: int) -> None:
        """Move a node from one partition to another.
        
        Args:
            node: The node to move
            from_partition: Source partition ID, or None if node is not in any partition
            to_partition: Target partition ID
        """
        if to_partition not in self.partitions:
            raise ValueError("Invalid target partition ID")
        
        if from_partition is not None:
            if from_partition not in self.partitions:
                raise ValueError("Invalid source partition ID")
            self.partitions[from_partition].remove_node(node)
            # Update source partition metrics
            self.partitions[from_partition].update_density(self)
            self.partitions[from_partition].update_conductance(self)
        
        # Add to target partition and update its metrics
        self.partitions[to_partition].add_node(node)
        self.partitions[to_partition].update_density(self)
        self.partitions[to_partition].update_conductance(self)
        
    def get_partition_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get metrics for all partitions."""
        metrics = {}
        for pid, partition in self.partitions.items():
            metrics[pid] = {
                'size': len(partition),
                'density': partition.density,
                'conductance': partition.conductance
            }
        return metrics
        
    def to_dict(self):
        return {
            'num_nodes': self.num_nodes,
            'edge_probability': self.edge_probability,
            'weight_range': self.weight_range,
            'adj_matrix': self.adj_matrix.tolist(),
            'node_features': self.node_features.tolist(),
            'partitions': [p.to_dict() for p in self.partitions.values()],
        }

    @staticmethod
    def from_dict(data):
        g = Graph(
            num_nodes=data['num_nodes'],
            edge_probability=data.get('edge_probability', 0.3),
            weight_range=tuple(data.get('weight_range', (0.1, 1.0)))
        )
        g.adj_matrix = torch.tensor(data['adj_matrix'])
        g.node_features = torch.tensor(data['node_features'])
        g.partitions = {p['id']: Partition.from_dict(p) for p in data.get('partitions', [])}
        return g

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return Graph.from_dict(data)

    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph for visualization."""
        G = nx.Graph()
        edge_index = self.get_edge_index()
        edge_weights = self.get_edge_weights()
        
        for i in range(edge_index.size(1)):
            u, v = edge_index[:, i].tolist()
            G.add_edge(u, v, weight=edge_weights[i].item())
            
        return G 