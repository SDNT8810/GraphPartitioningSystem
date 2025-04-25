import torch
import numpy as np
from typing import *
from scipy.sparse.linalg import *
from scipy.sparse import *
from .base_strategy import *
from ..core.graph import *
from ..utils.graph_metrics import compute_balance, compute_cut_size, compute_conductance

class SpectralPartitioningStrategy(BasePartitioningStrategy):
    """Spectral partitioning strategy using Laplacian eigenvectors."""
    
    def __init__(self, config):
        super().__init__(config)
        self.use_laplacian = config.use_laplacian
        
    def partition(self, graph: Graph) -> Dict[int, Partition]:
        """Partition the graph using spectral clustering."""
        # Pass config to graph
        graph.config = self.config
        # Convert graph to sparse matrix
        adj_matrix = self._graph_to_sparse(graph)
        
        if self.use_laplacian:
            # Compute Laplacian matrix
            degree_matrix = np.diag(adj_matrix.sum(axis=1))
            laplacian = degree_matrix - adj_matrix
        else:
            laplacian = adj_matrix
            
        # Compute k smallest eigenvectors
        k = self.config.num_partitions
        try:
            eigenvalues, eigenvectors = eigsh(laplacian, k=k, which='SM')
        except:
            # Fallback to random partitioning if spectral clustering fails
            return self._random_partitioning(graph)
            
        # Use k-means on eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(eigenvectors)
        
        # Create partitions
        partitions = {}
        for i in range(k):
            # Convert np.int64 to Python int for all node indices
            nodes = set(int(idx) for idx in np.where(labels == i)[0])
            partitions[i] = Partition(id=i, nodes=nodes)

        # --- Partition balancing/merging/splitting logic ---
        # Assign partitions to graph object for management
        if hasattr(graph, 'partitions'):
            # Overwrite graph partitions with new spectral result
            for pid in list(graph.partitions.keys()):
                del graph.partitions[pid]
            for pid, partition in partitions.items():
                graph.partitions[pid] = partition
            # Remove the dummy partition if it was added
            if 0 in graph.partitions and 0 not in partitions:
                del graph.partitions[0]
            # Balance if needed
            if hasattr(graph, 'is_balanced') and hasattr(graph, 'balance_partitions'):
                if not graph.is_balanced():
                    graph.balance_partitions()
            # Merge small partitions
            min_size = min(len(p.nodes) for p in graph.partitions.values())
            if min_size < 2 and len(graph.partitions) > 1 and hasattr(graph, 'merge_partitions'):
                small_pid = [p.id for p in graph.partitions.values() if len(p.nodes) == min_size][0]
                other_pid = [p.id for p in graph.partitions if p != small_pid][0]
                graph.merge_partitions(small_pid, other_pid)
            # Split large partitions
            max_size = max(len(p.nodes) for p in graph.partitions.values())
            if max_size > 2 * (len(graph.partitions) and graph.num_nodes // len(graph.partitions)):
                large_pid = [p.id for p in graph.partitions.values() if len(p.nodes) == max_size][0]
                if hasattr(graph, 'split_partition'):
                    graph.split_partition(large_pid)
            # Rebuild partitions dict for return
            partitions = {p.id: p for p in graph.partitions.values()}
        # --- End partition management logic ---
        return partitions
        
    def evaluate(self, graph: Graph, partitions: Dict[int, Partition]) -> Dict[str, float]:
        """Evaluate the spectral partitioning."""
        metrics = {
            'balance': compute_balance(partitions),
            'cut_size': compute_cut_size(graph, partitions)
        }
        
        # Calculate conductance
        metrics['conductance'] = compute_conductance(graph, partitions)
        
        return metrics
        
    def _graph_to_sparse(self, graph: Graph) -> np.ndarray:
        """Convert graph to sparse adjacency matrix."""
        n = graph.num_nodes
        adj = np.zeros((n, n))
        
        for i in range(n):
            neighbors = graph.get_neighbors(i)
            if isinstance(neighbors, int):
                neighbors = [neighbors]
            for j in neighbors:
                adj[i, j] = 1
                
        return adj
        
    def _random_partitioning(self, graph: Graph) -> Dict[int, Partition]:
        """Fallback to random partitioning if spectral clustering fails."""
        nodes = list(range(graph.num_nodes))
        np.random.shuffle(nodes)
        
        partitions = {}
        k = self.config.num_partitions
        partition_size = graph.num_nodes // k
        
        for i in range(k):
            start = i * partition_size
            end = start + partition_size if i < k - 1 else graph.num_nodes
            partition = Partition(id=i)
            partition.nodes = set(nodes[start:end])
            partitions[i] = partition
            
        return partitions