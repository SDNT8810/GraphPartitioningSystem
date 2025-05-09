import torch
import torch.nn.functional as F
from typing import Dict, Set, List
import numpy as np
from .base_strategy import BasePartitioningStrategy
from ..core.graph import Graph, Partition
from ..models.gnn import GraphNeuralNetwork
from ..config.system_config import GNNConfig

class GNNBasedPartitioningStrategy(BasePartitioningStrategy):
    """Partitioning strategy using Graph Neural Networks."""
    
    def __init__(self, config, gnn_config: GNNConfig):
        super().__init__(config)
        self.gnn_config = gnn_config
        self.model = GraphNeuralNetwork(gnn_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=gnn_config.learning_rate)
        
    def partition(self, graph: Graph) -> Dict[int, Set[int]]:
        """Partition the graph using GNN predictions."""
        # Get graph data
        x = graph.get_node_features()
        edge_index = graph.get_edge_index()
        
        # Get predictions for each node
        with torch.no_grad():
            predictions = self.model(x, edge_index)
            
        # Convert predictions to partition assignments
        k = self.config.num_partitions
        assignments = torch.argmax(predictions.view(-1, k), dim=1)
        
        # Create partitions
        partitions = {}
        for i in range(k):
            partitions[i] = set(torch.where(assignments == i)[0].tolist())

        # --- Partition balancing/merging/splitting logic ---
        if hasattr(graph, 'partitions'):
            # Overwrite graph partitions with new GNN result
            for pid in list(graph.partitions.keys()):
                del graph.partitions[pid]
            for pid, nodes in partitions.items():
                graph.partitions[pid] = Partition(id=pid, nodes=nodes)
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
            partitions = {p.id: set(p.nodes) for p in graph.partitions.values()}
        # --- End partition management logic ---
        
        return partitions
        
    def evaluate(self, graph: Graph, partitions: Dict[int, Set[int]]) -> Dict[str, float]:
        """Evaluate the GNN-based partitioning."""
        metrics = {
            'balance': self._calculate_balance(partitions, graph.num_nodes),
            'cut_size': self._calculate_cut_size(graph, partitions)
        }
        
        # Calculate average conductance and density
        conductances = []
        densities = []
        for partition in partitions.values():
            conductances.append(self._calculate_conductance(graph, partition))
            densities.append(self._calculate_density(graph, partition))
            
        metrics['avg_conductance'] = np.mean(conductances)
        metrics['avg_density'] = np.mean(densities)
        
        return metrics
        
    def train(self, graph: Graph, epochs: int = 500) -> List[float]:
        """Train the GNN model on the graph."""
        losses = []
        x = graph.get_node_features()
        edge_index = graph.get_edge_index()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(x, edge_index)
            
            # Compute loss
            loss = self._compute_loss(predictions, graph)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
        return losses
        
    def _compute_loss(self, predictions: torch.Tensor, graph: Graph) -> torch.Tensor:
        """Compute the loss for GNN training."""
        # Get current partitioning
        partitions = self.partition(graph)
        
        # Compute balance loss
        balance_loss = 1 - self._calculate_balance(partitions, graph.num_nodes)
        
        # Compute cut size loss
        cut_size = self._calculate_cut_size(graph, partitions)
        cut_size_loss = cut_size / (graph.num_nodes * (graph.num_nodes - 1) / 2)
        
        # Compute conductance loss
        conductances = []
        for partition in partitions.values():
            conductances.append(self._calculate_conductance(graph, partition))
        conductance_loss = np.mean(conductances)
        
        # Combine losses
        total_loss = (
            self.config.balance_weight * balance_loss +
            self.config.cut_size_weight * cut_size_loss +
            self.config.conductance_weight * conductance_loss
        )
        
        return torch.tensor(total_loss, requires_grad=True)
        
    def save_model(self, path: str) -> None:
        """Save the GNN model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load the GNN model."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state']) 