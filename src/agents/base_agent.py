import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.graph import Graph, Partition
from ..config.system_config import AgentConfig
from ..utils import graph_metrics
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AgentState:
    """State representation for the agent"""
    node_features: torch.Tensor
    partition_sizes: torch.Tensor
    partition_densities: torch.Tensor
    graph_metrics: Dict[str, float]
    local_metrics: Optional[Dict[str, float]] = None

class BaseAgent(nn.Module):
    """Base class for graph partitioning agents"""
    
    def __init__(self, config: AgentConfig, graph: Graph, node_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.graph = graph
        self.device = torch.device(getattr(config, 'device', 'cpu') or 'cpu')
        self.node_id = node_id
        self.current_partition = None
        self.neighbors = [] if node_id is None else graph.get_neighbors(node_id)
        
        # Initialize neural network components
        self.state_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.to(self.device)
    



    def _calculate_local_metrics(self, graph: Graph) -> Dict[str, float]:

        """Calculate metrics based on local graph structure."""
        if not self.neighbors:
            return {
                'degree': 0,
                'avg_neighbor_degree': 0,
                'clustering_coefficient': 0
            }
            
        # Calculate degree
        degree = len(self.neighbors)
        
        # Calculate average neighbor degree
        neighbor_degrees = [len(graph.get_neighbors(n)) for n in self.neighbors]
        avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees) if neighbor_degrees else 0
        
        # Calculate clustering coefficient
        triangles = 0
        for i, n1 in enumerate(self.neighbors):
            for n2 in self.neighbors[i+1:]:
                if graph.has_edge(n1, n2):
                    triangles += 1
        possible_triangles = degree * (degree - 1) / 2
        clustering_coefficient = triangles / possible_triangles if possible_triangles > 0 else 0
        
        return {
            'degree': degree,
            'avg_neighbor_degree': avg_neighbor_degree,
            'clustering_coefficient': clustering_coefficient
        }
    
    def get_state(self) -> AgentState:
        """Get the current state of the environment"""
        # Get node features
        node_features = self.graph.get_node_features()
        if node_features is None:
            node_features = torch.zeros((self.graph.num_nodes, 1))
        
        # Get partition metrics
        partition_sizes = torch.tensor([
            len(p) if isinstance(p, set) else len(p.nodes)
            for p in self.graph.partitions.values()
        ], dtype=torch.float32)
        
        # Handle partition densities
        partition_densities = torch.zeros_like(partition_sizes)
        for i, p in enumerate(self.graph.partitions.values()):
            if isinstance(p, Partition):
                partition_densities[i] = p.density
            else:
                # Compute density for set-based partitions
                nodes = p if isinstance(p, set) else p.nodes
                size = len(nodes)
                if size > 1:
                    edges = sum(1 for n1 in nodes for n2 in nodes 
                               if n1 < n2 and self.graph.adj_matrix[n1, n2] > 0)
                    partition_densities[i] = 2.0 * edges / (size * (size - 1))
        
        # Compute graph metrics
        graph_metrics_dict = {
            'partition_sizes': partition_sizes,
            'partition_densities': partition_densities,
            'balance': graph_metrics.compute_balance(self.graph.partitions),
            'cut_size': graph_metrics.compute_cut_size(self.graph, self.graph.partitions),
            'conductance': graph_metrics.compute_conductance(self.graph, self.graph.partitions),
            'node_id': self.node_id,
            'num_nodes': self.graph.num_nodes,
            'num_partitions': len(self.graph.partitions)
        }
        
        # Get local metrics if this is a local agent
        local_metrics = None
        if self.node_id is not None:
            local_metrics = self._calculate_local_metrics(self.graph)
        
        return AgentState(
            node_features=node_features,
            partition_sizes=partition_sizes,
            partition_densities=partition_densities,
            graph_metrics=graph_metrics_dict,
            local_metrics=local_metrics
        )
    
    def encode_state(self, state: AgentState) -> torch.Tensor:
        """Encode the state into a feature vector"""
        # Get node features (average across nodes)
        node_features = state.node_features.mean(dim=0)  # Shape: [feature_dim]
        
        # Get partition features
        partition_sizes = F.normalize(state.partition_sizes.unsqueeze(0), dim=1).squeeze(0)  # Shape: [num_partitions]
        partition_densities = F.normalize(state.partition_densities.unsqueeze(0), dim=1).squeeze(0)  # Shape: [num_partitions]
        
        # Get global metrics
        global_metrics = torch.tensor([
            state.graph_metrics['balance'],
            state.graph_metrics['cut_size'],
            state.graph_metrics['conductance']
        ], dtype=torch.float32, device=self.device)
        
        # Combine all features into state vector
        features = [node_features, partition_sizes, partition_densities, global_metrics]
        
        # Add local metrics if available
        if state.local_metrics is not None:
            local_features = torch.tensor([
                state.local_metrics['degree'],
                state.local_metrics['avg_neighbor_degree'],
                state.local_metrics['clustering_coefficient']
            ], dtype=torch.float32, device=self.device)
            features.append(local_features)
        
        # Concatenate features
        state_features = torch.cat(features)
        
        # Ensure we have the expected feature dimension
        if state_features.size(0) != self.config.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.config.feature_dim}, "
                f"but got {state_features.size(0)}. Check feature extraction logic."
            )
        
        # Encode state to desired dimension
        encoded_state = self.state_encoder(state_features.unsqueeze(0))  # Add batch dimension for encoder
        
        return encoded_state  # Shape: [1, state_dim]
    
    def forward(self, state: AgentState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        state_features = self.encode_state(state)
        action_logits = self.action_head(state_features)
        value = self.value_head(state_features)
        return action_logits, value
    
    def select_action(self, state: AgentState, epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        """Select an action using epsilon-greedy policy"""
        with torch.no_grad():
            action_logits, value = self.forward(state)
            
        if np.random.random() < epsilon:
            action = np.random.randint(self.config.action_dim)
        else:
            action = action_logits.argmax().item()
            
        return action, value
    
    def compute_loss(
        self,
        states: List[AgentState],
        actions: List[int],
        rewards: List[float],
        next_states: List[AgentState],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for a batch of transitions"""
        # Convert lists to tensors
        states_tensor = torch.stack([self.encode_state(s) for s in states])
        next_states_tensor = torch.stack([self.encode_state(s) for s in next_states])
        actions_tensor = torch.tensor(actions, device=self.device)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        dones_tensor = torch.tensor(dones, device=self.device)
        
        # Get current Q-values
        current_q_values = self.action_head(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1))
        
        # Get next Q-values
        with torch.no_grad():
            next_q_values = self.action_head(next_states_tensor)
            next_q_values = next_q_values.max(1)[0]
            next_q_values[dones_tensor] = 0.0
            target_q_values = rewards_tensor + self.config.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }
        
        return loss, metrics
    
    def update(self, loss: torch.Tensor) -> None:
        """Update the network parameters"""
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
    
    def communicate(self, other_agent: 'BaseAgent') -> Dict:
        """Exchange information with another agent."""
        return {
            'node_id': self.node_id,
            'current_partition': self.current_partition,
            'local_metrics': self._calculate_local_metrics(self.graph) if self.node_id is not None else None
        }

    def initialize(self, graph: Graph) -> None:
        """Initialize the agent with the graph structure."""
        self.neighbors = graph.get_neighbors(self.node_id)
        self._initialize_state(graph)
        
    def _initialize_state(self, graph: Graph) -> None:
        """Initialize the agent's local state."""
        self.local_state = {
            'node_features': graph.get_node_features()[self.node_id],
            'neighbor_features': graph.get_node_features()[self.neighbors],
            'partition_metrics': graph.get_partition_metrics(),
            'local_metrics': self._calculate_local_metrics(graph)
        }
        
    def observe(self, graph: Graph) -> torch.Tensor:
        """Observe the current state of the environment."""
        # Get node features
        node_features = graph.get_node_features()[self.node_id]
        
        # Get partition metrics
        partition_metrics = graph.get_partition_metrics()
        
        # Get local metrics
        local_metrics = self._calculate_local_metrics(graph)
        
        # Combine all features into a state vector
        state_vector = torch.cat([
            node_features,
            torch.tensor([
                local_metrics['degree'],
                local_metrics['avg_neighbor_degree'],
                local_metrics['clustering_coefficient']
            ]),
            torch.tensor([
                partition_metrics.get(self.current_partition, {}).get('density', 0),
                partition_metrics.get(self.current_partition, {}).get('conductance', 0)
            ])
        ])
        
        return state_vector
        
    def act(self, state: torch.Tensor, graph: Graph) -> Tuple[int, float]:
        """Choose an action based on the current state."""
        raise NotImplementedError("Subclasses must implement act()")
        
    def update_partition(self, new_partition: int) -> None:
        """Update the agent's current partition."""
        self.current_partition = new_partition 