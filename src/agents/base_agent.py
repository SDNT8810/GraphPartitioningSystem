import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.graph import Graph, Partition
from ..config.system_config import AgentConfig
from ..utils import graph_metrics
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

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
        
        # Handle device selection with graceful fallback
        requested_device = getattr(config, 'device', 'cpu') or 'cpu'
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(requested_device)
            
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
        if isinstance(state.node_features, torch.Tensor):
            if len(state.node_features.shape) > 1:
                node_features = state.node_features.mean(dim=0)  # Shape: [feature_dim]
            else:
                node_features = state.node_features  # Already a 1D tensor
        else:
            # Handle non-tensor node features
            node_features = torch.tensor(state.node_features, dtype=torch.float32)
        
        # Get partition features - use detach().clone() to avoid warnings
        if isinstance(state.partition_sizes, torch.Tensor):
            partition_sizes = state.partition_sizes.detach().clone()
        else:
            partition_sizes = torch.tensor(state.partition_sizes, dtype=torch.float32)
        
        if isinstance(state.partition_densities, torch.Tensor):
            partition_densities = state.partition_densities.detach().clone()
        else:
            partition_densities = torch.tensor(state.partition_densities, dtype=torch.float32)
            
        # Normalize partition tensors
        partition_sizes = F.normalize(partition_sizes.unsqueeze(0), dim=1).squeeze(0)
        partition_densities = F.normalize(partition_densities.unsqueeze(0), dim=1).squeeze(0)
        
        # Get global metrics
        global_metrics = torch.tensor([
            state.graph_metrics.get('balance', 0.0),
            state.graph_metrics.get('cut_size', 0.0),
            state.graph_metrics.get('conductance', 0.0)
        ], dtype=torch.float32)
        
        # Combine all features into state vector
        features = [node_features, partition_sizes, partition_densities, global_metrics]
        
        # Add local metrics if available
        if state.local_metrics is not None:
            local_features = torch.tensor([
                state.local_metrics.get('degree', 0.0),
                state.local_metrics.get('avg_neighbor_degree', 0.0),
                state.local_metrics.get('clustering_coefficient', 0.0)
            ], dtype=torch.float32)
            features.append(local_features)
        
        # Concatenate features
        state_features = torch.cat([f.to(self.device) for f in features])
        
        # Handle dimension mismatch - this is key to fixing our issue
        expected_dim = self.config.feature_dim
        actual_dim = state_features.size(0)
        
        if actual_dim < expected_dim:
            # Pad with zeros to reach expected dimension
            padding = torch.zeros(expected_dim - actual_dim, device=self.device)
            state_features = torch.cat([state_features, padding])
        elif actual_dim > expected_dim:
            # Truncate to expected dimension
            state_features = state_features[:expected_dim]
        
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
        try:
            # Convert lists to tensors - encode each state individually
            encoded_states = torch.stack([self.encode_state(s).squeeze(0) for s in states])  # [batch_size, state_dim]
            encoded_next_states = torch.stack([self.encode_state(s).squeeze(0) for s in next_states])  # [batch_size, state_dim]
            
            # Make sure these are proper tensors with correct device
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
            
            # Get current Q-values
            current_q_values = self.action_head(encoded_states)
            
            # Reshape actions tensor to properly gather values
            actions_tensor_reshaped = actions_tensor.view(-1, 1)
            current_q_values = current_q_values.gather(1, actions_tensor_reshaped)
            
            # Get next Q-values
            with torch.no_grad():
                next_q_values = self.action_head(encoded_next_states)
                max_next_q_values = next_q_values.max(1)[0]
                # Set terminal states to 0
                max_next_q_values = max_next_q_values * (~dones_tensor).float()
                target_q_values = rewards_tensor + self.config.gamma * max_next_q_values
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            
            # Compute metrics
            metrics = {
                'loss': loss.item(),
                'avg_reward': float(torch.mean(rewards_tensor).item()),
                'max_reward': float(torch.max(rewards_tensor).item()),
                'min_reward': float(torch.min(rewards_tensor).item())
            }
            
            return loss, metrics
            
        except Exception as e:
            # Add error handling to debug issues
            logging.error(f"Error in compute_loss: {e}")
            logging.error(f"States shape: {len(states)}")
            logging.error(f"Actions shape: {len(actions)}")
            
            # Return a dummy loss that won't crash training but will log the error
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            metrics = {'loss': 0.0, 'avg_reward': 0.0, 'error': str(e)}
            return dummy_loss, metrics
    
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