import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import random
from .base_agent import BaseAgent, AgentState
from ..core.graph import Graph
from ..config.system_config import AgentConfig
import torch.nn.functional as F
import torch

class AttentionQNetwork(nn.Module):
    """Enhanced Q-network with attention mechanism for better node relationship modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Feature embedding layers
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced processing layers with residual connections
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(3)
        ])
        
        # Final output layer with proper initialization
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Feature embedding
        embedded = self.feature_embed(x)
        
        # Self-attention (treating different features as sequence elements)
        attended, _ = self.attention(embedded, embedded, embedded)
        
        # Residual connection
        x_processed = embedded + attended
        
        # Process through layers with residual connections
        for layer in self.processing_layers:
            residual = x_processed
            x_processed = layer(x_processed) + residual
        
        # Final output
        output = self.output_layer(x_processed)
        
        # Return single sample if input was single sample
        if output.size(0) == 1:
            output = output.squeeze(0)
            
        return output

class QNetwork(nn.Module):
    """Standard Q-network for the local agent (fallback)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class LocalAgent(BaseAgent):
    """Agent that makes local decisions for graph partitioning with enhanced learning capabilities"""
    
    def __init__(self, config: AgentConfig, graph: Graph, node_id: int):
        super().__init__(config, graph, node_id)
        self.memory: List[Tuple[AgentState, int, float, AgentState, bool]] = []
        self.epsilon = config.epsilon_start
        
        # Enhanced Q-network with attention if enabled
        if getattr(config, 'use_attention', False):
            self.q_network = AttentionQNetwork(
                input_dim=config.feature_dim + config.state_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.action_dim,
                num_heads=getattr(config, 'num_heads', 4),
                dropout=getattr(config, 'dropout', 0.1)
            )
        else:
            self.q_network = QNetwork(
                input_dim=config.feature_dim + config.state_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.action_dim
            )
            
        # Target network for stable learning
        if getattr(config, 'use_attention', False):
            self.target_network = AttentionQNetwork(
                input_dim=config.feature_dim + config.state_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.action_dim,
                num_heads=getattr(config, 'num_heads', 4),
                dropout=getattr(config, 'dropout', 0.1)
            )
        else:
            self.target_network = QNetwork(
                input_dim=config.feature_dim + config.state_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.action_dim
            )
            
        # Copy weights from main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Enhanced optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=getattr(config, 'lr_step_size', 100), 
            gamma=getattr(config, 'lr_gamma', 0.95)
        )
        
        # Training metrics
        self.training_step = 0
        self.validation_scores = []
        self.best_validation_score = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = 20
        
    def select_partition(self, state: AgentState) -> Tuple[int, float]:
        """Select a partition to join based on current state"""
        action, value = self.select_action(state, self.epsilon)
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        # Map action to partition index
        num_partitions = len(state.partition_sizes)
        partition_idx = action % num_partitions
        
        return partition_idx, value.item() if hasattr(value, 'item') else float(value)
    
    def store_transition(
        self,
        state: AgentState,
        action: int,
        reward: float,
        next_state: AgentState,
        done: bool
    ) -> None:
        """Store a transition in the agent's memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.config.memory_size:
            self.memory.pop(0)
    
    def _state_to_tensor(self, state: AgentState) -> torch.Tensor:
        """Convert AgentState to tensor for neural network input."""
        # Handle node features - flatten if 2D
        node_features = state.node_features
        if node_features.dim() > 1:
            node_features = node_features.flatten()
        
        # Create a compact state representation
        graph_metrics = torch.tensor([
            state.graph_metrics.get('cut_size', 0.0),
            state.graph_metrics.get('balance', 0.0),
            state.graph_metrics.get('conductance', 0.0)
        ], dtype=torch.float32)
        
        # Concatenate partition sizes and densities (pad/truncate if needed)
        partition_info = torch.cat([
            state.partition_sizes[:4] if len(state.partition_sizes) >= 4 else torch.cat([
                state.partition_sizes, 
                torch.zeros(4 - len(state.partition_sizes))
            ]),
            state.partition_densities[:4] if len(state.partition_densities) >= 4 else torch.cat([
                state.partition_densities, 
                torch.zeros(4 - len(state.partition_densities))
            ])
        ])
        
        # Combine all features (ensure all are 1D)
        combined_features = torch.cat([
            node_features.flatten(),
            graph_metrics.flatten(),
            partition_info.flatten()
        ])
        
        # Ensure correct dimension
        expected_dim = self.config.feature_dim + self.config.state_dim
        if len(combined_features) < expected_dim:
            # Pad with zeros
            padding = torch.zeros(expected_dim - len(combined_features))
            combined_features = torch.cat([combined_features, padding])
        elif len(combined_features) > expected_dim:
            # Truncate
            combined_features = combined_features[:expected_dim]
        
        return combined_features
    
    def train_step(self) -> Dict[str, float]:
        """Perform a single training step with enhanced learning"""
        if len(self.memory) < self.config.batch_size:
            return {'loss': 0.0, 'avg_reward': 0.0, 'learning_rate': self.optimizer.param_groups[0]['lr']}
        
        # Sample batch
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert states to tensors
        state_tensors = torch.stack([self._state_to_tensor(s) for s in states])
        next_state_tensors = torch.stack([self._state_to_tensor(s) for s in next_states])
        
        # Convert to tensors
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)
        
        # Compute current Q-values
        current_q_values = self.q_network(state_tensors)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor.float()) * self.config.gamma * max_next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        self.training_step += 1
        
        # Update learning rate
        if self.training_step % getattr(self.config, 'lr_step_size', 100) == 0:
            self.scheduler.step()
        
        # Update target network
        if self.training_step % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Validation and early stopping check
        validation_score = self._compute_validation_score()
        self.validation_scores.append(validation_score)
        
        if validation_score > self.best_validation_score:
            self.best_validation_score = validation_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return {
            'loss': loss.item(),
            'avg_reward': rewards_tensor.mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'validation_score': validation_score,
            'early_stopping_patience': self.patience_counter
        }
    
    def _compute_validation_score(self) -> float:
        """Compute validation score for early stopping."""
        if len(self.memory) < 100:
            return 0.0
        
        # Use recent rewards as validation metric
        recent_rewards = [t[2] for t in self.memory[-100:]]
        return float(np.mean(recent_rewards))
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.early_stopping_patience
    
    def get_neighbor_info(self) -> Dict[str, torch.Tensor]:
        """Get information about neighboring nodes"""
        neighbor_features = []
        neighbor_partitions = []
        
        for neighbor in self.neighbors:
            # Get neighbor features
            node_features = self.graph.get_node_features()
            if node_features is not None:
                neighbor_features.append(node_features[neighbor])
            else:
                neighbor_features.append(torch.zeros(self.config.state_dim))
            
            # Get neighbor partition (simple heuristic - find which partition contains the neighbor)
            partition_id = -1
            if hasattr(self.graph, 'partitions'):
                for pid, partition in self.graph.partitions.items():
                    if hasattr(partition, 'nodes') and neighbor in partition.nodes:
                        partition_id = pid
                        break
            
            neighbor_partitions.append(partition_id)
        
        return {
            'features': torch.stack(neighbor_features) if neighbor_features else torch.tensor([]),
            'partitions': torch.tensor(neighbor_partitions, dtype=torch.long)
        }
    
    def evaluate_partition(self, partition_idx: int) -> float:
        """Evaluate the quality of a partition for the current node"""
        if partition_idx < 0 or not hasattr(self.graph, 'partitions'):
            return float('-inf')
        
        partitions = list(self.graph.partitions.keys())
        if partition_idx >= len(partitions):
            return float('-inf')
        
        # Simple heuristic: prefer partitions with more neighbors
        target_partition_id = partitions[partition_idx]
        
        if hasattr(self.graph, 'partitions') and target_partition_id in self.graph.partitions:
            target_partition = self.graph.partitions[target_partition_id]
            if hasattr(target_partition, 'nodes'):
                # Count neighbors in target partition
                neighbor_count = sum(1 for neighbor in self.neighbors 
                                   if neighbor in target_partition.nodes)
                
                # Prefer partitions with more neighbors (local connectivity)
                partition_size = len(target_partition.nodes)
                balance_score = 1.0 / (1.0 + abs(partition_size - self.graph.num_nodes / len(self.graph.partitions)))
                
                return neighbor_count * 0.7 + balance_score * 0.3
        
        return 0.0
        
    def act(self, state: torch.Tensor, graph: Graph) -> Tuple[int, float]:
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: choose random partition
            available_partitions = list(range(self.config.action_dim))
            action = random.choice(available_partitions)
            with torch.no_grad():
                q_values = self.q_network(state)
                q_value = q_values[action].item()
        else:
            # Exploit: choose best partition
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
                q_value = q_values[action].item()
                
        return action, q_value
        
    def get_partition_preference(self, graph: Graph) -> Dict[int, float]:
        """Get Q-values for each partition as preference scores."""
        # Create a simple state representation
        node_features = graph.get_node_features()[self.node_id] if hasattr(graph, 'get_node_features') else torch.zeros(self.config.feature_dim)
        
        # Simple state for preference computation
        simple_state = torch.cat([
            node_features,
            torch.zeros(self.config.state_dim)  # Placeholder for state features
        ])
        
        with torch.no_grad():
            q_values = self.q_network(simple_state)
            preferences = {i: q_values[i].item() for i in range(min(len(q_values), self.config.action_dim))}
        return preferences
        
    def save_checkpoint(self, path: str) -> None:
        """Save the agent's state."""
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'best_validation_score': self.best_validation_score,
            'patience_counter': self.patience_counter
        }, path)
        
    def load_checkpoint(self, path: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint.get('training_step', 0)
        self.best_validation_score = checkpoint.get('best_validation_score', float('-inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
