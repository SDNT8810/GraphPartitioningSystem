import torch
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

class QNetwork(nn.Module):
    """Q-network for the local agent."""
    
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
    """Agent that makes local decisions for graph partitioning"""
    
    def __init__(self, config: AgentConfig, graph: Graph, node_id: int):
        super().__init__(config, graph, node_id)
        self.memory: List[Tuple[AgentState, int, float, AgentState, bool]] = []
        self.epsilon = config.epsilon_start
        
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
        
        return partition_idx, value.item()
    
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
    
    def train_step(self) -> Dict[str, float]:
        """Perform a single training step"""
        if len(self.memory) < self.config.batch_size:
            return {'loss': 0.0, 'avg_reward': 0.0}
        
        # Sample batch
        batch = self.memory[-self.config.batch_size:]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Compute loss and update
        loss, metrics = self.compute_loss(states, actions, rewards, next_states, dones)
        self.update(loss)
        
        return metrics
    
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
            
            # Get neighbor partition
            partition = self.graph.get_node_partition(neighbor)
            if partition is not None:
                neighbor_partitions.append(partition)
            else:
                neighbor_partitions.append(-1)  # No partition
        
        return {
            'features': torch.stack(neighbor_features) if neighbor_features else torch.tensor([]),
            'partitions': torch.tensor(neighbor_partitions, dtype=torch.long)
        }
    
    def evaluate_partition(self, partition_idx: int) -> float:
        """Evaluate the quality of a partition for the current node"""
        if partition_idx < 0 or partition_idx >= len(self.graph.partitions):
            return float('-inf')
        
        # Get current metrics
        current_metrics = self.graph.get_partition_metrics()
        current_density = current_metrics.get('density', 0.0)
        current_balance = current_metrics.get('balance', 0.0)
        
        # Simulate moving to new partition
        self.graph.move_node(self.node_id, partition_idx)
        new_metrics = self.graph.get_partition_metrics()
        new_density = new_metrics.get('density', 0.0)
        new_balance = new_metrics.get('balance', 0.0)
        
        # Calculate score based on improvement
        density_improvement = new_density - current_density
        balance_improvement = new_balance - current_balance
        
        # Weight the improvements based on config
        score = (
            self.config.density_weight * density_improvement +
            self.config.balance_weight * balance_improvement
        )
        
        # Revert the simulation
        self.graph.revert_last_move()
        
        return score
    
    def update_partition(self, new_partition: int) -> None:
        """Update the agent's current partition"""
        if new_partition != self.current_partition:
            self.current_partition = new_partition
            self.graph.move_node(self.node_id, new_partition)
        
    def act(self, state: torch.Tensor, graph: Graph) -> Tuple[int, float]:
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: choose random partition
            available_partitions = list(graph.partitions.keys())
            action = random.choice(available_partitions)
            q_value = self.q_network(state)[action].item()
        else:
            # Exploit: choose best partition
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
                q_value = q_values[action].item()
                
        return action, q_value
        
    def update(self, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """Update the Q-network using experience replay."""
        # Store experience in memory
        self.memory.append((self.local_state, reward, next_state, done))
        
        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        
        # Sample random batch from memory
        if len(self.memory) < self.config.batch_size:
            return
            
        batch = random.sample(self.memory, self.config.batch_size)
        states, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones)
        
        # Compute Q-values
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.config.gamma * max_next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if len(self.memory) % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def get_partition_preference(self, graph: Graph) -> Dict[int, float]:
        """Get Q-values for each partition as preference scores."""
        state = self.observe(graph)
        with torch.no_grad():
            q_values = self.q_network(state)
            preferences = {i: q_values[i].item() for i in range(self.output_dim)}
        return preferences
        
    def save_checkpoint(self, path: str) -> None:
        """Save the agent's state."""
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load_checkpoint(self, path: str) -> None:
        """Load the agent's state."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon'] 